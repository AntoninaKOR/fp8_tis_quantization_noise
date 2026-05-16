"""
Main training script.

Supports all three experiment jobs:
  Job 1 – BF16 baseline              (gsm8k_bf16_baseline.yaml)
  Job 2 – FlashRL FP8 rollout + TIS  (gsm8k_flashrl_fp8_tis.yaml)
  Job 3 – FP8 + TIS + adaptive noise (gsm8k_flashrl_fp8_tis_quant_noise.yaml)

Usage:
  python train.py --config configs/gsm8k_bf16_baseline.yaml
  python train.py --config configs/gsm8k_flashrl_fp8_tis.yaml
  python train.py --config configs/gsm8k_flashrl_fp8_tis_quant_noise.yaml

  # Override any YAML key from the CLI:
  python train.py --config configs/gsm8k_bf16_baseline.yaml \
      --override training.max_steps=500 model.name_or_path=Qwen/Qwen2.5-3B-Instruct

  # Quick smoke test (small batch, few steps):
  python train.py --config configs/gsm8k_bf16_baseline.yaml --smoke
"""

from __future__ import annotations

import argparse
import os
import random
import time
import warnings
from pathlib import Path
from typing import Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.gsm8k import get_gsm8k_dataloader, gsm8k_reward_batch, apply_chat_template
from trainer.grpo_loss import compute_grpo_advantages, grpo_loss_with_tis
from trainer.logprob_recompute import recompute_train_logprobs_with_grad
from trainer.metrics import (
    MetricsLogger,
    gsm8k_metrics,
    response_length_metrics,
    memory_metrics,
    throughput_metrics,
    Timer,
)
from trainer.quant_noise import QuantNoiseConfig


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply CLI key=value overrides to a nested config dict."""
    for override in overrides:
        key, _, value = override.partition("=")
        keys = key.split(".")
        node = config
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        # Try to coerce value to int/float/bool before storing as string
        for coerce in (int, float):
            try:
                value = coerce(value)
                break
            except (ValueError, TypeError):
                pass
        if value == "true":
            value = True
        elif value == "false":
            value = False
        node[keys[-1]] = value
    return config


# ---------------------------------------------------------------------------
# Tokenisation / batch prep
# ---------------------------------------------------------------------------

def tokenize_batch(
    tokenizer,
    prompts: list[str],
    responses: list[str],
    max_prompt_length: int,
    max_response_length: int,
    device: torch.device,
):
    """
    Tokenize prompt+response pairs and build input_ids, attention_mask,
    response_mask tensors.
    """
    prompt_ids = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_prompt_length,
    ).to(device)

    response_ids = tokenizer(
        responses,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_response_length,
        add_special_tokens=False,
    ).to(device)

    input_ids = torch.cat([prompt_ids.input_ids, response_ids.input_ids], dim=1)
    attention_mask = torch.cat(
        [prompt_ids.attention_mask, response_ids.attention_mask], dim=1
    )

    # response_mask: 1 only for response tokens that are not padding
    prompt_len = prompt_ids.input_ids.shape[1]
    seq_len = input_ids.shape[1]
    response_mask = torch.zeros_like(attention_mask)
    response_mask[:, prompt_len:] = response_ids.attention_mask

    return input_ids, attention_mask, response_mask


# ---------------------------------------------------------------------------
# BF16 rollout (used when backend != flashrl)
# ---------------------------------------------------------------------------

@torch.no_grad()
def bf16_rollout(
    model,
    tokenizer,
    prompts: list[str],
    max_prompt_length: int,
    max_response_length: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    group_size: int = 1,
    autocast_dtype=None,
):
    """
    Generate responses using the training model.

    autocast_dtype: if set (e.g. torch.float16 on MPS), wraps generate() in
    torch.autocast for faster inference while keeping weights in float32.
    """
    model_device = next(model.parameters()).device

    prompt_ids = tokenizer(
        prompts * group_size,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    ).to(model_device)

    autocast_ctx = (
        torch.autocast(device_type=model_device.type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else torch.autocast(device_type=model_device.type, enabled=False)
    )

    with autocast_ctx:
        out = model.generate(
            **prompt_ids,
            max_new_tokens=max_response_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    response_ids = out[:, prompt_ids.input_ids.shape[1]:]
    decoded = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    return decoded, response_ids.to(device), prompt_ids.input_ids.to(device)


# ---------------------------------------------------------------------------
# FlashRL FP8 rollout stub
# ---------------------------------------------------------------------------

def get_rollout_worker(config: dict, model_path: str, tokenizer_path: str):
    """
    Return a rollout worker.  If backend == 'flashrl', use FlashRLRolloutWorker.
    Otherwise fall back to a BF16-model wrapper (for the baseline job).
    """
    backend = config.get("inference", {}).get("backend", "hf")
    if backend == "flashrl":
        try:
            from rollout.flashrl_worker import FlashRLRolloutWorker
            return FlashRLRolloutWorker(
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                fp8=config["inference"].get("quantization", "fp8") == "fp8",
                max_prompt_length=config["inference"].get("max_prompt_length", 256),
                max_response_length=config["inference"].get("max_response_length", 2048),
                temperature=config["inference"].get("temperature", 1.0),
                top_p=config["inference"].get("top_p", 1.0),
                seed=config.get("training", {}).get("seed", 42),
            )
        except ImportError:
            warnings.warn(
                "FlashRL not installed. Falling back to BF16 rollout.",
                RuntimeWarning,
            )
    return None   # caller uses bf16_rollout() directly


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: dict, smoke: bool = False):
    # ---- device / dtype ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    train_dtype = dtype_map.get(config["precision"]["train_dtype"], torch.bfloat16)

    # ---- model + tokeniser ----
    model_path = config["model"]["name_or_path"]
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # On MPS: keep weights in float32 for stable training.
    # float16 weights + Adam → NaN gradients within 1-2 steps on MPS.
    # Rollout generation uses autocast(float16) for speed (see bf16_rollout).
    rollout_autocast_dtype = None
    if device.type == "mps":
        if train_dtype in (torch.bfloat16, torch.float16):
            print("Note: MPS detected — training weights kept in float32 for stability. "
                  "Rollout will use autocast(float16) for speed.")
        train_dtype = torch.float32
        rollout_autocast_dtype = torch.float16
    elif device.type == "cuda" and train_dtype == torch.bfloat16:
        rollout_autocast_dtype = torch.bfloat16  # autocast on CUDA too (optional)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=train_dtype,
        device_map={"": device},
    )
    if config.get("training", {}).get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled (-30% memory, +20% time)", flush=True)
    model.train()

    # ---- optimizer ----
    t_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(t_cfg.get("learning_rate", 1e-6)),
        betas=(
            float(t_cfg.get("adam_beta1", 0.9)),
            float(t_cfg.get("adam_beta2", 0.99)),
        ),
        weight_decay=float(t_cfg.get("weight_decay", 0.1)),
    )

    # ---- rollout worker ----
    rollout_worker = get_rollout_worker(config, model_path, model_path)

    # ---- quant noise controller ----
    qn_cfg = config.get("quant_noise", {})
    noise_ctrl = QuantNoiseConfig.from_dict(qn_cfg).build_controller()

    # ---- logging ----
    logger = MetricsLogger.from_config(config)

    # ---- data ----
    inf_cfg  = config.get("inference", {})
    max_plen = int(inf_cfg.get("max_prompt_length", 512))
    max_rlen = int(inf_cfg.get("max_response_length", 1024))
    temp     = float(inf_cfg.get("temperature", 1.0))
    top_p    = float(inf_cfg.get("top_p", 1.0))

    group_size = int(t_cfg.get("group_size", 8))
    batch_size = int(t_cfg.get("batch_size", 32))
    max_steps  = int(t_cfg.get("max_steps", 2000))
    eval_every = int(t_cfg.get("eval_interval", 100))
    save_every = int(t_cfg.get("save_interval", 500))

    if smoke:
        # Absolute minimum: 1 prompt, smallest possible group, 32-token responses, 2 steps
        group_size = 2                        # minimum for GRPO mean/std (needs ≥ 2)
        batch_size = group_size               # 1 prompt × 2 responses
        max_steps  = 2
        eval_every = 999                      # skip mid-run eval
        max_rlen   = 32                       # 32-token responses
        max_plen   = min(max_plen, 128)       # truncate prompts too
        print(
            f"[smoke] group_size={group_size}  batch={batch_size}  "
            f"max_prompt_len={max_plen}  max_response_len={max_rlen}  steps={max_steps}",
            flush=True,
        )

    tis_enabled = config.get("rollout_correction", {}).get("enabled", False)
    tis_clip    = float(config.get("rollout_correction", {}).get("clip_threshold", 2.0))
    tis_maxlr   = float(config.get("rollout_correction", {}).get("max_log_ratio", 10.0))

    out_dir = Path("outputs") / config["experiment"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- save config ----
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    dataloader   = get_gsm8k_dataloader("train",  batch_size=batch_size // group_size)
    eval_batches = list(get_gsm8k_dataloader("test", batch_size=16, shuffle=False))

    step = 0
    for raw_batch in dataloader:
        if step >= max_steps:
            break

        chat_prompts = raw_batch["prompts"]   # list of chat message lists
        golds        = raw_batch["golds"]
        # Convert chat format → formatted strings via tokenizer chat template
        prompts = apply_chat_template(tokenizer, chat_prompts)

        print(f"[step {step:>6}]  rolling out...", flush=True)

        # ---- rollout ----
        with Timer() as rollout_timer:
            if rollout_worker is not None:
                # Same layout as bf16_rollout: one rollout per (prompt × group_size).
                rollout_prompts = prompts * group_size
                rb = rollout_worker.generate({"prompts": rollout_prompts})
                responses        = rb["decoded_text"]
                rollout_logprobs = rb["rollout_logprobs"].to(device)    # [B, R]
                response_ids     = rb["input_ids"].to(device)
                prompts = rollout_prompts
                golds   = golds * group_size
            else:
                responses, response_ids, _ = bf16_rollout(
                    model, tokenizer, prompts,
                    max_prompt_length=max_plen,
                    max_response_length=max_rlen,
                    temperature=temp,
                    top_p=top_p,
                    device=device,
                    group_size=group_size,
                    autocast_dtype=rollout_autocast_dtype,
                )
                rollout_logprobs = None
                prompts = prompts * group_size
                golds   = golds   * group_size

        n_rollout_tokens = int((response_ids != tokenizer.pad_token_id).sum().item())

        # ---- rewards ----
        rewards_list = gsm8k_reward_batch(responses, golds)
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)

        # ---- tokenise for training ----
        input_ids, attention_mask, response_mask = tokenize_batch(
            tokenizer, prompts, responses,
            max_prompt_length=max_plen,
            max_response_length=max_rlen,
            device=device,
        )

        # ---- advantages ----
        advantages = compute_grpo_advantages(rewards, group_size=group_size)

        # ---- noise scale (applied inside rollout worker on next step) ----
        if noise_ctrl is not None:
            # Compute policy entropy from current train logprobs estimate
            # (we get a proper estimate after the forward pass below)
            pass

        # ---- policy forward + loss ----
        with Timer() as train_timer:
            train_logprobs = recompute_train_logprobs_with_grad(
                model, input_ids, attention_mask, response_mask
            )

            # Align rollout_logprobs shape with train_logprobs if needed
            if rollout_logprobs is not None:
                rlen = train_logprobs.shape[1]
                rollout_logprobs = rollout_logprobs[:, :rlen]
                if rollout_logprobs.shape[1] < rlen:
                    pad = torch.zeros(
                        rollout_logprobs.shape[0], rlen - rollout_logprobs.shape[1],
                        device=device
                    )
                    rollout_logprobs = torch.cat([rollout_logprobs, pad], dim=1)

            resp_mask_trimmed = response_mask[:, -train_logprobs.shape[1]:]

            loss_out = grpo_loss_with_tis(
                train_logprobs=train_logprobs,
                rollout_logprobs=rollout_logprobs if tis_enabled else None,
                advantages=advantages,
                response_mask=resp_mask_trimmed,
                tis_clip_threshold=tis_clip,
                tis_max_log_ratio=tis_maxlr,
            )

            optimizer.zero_grad()
            loss_out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(t_cfg.get("grad_clip", 1.0)),
            )
            optimizer.step()

        # ---- noise controller update ----
        noise_scale_metrics: dict = {}
        if noise_ctrl is not None:
            new_scale = noise_ctrl.update(loss_out.entropy.item())
            noise_scale_metrics = noise_ctrl.metrics(loss_out.entropy.item())
            if rollout_worker is not None and hasattr(rollout_worker, "set_noise_scale"):
                rollout_worker.set_noise_scale(new_scale)

        # ---- sync weights into rollout worker ----
        if rollout_worker is not None and step % 1 == 0:
            rollout_worker.sync_weights(model.state_dict())

        # ---- log ----
        n_train_tokens = int(resp_mask_trimmed.sum().item())
        metrics = {
            **loss_out.as_dict(),
            **gsm8k_metrics(responses, golds, rewards_list, group_size=group_size),
            **response_length_metrics(resp_mask_trimmed),
            **throughput_metrics(n_rollout_tokens, rollout_timer.elapsed, prefix="rollout"),
            **throughput_metrics(n_train_tokens, train_timer.elapsed, prefix="train"),
            **memory_metrics(),
            **noise_scale_metrics,
            "actor_lr": optimizer.param_groups[0]["lr"],
        }
        logger.log_step(step, metrics)

        # ---- eval ----
        if step % eval_every == 0 and step > 0:
            eval_rewards = _evaluate(model, tokenizer, eval_batches[:5 if smoke else None],
                                     max_plen, max_rlen, temp, top_p, device, group_size,
                                     autocast_dtype=rollout_autocast_dtype)
            logger.log_step(step, {"reward/eval_exact_match": eval_rewards})
            print(f"[eval step {step}] exact_match={eval_rewards:.3f}")

        # ---- checkpoint ----
        if step % save_every == 0 and step > 0:
            ckpt_path = out_dir / f"checkpoint_step_{step}"
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"Saved checkpoint → {ckpt_path}")

        step += 1

    # ---- final eval + checkpoint ----
    final_eval = _evaluate(model, tokenizer, eval_batches[:5 if smoke else None],
                           max_plen, max_rlen, temp, top_p, device, group_size,
                           autocast_dtype=rollout_autocast_dtype)
    logger.log_step(step, {"reward/eval_exact_match": final_eval})
    print(f"\nFinal eval exact_match: {final_eval:.3f}")

    ckpt_path = out_dir / "checkpoint_final"
    model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)
    logger.finish()
    print(f"Done. Outputs saved to {out_dir}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate(model, tokenizer, eval_batches, max_plen, max_rlen, temp, top_p, device,
              group_size, autocast_dtype=None):
    if eval_batches is None:
        return 0.0
    model.eval()
    all_rewards = []
    for batch in eval_batches:
        eval_prompts = apply_chat_template(tokenizer, batch["prompts"])
        responses, _, _ = bf16_rollout(
            model, tokenizer, eval_prompts,
            max_prompt_length=max_plen,
            max_response_length=max_rlen,
            temperature=temp,
            top_p=top_p,
            device=device,
            group_size=1,
            autocast_dtype=autocast_dtype,
        )
        all_rewards.extend(gsm8k_reward_batch(responses, batch["golds"]))
    model.train()
    return sum(all_rewards) / max(len(all_rewards), 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FP8 TIS GSM8K Training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--override", nargs="*", default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. training.max_steps=100",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Minimal run: 1 prompt, group_size=2, 32-token responses, 2 steps",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.override)

    print(f"Experiment: {config['experiment']['name']}")
    print(f"Model:      {config['model']['name_or_path']}")
    print(f"Backend:    {config.get('inference', {}).get('backend', 'hf')}")
    print(f"TIS:        {config.get('rollout_correction', {}).get('enabled', False)}")
    print(f"Quant noise:{config.get('quant_noise', {}).get('enabled', False)}")
    print()

    train(config, smoke=args.smoke)
