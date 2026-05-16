"""
FlashRL FP8/INT8 Rollout Worker.

FlashRL (https://github.com/yaof20/Flash-RL) is a public MIT-licensed library
that patches vLLM to generate RL rollouts in FP8/INT8 with accurate log-probs.

Install:
    pip install flash-llm-rl

How it works:
    1. `import flash_rl` patches vLLM at import time.
    2. Set env var FLASHRL_CONFIG=fp8 (or int8 / bf16).
    3. Use vLLM as normal — FlashRL ensures the rollout log-probs returned
       are accurate despite quantized generation.
"""

from __future__ import annotations

import os
from typing import Any

import torch


class FlashRLRolloutWorker:
    """
    FP8/INT8 rollout worker backed by FlashRL + vLLM.

    Usage in train.py (set in YAML):
        inference:
          backend: flashrl
          fp8: true          # or int8 / bf16
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        fp8: bool = True,
        max_prompt_length: int = 512,
        max_response_length: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ):
        # Apply FlashRL patch before importing vLLM
        try:
            precision = "fp8" if fp8 else "bf16"
            os.environ.setdefault("FLASHRL_CONFIG", precision)
            import flash_rl  # noqa: F401  — patches vLLM at import time
        except ImportError:
            raise ImportError(
                "FlashRL is not installed.\n"
                "Install with: pip install flash-llm-rl\n"
                "See: https://github.com/yaof20/Flash-RL"
            )

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is required for FlashRLRolloutWorker.\n"
                "Install with: pip install vllm"
            )

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.7,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_response_length,
            logprobs=1,          # request per-token log-probs from vLLM
            prompt_logprobs=0,
        )
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length

    def generate(self, batch: dict) -> dict:
        """
        Run FP8 rollout for a batch of prompts.

        Args:
            batch: dict with key "prompts" — list of formatted prompt strings
                   (already passed through tokenizer.apply_chat_template)

        Returns dict with keys:
            decoded_text     : list[str]             — decoded response strings
            rollout_logprobs : torch.Tensor [B, R]   — per-token log-probs
            input_ids        : torch.Tensor [B, R]   — response token ids
        """
        from vllm import SamplingParams

        prompts = batch["prompts"]
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)

        decoded_text = []
        rollout_logprobs_list = []
        input_ids_list = []

        for out in outputs:
            seq = out.outputs[0]
            decoded_text.append(seq.text)
            token_ids = list(seq.token_ids)
            input_ids_list.append(token_ids)

            # Extract per-token log-probs returned by FlashRL-patched vLLM
            if seq.logprobs is not None:
                lp = [
                    list(step.values())[0].logprob
                    for step in seq.logprobs
                ]
            else:
                lp = [0.0] * len(token_ids)
            rollout_logprobs_list.append(lp)

        # Pad to max response length
        R = self.max_response_length
        B = len(decoded_text)

        ids_tensor = torch.zeros(B, R, dtype=torch.long)
        lp_tensor  = torch.zeros(B, R, dtype=torch.float32)

        for i, (ids, lp) in enumerate(zip(input_ids_list, rollout_logprobs_list)):
            L = min(len(ids), R)
            ids_tensor[i, :L] = torch.tensor(ids[:L])
            lp_tensor[i, :L]  = torch.tensor(lp[:L])

        return {
            "decoded_text":     decoded_text,
            "rollout_logprobs": lp_tensor,
            "input_ids":        ids_tensor,
        }

    def sync_weights(self, state_dict: dict[str, Any]) -> None:
        """
        Push updated training weights into the vLLM inference engine.

        vLLM supports weight updates via llm.llm_engine.model_executor.
        This is called every step in train.py to keep rollout policy fresh.
        """
        llm_model = (
            self.llm.llm_engine
                .model_executor
                .driver_worker
                .model_runner
                .model
        )
        weights = [(k, v.to(torch.float16)) for k, v in state_dict.items()]
        llm_model.load_weights(weights)

    def set_noise_scale(self, scale: float) -> None:
        """
        Adjust adaptive quantization noise scale (QeRL AQN).
        FlashRL exposes this via FLASHRL_CONFIG or direct scale injection.
        Currently a no-op — extend when FlashRL AQN API is available.
        """
        pass
