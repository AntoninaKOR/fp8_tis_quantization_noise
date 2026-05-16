"""
GRPO-style policy gradient loss with optional token-level TIS correction.

Reference: GRPO (Group Relative Policy Optimisation)

The loss computes per-token policy gradient weighted by:
  - GRPO group-relative advantages
  - optional TIS correction weights (for FP8 rollout / BF16 train mismatch)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from trainer.tis import (
    TISDiagnostics,
    MismatchDiagnostics,
    compute_tis_diagnostics,
    compute_mismatch_diagnostics,
    check_tis_safety,
)


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------

def compute_grpo_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    GRPO group-relative advantage normalisation.

    For each group of `group_size` responses generated from the same prompt,
    normalise rewards to zero mean and unit variance within the group.

    Args:
        rewards:    [batch]  scalar reward per response
        group_size: number of responses sampled per prompt
        eps:        numerical stability for std normalisation

    Returns:
        advantages: [batch]

    Example (group_size=4, 2 prompts → batch=8 responses total):

        Prompt 1: "Janet has 3 apples..."
          response 1 → reward 1.0  ┐
          response 2 → reward 0.0  │ mean=0.5, std=0.5
          response 3 → reward 1.0  │ → advantages: [+1, -1, +1, -1]
          response 4 → reward 0.0  ┘

        Prompt 2: "Tom bought 5 bags..."
          response 1 → reward 0.0  ┐
          response 2 → reward 0.0  │ mean=0.0, std=eps
          response 3 → reward 0.0  │ → advantages: [0, 0, 0, 0]
          response 4 → reward 0.0  ┘
          (model never got it right → gradient = 0 for this prompt)
    """
    batch = rewards.shape[0]
    assert batch % group_size == 0, (
        f"Batch size {batch} must be divisible by group_size {group_size}"
    )
    n_groups = batch // group_size

    rewards_grouped = rewards.view(n_groups, group_size)
    mean = rewards_grouped.mean(dim=1, keepdim=True)
    std = rewards_grouped.std(dim=1, keepdim=True).clamp_min(eps)
    advantages_grouped = (rewards_grouped - mean) / std
    return advantages_grouped.view(batch)



def recompute_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Recompute per-token log-probabilities from the BF16 training model.

    Args:
        model:          causal LM with a `forward` returning logits [batch, seq, vocab]
        input_ids:      [batch, prompt_len + response_len]
        attention_mask: [batch, prompt_len + response_len]
        response_mask:  [batch, prompt_len + response_len] — 1 for response tokens

    Returns:
        logprobs: [batch, response_len]  log π_train(a_t | s_t)
    """
    with torch.no_grad():
        pass  # placeholder — actual call happens in the training loop

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [batch, seq, vocab]

    # Shift: predict token t+1 from token t
    shift_logits = logits[:, :-1, :]          # [batch, seq-1, vocab]
    shift_labels = input_ids[:, 1:]           # [batch, seq-1]
    shift_resp_mask = response_mask[:, 1:]    # [batch, seq-1]

    log_probs_all = F.log_softmax(shift_logits, dim=-1)
    # Gather the log-prob of the actual next token
    token_logprobs = log_probs_all.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)  # [batch, seq-1]

    # Return only the response portion
    response_len = int(shift_resp_mask.sum(dim=-1).max().item())
    # Mask out non-response positions and take the last `response_len` columns
    # This assumes responses are right-aligned in the sequence
    token_logprobs = token_logprobs * shift_resp_mask
    # Trim to response length (assumes prompt is left-padded or fixed-length)
    return token_logprobs[:, -response_len:]


# ---------------------------------------------------------------------------
# GRPO loss
# ---------------------------------------------------------------------------

@dataclass
class GRPOLossOutput:
    loss: torch.Tensor
    policy_loss: torch.Tensor          # unweighted token-level PG loss
    entropy: torch.Tensor
    tis_diagnostics: Optional[TISDiagnostics] = None
    mismatch_diagnostics: Optional[MismatchDiagnostics] = None

    def as_dict(self) -> dict:
        out: dict = {
            "policy_loss": self.policy_loss.item(),
            "loss":        self.loss.item(),
            "entropy":     self.entropy.item(),
        }
        if self.tis_diagnostics is not None:
            out.update(self.tis_diagnostics.as_dict())
        if self.mismatch_diagnostics is not None:
            out.update(self.mismatch_diagnostics.as_dict())
        return out


def grpo_loss_with_tis(
    train_logprobs: torch.Tensor,
    rollout_logprobs: Optional[torch.Tensor],
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    tis_clip_threshold: float = 2.0,
    tis_max_log_ratio: float = 10.0,
    tis_abort_if_clipped_above: float = 0.80,
    tis_warn_if_ess_below: float = 0.30,
    entropy_logprobs: Optional[torch.Tensor] = None,
) -> GRPOLossOutput:
    """
    Token-level GRPO policy gradient loss with optional TIS correction.

    When `rollout_logprobs` is None the function behaves as a plain GRPO loss
    (BF16 baseline mode, no TIS).

    Args:
        train_logprobs:   [batch, response_len]  log π_train
        rollout_logprobs: [batch, response_len]  log π_rollout (None → no TIS)
        advantages:       [batch]                per-response GRPO advantage
        response_mask:    [batch, response_len]  1 for valid response tokens
        tis_clip_threshold: C for truncated IS
        tis_max_log_ratio:  hard clamp on log-ratio before exp
        tis_abort_if_clipped_above: fraction threshold that triggers TISSafetyError
        tis_warn_if_ess_below:      ESS fraction threshold for warning
        entropy_logprobs: [batch, response_len]  logprobs used for entropy estimate
                          (defaults to train_logprobs if None)

    Returns:
        GRPOLossOutput
    """
    adv = advantages.unsqueeze(1)  # [batch, 1] 
    # Per-token policy gradient objective: -adv * log π_train
    policy_logprob_term = train_logprobs  # [batch, response_len]
    token_loss = -adv * policy_logprob_term  # [batch, response_len]

    tis_diag = None  
    mm_diag = None   

    if rollout_logprobs is not None:
        tis_weights, raw_ratios, tis_diag = compute_tis_diagnostics(
            train_logprobs,
            rollout_logprobs,
            response_mask,
            clip_threshold=tis_clip_threshold,
            max_log_ratio=tis_max_log_ratio,
        )

        check_tis_safety(
            train_logprobs,
            rollout_logprobs,
            tis_weights,
            tis_diag,
            abort_if_clipped_fraction_above=tis_abort_if_clipped_above,
            warn_if_ess_fraction_below=tis_warn_if_ess_below,
        )

        # Apply TIS correction
        token_loss = token_loss * tis_weights

        mm_diag = compute_mismatch_diagnostics(
            train_logprobs,
            rollout_logprobs,
            response_mask,
        )

    # Mask and reduce
    token_loss = token_loss * response_mask
    loss = token_loss.sum() / response_mask.sum().clamp_min(1) #clamp_min(1) safe from division by zero

    # Entropy: H ≈ -E[log π]
    lp_for_entropy = entropy_logprobs if entropy_logprobs is not None else train_logprobs
    masked_lp = lp_for_entropy * response_mask
    entropy = -(masked_lp.sum() / response_mask.sum().clamp_min(1)) 

    # Unweighted policy loss for logging (no TIS, no advantage)
    policy_loss = (
        (-train_logprobs * response_mask).sum()
        / response_mask.sum().clamp_min(1)
    )

    return GRPOLossOutput(
        loss=loss,
        policy_loss=policy_loss,
        entropy=entropy,
        tis_diagnostics=tis_diag,
        mismatch_diagnostics=mm_diag,
    )
