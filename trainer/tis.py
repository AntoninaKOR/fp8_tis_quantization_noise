"""
Token-level Truncated Importance Sampling (TIS) correction.

Corrects the mismatch between an FP8 rollout policy (e.g. FlashRL) and the
BF16 training policy.  All tensors are assumed to be on the same device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch



def compute_tis_weights(
    train_logprobs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    clip_threshold: float = 2.0,
    max_log_ratio: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-token truncated importance sampling weights.

    Args:
        train_logprobs:   [batch, response_len]  log π_train(a_t | s_t)
        rollout_logprobs: [batch, response_len]  log π_rollout(a_t | s_t)
        response_mask:    [batch, response_len]  1 for valid response tokens
        clip_threshold:   C — maximum allowed ratio value (default 2.0)
        max_log_ratio:    hard clamp on log-ratio before exp, for NaN safety

    Returns:
        tis_weights: [batch, response_len]  clipped ratio * mask
        raw_ratios:  [batch, response_len]  unclipped ratio * mask
    """
    log_ratio = train_logprobs - rollout_logprobs
    log_ratio = torch.clamp(log_ratio, min=-max_log_ratio, max=max_log_ratio)

    raw_ratios = torch.exp(log_ratio)
    tis_weights = torch.clamp(raw_ratios, max=clip_threshold)

    tis_weights = tis_weights * response_mask
    raw_ratios = raw_ratios * response_mask

    return tis_weights, raw_ratios


def effective_sample_size(
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Effective sample size (ESS) of the importance weights.

    ESS = (\sum w_i)^2 / (\sum w_i^2)

    Applied only over masked positions.
    """
    w = weights * mask
    numerator = w.sum() ** 2
    denominator = (w ** 2).sum().clamp_min(1e-8)
    return numerator / denominator


def ess_fraction(
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """ESS as a fraction of the number of valid tokens."""
    n_valid = mask.sum().clamp_min(1.0)
    return effective_sample_size(weights, mask) / n_valid


# ---------------------------------------------------------------------------
# Diagnostics dataclass
# ---------------------------------------------------------------------------

@dataclass
class TISDiagnostics:
    ratio_mean: float
    ratio_std: float
    ratio_max: float
    ratio_min: float
    clipped_fraction: float
    weight_mean: float
    weight_std: float
    log_ratio_mean: float
    log_ratio_abs_mean: float
    effective_sample_size: float
    ess_fraction: float

    def as_dict(self, prefix: str = "tis/") -> dict:
        return {
            f"{prefix}ratio_mean":         self.ratio_mean,
            f"{prefix}ratio_std":          self.ratio_std,
            f"{prefix}ratio_max":          self.ratio_max,
            f"{prefix}ratio_min":          self.ratio_min,
            f"{prefix}clipped_fraction":   self.clipped_fraction,
            f"{prefix}weight_mean":        self.weight_mean,
            f"{prefix}weight_std":         self.weight_std,
            f"{prefix}log_ratio_mean":     self.log_ratio_mean,
            f"{prefix}log_ratio_abs_mean": self.log_ratio_abs_mean,
            f"{prefix}effective_sample_size": self.effective_sample_size,
            f"{prefix}ess_fraction":       self.ess_fraction,
        }


# ---------------------------------------------------------------------------
# Full diagnostics computation
# ---------------------------------------------------------------------------

def compute_tis_diagnostics(
    train_logprobs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    clip_threshold: float = 2.0,
    max_log_ratio: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor, TISDiagnostics]:
    """
    Compute TIS weights plus all diagnostic metrics in one call.

    Returns:
        tis_weights:  [batch, response_len]
        raw_ratios:   [batch, response_len]
        diagnostics:  TISDiagnostics
    """
    tis_weights, raw_ratios = compute_tis_weights(
        train_logprobs,
        rollout_logprobs,
        response_mask,
        clip_threshold=clip_threshold,
        max_log_ratio=max_log_ratio,
    )

    log_ratio = train_logprobs - rollout_logprobs
    log_ratio = torch.clamp(log_ratio, min=-max_log_ratio, max=max_log_ratio)

    valid_mask = response_mask.bool()
    n_valid = response_mask.sum().clamp_min(1.0)

    raw_valid = raw_ratios[valid_mask]
    weights_valid = tis_weights[valid_mask]
    log_ratio_valid = log_ratio[valid_mask]

    # Clipped fraction: tokens where the unclipped ratio exceeded the threshold
    n_clipped = (raw_valid > clip_threshold).float().sum()
    clipped_frac = (n_clipped / n_valid).item()

    ess = effective_sample_size(tis_weights, response_mask).item()
    ess_frac = ess / n_valid.item()

    diag = TISDiagnostics(
        ratio_mean=raw_valid.mean().item(),
        ratio_std=raw_valid.std().item(),
        ratio_max=raw_valid.max().item() if raw_valid.numel() > 0 else 0.0,
        ratio_min=raw_valid.min().item() if raw_valid.numel() > 0 else 0.0,
        clipped_fraction=clipped_frac,
        weight_mean=weights_valid.mean().item(),
        weight_std=weights_valid.std().item(),
        log_ratio_mean=log_ratio_valid.mean().item(),
        log_ratio_abs_mean=log_ratio_valid.abs().mean().item(),
        effective_sample_size=ess,
        ess_fraction=ess_frac,
    )

    return tis_weights, raw_ratios, diag


# ---------------------------------------------------------------------------
# Mismatch diagnostics
# ---------------------------------------------------------------------------

@dataclass
class MismatchDiagnostics:
    logprob_abs_diff_mean: float
    logprob_diff_std: float
    kl_approx: float

    def as_dict(self, prefix: str = "mismatch/") -> dict:
        return {
            f"{prefix}train_vs_rollout_logprob_abs_diff_mean": self.logprob_abs_diff_mean,
            f"{prefix}train_vs_rollout_logprob_diff_std":      self.logprob_diff_std,
            f"{prefix}kl_approx":                              self.kl_approx,
        }


def compute_mismatch_diagnostics(
    train_logprobs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
) -> MismatchDiagnostics:
    """
    Quantify the per-token logprob mismatch between the train and rollout policies.

    kl_approx uses the reverse-KL first-order approximation:
        KL(rollout || train) ≈ Σ exp(lp_rollout) * (lp_rollout - lp_train)
    averaged over valid tokens.
    """
    diff = (train_logprobs - rollout_logprobs) * response_mask
    valid_mask = response_mask.bool()
    diff_valid = diff[valid_mask]

    # KL approximation: E_{rollout}[log rollout - log train]
    # Using rollout logprobs as proxy for rollout distribution weight
    lp_rollout = rollout_logprobs * response_mask
    kl_terms = torch.exp(lp_rollout) * (-diff) * response_mask
    n_valid = response_mask.sum().clamp_min(1.0)
    kl_approx = kl_terms.sum() / n_valid

    return MismatchDiagnostics(
        logprob_abs_diff_mean=diff_valid.abs().mean().item(),
        logprob_diff_std=diff_valid.std().item(),
        kl_approx=kl_approx.item(),
    )


# ---------------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------------

class TISSafetyError(RuntimeError):
    pass


def check_tis_safety(
    train_logprobs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    tis_weights: torch.Tensor,
    diag: TISDiagnostics,
    abort_if_clipped_fraction_above: float = 0.80,
    warn_if_ess_fraction_below: float = 0.30,
) -> list[str]:
    """
    Run safety checks, emit RuntimeWarnings for soft conditions, and return
    a list of warning message strings.
    Raises TISSafetyError if a hard abort condition is triggered.
    """
    import warnings as _warnings

    messages: list[str] = []

    if not torch.isfinite(train_logprobs).all():
        raise TISSafetyError("Non-finite values in train_logprobs")
    if not torch.isfinite(rollout_logprobs).all():
        raise TISSafetyError("Non-finite values in rollout_logprobs")
    if not torch.isfinite(tis_weights).all():
        raise TISSafetyError("Non-finite values in tis_weights")

    if diag.clipped_fraction > abort_if_clipped_fraction_above:
        raise TISSafetyError(
            f"TIS clipped fraction {diag.clipped_fraction:.3f} exceeds abort threshold "
            f"{abort_if_clipped_fraction_above:.3f}. "
            "Consider: syncing weights more frequently, reducing LR, disabling FP8 KV cache, "
            "or lowering TIS clip threshold."
        )

    if diag.ess_fraction < warn_if_ess_fraction_below:
        msg = (
            f"TIS ESS fraction {diag.ess_fraction:.3f} is below warning threshold "
            f"{warn_if_ess_fraction_below:.3f}. "
            "Consider: reducing quant noise, reducing rollout length, normalising weights."
        )
        _warnings.warn(msg, RuntimeWarning, stacklevel=2)
        messages.append(msg)

    return messages
