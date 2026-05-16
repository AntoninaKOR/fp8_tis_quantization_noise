"""
Optional quantization-noise controller for FP8 rollouts (flashrl_fp8_adaptive_quant_noise).

Two strategies are implemented:

  A) Scale jitter  — stochastic perturbations to FP8 activation/weight scales
                     inside the FlashRL rollout worker.
  B) Logit noise   — additive Gaussian noise on logits before sampling;
                     easier to ablate but not true quantization noise.

The adaptive controller adjusts the noise scale based on observed policy entropy
relative to a target entropy value.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
# ---------------------------------------------------------------------------
# Strategy A: Quantization scale jitter
# ---------------------------------------------------------------------------

def apply_scale_jitter(
    scale: torch.Tensor,
    noise_scale: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Apply multiplicative log-normal jitter to an FP8 quantization scale tensor.

    Should only be called inside the FlashRL rollout engine / wrapper.
    Never called on BF16 training weights.

    Args:
        scale:       FP8 per-tensor or per-channel scale factor
        noise_scale: standard deviation of the log-normal perturbation
        generator:   optional RNG for reproducibility

    Returns:
        perturbed scale tensor (same shape and device as input)
    """
    if noise_scale <= 0.0:
        return scale
    eps = torch.randn_like(scale, generator=generator) * noise_scale
    jitter = torch.exp(eps)
    return scale * jitter


# ---------------------------------------------------------------------------
# Strategy QeRL: RMSNorm weight noise (copied from QeRL trl_trainer/noise_scheduler.py)
#
# Injects Gaussian noise into RMSNorm layer weights directly.
# Simpler than FP8 scale jitter — targets only normalization layers.
# Schedule: sigma_trend list defines sigma per training interval.
# ---------------------------------------------------------------------------

def get_sigma_by_step(step: int, total_steps: int, sigma_trend: list[float]):
    """Return (sigma_id, sigma) for the current training step."""
    step = min(step, total_steps)
    num_intervals = len(sigma_trend) + 1
    steps_per_interval = total_steps / num_intervals
    interval_id = int(step // steps_per_interval)
    if interval_id == 0:
        return 0, 0.0
    sigma_id = min(interval_id - 1, len(sigma_trend) - 1)
    return sigma_id, sigma_trend[sigma_id]


def apply_rmsnorm_noise(
    model: "torch.nn.Module",
    step: int,
    total_steps: int,
    sigma_trend: list[float],
) -> None:
    """
    Add Gaussian noise to RMSNorm weight tensors (in-place).

    Copied from QeRL trl_trainer/noise_scheduler.py generate_gaussian_noise().
    Applied to the rollout model each step to encourage exploration.

    Args:
        model:       the model whose RMSNorm weights will be perturbed
        step:        current training step
        total_steps: total number of training steps
        sigma_trend: list of sigma values for each training interval
                     e.g. [0.01, 0.005] → two intervals of increasing noise then decay
    """
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
        from transformers.models.llama.modeling_llama import LlamaRMSNorm
        rms_norm_types = (Qwen2RMSNorm, LlamaRMSNorm)
    except ImportError:
        import torch.nn as nn
        rms_norm_types = (nn.RMSNorm,)

    sigma_id, sigma = get_sigma_by_step(step, total_steps, sigma_trend)
    if sigma == 0:
        return

    for _, module in model.named_modules():
        if isinstance(module, rms_norm_types):
            noise = torch.randn_like(module.weight.float()) * sigma
            noise = noise.to(module.weight.dtype)
            with torch.no_grad():
                module.weight.add_(noise)


# ---------------------------------------------------------------------------
# Strategy B: Logit noise (baseline)
# ---------------------------------------------------------------------------

def add_controlled_logit_noise(
    logits: torch.Tensor,
    noise_scale: float,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Add scaled Gaussian noise to logits before sampling.

    Label this 'logit_noise_baseline' in experiments; it is NOT true
    quantization noise but is useful as a controlled ablation.

    Args:
        logits:      [batch, seq_len, vocab_size] or [batch, vocab_size]
        noise_scale: standard deviation of the additive noise
        temperature: scale factor applied to the noise amplitude
        generator:   optional RNG

    Returns:
        logits + noise * temperature  (same shape as input)
    """
    if noise_scale <= 0.0:
        return logits
    noise = torch.randn_like(logits, generator=generator) * noise_scale
    return logits + noise * temperature


# ---------------------------------------------------------------------------
# Adaptive entropy-based noise controller
# ---------------------------------------------------------------------------

class AdaptiveQuantNoiseController:
    """
    Adjusts the quantization noise scale based on observed policy entropy.

    Semantics:
        observed_entropy < target_entropy  →  increase noise (more exploration)
        observed_entropy > target_entropy  →  decrease noise (less distortion)

    The controller is stateful; call `update(observed_entropy)` each training
    step to get the current noise scale.
    """

    def __init__(
        self,
        target_entropy: float,
        initial_scale: float = 1.0,
        min_scale: float = 0.0,
        max_scale: float = 2.0,
        update_rate: float = 0.01,
        warmup_steps: int = 100,
        decay_steps: int = 0,
    ):
        self.target_entropy = target_entropy
        self.scale = float(initial_scale)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.update_rate = float(update_rate)
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self._step = 0

    def update(self, observed_entropy: float) -> float:
        """
        Update internal scale based on observed entropy and return the new scale.

        During warmup the scale stays fixed at its initial value.
        If decay_steps > 0, the max_scale decays linearly to min_scale after
        warmup, which prevents the noise from remaining high forever.
        """
        self._step += 1

        if self._step <= self.warmup_steps:
            return self.scale

        error = self.target_entropy - observed_entropy
        self.scale += self.update_rate * error

        # Optional linear decay on the max_scale ceiling
        if self.decay_steps > 0:
            decay_progress = min(
                1.0,
                (self._step - self.warmup_steps) / self.decay_steps,
            )
            effective_max = self.max_scale * (1.0 - decay_progress) + self.min_scale * decay_progress
        else:
            effective_max = self.max_scale

        self.scale = float(max(self.min_scale, min(effective_max, self.scale)))
        return self.scale

    def metrics(self, observed_entropy: float) -> dict:
        return {
            "quant_noise/enabled":          True,
            "quant_noise/scale":            self.scale,
            "quant_noise/target_entropy":   self.target_entropy,
            "quant_noise/observed_entropy": observed_entropy,
            "quant_noise/update_delta":     self.update_rate * (self.target_entropy - observed_entropy),
        }


# ---------------------------------------------------------------------------
# Noise config dataclass
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class QuantNoiseConfig:
    enabled: bool = False
    mode: str = "none"           # "none" | "scale_jitter" | "logit_noise"
    target: str = "rollout_only"
    target_entropy: float = 0.8
    initial_scale: float = 0.0
    min_scale: float = 0.0
    max_scale: float = 0.0
    update_rate: float = 0.0
    warmup_steps: int = 0
    decay_steps: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "QuantNoiseConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def build_controller(self) -> Optional[AdaptiveQuantNoiseController]:
        if not self.enabled or self.mode == "none":
            return None
        return AdaptiveQuantNoiseController(
            target_entropy=self.target_entropy,
            initial_scale=self.initial_scale,
            min_scale=self.min_scale,
            max_scale=self.max_scale,
            update_rate=self.update_rate,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
        )
