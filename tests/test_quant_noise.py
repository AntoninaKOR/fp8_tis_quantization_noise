"""
Unit tests for the quantization noise module.
"""

import pytest
import torch

from trainer.quant_noise import (
    apply_scale_jitter,
    add_controlled_logit_noise,
    AdaptiveQuantNoiseController,
    QuantNoiseConfig,
)


class TestApplyScaleJitter:
    def test_zero_noise_returns_unchanged(self):
        scale = torch.ones(4)
        result = apply_scale_jitter(scale, noise_scale=0.0)
        assert torch.allclose(result, scale)

    def test_nonzero_noise_changes_scale(self):
        torch.manual_seed(0)
        scale = torch.ones(100)
        result = apply_scale_jitter(scale, noise_scale=0.1)
        assert not torch.allclose(result, scale)

    def test_result_is_finite(self):
        torch.manual_seed(42)
        scale = torch.ones(64)
        result = apply_scale_jitter(scale, noise_scale=1.0)
        assert torch.isfinite(result).all()

    def test_result_is_positive(self):
        # log-normal jitter should always be positive
        torch.manual_seed(1)
        scale = torch.ones(64)
        result = apply_scale_jitter(scale, noise_scale=2.0)
        assert (result > 0).all()


class TestAddControlledLogitNoise:
    def test_zero_noise_returns_unchanged(self):
        logits = torch.randn(2, 4, 32000)
        result = add_controlled_logit_noise(logits, noise_scale=0.0)
        assert torch.allclose(result, logits)

    def test_nonzero_noise_changes_logits(self):
        torch.manual_seed(0)
        logits = torch.zeros(2, 4, 32000)
        result = add_controlled_logit_noise(logits, noise_scale=1.0)
        assert not torch.allclose(result, logits)

    def test_output_shape_preserved(self):
        logits = torch.randn(3, 5, 1000)
        result = add_controlled_logit_noise(logits, noise_scale=0.5)
        assert result.shape == logits.shape

    def test_finite_output(self):
        torch.manual_seed(7)
        logits = torch.randn(4, 16, 512)
        result = add_controlled_logit_noise(logits, noise_scale=1.0)
        assert torch.isfinite(result).all()


class TestAdaptiveQuantNoiseController:
    def test_scale_increases_when_entropy_too_low(self):
        ctrl = AdaptiveQuantNoiseController(
            target_entropy=1.0,
            initial_scale=0.5,
            warmup_steps=0,
            update_rate=0.1,
        )
        scale_before = ctrl.scale
        ctrl.update(observed_entropy=0.0)  # much lower than target
        assert ctrl.scale > scale_before

    def test_scale_decreases_when_entropy_too_high(self):
        ctrl = AdaptiveQuantNoiseController(
            target_entropy=0.5,
            initial_scale=1.0,
            warmup_steps=0,
            update_rate=0.1,
        )
        scale_before = ctrl.scale
        ctrl.update(observed_entropy=2.0)  # much higher than target
        assert ctrl.scale < scale_before

    def test_scale_clamped_to_max(self):
        ctrl = AdaptiveQuantNoiseController(
            target_entropy=10.0,
            initial_scale=0.0,
            max_scale=2.0,
            warmup_steps=0,
            update_rate=1.0,
        )
        for _ in range(100):
            ctrl.update(observed_entropy=0.0)
        assert ctrl.scale <= 2.0

    def test_scale_clamped_to_min(self):
        ctrl = AdaptiveQuantNoiseController(
            target_entropy=0.0,
            initial_scale=2.0,
            min_scale=0.1,
            warmup_steps=0,
            update_rate=1.0,
        )
        for _ in range(100):
            ctrl.update(observed_entropy=10.0)
        assert ctrl.scale >= 0.1

    def test_warmup_freezes_scale(self):
        ctrl = AdaptiveQuantNoiseController(
            target_entropy=1.0,
            initial_scale=0.5,
            warmup_steps=10,
            update_rate=1.0,
        )
        for _ in range(10):
            ctrl.update(observed_entropy=0.0)
        assert ctrl.scale == pytest.approx(0.5)

    def test_metrics_dict_has_expected_keys(self):
        ctrl = AdaptiveQuantNoiseController(target_entropy=0.8, warmup_steps=0)
        d = ctrl.metrics(observed_entropy=0.5)
        assert "quant_noise/scale"            in d
        assert "quant_noise/target_entropy"   in d
        assert "quant_noise/observed_entropy" in d
        assert "quant_noise/update_delta"     in d


class TestQuantNoiseConfig:
    def test_disabled_config_builds_no_controller(self):
        cfg = QuantNoiseConfig(enabled=False)
        assert cfg.build_controller() is None

    def test_none_mode_builds_no_controller(self):
        cfg = QuantNoiseConfig(enabled=True, mode="none")
        assert cfg.build_controller() is None

    def test_enabled_builds_controller(self):
        cfg = QuantNoiseConfig(
            enabled=True,
            mode="scale_jitter",
            target_entropy=0.8,
            initial_scale=0.3,
            min_scale=0.0,
            max_scale=1.5,
            update_rate=0.01,
            warmup_steps=10,
        )
        ctrl = cfg.build_controller()
        assert ctrl is not None
        assert isinstance(ctrl, AdaptiveQuantNoiseController)

    def test_from_dict_round_trip(self):
        d = {
            "enabled": True,
            "mode": "scale_jitter",
            "target_entropy": 0.8,
            "initial_scale": 0.3,
            "min_scale": 0.0,
            "max_scale": 1.5,
            "update_rate": 0.01,
            "warmup_steps": 50,
            "decay_steps": 100,
        }
        cfg = QuantNoiseConfig.from_dict(d)
        assert cfg.enabled is True
        assert cfg.target_entropy == 0.8
        assert cfg.decay_steps == 100
