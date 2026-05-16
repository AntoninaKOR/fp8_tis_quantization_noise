"""
Unit tests for the TIS (Truncated Importance Sampling) module.
Covers: shapes, clipping, masking, NaN safety, ESS, and mismatch diagnostics.
"""

import pytest
import torch

from trainer.tis import (
    compute_tis_weights,
    compute_tis_diagnostics,
    compute_mismatch_diagnostics,
    effective_sample_size,
    ess_fraction,
    check_tis_safety,
    TISSafetyError,
)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestTISShapes:
    def test_output_shapes_match_input(self):
        bsz, seqlen = 4, 16
        train_lp   = torch.randn(bsz, seqlen)
        rollout_lp = torch.randn(bsz, seqlen)
        mask       = torch.ones(bsz, seqlen)

        weights, ratios = compute_tis_weights(train_lp, rollout_lp, mask)

        assert weights.shape == train_lp.shape
        assert ratios.shape  == train_lp.shape

    def test_diagnostics_returns_three_values(self):
        bsz, seqlen = 2, 8
        train_lp   = torch.randn(bsz, seqlen)
        rollout_lp = torch.randn(bsz, seqlen)
        mask       = torch.ones(bsz, seqlen)

        weights, ratios, diag = compute_tis_diagnostics(train_lp, rollout_lp, mask)

        assert weights.shape == (bsz, seqlen)
        assert ratios.shape  == (bsz, seqlen)
        assert diag is not None


# ---------------------------------------------------------------------------
# Clipping tests
# ---------------------------------------------------------------------------

class TestTISClipping:
    def test_large_log_ratio_is_clipped(self):
        train_lp   = torch.tensor([[10.0]])
        rollout_lp = torch.tensor([[0.0]])
        mask       = torch.tensor([[1.0]])

        weights, ratios = compute_tis_weights(
            train_lp, rollout_lp, mask, clip_threshold=2.0
        )

        assert weights.item() == pytest.approx(2.0)

    def test_small_ratio_not_clipped(self):
        # log ratio = 0.5, ratio = exp(0.5) ≈ 1.65 < 2.0
        train_lp   = torch.tensor([[0.5]])
        rollout_lp = torch.tensor([[0.0]])
        mask       = torch.tensor([[1.0]])

        weights, ratios = compute_tis_weights(
            train_lp, rollout_lp, mask, clip_threshold=2.0
        )

        assert weights.item() == pytest.approx(ratios.item(), rel=1e-5)
        assert weights.item() < 2.0

    def test_clip_fraction_in_diagnostics(self):
        # All tokens should be clipped (ratio >> 2.0)
        train_lp   = torch.full((1, 4), 10.0)
        rollout_lp = torch.zeros(1, 4)
        mask       = torch.ones(1, 4)

        _, _, diag = compute_tis_diagnostics(
            train_lp, rollout_lp, mask, clip_threshold=2.0
        )

        assert diag.clipped_fraction == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Masking tests
# ---------------------------------------------------------------------------

class TestTISMasking:
    def test_masked_tokens_zero(self):
        train_lp   = torch.tensor([[1.0, 1.0]])
        rollout_lp = torch.tensor([[0.0, 0.0]])
        mask       = torch.tensor([[1.0, 0.0]])

        weights, ratios = compute_tis_weights(train_lp, rollout_lp, mask)

        assert weights[0, 1].item() == pytest.approx(0.0)
        assert ratios[0, 1].item()  == pytest.approx(0.0)

    def test_unmasked_tokens_nonzero(self):
        train_lp   = torch.tensor([[1.0, 1.0]])
        rollout_lp = torch.tensor([[0.0, 0.0]])
        mask       = torch.tensor([[1.0, 0.0]])

        weights, _ = compute_tis_weights(train_lp, rollout_lp, mask)

        assert weights[0, 0].item() > 0.0


# ---------------------------------------------------------------------------
# NaN / numerical safety
# ---------------------------------------------------------------------------

class TestTISNumericalSafety:
    def test_no_nan_with_large_values(self):
        train_lp   = torch.tensor([[1000.0]])
        rollout_lp = torch.tensor([[-1000.0]])
        mask       = torch.tensor([[1.0]])

        weights, ratios = compute_tis_weights(
            train_lp, rollout_lp, mask, max_log_ratio=10.0
        )

        assert torch.isfinite(weights).all()
        assert torch.isfinite(ratios).all()

    def test_identical_logprobs_give_ratio_one(self):
        lp   = torch.randn(2, 8)
        mask = torch.ones(2, 8)

        weights, ratios = compute_tis_weights(lp, lp, mask)

        assert torch.allclose(weights, torch.ones_like(weights), atol=1e-5)

    def test_no_nan_zero_mask(self):
        train_lp   = torch.randn(2, 8)
        rollout_lp = torch.randn(2, 8)
        mask       = torch.zeros(2, 8)

        weights, ratios = compute_tis_weights(train_lp, rollout_lp, mask)

        assert torch.isfinite(weights).all()
        assert torch.isfinite(ratios).all()


# ---------------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------------

class TestESS:
    def test_uniform_weights_ess_equals_n(self):
        weights = torch.ones(1, 10)
        mask    = torch.ones(1, 10)

        ess = effective_sample_size(weights, mask)

        assert ess.item() == pytest.approx(10.0, rel=1e-4)

    def test_ess_fraction_uniform(self):
        weights = torch.ones(1, 10)
        mask    = torch.ones(1, 10)

        frac = ess_fraction(weights, mask)

        assert frac.item() == pytest.approx(1.0, rel=1e-4)

    def test_ess_degenerate_single_weight(self):
        # One token gets all the weight; ESS should be ~1
        weights = torch.zeros(1, 10)
        weights[0, 0] = 10.0
        mask = torch.ones(1, 10)

        ess = effective_sample_size(weights, mask)

        assert ess.item() == pytest.approx(1.0, rel=1e-4)

    def test_ess_in_diagnostics_matches_standalone(self):
        bsz, seqlen = 2, 16
        train_lp   = torch.randn(bsz, seqlen)
        rollout_lp = torch.randn(bsz, seqlen)
        mask       = torch.ones(bsz, seqlen)

        weights, _, diag = compute_tis_diagnostics(train_lp, rollout_lp, mask)
        standalone_ess   = effective_sample_size(weights, mask).item()

        assert diag.effective_sample_size == pytest.approx(standalone_ess, rel=1e-4)


# ---------------------------------------------------------------------------
# Mismatch diagnostics
# ---------------------------------------------------------------------------

class TestMismatchDiagnostics:
    def test_zero_mismatch_identical_logprobs(self):
        lp   = torch.randn(2, 8)
        mask = torch.ones(2, 8)

        diag = compute_mismatch_diagnostics(lp, lp, mask)

        assert diag.logprob_abs_diff_mean == pytest.approx(0.0, abs=1e-6)
        assert diag.logprob_diff_std      == pytest.approx(0.0, abs=1e-5)

    def test_nonzero_mismatch(self):
        train_lp   = torch.ones(2, 8)
        rollout_lp = torch.zeros(2, 8)
        mask       = torch.ones(2, 8)

        diag = compute_mismatch_diagnostics(train_lp, rollout_lp, mask)

        assert diag.logprob_abs_diff_mean == pytest.approx(1.0, rel=1e-5)

    def test_as_dict_has_expected_keys(self):
        lp   = torch.randn(2, 8)
        mask = torch.ones(2, 8)

        diag = compute_mismatch_diagnostics(lp, lp, mask)
        d    = diag.as_dict()

        assert "mismatch/train_vs_rollout_logprob_abs_diff_mean" in d
        assert "mismatch/train_vs_rollout_logprob_diff_std"      in d
        assert "mismatch/kl_approx"                              in d


# ---------------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------------

class TestTISSafetyChecks:
    def test_abort_on_high_clipped_fraction(self):
        # Force clipped_fraction = 1.0 by making all log-ratios huge
        train_lp   = torch.full((2, 8), 10.0)
        rollout_lp = torch.zeros(2, 8)
        mask       = torch.ones(2, 8)

        weights, _, diag = compute_tis_diagnostics(
            train_lp, rollout_lp, mask, clip_threshold=2.0
        )

        with pytest.raises(TISSafetyError, match="clipped fraction"):
            check_tis_safety(
                train_lp, rollout_lp, weights, diag,
                abort_if_clipped_fraction_above=0.5,
            )

    def test_warn_on_low_ess(self):
        # Degenerate weights → very low ESS
        train_lp   = torch.zeros(1, 10)
        rollout_lp = torch.zeros(1, 10)
        mask       = torch.ones(1, 10)
        train_lp[0, 0] = 100.0   # one dominant token

        weights, _, diag = compute_tis_diagnostics(
            train_lp, rollout_lp, mask, clip_threshold=1e6
        )

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_tis_safety(
                train_lp, rollout_lp, weights, diag,
                abort_if_clipped_fraction_above=1.1,   # never abort
                warn_if_ess_fraction_below=0.99,       # always warn
            )
            assert any(issubclass(warning.category, RuntimeWarning) for warning in w)

    def test_non_finite_train_logprobs_raises(self):
        train_lp   = torch.tensor([[float("nan")]])
        rollout_lp = torch.tensor([[0.0]])
        mask       = torch.tensor([[1.0]])
        weights    = torch.tensor([[1.0]])

        from trainer.tis import TISDiagnostics
        dummy_diag = TISDiagnostics(
            ratio_mean=1.0, ratio_std=0.0, ratio_max=1.0, ratio_min=1.0,
            clipped_fraction=0.0, weight_mean=1.0, weight_std=0.0,
            log_ratio_mean=0.0, log_ratio_abs_mean=0.0,
            effective_sample_size=1.0, ess_fraction=1.0,
        )

        with pytest.raises(TISSafetyError, match="train_logprobs"):
            check_tis_safety(train_lp, rollout_lp, weights, dummy_diag)


# ---------------------------------------------------------------------------
# Diagnostics dict
# ---------------------------------------------------------------------------

class TestDiagnosticsDict:
    def test_as_dict_has_all_tis_keys(self):
        bsz, seqlen = 2, 8
        train_lp   = torch.randn(bsz, seqlen)
        rollout_lp = torch.randn(bsz, seqlen)
        mask       = torch.ones(bsz, seqlen)

        _, _, diag = compute_tis_diagnostics(train_lp, rollout_lp, mask)
        d = diag.as_dict()

        expected_keys = [
            "tis/ratio_mean", "tis/ratio_std", "tis/ratio_max", "tis/ratio_min",
            "tis/clipped_fraction", "tis/weight_mean", "tis/weight_std",
            "tis/log_ratio_mean", "tis/log_ratio_abs_mean",
            "tis/effective_sample_size", "tis/ess_fraction",
        ]
        for k in expected_keys:
            assert k in d, f"Missing key: {k}"
