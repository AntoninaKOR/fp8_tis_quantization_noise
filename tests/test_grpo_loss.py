"""
Unit tests for the GRPO loss with TIS integration.
"""

import pytest
import torch

from trainer.grpo_loss import (
    compute_grpo_advantages,
    grpo_loss_with_tis,
    GRPOLossOutput,
)


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------

class TestGRPOAdvantages:
    def test_zero_mean_per_group(self):
        rewards    = torch.tensor([1.0, 0.0, 1.0, 0.0])
        mask       = torch.ones(4, 8)
        advantages = compute_grpo_advantages(rewards, group_size=2)

        # Each group of 2 should have zero mean
        assert advantages[:2].mean().item() == pytest.approx(0.0, abs=1e-5)
        assert advantages[2:].mean().item() == pytest.approx(0.0, abs=1e-5)

    def test_unit_std_per_group(self):
        rewards    = torch.tensor([1.0, 0.0, 2.0, 0.0])
        mask       = torch.ones(4, 8)
        advantages = compute_grpo_advantages(rewards, group_size=2)

        assert advantages[:2].std().item() == pytest.approx(1.0, abs=1e-4)

    def test_invalid_group_size_raises(self):
        rewards = torch.tensor([1.0, 0.0, 1.0])  # 3 not divisible by 2
        mask    = torch.ones(3, 8)

        with pytest.raises(AssertionError):
            compute_grpo_advantages(rewards, group_size=2)


# ---------------------------------------------------------------------------
# Loss output shapes / types
# ---------------------------------------------------------------------------

class TestGRPOLossOutput:
    def _make_batch(self, bsz=4, seqlen=8):
        train_lp   = torch.randn(bsz, seqlen, requires_grad=True)
        rollout_lp = torch.randn(bsz, seqlen)
        mask       = torch.ones(bsz, seqlen)
        advantages = torch.randn(bsz)
        return train_lp, rollout_lp, mask, advantages

    def test_returns_grpo_loss_output(self):
        train_lp, rollout_lp, mask, advantages = self._make_batch()
        out = grpo_loss_with_tis(train_lp, rollout_lp, advantages, mask)
        assert isinstance(out, GRPOLossOutput)

    def test_loss_is_scalar(self):
        train_lp, rollout_lp, mask, advantages = self._make_batch()
        out = grpo_loss_with_tis(train_lp, rollout_lp, advantages, mask)
        assert out.loss.shape == ()

    def test_loss_is_finite(self):
        train_lp, rollout_lp, mask, advantages = self._make_batch()
        out = grpo_loss_with_tis(train_lp, rollout_lp, advantages, mask)
        assert torch.isfinite(out.loss)

    def test_tis_diagnostics_present_when_rollout_given(self):
        train_lp, rollout_lp, mask, advantages = self._make_batch()
        out = grpo_loss_with_tis(train_lp, rollout_lp, advantages, mask)
        assert out.tis_diagnostics is not None

    def test_tis_diagnostics_absent_when_no_rollout(self):
        train_lp, _, mask, advantages = self._make_batch()
        out = grpo_loss_with_tis(train_lp, None, advantages, mask)
        assert out.tis_diagnostics is None

    def test_mismatch_diagnostics_present_when_rollout_given(self):
        train_lp, rollout_lp, mask, advantages = self._make_batch()
        out = grpo_loss_with_tis(train_lp, rollout_lp, advantages, mask)
        assert out.mismatch_diagnostics is not None

    def test_entropy_is_positive(self):
        # Entropy should generally be positive for random logprobs
        bsz, seqlen = 8, 16
        # Use negative logprobs (log π ≤ 0 for valid probs)
        train_lp   = -torch.rand(bsz, seqlen)
        rollout_lp = -torch.rand(bsz, seqlen)
        mask       = torch.ones(bsz, seqlen)
        advantages = torch.randn(bsz)

        out = grpo_loss_with_tis(train_lp, rollout_lp, advantages, mask)
        assert out.entropy.item() > 0.0


# ---------------------------------------------------------------------------
# Masking: only response tokens contribute
# ---------------------------------------------------------------------------

class TestGRPOLossMasking:
    def test_zero_mask_produces_finite_loss(self):
        bsz, seqlen = 4, 8
        train_lp   = torch.randn(bsz, seqlen)
        rollout_lp = torch.randn(bsz, seqlen)
        mask       = torch.zeros(bsz, seqlen)
        advantages = torch.randn(bsz)

        out = grpo_loss_with_tis(train_lp, rollout_lp, advantages, mask)
        assert torch.isfinite(out.loss)

    def test_masked_out_tokens_dont_affect_loss(self):
        # Two batches identical except for positions beyond the mask
        bsz, seqlen = 2, 8
        train_lp_a = torch.randn(bsz, seqlen)
        train_lp_b = train_lp_a.clone()
        train_lp_b[:, 4:] = 999.0   # garbage beyond mask

        rollout_lp = torch.randn(bsz, seqlen)
        mask       = torch.zeros(bsz, seqlen)
        mask[:, :4] = 1.0
        advantages = torch.randn(bsz)

        out_a = grpo_loss_with_tis(train_lp_a, rollout_lp, advantages, mask)
        out_b = grpo_loss_with_tis(train_lp_b, rollout_lp, advantages, mask)

        assert out_a.loss.item() == pytest.approx(out_b.loss.item(), rel=1e-5)


# ---------------------------------------------------------------------------
# as_dict
# ---------------------------------------------------------------------------

class TestGRPOLossAsDict:
    def test_as_dict_has_required_keys(self):
        bsz, seqlen = 4, 8
        train_lp   = torch.randn(bsz, seqlen)
        rollout_lp = torch.randn(bsz, seqlen)
        mask       = torch.ones(bsz, seqlen)
        advantages = torch.randn(bsz)

        out = grpo_loss_with_tis(train_lp, rollout_lp, advantages, mask)
        d   = out.as_dict()

        assert "loss"        in d
        assert "policy_loss" in d
        assert "entropy"     in d
        assert "tis/ratio_mean" in d
