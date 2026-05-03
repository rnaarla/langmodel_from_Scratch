"""Unit tests for the TransformerLM architecture.

Tests are intentionally lightweight (CPU, tiny model) so they run quickly in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model.architecture import RMSNorm, TransformerLM, _precompute_rope_freqs, apply_rope
from model.config import ModelConfig


@pytest.fixture
def tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=256,
        max_seq_len=64,
        tie_embeddings=True,
        dropout=0.0,
    )


@pytest.fixture
def tiny_model(tiny_cfg: ModelConfig) -> TransformerLM:
    return TransformerLM.from_config(tiny_cfg)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


class TestShapes:
    def test_forward_logits_shape(self, tiny_model: TransformerLM) -> None:
        """Output tensor must be (B, T, vocab_size)."""
        B, T = 2, 32
        ids = torch.randint(0, 256, (B, T))
        logits = tiny_model(ids)
        assert logits.shape == (B, T, 256)

    def test_forward_loss_scalar(self, tiny_model: TransformerLM) -> None:
        """With labels, forward must return a scalar loss."""
        ids = torch.randint(0, 256, (2, 32))
        loss = tiny_model(ids, labels=ids)
        assert loss.ndim == 0, "Expected scalar loss"

    def test_loss_is_finite(self, tiny_model: TransformerLM) -> None:
        ids = torch.randint(0, 256, (2, 32))
        loss = tiny_model(ids, labels=ids)
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"

    def test_num_params(self, tiny_model: TransformerLM) -> None:
        n = tiny_model.num_params(non_embedding=False)
        assert n > 0

    def test_tied_embeddings(self, tiny_cfg: ModelConfig) -> None:
        tiny_cfg.tie_embeddings = True
        m = TransformerLM.from_config(tiny_cfg)
        assert m.embed.weight is m.lm_head.weight, "Tied weights must share storage"

    def test_untied_embeddings(self, tiny_cfg: ModelConfig) -> None:
        tiny_cfg.tie_embeddings = False
        m = TransformerLM.from_config(tiny_cfg)
        assert m.embed.weight is not m.lm_head.weight


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class TestRMSNorm:
    def test_output_shape(self) -> None:
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        assert norm(x).shape == x.shape

    def test_normalises(self) -> None:
        norm = RMSNorm(64)
        x = torch.randn(4, 64) * 100
        y = norm(x)
        # After RMSNorm the RMS ≈ 1 (times the weight, which starts at 1)
        rms = y.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


class TestRoPE:
    def test_cos_sin_shape(self) -> None:
        cos, sin = _precompute_rope_freqs(head_dim=32, max_seq_len=128)
        assert cos.shape == (128, 32)
        assert sin.shape == (128, 32)

    def test_apply_rope_shape(self) -> None:
        cos, sin = _precompute_rope_freqs(32, 64)
        x = torch.randn(2, 4, 16, 32)  # (B, H, T, D)
        out = apply_rope(x, cos, sin)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Causal masking / gradient flow
# ---------------------------------------------------------------------------


class TestCausalMask:
    def test_future_tokens_not_attended(self, tiny_model: TransformerLM) -> None:
        """Changing a future token must NOT change the output at the current position."""
        tiny_model.eval()
        ids = torch.randint(0, 256, (1, 16))

        with torch.no_grad():
            logits_orig = tiny_model(ids.clone())

        ids_mod = ids.clone()
        ids_mod[0, -1] = (ids_mod[0, -1] + 1) % 256  # change last token
        with torch.no_grad():
            logits_mod = tiny_model(ids_mod)

        # Position 0 should be identical (can't see position 15)
        assert torch.allclose(logits_orig[0, 0], logits_mod[0, 0], atol=1e-6), (
            "Causal mask violated: future token affects current position"
        )

    def test_gradient_flows_to_all_params(self, tiny_model: TransformerLM) -> None:
        """Every parameter with requires_grad must receive a gradient."""
        ids = torch.randint(0, 256, (1, 16))
        loss = tiny_model(ids, labels=ids)
        loss.backward()

        no_grad = [
            n
            for n, p in tiny_model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert not no_grad, f"Parameters with no gradient: {no_grad}"


# ---------------------------------------------------------------------------
# from_config constructor
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_tiny_preset(self) -> None:
        m = TransformerLM.from_config(ModelConfig.tiny())
        assert m is not None

    def test_gqa_heads(self) -> None:
        cfg = ModelConfig(d_model=128, n_heads=4, n_kv_heads=2, n_layers=1, d_ff=256)
        m = TransformerLM.from_config(cfg)
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        assert m(ids).shape == (1, 8, cfg.vocab_size)
