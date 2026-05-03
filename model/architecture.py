"""Decoder-only Transformer architecture with RMSNorm, RoPE, GQA, SwiGLU, and SDPA.

Design choices
--------------
* **RMSNorm** instead of LayerNorm — faster and equally stable.
* **Rotary Position Embeddings (RoPE)** — applied to Q and K before attention.
* **Grouped-Query Attention (GQA)** — n_kv_heads ≤ n_heads for KV-cache savings.
* **SwiGLU MLP** — three-linear-layer gated activation (w1, w2 gate; w3 down-proj).
* **scaled_dot_product_attention (SDPA)** — uses FlashAttention kernel when available.
* **Tied embeddings** — optional weight sharing between token embed and output proj.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x * norm


# ---------------------------------------------------------------------------
# Rotary Position Embeddings
# ---------------------------------------------------------------------------


def _precompute_rope_freqs(
    head_dim: int, max_seq_len: int, theta: float = 10000.0, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) caches of shape (max_seq_len, head_dim)."""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)  # (T, half)
    freqs = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
    return freqs.cos(), freqs.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to *x* of shape (..., T, head_dim) using cached (cos, sin)."""
    T = x.shape[-2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Grouped-Query Attention
# ---------------------------------------------------------------------------


class GroupedQueryAttention(nn.Module):
    """Multi-head (or grouped-query) causal self-attention.

    When ``n_kv_heads == n_heads`` this is standard MHA.
    When ``n_kv_heads < n_heads`` each KV head is shared across a group of Q heads
    (GQA — Ainslie et al., 2023).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.n_heads % cfg.n_kv_heads == 0  # type: ignore[operator]
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads  # type: ignore[assignment]
        self.head_dim = cfg.d_model // cfg.n_heads
        self.groups = cfg.n_heads // cfg.n_kv_heads  # type: ignore[operator]
        self.dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Expand KV heads to match Q heads (GQA broadcast)
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)

        # SDPA — uses FlashAttention2 when available (torch >= 2.0)
        attn_drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=attn_drop)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward
# ---------------------------------------------------------------------------


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward: out = W3( SiLU(W1(x)) * W2(x) )."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)  # gate
        self.w2 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)  # value
        self.w3 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)  # down
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Pre-norm residual block: Attention + MLP."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = GroupedQueryAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLUMLP(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), cos, sin))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full Language Model
# ---------------------------------------------------------------------------


class TransformerLM(nn.Module):
    """Causal decoder-only language model.

    Usage::

        cfg = ModelConfig.tiny()
        model = TransformerLM.from_config(cfg)
        logits = model(input_ids)   # (B, T, vocab_size)
        loss   = model(input_ids, labels)  # scalar
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # Pre-compute RoPE cache
        head_dim = cfg.d_model // cfg.n_heads
        cos, sin = _precompute_rope_freqs(head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02
        nn.init.normal_(self.embed.weight, std=std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> TransformerLM:
        """Construct from a ``ModelConfig`` instance."""
        return cls(cfg)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        input_ids:
            Long tensor of shape ``(B, T)``.
        labels:
            Optional target ids of shape ``(B, T)``.  When provided the method
            returns the cross-entropy loss scalar instead of logits.
        """
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"
        )

        x = self.drop(self.embed(input_ids))
        cos = self.rope_cos[:T]  # type: ignore[index]
        sin = self.rope_sin[:T]  # type: ignore[index]

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if labels is not None:
            # Shift so that token i predicts i+1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            return loss  # type: ignore[return-value]

        return logits

    def num_params(self, non_embedding: bool = True) -> int:
        """Count trainable parameters (optionally excluding embeddings)."""
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.embed.weight.numel()
        return total

    @staticmethod
    def estimate_mfu(n_params: int, tokens_per_sec: float, gpu_flops: float = 312e12) -> float:
        """Estimate Model FLOPs Utilisation (MFU).

        ``gpu_flops`` defaults to A100 BF16 peak (312 TFLOPs/s).
        Formula: MFU ≈ 6·N·tokens_per_sec / gpu_flops.
        """
        return 6 * n_params * tokens_per_sec / gpu_flops
