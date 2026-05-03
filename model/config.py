"""Model configuration dataclass for the decoder-only Transformer LLM."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Hyperparameters that fully specify a TransformerLM architecture.

    All fields have sensible defaults for a tiny debug model (≈2 M params).
    Override via ``from_dict`` or by passing keyword arguments.
    """

    # Vocabulary
    vocab_size: int = 32000
    # Hidden dimension
    d_model: int = 128
    # Number of transformer layers
    n_layers: int = 2
    # Number of query heads
    n_heads: int = 4
    # Number of key/value heads (GQA); must evenly divide n_heads. None → MHA (n_heads).
    n_kv_heads: int | None = None
    # Feed-forward hidden dimension (defaults to 4 × d_model if left at 0)
    d_ff: int = 0
    # Maximum sequence length
    max_seq_len: int = 512
    # RoPE base frequency
    rope_theta: float = 10000.0
    # Tie input embeddings ↔ output projection weights
    tie_embeddings: bool = True
    # Dropout probability (0 = disabled, good for large-scale training)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.d_ff == 0:
            # SwiGLU convention: use 8/3 × d_model, rounded to nearest multiple of 256
            raw = int(8 * self.d_model / 3)
            self.d_ff = ((raw + 255) // 256) * 256

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """Construct from a plain dictionary (e.g., loaded from YAML)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    # Convenience presets --------------------------------------------------------

    @classmethod
    def tiny(cls) -> "ModelConfig":
        """Tiny debug model — runs on CPU in seconds."""
        return cls(d_model=128, n_layers=2, n_heads=4, max_seq_len=128)

    @classmethod
    def small_125m(cls) -> "ModelConfig":
        """~125 M parameter model (GPT-2 scale)."""
        return cls(
            vocab_size=32000,
            d_model=768,
            n_layers=12,
            n_heads=12,
            n_kv_heads=12,
            d_ff=3072,
            max_seq_len=2048,
        )

    @classmethod
    def medium_1b(cls) -> "ModelConfig":
        """~1 B parameter model."""
        return cls(
            vocab_size=32000,
            d_model=2048,
            n_layers=24,
            n_heads=16,
            n_kv_heads=8,  # GQA: 2× compression
            d_ff=5632,
            max_seq_len=4096,
        )
