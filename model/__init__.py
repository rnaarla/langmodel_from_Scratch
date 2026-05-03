"""LLM model package — architecture, config, training, and checkpointing."""

from model.architecture import TransformerLM
from model.config import ModelConfig

__all__ = ["TransformerLM", "ModelConfig"]
