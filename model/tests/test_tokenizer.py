"""Unit tests for tokenizer round-trip and special tokens.

These tests use the HuggingFace ``tokenizers`` library's ByteLevel BPE tokenizer
that the project trains in ``tokenizer/train_tokenizer.py``.

When the trained tokenizer artifacts don't exist yet (fresh CI clone), the tests
fall back to a minimal in-memory tokenizer trained on a tiny corpus so that CI
always passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "tokenizer" / "artifacts"
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


# ---------------------------------------------------------------------------
# Fixture: load or build a tokenizer
# ---------------------------------------------------------------------------


def _build_tiny_tokenizer():
    """Return a ByteLevel BPE tokenizer trained on a tiny corpus."""
    from tokenizers import ByteLevelBPETokenizer

    tok = ByteLevelBPETokenizer()
    corpus = [
        "Hello world! This is a test sentence.",
        "The quick brown fox jumps over the lazy dog.",
        "Building an LLM from scratch is a fascinating journey.",
        "Transformers use attention mechanisms for sequence modelling.",
        "Data quality dominates model quality in large language models.",
    ]
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(corpus))
        tmp_path = f.name

    try:
        tok.train(
            files=[tmp_path],
            vocab_size=500,
            min_frequency=1,
            special_tokens=SPECIAL_TOKENS,
        )
    finally:
        os.unlink(tmp_path)
    return tok


@pytest.fixture(scope="module")
def tokenizer():
    """Load from artifacts if available, else build a tiny in-memory tokenizer."""
    vocab_file = ARTIFACTS_DIR / "vocab.json"
    merges_file = ARTIFACTS_DIR / "merges.txt"

    if vocab_file.exists() and merges_file.exists():
        from tokenizers import ByteLevelBPETokenizer

        return ByteLevelBPETokenizer(str(vocab_file), str(merges_file))

    return _build_tiny_tokenizer()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_basic_roundtrip(self, tokenizer) -> None:
        """Encoding then decoding must recover the original string."""
        text = "Hello, world!"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        # ByteLevel BPE may add a leading space; strip for comparison
        assert text in decoded or decoded.strip() == text.strip()

    def test_roundtrip_longer_text(self, tokenizer) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        ids = tokenizer.encode(text).ids
        assert len(ids) > 0
        decoded = tokenizer.decode(ids)
        assert text.replace(" ", "") in decoded.replace(" ", "")

    def test_ids_are_integers(self, tokenizer) -> None:
        ids = tokenizer.encode("test sentence").ids
        assert all(isinstance(i, int) for i in ids)

    def test_empty_string(self, tokenizer) -> None:
        """Encoding an empty string should not raise."""
        ids = tokenizer.encode("").ids
        assert isinstance(ids, list)


class TestSpecialTokens:
    def test_special_tokens_in_vocab(self, tokenizer) -> None:
        vocab = tokenizer.get_vocab()
        for tok in SPECIAL_TOKENS:
            assert tok in vocab, f"Special token {tok!r} missing from vocabulary"

    def test_special_token_ids_unique(self, tokenizer) -> None:
        vocab = tokenizer.get_vocab()
        ids = [vocab[t] for t in SPECIAL_TOKENS]
        assert len(ids) == len(set(ids)), "Special token IDs are not unique"

    def test_vocab_size_positive(self, tokenizer) -> None:
        assert tokenizer.get_vocab_size() > 0


class TestBytesPerToken:
    def test_bytes_per_token_range(self, tokenizer) -> None:
        """For English text, bytes/token should be in [1, 10] range."""
        text = (
            "Building a large language model from scratch requires careful attention to "
            "data quality, model architecture, and training stability."
        )
        ids = tokenizer.encode(text).ids
        bpt = len(text.encode("utf-8")) / max(1, len(ids))
        assert 1.0 <= bpt <= 10.0, f"Unexpected bytes/token: {bpt:.2f}"
