"""Train a 32k ByteLevel BPE tokenizer on a directory of .txt files.

The trained tokenizer is saved to ``tokenizer/artifacts/`` and can be loaded
by the HuggingFace ``tokenizers`` library or converted to a ``PreTrainedTokenizerFast``.

Usage::

    python tokenizer/train_tokenizer.py \\
        --input-dir data/cleaned \\
        --vocab-size 32000 \\
        --output-dir tokenizer/artifacts
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def train_tokenizer(
    input_dir: Path,
    vocab_size: int,
    output_dir: Path,
    min_frequency: int = 2,
    glob_pattern: str = "*.txt",
) -> None:
    """Train a ByteLevelBPETokenizer and save artifacts.

    Parameters
    ----------
    input_dir:
        Directory containing ``.txt`` training files.
    vocab_size:
        Target vocabulary size (e.g., 32000 for LLaMA-style).
    output_dir:
        Directory where ``vocab.json`` and ``merges.txt`` will be saved.
    min_frequency:
        Minimum pair frequency for a merge to be included.
    glob_pattern:
        Glob pattern to select training files.
    """
    from tokenizers import ByteLevelBPETokenizer  # noqa: PLC0415

    files = sorted(input_dir.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{glob_pattern}' found in {input_dir}. "
            "Please populate data/cleaned/ with .txt files first."
        )
    logger.info("Found %d training files in %s", len(files), input_dir)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(f) for f in files],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(output_dir))
    logger.info("Tokenizer saved to %s (vocab_size=%d)", output_dir, tokenizer.get_vocab_size())

    # Validation: bytes-per-token on a sample from the first file
    sample_text = files[0].read_text(encoding="utf-8", errors="ignore")[:5000]
    if sample_text:
        ids = tokenizer.encode(sample_text).ids
        bpt = len(sample_text.encode("utf-8")) / max(1, len(ids))
        logger.info("Validation — bytes/token on sample: %.2f (target: 3.5–4.5 for English)", bpt)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a ByteLevel BPE tokenizer.")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/cleaned"),
        help="Directory of .txt training files.",
    )
    p.add_argument("--vocab-size", type=int, default=32000, help="Target vocabulary size.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tokenizer/artifacts"),
        help="Output directory for vocab.json and merges.txt.",
    )
    p.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum merge pair frequency.",
    )
    p.add_argument(
        "--glob",
        default="*.txt",
        help="Glob pattern to select training files (default: *.txt).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_tokenizer(
        input_dir=args.input_dir,
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
        min_frequency=args.min_frequency,
        glob_pattern=args.glob,
    )
