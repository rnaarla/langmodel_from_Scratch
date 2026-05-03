"""Tokenize cleaned text and write memory-mapped binary shards.

Reads a directory of ``.parquet`` / ``.txt`` cleaned files, tokenises each document
with the trained tokenizer, and concatenates tokens into fixed-size ``uint16`` or
``uint32`` NumPy memmap ``.bin`` shards.  An accompanying ``.json`` index records
the shard manifest.

Usage::

    python data/shard.py \\
        --input-dir data/cleaned \\
        --tokenizer-dir tokenizer/artifacts \\
        --output-dir data/tokenized \\
        --shard-size 500000000  # tokens per shard (~1 GB at uint16)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

DEFAULT_SHARD_SIZE = 500_000_000  # tokens per shard


def load_tokenizer(tokenizer_dir: Path):
    """Load the ByteLevel BPE tokenizer from artifacts."""
    from tokenizers import ByteLevelBPETokenizer  # noqa: PLC0415

    vocab = tokenizer_dir / "vocab.json"
    merges = tokenizer_dir / "merges.txt"
    if not (vocab.exists() and merges.exists()):
        raise FileNotFoundError(
            f"Tokenizer artifacts not found in {tokenizer_dir}. "
            "Run `make tokenizer` first."
        )
    return ByteLevelBPETokenizer(str(vocab), str(merges))


def iter_texts(input_dir: Path):
    """Yield raw text strings from .parquet and .txt files."""
    for path in sorted(input_dir.rglob("*.parquet")):
        try:
            import pyarrow.parquet as pq  # noqa: PLC0415

            tbl = pq.read_table(path, columns=["text"])
            for txt in tbl.column("text").to_pylist():
                if txt:
                    yield txt
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s: %s", path, exc)

    for path in sorted(input_dir.rglob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            yield text


def shard(
    input_dir: Path,
    tokenizer_dir: Path,
    output_dir: Path,
    shard_size: int = DEFAULT_SHARD_SIZE,
    dtype: str = "uint16",
) -> None:
    """Tokenise corpus and write memmap shards.

    Parameters
    ----------
    dtype:
        ``uint16`` (max vocab ~65 k) or ``uint32`` for larger vocabularies.
    """
    np_dtype = np.dtype(dtype)
    max_id = 2 ** (np_dtype.itemsize * 8) - 1

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(tokenizer_dir)

    # Check vocab fits in dtype
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size - 1 > max_id:
        logger.warning(
            "Vocab size %d exceeds max for dtype %s (%d). Consider using uint32.",
            vocab_size,
            dtype,
            max_id,
        )

    shard_idx = 0
    buf: list[int] = []
    manifest: list[dict] = []
    total_tokens = 0

    def _flush(buf: list[int]) -> None:
        nonlocal shard_idx
        arr = np.array(buf, dtype=np_dtype)
        out_path = output_dir / f"shard_{shard_idx:04d}.bin"
        fp = np.memmap(str(out_path), dtype=np_dtype, mode="w+", shape=(len(arr),))
        fp[:] = arr
        del fp  # flush to disk
        manifest.append({"shard": out_path.name, "tokens": len(arr)})
        logger.info("Wrote shard %s (%d tokens)", out_path.name, len(arr))
        shard_idx += 1

    for text in iter_texts(input_dir):
        ids = tokenizer.encode(text).ids
        buf.extend(ids)
        total_tokens += len(ids)

        while len(buf) >= shard_size:
            _flush(buf[:shard_size])
            buf = buf[shard_size:]

    if buf:
        _flush(buf)

    index_path = output_dir / "index.json"
    index_path.write_text(
        json.dumps(
            {"dtype": dtype, "vocab_size": vocab_size, "total_tokens": total_tokens,
             "shards": manifest},
            indent=2,
        )
    )
    logger.info(
        "Sharding complete — %d shards, %d total tokens. Index → %s",
        shard_idx,
        total_tokens,
        index_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tokenize and shard a cleaned text corpus.")
    p.add_argument("--input-dir", type=Path, default=Path("data/cleaned"))
    p.add_argument("--tokenizer-dir", type=Path, default=Path("tokenizer/artifacts"))
    p.add_argument("--output-dir", type=Path, default=Path("data/tokenized"))
    p.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help="Number of tokens per shard.",
    )
    p.add_argument(
        "--dtype",
        choices=["uint16", "uint32"],
        default="uint16",
        help="NumPy dtype for token IDs.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    shard(
        input_dir=args.input_dir,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        dtype=args.dtype,
    )
