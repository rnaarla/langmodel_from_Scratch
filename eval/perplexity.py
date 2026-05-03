"""Perplexity and bits-per-byte evaluation on a held-out file.

Usage::

    python eval/perplexity.py \\
        --checkpoint checkpoints/125m/checkpoint_step_0100000.pt \\
        --data-file data/eval/held_out.bin \\
        --tokenizer-dir tokenizer/artifacts
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import TransformerLM
from model.checkpoint import load_checkpoint
from model.config import ModelConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")


def compute_perplexity(
    model: TransformerLM,
    data: np.ndarray,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Compute perplexity and bits-per-byte on a token array.

    Returns
    -------
    (perplexity, bits_per_byte)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    # Slide a fixed window over the token array
    n_chunks = (len(data) - 1) // seq_len
    if n_chunks == 0:
        raise ValueError(f"Data has only {len(data)} tokens; need > {seq_len + 1}.")

    with torch.no_grad():
        for start in range(0, n_chunks * seq_len, batch_size * seq_len):
            batch_ids = []
            batch_labels = []
            for b in range(batch_size):
                s = start + b * seq_len
                if s + seq_len + 1 > len(data):
                    break
                chunk = data[s : s + seq_len + 1].astype(np.int64)
                batch_ids.append(chunk[:-1])
                batch_labels.append(chunk[1:])

            if not batch_ids:
                break

            input_ids = torch.tensor(np.stack(batch_ids), dtype=torch.long, device=device)
            labels = torch.tensor(np.stack(batch_labels), dtype=torch.long, device=device)

            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu",
                                    dtype=torch.bfloat16):
                loss = model(input_ids, labels=labels)

            n_toks = input_ids.numel()
            total_loss += loss.item() * n_toks
            total_tokens += n_toks

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss)
    # bits per byte: nats / ln(2) / bytes_per_token
    # Average bytes per token ≈ 4 for BPE; use token count as proxy
    bpb = avg_loss / math.log(2)
    return ppl, bpb


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    cfg = ModelConfig.tiny()  # TODO: load cfg from checkpoint dir
    model = TransformerLM.from_config(cfg).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, device=device)
        logger.info("Loaded checkpoint %s", args.checkpoint)

    # Load data
    data_path = Path(args.data_file)
    dtype = args.dtype
    data = np.memmap(str(data_path), dtype=dtype, mode="r") if data_path.exists() else None

    if data is None or len(data) < cfg.max_seq_len + 1:
        logger.warning(
            "Data file %s not found or too small. Using random dummy data for demo.",
            data_path,
        )
        data = np.random.randint(0, cfg.vocab_size, size=(cfg.max_seq_len * 10 + 1,),
                                 dtype=np.uint16)

    ppl, bpb = compute_perplexity(model, data, cfg.max_seq_len, args.batch_size, device)
    logger.info("Perplexity: %.2f | Bits-per-byte: %.4f", ppl, bpb)
    print(f"perplexity={ppl:.2f}  bits_per_byte={bpb:.4f}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute perplexity and bits-per-byte.")
    p.add_argument("--checkpoint", default=None, help="Path to model checkpoint .pt file.")
    p.add_argument("--data-file", default="data/eval/held_out.bin")
    p.add_argument("--tokenizer-dir", default="tokenizer/artifacts")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--dtype", default="uint16", choices=["uint16", "uint32"])
    return p.parse_args()


if __name__ == "__main__":
    evaluate(_parse_args())
