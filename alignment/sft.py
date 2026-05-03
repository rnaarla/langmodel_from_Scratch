"""Supervised Fine-Tuning (SFT) skeleton for instruction-following alignment.

Loads ``instruction / input / output`` JSONL triples, formats them with a
prompt template, and fine-tunes a TransformerLM while masking the loss so that
only response tokens contribute to the gradient.

Usage::

    python alignment/sft.py \\
        --model-dir checkpoints/pretrained \\
        --data-path data/eval/instructions.jsonl \\
        --output-dir checkpoints/sft \\
        --epochs 3 --lr 2e-5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import TransformerLM
from model.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from model.config import ModelConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)
IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class InstructionDataset(torch.utils.data.Dataset):
    """Reads JSONL with ``instruction``, ``input``, ``output`` fields."""

    def __init__(self, path: Path, tokenizer, max_seq_len: int = 2048) -> None:
        self.samples: list[dict] = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        logger.info("Loaded %d instruction samples from %s", len(self.samples), path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.samples[idx]
        prompt = PROMPT_TEMPLATE.format(
            instruction=ex.get("instruction", ""),
            input=ex.get("input", ""),
        )
        response = ex.get("output", "")
        full_text = prompt + response

        prompt_ids = self.tokenizer.encode(prompt).ids
        full_ids = self.tokenizer.encode(full_text).ids

        # Truncate to max_seq_len
        full_ids = full_ids[: self.max_seq_len]
        prompt_len = min(len(prompt_ids), len(full_ids))

        input_ids = torch.tensor(full_ids, dtype=torch.long)

        # Labels: mask prompt tokens with IGNORE_INDEX so loss only covers response
        labels = input_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX

        return {"input_ids": input_ids, "labels": labels}


def _collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Left-pad all tensors in the batch to the longest sequence."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids_list, labels_list = [], []
    for b in batch:
        pad_len = max_len - b["input_ids"].shape[0]
        input_ids_list.append(F.pad(b["input_ids"], (pad_len, 0), value=0))
        labels_list.append(F.pad(b["labels"], (pad_len, 0), value=IGNORE_INDEX))
    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def sft_train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    from tokenizers import ByteLevelBPETokenizer  # noqa: PLC0415

    tok_dir = Path(args.tokenizer_dir)
    tokenizer = ByteLevelBPETokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))

    # Load model
    model_dir = Path(args.model_dir)
    cfg = ModelConfig.small_125m()  # TODO: load from model_dir/config.yaml
    model = TransformerLM.from_config(cfg).to(device)

    latest = find_latest_checkpoint(model_dir)
    if latest:
        load_checkpoint(latest, model, device=device)
        logger.info("Loaded base model from %s", latest)
    else:
        logger.warning("No checkpoint found in %s — training from scratch.", model_dir)

    dataset = InstructionDataset(Path(args.data_path), tokenizer, cfg.max_seq_len)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    output_dir = Path(args.output_dir)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu",
                                    dtype=torch.bfloat16):
                # Use model's built-in loss (it handles IGNORE_INDEX via F.cross_entropy)
                loss = model(input_ids, labels=labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()

            if step % 50 == 0:
                logger.info("epoch=%d step=%d loss=%.4f", epoch + 1, step, loss.item())

        avg_loss = total_loss / max(1, len(loader))
        logger.info("Epoch %d complete — avg_loss=%.4f", epoch + 1, avg_loss)
        save_checkpoint(model, optimizer, None, epoch, avg_loss, output_dir)

    logger.info("SFT complete. Checkpoints saved to %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) for instruction following.")
    p.add_argument("--model-dir", default="checkpoints/pretrained", help="Base model checkpoint dir.")
    p.add_argument("--tokenizer-dir", default="tokenizer/artifacts")
    p.add_argument(
        "--data-path",
        default="data/eval/instructions.jsonl",
        help="Path to JSONL instruction dataset.",
    )
    p.add_argument("--output-dir", default="checkpoints/sft")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    return p.parse_args()


if __name__ == "__main__":
    sft_train(_parse_args())
