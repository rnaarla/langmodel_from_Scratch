"""Direct Preference Optimization (DPO) alignment skeleton.

Implements the DPO loss (Rafailov et al., 2023) against a frozen reference model.

Dataset format (JSONL)::

    {"prompt": "...", "chosen": "...", "rejected": "..."}

Usage::

    python alignment/dpo.py \\
        --model-dir checkpoints/sft \\
        --ref-model-dir checkpoints/sft \\
        --data-path data/eval/preferences.jsonl \\
        --output-dir checkpoints/dpo \\
        --beta 0.1
"""

from __future__ import annotations

import argparse
import copy
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

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------


def dpo_loss(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    input_ids_chosen: torch.Tensor,
    input_ids_rejected: torch.Tensor,
    labels_chosen: torch.Tensor,
    labels_rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Compute the DPO loss for a batch of chosen/rejected pairs.

    DPO objective (simplified)::

        loss = -E[ log σ( β · (log π(chosen) - log π(rejected)
                             - log π_ref(chosen) + log π_ref(rejected)) ) ]

    Parameters
    ----------
    beta:
        Temperature parameter controlling deviation from the reference policy.
        Lower β → closer to reference; higher β → more aggressive preference.
    """

    def _log_prob(m: torch.nn.Module, ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Sum of per-token log-probabilities for response tokens (labels != IGNORE_INDEX)."""
        logits = m(ids)  # (B, T, V)
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, T-1, V)
        tgt = labels[:, 1:].clone()  # (B, T-1)

        mask = tgt != IGNORE_INDEX
        tgt[~mask] = 0  # avoid index out of bounds
        gathered = log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
        return (gathered * mask).sum(-1)  # (B,)

    with torch.no_grad():
        logp_chosen_ref = _log_prob(ref_model, input_ids_chosen, labels_chosen)
        logp_rejected_ref = _log_prob(ref_model, input_ids_rejected, labels_rejected)

    logp_chosen = _log_prob(model, input_ids_chosen, labels_chosen)
    logp_rejected = _log_prob(model, input_ids_rejected, labels_rejected)

    reward = beta * (
        (logp_chosen - logp_chosen_ref) - (logp_rejected - logp_rejected_ref)
    )
    loss = -F.logsigmoid(reward).mean()
    return loss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PreferenceDataset(torch.utils.data.Dataset):
    """Reads JSONL with ``prompt``, ``chosen``, ``rejected`` fields."""

    def __init__(self, path: Path, tokenizer, max_seq_len: int = 1024) -> None:
        self.samples: list[dict] = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        logger.info("Loaded %d preference pairs from %s", len(self.samples), path)

    def _encode(self, prompt: str, response: str) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_ids = self.tokenizer.encode(prompt).ids
        full_ids = self.tokenizer.encode(prompt + response).ids[: self.max_seq_len]
        prompt_len = min(len(prompt_ids), len(full_ids))
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX
        return input_ids, labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.samples[idx]
        ids_c, lbl_c = self._encode(ex["prompt"], ex["chosen"])
        ids_r, lbl_r = self._encode(ex["prompt"], ex["rejected"])
        return {
            "input_ids_chosen": ids_c,
            "labels_chosen": lbl_c,
            "input_ids_rejected": ids_r,
            "labels_rejected": lbl_r,
        }


def _pad(tensors: list[torch.Tensor], value: int) -> torch.Tensor:
    max_len = max(t.shape[0] for t in tensors)
    return torch.stack([F.pad(t, (max_len - t.shape[0], 0), value=value) for t in tensors])


def _collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "input_ids_chosen": _pad([b["input_ids_chosen"] for b in batch], 0),
        "labels_chosen": _pad([b["labels_chosen"] for b in batch], IGNORE_INDEX),
        "input_ids_rejected": _pad([b["input_ids_rejected"] for b in batch], 0),
        "labels_rejected": _pad([b["labels_rejected"] for b in batch], IGNORE_INDEX),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def dpo_train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from tokenizers import ByteLevelBPETokenizer  # noqa: PLC0415

    tok_dir = Path(args.tokenizer_dir)
    tokenizer = ByteLevelBPETokenizer(str(tok_dir / "vocab.json"), str(tok_dir / "merges.txt"))

    cfg = ModelConfig.small_125m()  # TODO: load from config
    model = TransformerLM.from_config(cfg).to(device)

    latest = find_latest_checkpoint(Path(args.model_dir))
    if latest:
        load_checkpoint(latest, model, device=device)
        logger.info("Loaded policy model from %s", latest)

    # Frozen reference model (deep copy)
    ref_model = copy.deepcopy(model)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    dataset = PreferenceDataset(Path(args.data_path), tokenizer)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    output_dir = Path(args.output_dir)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for step, batch in enumerate(loader):
            ids_c = batch["input_ids_chosen"].to(device)
            lbl_c = batch["labels_chosen"].to(device)
            ids_r = batch["input_ids_rejected"].to(device)
            lbl_r = batch["labels_rejected"].to(device)

            loss = dpo_loss(model, ref_model, ids_c, ids_r, lbl_c, lbl_r, beta=args.beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()

            if step % 50 == 0:
                logger.info("epoch=%d step=%d loss=%.4f", epoch + 1, step, loss.item())

        avg = total_loss / max(1, len(loader))
        logger.info("Epoch %d — avg DPO loss=%.4f", epoch + 1, avg)
        save_checkpoint(model, optimizer, None, epoch, avg, output_dir)

    logger.info("DPO training complete. Output → %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO preference optimisation.")
    p.add_argument("--model-dir", default="checkpoints/sft")
    p.add_argument("--tokenizer-dir", default="tokenizer/artifacts")
    p.add_argument("--data-path", default="data/eval/preferences.jsonl")
    p.add_argument("--output-dir", default="checkpoints/dpo")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--beta", type=float, default=0.1, help="DPO temperature.")
    return p.parse_args()


if __name__ == "__main__":
    dpo_train(_parse_args())
