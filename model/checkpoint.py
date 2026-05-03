"""Checkpoint helpers: save, load, and auto-resume for distributed training."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    loss: float,
    out_dir: str | Path,
    keep_last: int = 3,
) -> Path:
    """Persist training state to disk.

    Saves a ``checkpoint_step_{step:07d}.pt`` file containing model weights,
    optimiser state, scheduler state, and metadata.

    Parameters
    ----------
    keep_last:
        Retain only this many most-recent checkpoints (plus the best).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "step": step,
        "loss": loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
    }
    path = out_dir / f"checkpoint_step_{step:07d}.pt"
    torch.save(ckpt, path)
    logger.info("Saved checkpoint → %s", path)

    # Prune old checkpoints
    _prune_checkpoints(out_dir, keep_last)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load checkpoint and restore model/optimizer/scheduler states.

    Returns the metadata dict with at least ``step`` and ``loss`` keys.
    """
    path = Path(path)
    logger.info("Loading checkpoint from %s", path)
    ckpt = torch.load(path, map_location=device, weights_only=True)

    # Strip DDP/FSDP prefix if present
    state = ckpt["model_state"]
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state)

    if optimizer is not None and ckpt.get("optimizer_state"):
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])

    return {"step": ckpt["step"], "loss": ckpt["loss"]}


def find_latest_checkpoint(ckpt_dir: str | Path) -> Path | None:
    """Return the path of the most recent checkpoint in *ckpt_dir*, or None."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None
    pattern = re.compile(r"checkpoint_step_(\d+)\.pt$")
    candidates = []
    for f in ckpt_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            candidates.append((int(m.group(1)), f))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _prune_checkpoints(ckpt_dir: Path, keep_last: int) -> None:
    """Delete old checkpoints, keeping *keep_last* most-recent files."""
    pattern = re.compile(r"checkpoint_step_(\d+)\.pt$")
    candidates = sorted(
        [(int(m.group(1)), f) for f in ckpt_dir.iterdir() if (m := pattern.match(f.name))],
        key=lambda x: x[0],
    )
    for _, path in candidates[:-keep_last]:
        path.unlink(missing_ok=True)
        logger.debug("Pruned old checkpoint %s", path)
