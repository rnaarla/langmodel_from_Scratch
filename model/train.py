"""Distributed pre-training loop for the decoder-only TransformerLM.

Designed to be launched with ``torchrun`` for multi-GPU / multi-node training,
with a transparent single-GPU fallback when only one device is available.

Features
--------
* BF16 autocast with ``torch.amp.GradScaler`` (no-op in BF16 mode).
* AdamW with β=(0.9, 0.95) and weight decay.
* Cosine LR schedule with linear warm-up.
* Gradient clipping at 1.0.
* Periodic checkpointing with auto-resume.
* Logs loss, LR, grad_norm, tokens/s, and MFU to stdout + MLflow (optional).

Launch example::

    torchrun --nproc_per_node=8 model/train.py \\
        --config config/pretrain_125m.yaml \\
        --data-dir data/tokenized \\
        --checkpoint-dir checkpoints/125m
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import yaml

# Add project root so that `model` package is importable when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import TransformerLM
from model.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from model.config import ModelConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

# ---------------------------------------------------------------------------
# Dataset helpers (simple memmap streaming)
# ---------------------------------------------------------------------------


class MemMapDataset(torch.utils.data.Dataset):
    """Reads a flat uint16 / uint32 binary shard produced by ``data/shard.py``."""

    def __init__(self, path: str | Path, seq_len: int, dtype: str = "uint16") -> None:
        import numpy as np

        self.seq_len = seq_len
        data = np.memmap(str(path), dtype=dtype, mode="r")
        # Truncate to a multiple of (seq_len + 1) so every sample is complete.
        n = (len(data) // (seq_len + 1)) * (seq_len + 1)
        self.data = data[:n]

    def __len__(self) -> int:
        return len(self.data) // (self.seq_len + 1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * (self.seq_len + 1)
        chunk = torch.from_numpy(self.data[start : start + self.seq_len + 1].astype("int64"))
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


def _build_dataloader(data_dir: Path, seq_len: int, batch_size: int, rank: int, world_size: int):
    """Construct a DataLoader over all .bin shards in *data_dir*."""
    shards = sorted(data_dir.glob("*.bin"))
    if not shards:
        raise FileNotFoundError(f"No .bin shards found in {data_dir}")

    datasets = [MemMapDataset(s, seq_len) for s in shards]
    dataset = torch.utils.data.ConcatDataset(datasets)
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def cosine_lr_lambda(step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    """Return the LR multiplier for a given *step*."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(min_lr_ratio, cosine)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _try_log_mlflow(metrics: dict, step: int) -> None:
    """Log *metrics* to MLflow if available (silently skipped otherwise)."""
    try:
        import mlflow  # noqa: PLC0415

        for k, v in metrics.items():
            mlflow.log_metric(k, v, step=step)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    # ---- Distributed setup ----
    dist_available = torch.distributed.is_available()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = rank == 0

    if dist_available and world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main:
        logger.info("Device: %s | World size: %d", device, world_size)

    # ---- Config ----
    if args.config:
        with open(args.config) as f:
            cfg_dict = yaml.safe_load(f)
        model_cfg = ModelConfig.from_dict(cfg_dict.get("model", {}))
    else:
        model_cfg = ModelConfig.tiny()

    batch_size = args.batch_size
    total_steps = args.total_steps
    warmup_steps = args.warmup_steps
    grad_clip = args.grad_clip
    log_interval = args.log_interval
    ckpt_interval = args.checkpoint_interval
    peak_lr = args.lr

    # ---- Model ----
    model = TransformerLM.from_config(model_cfg).to(device)
    n_params = model.num_params()
    if is_main:
        logger.info("Model params (non-embed): %s M", f"{n_params/1e6:.2f}")

    # Wrap with DDP for multi-GPU
    if dist_available and world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_lr_lambda(s, warmup_steps, total_steps),
    )

    # ---- Auto-resume ----
    start_step = 0
    ckpt_dir = Path(args.checkpoint_dir)
    latest = find_latest_checkpoint(ckpt_dir)
    if latest:
        meta = load_checkpoint(latest, model, optimizer, scheduler, device=device)
        start_step = meta["step"] + 1
        if is_main:
            logger.info("Resumed from step %d (loss=%.4f)", meta["step"], meta["loss"])

    # ---- Data ----
    if args.data_dir and Path(args.data_dir).exists():
        loader = _build_dataloader(
            Path(args.data_dir), model_cfg.max_seq_len, batch_size, rank, world_size
        )
        data_iter = iter(loader)
    else:
        if is_main:
            logger.warning(
                "--data-dir not set or missing; using random dummy batches for smoke testing."
            )
        data_iter = None  # type: ignore[assignment]

    def _next_batch() -> dict[str, torch.Tensor]:
        if data_iter is not None:
            try:
                return next(data_iter)
            except StopIteration:
                return next(iter(loader))
        # Dummy random batch for smoke-test / tiny training
        ids = torch.randint(0, model_cfg.vocab_size, (batch_size, model_cfg.max_seq_len))
        return {"input_ids": ids, "labels": ids}

    # ---- MLflow (optional) ----
    if is_main:
        try:
            import mlflow  # noqa: PLC0415

            mlflow.start_run()
            mlflow.log_params({"n_params_M": n_params / 1e6, **vars(args)})
        except Exception:  # noqa: BLE001
            pass

    # ---- Main loop ----
    model.train()
    t0 = time.perf_counter()
    tokens_seen = 0

    for step in range(start_step, total_steps):
        batch = _next_batch()
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16):
            loss = model(input_ids, labels)

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        tokens_seen += batch_size * model_cfg.max_seq_len * world_size

        if is_main and step % log_interval == 0:
            elapsed = time.perf_counter() - t0
            tps = tokens_seen / elapsed
            lr_now = scheduler.get_last_lr()[0]
            mfu = TransformerLM.estimate_mfu(n_params, tps)
            metrics = {
                "loss": loss.item(),
                "lr": lr_now,
                "grad_norm": grad_norm,
                "tokens_per_sec": tps,
                "mfu": mfu,
                "step": step,
            }
            logger.info(
                "step=%d  loss=%.4f  lr=%.2e  grad_norm=%.3f  tok/s=%.0f  MFU=%.2f%%",
                step,
                loss.item(),
                lr_now,
                grad_norm,
                tps,
                mfu * 100,
            )
            _try_log_mlflow(metrics, step)

        if is_main and step % ckpt_interval == 0 and step > 0:
            raw_model = model.module if hasattr(model, "module") else model
            save_checkpoint(raw_model, optimizer, scheduler, step, loss.item(), ckpt_dir)

    # Final checkpoint
    if is_main:
        raw_model = model.module if hasattr(model, "module") else model
        save_checkpoint(raw_model, optimizer, scheduler, total_steps, loss.item(), ckpt_dir)
        logger.info("Training complete.")
        try:
            import mlflow  # noqa: PLC0415

            mlflow.end_run()
        except Exception:  # noqa: BLE001
            pass

    if dist_available and world_size > 1:
        torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-train a TransformerLM from scratch.")
    p.add_argument("--config", default=None, help="Path to YAML config file.")
    p.add_argument("--data-dir", default=None, help="Directory of .bin memmap shards.")
    p.add_argument(
        "--checkpoint-dir", default="checkpoints/run", help="Directory for checkpoints."
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--total-steps", type=int, default=100)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--checkpoint-interval", type=int, default=500)
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
