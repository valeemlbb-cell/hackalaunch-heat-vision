"""Training loop for HeatNet-S (CPU friendly)."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import DEFAULT, CellConfig, TrainConfig
from ..sim.dataset import Sample, get_split
from .infer import CHECKPOINT_NAME, save_checkpoint
from .model import HeatNetS, focal_loss, masked_l1
from .targets import HeatVisionDataset


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))


def _lr_at(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def compute_losses(
    outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], tcfg: TrainConfig
) -> dict[str, torch.Tensor]:
    hm = focal_loss(outputs["hm"], batch["hm"])
    wh = masked_l1(outputs["wh"], batch["wh"], batch["mask"])
    off = masked_l1(outputs["off"], batch["off"], batch["mask"])
    total = tcfg.hm_weight * hm + tcfg.wh_weight * wh + tcfg.off_weight * off
    return {"total": total, "hm": hm, "wh": wh, "off": off}


@torch.no_grad()
def evaluate_loss(model: HeatNetS, loader: DataLoader, tcfg: TrainConfig) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    n = 0
    for batch in loader:
        losses = compute_losses(model(batch["image"]), batch, tcfg)
        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach())
        n += 1
    model.train()
    return {k: v / max(1, n) for k, v in sums.items()}


def train(
    out_dir: Path,
    *,
    train_samples: list[Sample] | None = None,
    val_samples: list[Sample] | None = None,
    n_train: int = 2400,
    n_val: int = 300,
    cfg: CellConfig = DEFAULT,
    tcfg: TrainConfig | None = None,
    cache_dir: Path | None = None,
    zero_channels: tuple[int, ...] = (),
    verbose: bool = True,
) -> dict:
    tcfg = tcfg or TrainConfig()
    set_seed(tcfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if train_samples is None:
        if verbose:
            print(f"rendering {n_train} training frames ...", flush=True)
        train_samples = get_split("train", n_train, cfg, cache_dir=cache_dir, progress=verbose)
    if val_samples is None:
        if verbose:
            print(f"rendering {n_val} validation frames ...", flush=True)
        val_samples = get_split("val", n_val, cfg, cache_dir=cache_dir, progress=verbose)

    train_ds = HeatVisionDataset(
        train_samples, cfg.detector, augment=True, seed=tcfg.seed, zero_channels=zero_channels
    )
    val_ds = HeatVisionDataset(val_samples, cfg.detector, augment=False, zero_channels=zero_channels)
    train_loader = DataLoader(
        train_ds, batch_size=tcfg.batch_size, shuffle=True, num_workers=tcfg.num_workers, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, num_workers=tcfg.num_workers)

    model = HeatNetS(cfg.detector)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    total_steps = tcfg.epochs * len(train_loader)
    if verbose:
        print(
            f"HeatNet-S: {model.num_parameters():,} params | "
            f"{len(train_ds)} train / {len(val_ds)} val | {total_steps} steps",
            flush=True,
        )

    history: list[dict] = []
    best = float("inf")
    step = 0
    started = time.time()
    model.train()
    for epoch in range(tcfg.epochs):
        running: dict[str, float] = {}
        for batch in train_loader:
            lr = _lr_at(step, total_steps, tcfg.lr, tcfg.warmup_steps)
            for group in opt.param_groups:
                group["lr"] = lr
            losses = compute_losses(model(batch["image"]), batch, tcfg)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach())
            step += 1
        train_stats = {k: v / len(train_loader) for k, v in running.items()}
        val_stats = evaluate_loss(model, val_loader, tcfg)
        row = {
            "epoch": epoch + 1,
            "lr": lr,
            "train": train_stats,
            "val": val_stats,
            "elapsed_s": round(time.time() - started, 1),
        }
        history.append(row)
        if verbose:
            print(
                f"epoch {epoch + 1:2d}/{tcfg.epochs}  "
                f"train {train_stats['total']:.4f}  val {val_stats['total']:.4f}  "
                f"({row['elapsed_s']:.0f}s)",
                flush=True,
            )
        if val_stats["total"] < best:
            best = val_stats["total"]
            save_checkpoint(
                out_dir / CHECKPOINT_NAME,
                model,
                meta={
                    "epoch": epoch + 1,
                    "val_loss": best,
                    "params": model.num_parameters(),
                    "train_config": asdict(tcfg),
                    "n_train": len(train_ds),
                    "n_val": len(val_ds),
                    "zero_channels": list(zero_channels),
                },
            )

    report = {
        "best_val_loss": best,
        "epochs": tcfg.epochs,
        "params": model.num_parameters(),
        "train_seconds": round(time.time() - started, 1),
        "zero_channels": list(zero_channels),
        "history": history,
        "checkpoint": str(out_dir / CHECKPOINT_NAME),
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
