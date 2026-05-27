"""Training script for tarp segmentation.

Usage:
    python -m segmentation.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from segmentation.data.dataset import TarpDataset, ResolutionBatchSampler
from segmentation.evaluate import evaluate
from segmentation.data.transforms import get_train_transforms, get_val_transforms
from segmentation.losses.losses import get_loss
from segmentation.metrics.metrics import SegmentationMetrics
from segmentation.models.unet import build_model


def resolve_device() -> torch.device:
    """Pick the best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg: dict, device: torch.device) -> tuple[DataLoader, DataLoader]:
    data_cfg  = cfg["data"]
    use_fixed_size = data_cfg.get("image_size") is not None
    task       = data_cfg.get("task", "binary")
    batch_size = cfg["training"]["batch_size"]
    seed       = cfg["experiment"].get("seed", 42)
    metadata_dir = data_cfg.get("metadata_dir")
    num_workers  = data_cfg.get("num_workers", 4)
    # pin_memory speeds up host→GPU transfers; not supported on MPS
    pin_memory = device.type == "cuda"

    class_mapper = data_cfg.get("class_mapper")
    max_samples  = data_cfg.get("max_samples")

    train_ds = TarpDataset(
        split_csv=data_cfg["train_csv"],
        task=task,
        transform=get_train_transforms(cfg),
        metadata_dir=metadata_dir,
        class_mapper=class_mapper,
        max_samples=max_samples,
    )
    val_ds = TarpDataset(
        split_csv=data_cfg["val_csv"],
        task=task,
        transform=get_val_transforms(cfg),
        metadata_dir=metadata_dir,
        class_mapper=class_mapper,
        max_samples=max_samples,
    )

    if use_fixed_size:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
    else:
        train_sampler = ResolutionBatchSampler(train_ds, batch_size=batch_size, seed=seed)
        val_sampler   = ResolutionBatchSampler(val_ds,   batch_size=batch_size, drop_last=False)
        train_loader  = DataLoader(train_ds, batch_sampler=train_sampler,
                                   num_workers=num_workers, pin_memory=pin_memory)
        val_loader    = DataLoader(val_ds,   batch_sampler=val_sampler,
                                   num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"].get("weight_decay", 1e-5),
    )


def build_scheduler(optimizer, cfg: dict):
    name = cfg["training"].get("scheduler", "cosine")
    epochs = cfg["training"]["epochs"]
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.1)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)
    return None


# ── Mixed-precision helpers ───────────────────────────────────────────────────

def _make_autocast(device: torch.device, enabled: bool):
    """Return an autocast context manager, or a no-op if AMP is disabled.

    * CUDA  → torch.amp.autocast('cuda', float16)   – full AMP
    * MPS   → torch.amp.autocast('mps',  float16)   – autocast only (no scaler)
    * CPU   → nullcontext                            – AMP not useful on CPU
    """
    if not enabled or device.type == "cpu":
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=torch.float16)


def _make_scaler(device: torch.device, enabled: bool) -> torch.amp.GradScaler | None:
    """Return a GradScaler for CUDA AMP, or None otherwise.

    GradScaler is only meaningful on CUDA. MPS and CPU run without it.
    """
    if enabled and device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    return None


# ─────────────────────────────────────────────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str,
    grad_clip: float,
    autocast_ctx,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, float]:
    model.train()
    metrics = SegmentationMetrics(task=task)
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)

        with autocast_ctx:
            logits = model(images)
            loss   = _compute_loss(criterion, logits, masks, task)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        metrics.update(logits.detach(), masks)

    result = metrics.compute()
    result["loss"] = total_loss / len(loader)
    return result


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task: str,
    autocast_ctx,
) -> dict[str, float]:
    model.eval()
    metrics = SegmentationMetrics(task=task)
    total_loss = 0.0

    for batch in tqdm(loader, desc="Val", leave=False):
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)
        with autocast_ctx:
            logits = model(images)
            total_loss += _compute_loss(criterion, logits, masks, task).item()
        metrics.update(logits, masks)

    result = metrics.compute()
    result["loss"] = total_loss / len(loader)
    return result


def _compute_loss(criterion, logits, masks, task):
    if task == "binary":
        return criterion(logits.squeeze(1), masks.float())
    return criterion(logits, masks)


def _prefix(prefix: str, d: dict) -> dict:
    return {f"{prefix}/{k}": v for k, v in d.items()}


def _flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = {}
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten(v, key, sep))
        elif v is not None:
            items[key] = v
    return items


def train(cfg: dict):
    set_seed(cfg["experiment"].get("seed", 42))
    device = resolve_device()
    task   = cfg["data"].get("task", "binary")

    # ── CUDA-specific backend optimisations ───────────────────────────────────
    if device.type == "cuda":
        # TF32 on Ampere (A100/A10/A30/…): faster matmul with negligible precision loss.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

    # ── Mixed precision ───────────────────────────────────────────────────────
    use_amp      = cfg["training"].get("mixed_precision", False)
    autocast_ctx = _make_autocast(device, use_amp)
    scaler       = _make_scaler(device, use_amp)
    amp_label    = (
        "AMP enabled (autocast + GradScaler)" if scaler is not None
        else f"AMP enabled (autocast only, no scaler on {device.type})" if use_amp and device.type != "cpu"
        else "AMP disabled"
    )
    print(f"  Device: {device}  |  {amp_label}")

    mlflow.set_tracking_uri(cfg["experiment"].get("mlflow_tracking_uri", "mlruns"))
    mlflow.set_experiment(cfg["experiment"]["name"])

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(_flatten(cfg))

        train_loader, val_loader = build_dataloaders(cfg, device)
        model = build_model(cfg).to(device)

        # ── torch.compile (PyTorch 2.x, CUDA only) ────────────────────────────
        if cfg["training"].get("compile", False):
            if hasattr(torch, "compile") and device.type == "cuda":
                model = torch.compile(model)
                print("  torch.compile() applied — first batch will be slower (tracing)")
            else:
                print("  torch.compile skipped (requires PyTorch ≥ 2.0 and CUDA)")
        criterion = get_loss(cfg, device)
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)

        ckpt_dir = Path("checkpoints") / cfg["experiment"]["name"]
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        best_iou        = 0.0
        patience        = cfg["training"].get("early_stopping_patience", 15)
        patience_counter = 0
        grad_clip       = cfg["training"].get("grad_clip", 1.0)
        ckpt_path       = ckpt_dir / "best.pt"

        for epoch in range(cfg["training"]["epochs"]):
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer,
                device, task, grad_clip, autocast_ctx, scaler,
            )
            val_metrics = validate(
                model, val_loader, criterion, device, task, autocast_ctx,
            )

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["iou"])
            elif scheduler is not None:
                scheduler.step()

            mlflow.log_metrics(
                {**_prefix("train", train_metrics), **_prefix("val", val_metrics)},
                step=epoch,
            )
            mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)

            print(
                f"Epoch {epoch+1:3d} | "
                f"train_loss={train_metrics['loss']:.4f}  train_iou={train_metrics['iou']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f}  val_iou={val_metrics['iou']:.4f}"
            )

            if val_metrics["iou"] > best_iou:
                best_iou = val_metrics["iou"]
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_iou": best_iou,
                        "cfg": cfg,
                    },
                    ckpt_path,
                )
                mlflow.log_artifact(str(ckpt_path))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        mlflow.log_metric("best_val_iou", best_iou)

        # ── Automatic test evaluation with the best checkpoint ────────────
        print("\n=== Running test evaluation on best checkpoint ===")
        test_metrics = evaluate(cfg, str(ckpt_path))
        mlflow.log_metrics(_prefix("test", test_metrics))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
