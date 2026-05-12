"""Training script for tarp segmentation.

Usage:
    python -m segmentation.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from segmentation.dataset.dataset import TarpDataset, ResolutionBatchSampler
from segmentation.evaluate import evaluate
from segmentation.dataset.transforms import get_train_transforms, get_val_transforms
from segmentation.losses.losses import get_loss
from segmentation.metrics.metrics import SegmentationMetrics
from segmentation.models.unet import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    use_fixed_size = data_cfg.get("image_size") is not None
    task = data_cfg.get("task", "binary")
    batch_size = cfg["training"]["batch_size"]
    seed = cfg["experiment"].get("seed", 42)
    metadata_dir = data_cfg.get("metadata_dir")

    class_mapper = data_cfg.get("class_mapper")

    train_ds = TarpDataset(
        split_csv=data_cfg["train_csv"],
        task=task,
        transform=get_train_transforms(cfg),
        metadata_dir=metadata_dir,
        class_mapper=class_mapper,
    )
    val_ds = TarpDataset(
        split_csv=data_cfg["val_csv"],
        task=task,
        transform=get_val_transforms(cfg),
        metadata_dir=metadata_dir,
        class_mapper=class_mapper,
    )

    if use_fixed_size:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
    else:
        train_sampler = ResolutionBatchSampler(train_ds, batch_size=batch_size, seed=seed)
        val_sampler = ResolutionBatchSampler(val_ds, batch_size=batch_size, drop_last=False)
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=4, pin_memory=True)

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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    metrics = SegmentationMetrics(task=task)
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = _compute_loss(criterion, logits, masks, task)

        optimizer.zero_grad()
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
) -> dict[str, float]:
    model.eval()
    metrics = SegmentationMetrics(task=task)
    total_loss = 0.0

    for batch in tqdm(loader, desc="Val", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = cfg["data"].get("task", "binary")

    mlflow.set_tracking_uri(cfg["experiment"].get("mlflow_tracking_uri", "mlruns"))
    mlflow.set_experiment(cfg["experiment"]["name"])

    with mlflow.start_run():
        mlflow.log_params(_flatten(cfg))

        train_loader, val_loader = build_dataloaders(cfg)
        model = build_model(cfg).to(device)
        criterion = get_loss(cfg, device)
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)

        ckpt_dir = Path("checkpoints") / cfg["experiment"]["name"]
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        best_iou = 0.0
        patience = cfg["training"].get("early_stopping_patience", 15)
        patience_counter = 0
        grad_clip = cfg["training"].get("grad_clip", 1.0)
        ckpt_path = ckpt_dir / "best.pt"

        for epoch in range(cfg["training"]["epochs"]):
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, task, grad_clip
            )
            val_metrics = validate(model, val_loader, criterion, device, task)

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
