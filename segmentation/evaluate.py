"""Evaluate a trained model on the test split.

Usage:
    python -m segmentation.evaluate \
        --config configs/default.yaml \
        --checkpoint checkpoints/unet_resnet34_binary_baseline/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from segmentation.dataset.dataset import TarpDataset, ResolutionBatchSampler
from segmentation.dataset.transforms import get_val_transforms
from segmentation.losses.losses import get_loss
from segmentation.metrics.metrics import SegmentationMetrics
from segmentation.models.unet import load_checkpoint


def evaluate(cfg: dict, checkpoint_path: str) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = cfg["data"].get("task", "binary")
    data_cfg = cfg["data"]
    use_fixed_size = data_cfg.get("image_size") is not None
    batch_size = cfg["training"]["batch_size"]

    test_ds = TarpDataset(
        split_csv=data_cfg["test_csv"],
        task=task,
        transform=get_val_transforms(cfg),
        metadata_dir=data_cfg.get("metadata_dir"),
    )

    if use_fixed_size:
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        sampler = ResolutionBatchSampler(test_ds, batch_size=batch_size)
        test_loader = DataLoader(test_ds, batch_sampler=sampler, num_workers=4)

    model = load_checkpoint(cfg, checkpoint_path, device)
    model.eval()
    criterion = get_loss(cfg, device)
    metrics = SegmentationMetrics(task=task)
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)

            if task == "binary":
                loss = criterion(logits.squeeze(1), masks.float())
            else:
                loss = criterion(logits, masks)

            total_loss += loss.item()
            metrics.update(logits, masks)

    result = metrics.compute()
    result["loss"] = total_loss / len(test_loader)

    print("\n=== Test Results ===")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")

    output_dir = Path(cfg["inference"].get("output_dir", "outputs/predictions"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
