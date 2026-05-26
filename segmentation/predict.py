"""Run inference on a single image or a directory of images.

Usage:
    # Single image
    python -m segmentation.predict --config configs/default.yaml --checkpoint best.pt --input data/images/tile.png

    # Directory
    python -m segmentation.predict --config configs/default.yaml --checkpoint best.pt --input data/images/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from segmentation.data.transforms import get_val_transforms
from segmentation.models.unet import load_checkpoint

_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    transform,
    device: torch.device,
    task: str,
    threshold: float = 0.5,
) -> np.ndarray:
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    result = transform(image=image, mask=np.zeros((h, w), dtype=np.uint8))
    tensor = result["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        if task == "binary":
            prob = torch.sigmoid(logits.squeeze()).cpu().numpy()
            mask = (prob > threshold).astype(np.uint8)
        else:
            pred = torch.argmax(logits.squeeze(0), dim=0).cpu().numpy()
            mask = pred.astype(np.uint8)

    return mask


def save_mask(mask: np.ndarray, out_path: Path, save_overlay: bool, original_path: Path):
    cv2.imwrite(str(out_path), mask * 255)

    if save_overlay:
        original = cv2.imread(str(original_path))
        overlay = original.copy()
        overlay[mask == 1] = [0, 0, 255]  # red for tarp
        blended = cv2.addWeighted(original, 0.6, overlay, 0.4, 0)
        overlay_path = out_path.with_name(out_path.stem + "_overlay" + out_path.suffix)
        cv2.imwrite(str(overlay_path), blended)


def predict(cfg: dict, checkpoint_path: str, input_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = cfg["data"].get("task", "binary")
    threshold = cfg["inference"].get("threshold", 0.5)
    save_overlay = cfg["inference"].get("save_overlay", True)
    output_dir = Path(cfg["inference"].get("output_dir", "outputs/predictions"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_checkpoint(cfg, checkpoint_path, device)
    model.eval()
    transform = get_val_transforms(cfg)

    input_path = Path(input_path)
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = [p for p in input_path.rglob("*") if p.suffix.lower() in _IMG_EXTENSIONS]

    for img_path in tqdm(image_paths, desc="Predicting"):
        mask = predict_image(model, img_path, transform, device, task, threshold)
        out_path = output_dir / img_path.name
        save_mask(mask, out_path, save_overlay, img_path)

    print(f"Predictions saved to {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Image file or directory")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    predict(cfg, args.checkpoint, args.input)


if __name__ == "__main__":
    main()
