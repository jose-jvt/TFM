"""Generate building-level damage reports from scored DataFrames.

Usage:
    python -m building_analysis.report \
        --predictions outputs/predictions \
        --building-masks data/building_masks \
        --images data/images \
        --output outputs/building_scores \
        --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from building_analysis.scoring import score_all_images
from building_analysis.overlay import save_overlay


def add_filename_metadata(df: pd.DataFrame, filename_pattern: str) -> pd.DataFrame:
    """Parse year, region, tile from image_filename and add as columns."""
    import re

    pattern = re.compile(filename_pattern, re.IGNORECASE)

    def _parse(fname):
        stem = Path(fname).stem
        m = pattern.search(stem)
        if m:
            g = m.groupdict()
            return pd.Series(
                {
                    "year": int(g["year"]) if "year" in g and g["year"] else None,
                    "region": g.get("region", "unknown"),
                    "tile": g.get("tile", stem),
                    "image_type": g.get("image_type", "unknown"),
                }
            )
        return pd.Series({"year": None, "region": "unknown", "tile": stem, "image_type": "unknown"})

    meta = df["image_filename"].apply(_parse)
    return pd.concat([df, meta], axis=1)


def plot_score_distribution(df: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df["score"], bins=50, edgecolor="black")
    axes[0].set_title("Building Tarp Coverage Score Distribution")
    axes[0].set_xlabel("Score (tarp area / building area)")
    axes[0].set_ylabel("Count")

    score_bins = pd.cut(df["score"], bins=[0, 0.1, 0.3, 0.6, 1.0], labels=["<10%", "10-30%", "30-60%", ">60%"])
    score_bins.value_counts().sort_index().plot(kind="bar", ax=axes[1], edgecolor="black")
    axes[1].set_title("Buildings by Damage Category")
    axes[1].set_xlabel("Tarp Coverage")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_report(
    predictions_dir: str | Path,
    building_masks_dir: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    filename_pattern: str,
    min_building_area_px: int = 50,
    save_overlays: bool = True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Scoring buildings...")
    df = score_all_images(predictions_dir, building_masks_dir, min_building_area_px)

    if df.empty:
        print("No buildings found. Check that prediction and building mask filenames match.")
        return

    df = add_filename_metadata(df, filename_pattern)

    csv_path = output_dir / "building_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved scores to {csv_path}  ({len(df)} buildings)")

    plot_score_distribution(df, output_dir / "score_distribution.png")

    if save_overlays:
        import cv2, numpy as np

        overlay_dir = output_dir / "overlays"
        overlay_dir.mkdir(exist_ok=True)
        for fname in df["image_filename"].unique():
            img_path = Path(images_dir) / fname
            tarp_path = Path(predictions_dir) / fname
            building_path = Path(building_masks_dir) / fname
            if not all(p.exists() for p in [img_path, tarp_path, building_path]):
                continue
            tarp_mask = (cv2.imread(str(tarp_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
            building_mask = (cv2.imread(str(building_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
            save_overlay(img_path, tarp_mask, building_mask, overlay_dir / fname)

    summary = df.groupby("image_filename")["score"].agg(["mean", "max", "count"]).reset_index()
    summary.columns = ["image_filename", "mean_score", "max_score", "num_buildings"]
    summary = add_filename_metadata(summary, filename_pattern)
    summary.to_csv(output_dir / "image_summary.csv", index=False)

    print(f"\n=== Summary ===")
    print(f"  Total buildings:   {len(df)}")
    print(f"  Mean score:        {df['score'].mean():.3f}")
    print(f"  Buildings >30% coverage: {(df['score'] > 0.3).sum()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--building-masks", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    generate_report(
        predictions_dir=args.predictions,
        building_masks_dir=args.building_masks,
        images_dir=args.images,
        output_dir=args.output,
        filename_pattern=cfg["data"]["filename_pattern"],
        min_building_area_px=cfg["building_analysis"].get("min_building_area_px", 50),
    )


if __name__ == "__main__":
    main()
