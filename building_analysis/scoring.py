"""Per-building damage scoring.

Given a tarp prediction mask and a binary building mask (already computed by
an external building segmentation model), computes score = tarp_area / building_area
for each individual building detected via connected components.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage


def score_buildings(
    tarp_mask: np.ndarray,
    building_mask: np.ndarray,
    image_filename: str = "",
    min_building_area_px: int = 50,
) -> pd.DataFrame:
    """Assign a damage score to each building in the image.

    Args:
        tarp_mask:          (H, W) uint8. 1 = tarp pixel (any class), 0 = background.
                            For multiclass masks, pass (tarp_mask > 0).astype(np.uint8).
        building_mask:      (H, W) uint8. 1 = building pixel, 0 = background.
        image_filename:     Source image name (stored in output for traceability).
        min_building_area_px: Buildings smaller than this are skipped (noise filter).

    Returns:
        DataFrame with one row per building:
            building_id, building_area_px, tarp_area_px, score, image_filename
    """
    if tarp_mask.shape != building_mask.shape:
        raise ValueError(
            f"Mask shapes do not match: tarp={tarp_mask.shape}, building={building_mask.shape}"
        )

    labeled, num_buildings = ndimage.label(building_mask > 0)

    records = []
    for building_id in range(1, num_buildings + 1):
        building_region = labeled == building_id
        building_area = int(building_region.sum())

        if building_area < min_building_area_px:
            continue

        tarp_pixels = int((building_region & (tarp_mask > 0)).sum())
        score = tarp_pixels / building_area

        records.append(
            {
                "image_filename": image_filename,
                "building_id": building_id,
                "building_area_px": building_area,
                "tarp_area_px": tarp_pixels,
                "score": round(score, 4),
            }
        )

    return pd.DataFrame(records)


def score_all_images(
    predictions_dir: str | Path,
    building_masks_dir: str | Path,
    min_building_area_px: int = 50,
) -> pd.DataFrame:
    """Batch scoring over a directory of prediction masks and building masks.

    Expects prediction mask filename to match building mask filename exactly.
    """
    import cv2

    predictions_dir = Path(predictions_dir)
    building_masks_dir = Path(building_masks_dir)

    all_records = []
    for tarp_path in sorted(predictions_dir.glob("*.png")):
        building_path = building_masks_dir / tarp_path.name
        if not building_path.exists():
            continue

        tarp_mask = cv2.imread(str(tarp_path), cv2.IMREAD_GRAYSCALE)
        building_mask = cv2.imread(str(building_path), cv2.IMREAD_GRAYSCALE)

        # Binarize in case masks have 255 instead of 1
        tarp_binary = (tarp_mask > 127).astype(np.uint8)
        building_binary = (building_mask > 127).astype(np.uint8)

        df = score_buildings(
            tarp_binary,
            building_binary,
            image_filename=tarp_path.name,
            min_building_area_px=min_building_area_px,
        )
        all_records.append(df)

    if not all_records:
        return pd.DataFrame()

    return pd.concat(all_records, ignore_index=True)
