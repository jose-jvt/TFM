"""Visualization utilities: overlay tarp and building masks on the original image."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# BGR colors for overlay
_TARP_COLOR = (0, 0, 255)       # red  → tarp pixels
_BUILDING_COLOR = (0, 255, 0)   # green → building contour
_OVERLAP_COLOR = (0, 165, 255)  # orange → tarp inside building


def create_overlay(
    image: np.ndarray,
    tarp_mask: np.ndarray,
    building_mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Draw tarp and building masks over the original RGB image.

    Args:
        image:         (H, W, 3) BGR image.
        tarp_mask:     (H, W) binary mask; 1 = tarp.
        building_mask: (H, W) binary mask; 1 = building.
        alpha:         Blend weight for the overlay layer.

    Returns:
        (H, W, 3) BGR image with colored overlays.
    """
    overlay = image.copy()

    tarp = tarp_mask > 0
    building = building_mask > 0
    overlap = tarp & building

    overlay[tarp & ~overlap] = _TARP_COLOR
    overlay[overlap] = _OVERLAP_COLOR

    # Draw building contours
    contours, _ = cv2.findContours(
        building_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, _BUILDING_COLOR, thickness=2)

    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)


def save_overlay(
    image_path: str | Path,
    tarp_mask: np.ndarray,
    building_mask: np.ndarray,
    output_path: str | Path,
    alpha: float = 0.45,
):
    """Load image from disk, create overlay, and save."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    result = create_overlay(image, tarp_mask, building_mask, alpha)
    cv2.imwrite(str(output_path), result)
