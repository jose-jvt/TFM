"""
Filename parsing and annotation loading utilities.

Annotation CSV expected columns:
    image_filename  : filename of the image (e.g. CAP-HAITIEN_2021_RGB_0234.png)
    class           : tarp class label (tarp_blue, tarp_black, ...)
    wkt_polygon     : polygon in WKT format, CRS EPSG:4326 (optional, for geo-analysis)
    pixel_coords    : JSON array of [x, y] polygon vertices in pixel space
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import cv2


TARP_CLASSES = {
    "background": 0,
    "tarp_blue": 1,
    "tarp_black": 2,
    "tarp_green": 3,
    "tarp_white": 4,
    "tarp_repair": 5,
    "tarp_other": 6,
}

BINARY_CLASS_ID = 1  # all tarps → this label in binary mode


@dataclass
class ImageMetadata:
    filename: str
    tile: str
    year: Optional[int]
    region: str
    image_type: str
    resolution: int  # 1024 or 2048


class FilenameParser:
    """Parses image filenames using a configurable regex pattern.

    The pattern must define named groups: year, region, image_type, tile.
    """

    def __init__(self, pattern: str):
        self._pattern = re.compile(pattern, re.IGNORECASE)

    def parse(self, filename: str) -> ImageMetadata:
        stem = Path(filename).stem
        match = self._pattern.search(stem)
        if match is None:
            raise ValueError(f"Filename '{filename}' does not match pattern '{self._pattern.pattern}'")

        groups = match.groupdict()
        return ImageMetadata(
            filename=filename,
            tile=groups.get("tile", stem),
            year=int(groups["year"]) if "year" in groups and groups["year"] else None,
            region=groups.get("region", "unknown"),
            image_type=groups.get("image_type", "unknown"),
            resolution=-1,  # filled in when image is loaded
        )


def load_annotations(annotations_file: str | Path) -> pd.DataFrame:
    """Load annotation CSV and validate required columns."""
    df = pd.read_csv(annotations_file)
    required = {"image_filename", "class", "pixel_coords"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Annotation file missing columns: {missing}")
    df["class"] = df["class"].str.strip().str.lower()
    return df


def pixel_coords_to_mask(
    pixel_coords_json: str,
    height: int,
    width: int,
    class_id: int,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Rasterize a polygon (given as JSON [[x,y],...]) onto a mask array.

    If `mask` is provided, the polygon is drawn on top (in-place modification).
    """
    if mask is None:
        mask = np.zeros((height, width), dtype=np.uint8)

    coords = json.loads(pixel_coords_json)
    polygon = np.array(coords, dtype=np.int32)
    if polygon.ndim == 1:
        polygon = polygon.reshape(-1, 2)

    cv2.fillPoly(mask, [polygon], color=int(class_id))
    return mask


def build_mask(
    annotation_rows: pd.DataFrame,
    height: int,
    width: int,
    task: str = "binary",
) -> np.ndarray:
    """Build a segmentation mask from all annotation rows for a single image.

    Args:
        annotation_rows: Subset of annotation DataFrame for one image.
        height, width: Mask dimensions.
        task: "binary" or "multiclass".

    Returns:
        uint8 mask of shape (H, W).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for _, row in annotation_rows.iterrows():
        if task == "binary":
            class_id = BINARY_CLASS_ID
        else:
            class_id = TARP_CLASSES.get(row["class"], TARP_CLASSES["tarp_other"])
        pixel_coords_to_mask(row["pixel_coords"], height, width, class_id, mask)
    return mask
