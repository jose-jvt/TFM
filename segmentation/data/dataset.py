"""TarpDataset and resolution-aware batch sampler.

Dataset structure expected:
    dataset/
        images/       {base}_rgb.png  (or any suffix)
        masks/        {base}_mask.png (single-channel, class IDs)
        annotations/  {base}_annotation.json
        metadata/     {base}_metadata.json

Split CSVs (train.csv / val.csv / test.csv) must have columns:
    image_path, mask_path, annotation_path  (paths can be absolute or relative to cwd)
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import BatchSampler, Dataset


class TarpDataset(Dataset):
    """PyTorch Dataset that reads pre-generated masks from a split CSV.

    Each item returns:
        image : float32 tensor (C, H, W)  – transformed
        mask  : int64  tensor (H, W)      – class IDs (binary: 0/1, multiclass: 0-N)
        meta  : dict   with image_path, mask_path, resolution and any metadata JSON fields

    Class remapping (optional):
        Pass class_mapper as a dict {original_id: new_id, ...} defined directly
        in the YAML config (data.class_mapper). When provided it takes full control
        of class encoding and the automatic binary binarisation (mask > 0) is skipped.
    """

    def __init__(
        self,
        split_csv: str | Path,
        task: str = "binary",
        transform=None,
        metadata_dir: str | Path | None = None,
        class_mapper: dict | None = None,
        max_samples: int | None = None,
    ):
        self.df = pd.read_csv(split_csv)
        _validate_columns(self.df, split_csv)
        if max_samples is not None:
            self.df = self.df.head(max_samples).reset_index(drop=True)

        self.task = task
        self.transform = transform
        self.metadata_dir = Path(metadata_dir) if metadata_dir else None
        self._class_lut: np.ndarray | None = _build_class_lut(class_mapper)

        # Read only image headers (PIL lazy open) to get resolution without loading pixels.
        self._resolutions = self._detect_resolutions()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        image = _load_rgb(row["image_path"])
        mask = _load_mask(row["mask_path"], image.shape[:2])

        if self._class_lut is not None:
            mask = self._class_lut[mask]
        elif self.task == "binary":
            mask = (mask > 0).astype(np.uint8)

        if self.transform:
            result = self.transform(image=image, mask=mask)
            image = result["image"]      # Tensor (C, H, W) – albumentations ToTensorV2
            mask = result["mask"]        # Tensor (H, W)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask)

        meta = {
            "image_path": str(row["image_path"]),
            "mask_path": str(row["mask_path"]),
            "resolution": self._resolutions[idx],
        }
        meta.update(self._load_metadata(row["image_path"]))

        return {"image": image, "mask": mask.long(), "meta": meta}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_resolutions(self) -> list[int]:
        """Read image dimensions from file headers (no pixel loading)."""
        resolutions = []
        for path in self.df["image_path"]:
            with Image.open(path) as img:
                w, h = img.size
            resolutions.append(max(h, w))
        return resolutions

    def _load_metadata(self, image_path: str) -> dict:
        if self.metadata_dir is None:
            return {}
        # Derive base name by stripping the last underscore-separated token
        # e.g. "CAP-HAITIEN_2021_RGB_0234_rgb" → "CAP-HAITIEN_2021_RGB_0234"
        stem = Path(image_path).stem
        base = re.sub(r"_[^_]+$", "", stem)
        meta_path = self.metadata_dir / f"{base}_metadata.json"
        if not meta_path.exists():
            return {}
        with open(meta_path) as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, csv_path):
    required = {"image_path", "mask_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV '{csv_path}' is missing columns: {missing}")


def _load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_mask(path: str, fallback_shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros(fallback_shape, dtype=np.uint8)
    return mask


def _build_class_lut(class_mapper: dict | None) -> np.ndarray | None:
    """Build a uint8 numpy LUT from a class_mapper dict.

    The dict maps original pixel values to new class IDs:
        {0: 0, 1: 1, 2: 1, 3: 1, ...}
    Keys may be ints or strings (YAML loads numeric keys as ints).

    The LUT has 256 entries (full uint8 range). Any pixel value not listed
    defaults to 0 (background).
    Returns None if class_mapper is None.
    """
    if class_mapper is None:
        return None

    lut = np.zeros(256, dtype=np.uint8)
    for k, v in class_mapper.items():
        idx = int(k)
        if not (0 <= idx <= 255):
            raise ValueError(f"class_mapper key {k!r} is out of uint8 range [0, 255].")
        lut[idx] = int(v)
    return lut


class ResolutionBatchSampler(BatchSampler):
    """Creates same-resolution batches when image_size is None.

    Builds batches by grouping indices that share the same resolution,
    then shuffles the batch list so the training loop sees a mix.
    """

    def __init__(
        self,
        dataset: TarpDataset,
        batch_size: int,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self._seed = seed

        groups: dict[int, list[int]] = {}
        for i, res in enumerate(dataset._resolutions):
            groups.setdefault(res, []).append(i)
        self._groups = groups

    def _make_batches(self) -> list[list[int]]:
        rng = random.Random(self._seed)
        batches: list[list[int]] = []
        for indices in self._groups.values():
            shuffled = indices[:]
            rng.shuffle(shuffled)
            for i in range(0, len(shuffled), self.batch_size):
                batch = shuffled[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        for batch in self._make_batches():
            yield batch

    def __len__(self) -> int:
        return len(self._make_batches())
