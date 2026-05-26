"""Stratified dataset splitter for tarp segmentation.

Filename format:
    {tipo_proyecto}_{año}_{tipo_imagen}_{tipo_ortho}_{zoom}_{tile_x}_{tile_y}.png

Splitting constraints (by priority):
  1. Anti-leakage  : all images sharing (tile_x, tile_y) go to the same split,
                     regardless of year, to prevent geographic data leakage.
  2. Class balance : pixel-level class distribution in each split approximates
                     the overall distribution (computed from masks).
  3. Proyecto balance: BLUESKY/GRAYSKY ratio is kept consistent across splits;
                     this is the most important structural constraint after class
                     stratification.

Algorithm:
  - Atomic unit  : TileGroup  — all images with the same (tile_x, tile_y).
  - Stratum      : (dominant_tipo_proyecto, dominant_tarp_class).
  - Assignment   : within each stratum, tile groups are shuffled and distributed
                   proportionally. This is repeated n_trials times and the trial
                   with the lowest total variation distance (class + proyecto) is kept.

Usage:
    python -m segmentation.dataset.generate_splits \\
        --dataset-dir dataset \\
        --output-dir  dataset \\
        --train 0.70 --val 0.15 \\
        --trials 200 --seed 42
"""
from __future__ import annotations

import argparse
import re
import random
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Background class is excluded from dominant-class computation
_BACKGROUND_CLASS = 0

# Filename regex — named groups map to the 7 filename fields
_FILENAME_RE = re.compile(
    r"^(?P<tipo_proyecto>[^_]+)"
    r"_(?P<año>\d{4})"
    r"_(?P<tipo_imagen>[^_]+)"
    r"_(?P<tipo_ortho>[^_]+)"
    r"_(?P<zoom>\d+)"
    r"_(?P<tile_x>\d+)"
    r"_(?P<tile_y>\d+)$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ImageRecord:
    image_path: Path
    mask_path: Path
    annotation_path: str
    metadata_path: str
    tipo_proyecto: str
    año: int
    tipo_imagen: str
    tipo_ortho: str
    zoom: int
    tile_x: int
    tile_y: int


@dataclass
class TileGroup:
    tile_x: int
    tile_y: int
    records: list[ImageRecord] = field(default_factory=list)
    # Filled by compute_class_stats
    class_pixel_counts: Counter = field(default_factory=Counter)

    @property
    def tile_key(self) -> tuple[int, int]:
        return (self.tile_x, self.tile_y)

    @property
    def dominant_tipo_proyecto(self) -> str:
        counts = Counter(r.tipo_proyecto for r in self.records)
        return counts.most_common(1)[0][0]

    @property
    def dominant_tarp_class(self) -> int:
        """Class with most pixels, excluding background."""
        non_bg = {k: v for k, v in self.class_pixel_counts.items() if k != _BACKGROUND_CLASS}
        return max(non_bg, key=non_bg.get) if non_bg else _BACKGROUND_CLASS

    @property
    def stratum(self) -> str:
        return f"{self.dominant_tipo_proyecto}_{self.dominant_tarp_class}"

    @property
    def n_images(self) -> int:
        return len(self.records)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_filename(filename: str) -> dict | None:
    stem = Path(filename).stem
    m = _FILENAME_RE.match(stem)
    if m is None:
        return None
    g = m.groupdict()
    return {
        "tipo_proyecto": g["tipo_proyecto"].upper(),
        "año": int(g["año"]),
        "tipo_imagen": g["tipo_imagen"],
        "tipo_ortho": g["tipo_ortho"],
        "zoom": int(g["zoom"]),
        "tile_x": int(g["tile_x"]),
        "tile_y": int(g["tile_y"]),
    }


def scan_dataset(
    images_dir: Path,
    masks_dir: Path,
    annotations_dir: Path,
    metadata_dir: Path,
    mask_suffix: str,
    annotation_suffix: str,
    metadata_suffix: str,
) -> list[ImageRecord]:
    """Scan images/ and pair each image with its mask, annotation and metadata."""
    records: list[ImageRecord] = []
    skipped = 0

    for img_path in sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in _IMG_EXTENSIONS):
        parsed = parse_filename(img_path.name)
        if parsed is None:
            log.warning("Skipping '%s': filename does not match expected pattern.", img_path.name)
            skipped += 1
            continue

        # Mask: same stem + _mask suffix
        mask_path: Path | None = None
        for ext in _IMG_EXTENSIONS:
            candidate = masks_dir / f"{img_path.stem}_{mask_suffix}{ext}"
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is None:
            log.debug("No mask for '%s'; skipping.", img_path.name)
            skipped += 1
            continue

        ann = annotations_dir / f"{img_path.stem}_{annotation_suffix}.json"
        meta = metadata_dir / f"{img_path.stem}_{metadata_suffix}.json"

        records.append(
            ImageRecord(
                image_path=img_path.resolve(),
                mask_path=mask_path.resolve(),
                annotation_path=str(ann.resolve()) if ann.exists() else "",
                metadata_path=str(meta.resolve()) if meta.exists() else "",
                **parsed,
            )
        )

    log.info("Found %d valid image/mask pairs (%d skipped).", len(records), skipped)
    return records


# ---------------------------------------------------------------------------
# Tile groups
# ---------------------------------------------------------------------------

def build_tile_groups(records: list[ImageRecord]) -> list[TileGroup]:
    """Group records by (tile_x, tile_y) — the anti-leakage unit."""
    groups: dict[tuple[int, int], TileGroup] = {}
    for rec in records:
        key = (rec.tile_x, rec.tile_y)
        if key not in groups:
            groups[key] = TileGroup(tile_x=rec.tile_x, tile_y=rec.tile_y)
        groups[key].records.append(rec)
    return list(groups.values())


def compute_class_stats(tile_groups: list[TileGroup]) -> None:
    """Read all masks in each tile group and accumulate pixel counts per class."""
    import cv2

    for tg in tqdm(tile_groups, desc="Reading masks", unit="tile"):
        for rec in tg.records:
            mask = cv2.imread(str(rec.mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                log.warning("Cannot read mask: %s", rec.mask_path)
                continue
            counts = np.bincount(mask.ravel(), minlength=256)
            for cls_id, cnt in enumerate(counts):
                if cnt > 0:
                    tg.class_pixel_counts[cls_id] += int(cnt)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _class_distribution(tile_groups: list[TileGroup], n_classes: int = 7) -> np.ndarray:
    """Normalised pixel count per class (background excluded)."""
    total = np.zeros(n_classes, dtype=np.float64)
    for tg in tile_groups:
        for cls_id in range(n_classes):
            total[cls_id] += tg.class_pixel_counts.get(cls_id, 0)
    s = total.sum()
    return total / s if s > 0 else total


def _proyecto_ratio(tile_groups: list[TileGroup]) -> float:
    """Fraction of BLUESKY images (0.0–1.0)."""
    counts = Counter(r.tipo_proyecto for tg in tile_groups for r in tg.records)
    total = sum(counts.values())
    return counts.get("BLUESKY", 0) / total if total > 0 else 0.0


def score_split(
    splits: list[list[TileGroup]],
    overall_class_dist: np.ndarray,
    overall_proyecto_ratio: float,
    class_weight: float = 0.6,
) -> float:
    """
    Combined imbalance score (lower = better).

    Metrics:
      - Total variation distance of class distribution vs. overall (per split)
      - Absolute deviation of BLUESKY ratio vs. overall (per split)
    """
    tv_class = 0.0
    tv_proyecto = 0.0
    for split in splits:
        if not split:
            continue
        cd = _class_distribution(split)
        tv_class += 0.5 * np.abs(cd - overall_class_dist).sum()
        tv_proyecto += abs(_proyecto_ratio(split) - overall_proyecto_ratio)

    return class_weight * tv_class + (1.0 - class_weight) * tv_proyecto


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def _distribute(groups: list[TileGroup], ratios: list[float], rng: random.Random) -> list[list[TileGroup]]:
    """Shuffle `groups` and distribute proportionally across splits."""
    shuffled = groups[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    splits: list[list[TileGroup]] = [[] for _ in ratios]

    # Floor allocation + remainder to highest-ratio splits
    counts = [int(n * r) for r in ratios]
    remainder = n - sum(counts)
    for i in sorted(range(len(ratios)), key=lambda x: ratios[x], reverse=True):
        if remainder <= 0:
            break
        counts[i] += 1
        remainder -= 1

    idx = 0
    for i, cnt in enumerate(counts):
        splits[i] = shuffled[idx : idx + cnt]
        idx += cnt
    return splits


def stratified_split(
    tile_groups: list[TileGroup],
    ratios: list[float],
    n_trials: int,
    seed: int,
) -> list[list[TileGroup]]:
    """
    Stratified split with random search for the best trial.

    Strategy:
      1. Group tile groups by stratum (dominant_tipo_proyecto, dominant_tarp_class).
      2. For each trial, shuffle within each stratum and distribute proportionally.
      3. Keep the trial with the lowest combined imbalance score.
    """
    overall_class_dist = _class_distribution(tile_groups)
    overall_proyecto_ratio = _proyecto_ratio(tile_groups)

    # Build strata
    strata: dict[str, list[TileGroup]] = defaultdict(list)
    for tg in tile_groups:
        strata[tg.stratum].append(tg)

    log.info("Strata found: %s", {k: len(v) for k, v in sorted(strata.items())})

    best_splits: list[list[TileGroup]] | None = None
    best_score = float("inf")

    for trial in range(n_trials):
        rng = random.Random(seed + trial)
        trial_splits: list[list[TileGroup]] = [[] for _ in ratios]

        for groups in strata.values():
            partial = _distribute(groups, ratios, rng)
            for i, part in enumerate(partial):
                trial_splits[i].extend(part)

        sc = score_split(trial_splits, overall_class_dist, overall_proyecto_ratio)
        if sc < best_score:
            best_score = sc
            best_splits = trial_splits

    log.info("Best score after %d trials: %.6f", n_trials, best_score)
    assert best_splits is not None
    return best_splits


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_split_stats(
    splits: list[list[TileGroup]],
    split_names: list[str],
    n_classes: int = 7,
):
    class_names = {0: "background", 1: "tarp_blue", 2: "tarp_black", 3: "tarp_green",
                   4: "tarp_white", 5: "tarp_repair", 6: "tarp_other"}

    total_images = sum(tg.n_images for s in splits for tg in s)
    total_tiles = sum(len(s) for s in splits)

    header = f"{'Split':<8} {'Tiles':>6} {'Images':>7} {'BLUESKY%':>9} {'GRAYSKY%':>9}"
    for c in range(1, n_classes):
        header += f" {class_names.get(c, str(c)):>10}"
    log.info("\n" + header)
    log.info("-" * len(header))

    for name, split in zip(split_names, splits):
        n_images = sum(tg.n_images for tg in split)
        proyecto = Counter(r.tipo_proyecto for tg in split for r in tg.records)
        total_p = sum(proyecto.values())
        blue_pct = 100 * proyecto.get("BLUESKY", 0) / total_p if total_p else 0
        gray_pct = 100 * proyecto.get("GRAYSKY", 0) / total_p if total_p else 0

        cd = _class_distribution(split, n_classes)
        row = f"{name:<8} {len(split):>6} {n_images:>7} {blue_pct:>8.1f}% {gray_pct:>8.1f}%"
        for c in range(1, n_classes):
            row += f" {100*cd[c]:>9.2f}%"
        log.info(row)

    log.info("-" * len(header))
    log.info(
        "Total: %d tile groups, %d images.", total_tiles, total_images
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def tile_groups_to_records(tile_groups: list[TileGroup]) -> list[dict]:
    rows = []
    for tg in tile_groups:
        for rec in tg.records:
            rows.append({
                "image_path": str(rec.image_path),
                "mask_path": str(rec.mask_path),
                "annotation_path": rec.annotation_path,
                "metadata_path": rec.metadata_path,
                "tipo_proyecto": rec.tipo_proyecto,
                "año": rec.año,
                "zoom": rec.zoom,
                "tile_x": rec.tile_x,
                "tile_y": rec.tile_y,
            })
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stratified dataset splitter respecting tile-group anti-leakage."
    )
    parser.add_argument("--dataset-dir", required=True, help="Root of the dataset folder.")
    parser.add_argument("--output-dir", default=None, help="Where to write CSVs (default: dataset-dir).")
    parser.add_argument("--train", type=float, default=0.70, help="Train ratio (default: 0.70).")
    parser.add_argument("--val", type=float, default=0.15, help="Validation ratio (default: 0.15).")
    parser.add_argument("--trials", type=int, default=200, help="Random search trials (default: 200).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-suffix", default="mask")
    parser.add_argument("--annotation-suffix", default="annotation")
    parser.add_argument("--metadata-suffix", default="metadata")
    args = parser.parse_args()

    train_r, val_r = args.train, args.val
    test_r = round(1.0 - train_r - val_r, 10)
    if test_r < 0:
        raise ValueError("train + val ratios exceed 1.0")
    ratios = [train_r, val_r, test_r]
    split_names = ["train", "val", "test"]

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"
    annotations_dir = dataset_dir / "annotations"
    metadata_dir = dataset_dir / "metadata"

    for d in (images_dir, masks_dir):
        if not d.exists():
            raise FileNotFoundError(f"Required directory not found: {d}")

    # 1. Scan dataset
    records = scan_dataset(
        images_dir, masks_dir, annotations_dir, metadata_dir,
        args.mask_suffix, args.annotation_suffix, args.metadata_suffix,
    )
    if not records:
        log.error("No valid image/mask pairs found.")
        return

    # 2. Build tile groups
    tile_groups = build_tile_groups(records)
    log.info("Tile groups (anti-leak units): %d", len(tile_groups))

    # 3. Compute class stats from masks
    compute_class_stats(tile_groups)

    # 4. Stratified split
    splits = stratified_split(tile_groups, ratios, n_trials=args.trials, seed=args.seed)

    # 5. Report
    print_split_stats(splits, split_names)

    # 6. Write CSVs
    for name, split in zip(split_names, splits):
        rows = tile_groups_to_records(split)
        path = output_dir / f"{name}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        n_images = sum(tg.n_images for tg in split)
        log.info("Wrote %s: %d tile groups, %d images.", path, len(split), n_images)


if __name__ == "__main__":
    main()
