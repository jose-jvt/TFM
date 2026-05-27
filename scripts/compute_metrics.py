"""Compute pixel-level and instance-level segmentation metrics.

Pairs predicted masks with ground-truth masks, feeds them through
``DetailedSegmentationMetrics``, prints a formatted report, and writes
JSON/CSV summaries to disk.

Usage examples
--------------
# Binary – predictions and GT in separate directories, matched by filename stem:
    python -m scripts.compute_metrics \\
        --predictions outputs/predictions \\
        --masks       dataset/masks/test \\
        --output-dir  outputs/metrics/run01

# Multiclass (4 classes including background):
    python -m scripts.compute_metrics \\
        --predictions outputs/predictions \\
        --masks       dataset/masks/test \\
        --num-classes 4 \\
        --class-names 0:background,1:tarp,2:debris,3:water

# GT from a CSV file (must have a ``mask_path`` column; optionally also
# ``image_path`` – only the mask_path column is used here):
    python -m scripts.compute_metrics \\
        --predictions outputs/predictions \\
        --masks       dataset/test.csv \\
        --output-dir  outputs/metrics/run01

# Predictions stored as float32 .npy logit arrays instead of PNG masks:
    python -m scripts.compute_metrics \\
        --predictions outputs/logits \\
        --masks       dataset/masks/test \\
        --pred-format logits

Prediction formats
------------------
``mask``   (default)  PNG/TIFF/BMP with integer class indices.
           Binary masks may have values in {0, 255} — they are automatically
           remapped to {0, 1}.
``logits`` .npy files containing float arrays of shape (H,W) for binary or
           (C,H,W) for multiclass. Thresholded using ``--threshold``.

Output
------
<output-dir>/metrics.json  — all metric keys and their float values
<output-dir>/metrics.csv   — two-column (metric, value) table
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Allow running as a top-level script as well
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from segmentation.metrics.detailed_metrics import DetailedSegmentationMetrics


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

_MASK_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}
_LOGIT_EXTENSIONS = {".npy"}


def _load_mask_png(path: Path) -> np.ndarray:
    """Load a PNG/TIFF mask as a uint8 integer class-index array.

    Binary masks with values in {0, 255} are automatically normalised to
    {0, 1}.
    """
    img = np.array(Image.open(path))
    if img.ndim == 3:
        # Some exporters write RGBA/RGB – take first channel
        img = img[..., 0]
    img = img.astype(np.int32)
    # Remap 0/255 binary masks
    if img.max() == 255 and np.isin(np.unique(img), [0, 255]).all():
        img = (img // 255).astype(np.int32)
    return img


def _load_logit_npy(path: Path) -> np.ndarray:
    """Load a float logit array from an .npy file."""
    arr = np.load(path)
    return arr.astype(np.float32)


def _collect_predictions(pred_dir: Path, fmt: str) -> dict[str, Path]:
    """Return {stem: path} for all prediction files in *pred_dir*."""
    exts = _LOGIT_EXTENSIONS if fmt == "logits" else _MASK_EXTENSIONS
    files: dict[str, Path] = {}
    for f in sorted(pred_dir.iterdir()):
        if f.suffix.lower() in exts:
            files[f.stem] = f
    return files


def _collect_gt_masks(masks_src: Path) -> dict[str, Path]:
    """Return {stem: path} for GT masks.

    *masks_src* can be:
    - A directory containing mask image files.
    - A CSV file with a ``mask_path`` column.
    """
    result: dict[str, Path] = {}

    if masks_src.suffix.lower() == ".csv":
        import csv as _csv
        with open(masks_src) as fh:
            reader = _csv.DictReader(fh)
            if "mask_path" not in (reader.fieldnames or []):
                raise ValueError(
                    f"CSV '{masks_src}' must contain a 'mask_path' column. "
                    f"Found: {reader.fieldnames}"
                )
            for row in reader:
                p = Path(row["mask_path"])
                result[p.stem] = p
    else:
        # Directory
        for f in sorted(masks_src.iterdir()):
            if f.suffix.lower() in _MASK_EXTENSIONS:
                result[f.stem] = f

    return result


def _pair_files(
    predictions: dict[str, Path],
    gt_masks: dict[str, Path],
) -> list[tuple[Path, Path]]:
    """Match predictions to GT masks by stem name."""
    pred_stems = set(predictions)
    gt_stems   = set(gt_masks)

    common = pred_stems & gt_stems
    only_pred = pred_stems - gt_stems
    only_gt   = gt_stems - pred_stems

    if only_pred:
        warnings.warn(
            f"{len(only_pred)} prediction(s) have no matching GT mask and will "
            f"be skipped: {sorted(only_pred)[:5]}{'…' if len(only_pred)>5 else ''}",
            stacklevel=2,
        )
    if only_gt:
        warnings.warn(
            f"{len(only_gt)} GT mask(s) have no matching prediction and will "
            f"be skipped: {sorted(only_gt)[:5]}{'…' if len(only_gt)>5 else ''}",
            stacklevel=2,
        )

    if not common:
        raise RuntimeError(
            "No matching (prediction, GT-mask) pairs found. "
            "Check that filenames (without extension) match between "
            "--predictions and --masks."
        )

    return [(predictions[s], gt_masks[s]) for s in sorted(common)]


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    pairs: list[tuple[Path, Path]],
    pred_format: str,
    metrics: DetailedSegmentationMetrics,
) -> None:
    """Feed all image pairs into *metrics*."""
    for pred_path, gt_path in tqdm(pairs, desc="Evaluating", unit="img"):
        gt = _load_mask_png(gt_path)

        if pred_format == "logits":
            pred = _load_logit_npy(pred_path)
            metrics.update(pred, gt, is_logits=True)
        else:
            pred = _load_mask_png(pred_path)
            metrics.update(pred, gt, is_logits=False)


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_json(result: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"  JSON saved → {path}")


def _save_csv(result: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for k, v in sorted(result.items()):
            writer.writerow([k, f"{v:.6f}"])
    print(f"  CSV  saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_class_names(raw: str | None) -> dict[int, str] | None:
    """Parse ``0:background,1:tarp,2:debris`` into {0: 'background', …}."""
    if not raw:
        return None
    result: dict[int, str] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise argparse.ArgumentTypeError(
                f"Invalid class-name token '{token}'. Expected format: id:name"
            )
        idx, name = token.split(":", 1)
        result[int(idx)] = name.strip()
    return result or None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute pixel-level and instance-level segmentation metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--predictions", "-p",
        type=Path, required=True,
        metavar="DIR",
        help="Directory containing predicted mask files (PNG/TIFF) or logit .npy files.",
    )
    p.add_argument(
        "--masks", "-m",
        type=Path, required=True,
        metavar="DIR_OR_CSV",
        help=(
            "Directory of ground-truth mask PNGs, or a CSV file with a "
            "'mask_path' column."
        ),
    )
    p.add_argument(
        "--output-dir", "-o",
        type=Path, default=None,
        metavar="DIR",
        help=(
            "Directory for metrics.json and metrics.csv. "
            "Defaults to <predictions>/../metrics."
        ),
    )
    p.add_argument(
        "--pred-format",
        choices=["mask", "logits"], default="mask",
        help=(
            "'mask' (default): PNG/TIFF integer class-index images. "
            "'logits': float32 .npy arrays (H,W) binary or (C,H,W) multiclass."
        ),
    )
    p.add_argument(
        "--num-classes", "-k",
        type=int, default=2,
        metavar="K",
        help=(
            "Total number of classes including background (class 0). "
            "Binary segmentation: 2 (default)."
        ),
    )
    p.add_argument(
        "--threshold", "-t",
        type=float, default=0.5,
        metavar="T",
        help="Sigmoid/softmax threshold for logit predictions (default: 0.5).",
    )
    p.add_argument(
        "--instance-threshold",
        type=float, default=0.10,
        metavar="T",
        help=(
            "Minimum overlap/GT-area ratio to count a (GT, pred) blob pair as "
            "a True Positive (default: 0.10 = 10%%)."
        ),
    )
    p.add_argument(
        "--min-pred-size",
        type=int, default=10,
        metavar="PX",
        help="Predicted blobs smaller than this many pixels are discarded (default: 10).",
    )
    p.add_argument(
        "--class-names",
        type=str, default=None,
        metavar="0:bg,1:tarp",
        help=(
            "Comma-separated id:name pairs for the report, e.g. "
            "'0:background,1:tarp'. Background (0) is not printed."
        ),
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Print the report but do not save JSON/CSV files.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args   = parser.parse_args(argv)

    # ── Resolve output directory ──────────────────────────────────────────────
    output_dir: Path = args.output_dir or (args.predictions.parent / "metrics")

    # ── Parse class names ─────────────────────────────────────────────────────
    class_names = _parse_class_names(args.class_names)

    # ── Collect files ─────────────────────────────────────────────────────────
    print(f"\nCollecting prediction files from: {args.predictions}")
    predictions = _collect_predictions(args.predictions, args.pred_format)
    print(f"  Found {len(predictions)} prediction file(s).")

    print(f"Collecting GT mask files from:    {args.masks}")
    gt_masks = _collect_gt_masks(args.masks)
    print(f"  Found {len(gt_masks)} GT mask file(s).")

    pairs = _pair_files(predictions, gt_masks)
    print(f"  Matched {len(pairs)} pair(s).\n")

    # ── Initialise metrics ────────────────────────────────────────────────────
    metrics = DetailedSegmentationMetrics(
        num_classes            = args.num_classes,
        threshold              = args.threshold,
        instance_iou_threshold = args.instance_threshold,
        min_pred_size_px       = args.min_pred_size,
        class_names            = class_names,
    )

    # ── Run evaluation ────────────────────────────────────────────────────────
    evaluate(pairs, args.pred_format, metrics)

    # ── Report ────────────────────────────────────────────────────────────────
    result = metrics.compute()
    metrics.print_report(result)

    # ── Save outputs ──────────────────────────────────────────────────────────
    if not args.no_save:
        _save_json(result, output_dir / "metrics.json")
        _save_csv(result,  output_dir / "metrics.csv")


if __name__ == "__main__":
    main()
