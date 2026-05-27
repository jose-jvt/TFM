"""Detailed segmentation metrics: pixel-level and instance-level.

Pixel metrics  (confusion-matrix-based, class 0 / background always excluded):
    Per class  : IoU, Precision, Recall, F1
    Micro      : aggregate TP/FP/FN numerically across all foreground classes
    Macro      : mean of per-class values

Instance metrics (connected-component blob matching, background excluded):
    A GT blob is a True Positive  if any predicted blob covers
    ≥ ``instance_iou_threshold`` of its area.
    A predicted blob smaller than ``min_pred_size_px`` pixels is discarded
    before matching.
    Per class  : Precision, Recall, F1  + raw TP / FP / FN counts
    Macro      : mean over foreground classes

Both metric families accumulate state across calls to ``update()`` so they
can be used in a streaming fashion (batch-by-batch) or fed full arrays at once.
``reset()`` clears all accumulated state.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from scipy import ndimage


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

_EPS = 1e-7


def _filter_small_blobs(binary_mask: np.ndarray, min_size: int) -> np.ndarray:
    """Zero out connected components smaller than *min_size* pixels."""
    labeled, n = ndimage.label(binary_mask)
    if n == 0:
        return binary_mask
    sizes = ndimage.sum(binary_mask, labeled, range(1, n + 1))
    remove_ids = np.where(np.array(sizes) < min_size)[0] + 1
    for rid in remove_ids:
        labeled[labeled == rid] = 0
    return (labeled > 0).astype(np.uint8)


def _match_blobs(
    gt_binary: np.ndarray,
    pred_binary: np.ndarray,
    threshold: float,
) -> tuple[int, int, int]:
    """Greedy blob matching for one class.

    Parameters
    ----------
    gt_binary, pred_binary :
        (H, W) uint8 masks for a single class (already small-blob-filtered
        for pred).
    threshold :
        Minimum ``overlap / gt_blob_area`` to consider a match a TP.

    Returns
    -------
    (tp, fp, fn)
    """
    gt_labeled, n_gt   = ndimage.label(gt_binary)
    pred_labeled, n_pred = ndimage.label(pred_binary)

    if n_gt == 0 and n_pred == 0:
        return 0, 0, 0
    if n_gt == 0:
        return 0, n_pred, 0
    if n_pred == 0:
        return 0, 0, n_gt

    # Vectorised overlap count via flat-index encoding
    gt_flat   = gt_labeled.ravel()
    pred_flat = pred_labeled.ravel()
    both      = (gt_flat > 0) & (pred_flat > 0)

    if not both.any():
        # No spatial overlap at all
        return 0, n_pred, n_gt

    gt_sel   = gt_flat[both] - 1   # 0-indexed
    pred_sel = pred_flat[both] - 1

    pair_ids    = gt_sel * n_pred + pred_sel
    pair_counts = np.bincount(pair_ids, minlength=n_gt * n_pred).reshape(n_gt, n_pred)

    gt_areas      = np.bincount(gt_flat, minlength=n_gt + 1)[1:]   # pixels per GT blob
    overlap_ratio = pair_counts / (gt_areas[:, np.newaxis] + _EPS)  # (n_gt, n_pred)

    # Greedy matching: sort candidate pairs by overlap ratio descending
    gi_arr, pi_arr = np.where(overlap_ratio >= threshold)
    if len(gi_arr) == 0:
        return 0, n_pred, n_gt

    order  = np.argsort(-overlap_ratio[gi_arr, pi_arr])
    gi_arr = gi_arr[order]
    pi_arr = pi_arr[order]

    matched_gt   = set()
    matched_pred = set()
    tp = 0

    for gi, pi in zip(gi_arr.tolist(), pi_arr.tolist()):
        if gi not in matched_gt and pi not in matched_pred:
            matched_gt.add(gi)
            matched_pred.add(pi)
            tp += 1

    fn = n_gt   - tp
    fp = n_pred - len(matched_pred)
    return tp, fp, fn


# ─────────────────────────────────────────────────────────────────────────────
# Public class
# ─────────────────────────────────────────────────────────────────────────────

class DetailedSegmentationMetrics:
    """Accumulates pixel-level and (optionally) instance-level metrics across images.

    Parameters
    ----------
    num_classes :
        Total number of classes **including** background (class 0).
        Binary segmentation → ``num_classes=2``.
    threshold :
        Sigmoid / softmax threshold used when converting logits to a hard
        prediction mask.  Ignored when ``update()`` receives integer arrays
        directly.
    instance_iou_threshold :
        Minimum ``overlap_area / gt_instance_area`` ratio to count a
        (GT, pred) blob pair as a True Positive.  Default 0.10 (10 %).
    min_pred_size_px :
        Predicted blobs with fewer pixels than this are discarded before
        instance matching (noise filter).  Default 10.
    class_names :
        Optional mapping ``{class_id: name}`` used in printed output.
        Background (0) is never included in the report.
    pixel_only :
        When ``True``, instance-level metrics (blob matching) are **not**
        computed — only the confusion-matrix pixel metrics are accumulated.
        Use this for the train/val loops where instance matching is too
        expensive to run every epoch; set to ``False`` (default) for the
        final test evaluation.
    """

    def __init__(
        self,
        num_classes: int = 2,
        threshold: float = 0.5,
        instance_iou_threshold: float = 0.10,
        min_pred_size_px: int = 10,
        class_names: dict[int, str] | None = None,
        pixel_only: bool = False,
    ):
        if num_classes < 2:
            raise ValueError("num_classes must be ≥ 2 (including background).")

        self.num_classes             = num_classes
        self.threshold               = threshold
        self.instance_iou_threshold  = instance_iou_threshold
        self.min_pred_size_px        = min_pred_size_px
        self.class_names             = class_names or {}
        self.pixel_only              = pixel_only
        self._fg_classes             = list(range(1, num_classes))

        self.reset()

    # ── State management ─────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        # (K, K) confusion matrix: cm[true, pred]
        self._cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        # Instance counts per foreground class
        self._inst_tp: dict[int, int] = {c: 0 for c in self._fg_classes}
        self._inst_fp: dict[int, int] = {c: 0 for c in self._fg_classes}
        self._inst_fn: dict[int, int] = {c: 0 for c in self._fg_classes}

    # ── Update API ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def update(
        self,
        predictions: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        is_logits: bool = True,
    ) -> None:
        """Accumulate one batch or one image.

        Parameters
        ----------
        predictions :
            * If ``is_logits=True``:
              ``(B, C, H, W)`` float logits  (multiclass)
              ``(B, 1, H, W)`` or ``(B, H, W)`` float logits  (binary)
            * If ``is_logits=False``:
              ``(B, H, W)`` or ``(H, W)`` integer class-index mask (0–K-1)
        targets :
            ``(B, H, W)`` or ``(H, W)`` integer class-index mask (0–K-1).
        is_logits :
            Whether *predictions* are raw model outputs that need thresholding.
        """
        # ── Convert to numpy integer masks ───────────────────────────────────
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        if is_logits:
            pred_mask = self._logits_to_mask(predictions)
        else:
            pred_mask = predictions.astype(np.int32)

        gt_mask = targets.astype(np.int32)

        # Handle single-image (H, W) vs batch (B, H, W)
        if pred_mask.ndim == 2:
            pred_mask = pred_mask[np.newaxis]
            gt_mask   = gt_mask[np.newaxis]

        for pred, gt in zip(pred_mask, gt_mask):
            self._update_pixel(pred.astype(np.int32), gt.astype(np.int32))
            if not self.pixel_only:
                self._update_instance(pred.astype(np.uint8), gt.astype(np.uint8))

    # ── Compute ──────────────────────────────────────────────────────────────

    def compute(self) -> dict[str, float]:
        """Return a flat dict with all metrics.

        Keys follow the pattern:
            ``pixel/<metric>_c<id>``     per-class pixel metrics
            ``pixel/<metric>_micro``     micro-averaged pixel metrics
            ``pixel/<metric>_macro``     macro-averaged pixel metrics
            ``instance/<metric>_c<id>``  per-class instance metrics
            ``instance/<metric>_macro``  macro-averaged instance metrics

        For binary tasks (one foreground class) top-level aliases are also
        included: ``pixel/iou``, ``instance/recall``, etc.
        """
        result: dict[str, float] = {}

        # ── Pixel metrics from confusion matrix ───────────────────────────────
        cm   = self._cm
        fg   = np.array(self._fg_classes)

        tp_all = np.diag(cm)[fg]
        fp_all = cm.sum(axis=0)[fg] - tp_all   # predicted-as-c minus correct
        fn_all = cm.sum(axis=1)[fg] - tp_all   # actually-c minus correct

        iou_c  = tp_all / (tp_all + fp_all + fn_all + _EPS)
        pre_c  = tp_all / (tp_all + fp_all + _EPS)
        rec_c  = tp_all / (tp_all + fn_all + _EPS)
        f1_c   = 2 * tp_all / (2 * tp_all + fp_all + fn_all + _EPS)

        for i, c in enumerate(self._fg_classes):
            tag = f"c{c}"
            result[f"pixel/iou_{tag}"]       = float(iou_c[i])
            result[f"pixel/precision_{tag}"]  = float(pre_c[i])
            result[f"pixel/recall_{tag}"]     = float(rec_c[i])
            result[f"pixel/f1_{tag}"]         = float(f1_c[i])

        # Micro
        tp_m = int(tp_all.sum())
        fp_m = int(fp_all.sum())
        fn_m = int(fn_all.sum())
        result["pixel/iou_micro"]       = float(tp_m / (tp_m + fp_m + fn_m + _EPS))
        result["pixel/precision_micro"] = float(tp_m / (tp_m + fp_m + _EPS))
        result["pixel/recall_micro"]    = float(tp_m / (tp_m + fn_m + _EPS))
        result["pixel/f1_micro"]        = float(2 * tp_m / (2 * tp_m + fp_m + fn_m + _EPS))

        # Macro
        result["pixel/iou_macro"]       = float(iou_c.mean())
        result["pixel/precision_macro"] = float(pre_c.mean())
        result["pixel/recall_macro"]    = float(rec_c.mean())
        result["pixel/f1_macro"]        = float(f1_c.mean())

        # ── Instance metrics (test set only — skipped when pixel_only=True) ────
        if not self.pixel_only:
            inst_pre_list, inst_rec_list, inst_f1_list = [], [], []

            for c in self._fg_classes:
                tp = self._inst_tp[c]
                fp = self._inst_fp[c]
                fn = self._inst_fn[c]
                pre = tp / (tp + fp + _EPS)
                rec = tp / (tp + fn + _EPS)
                f1  = 2 * pre * rec / (pre + rec + _EPS)

                tag = f"c{c}"
                result[f"instance/precision_{tag}"] = float(pre)
                result[f"instance/recall_{tag}"]    = float(rec)
                result[f"instance/f1_{tag}"]        = float(f1)
                result[f"instance/tp_{tag}"]        = float(tp)
                result[f"instance/fp_{tag}"]        = float(fp)
                result[f"instance/fn_{tag}"]        = float(fn)

                inst_pre_list.append(pre)
                inst_rec_list.append(rec)
                inst_f1_list.append(f1)

            result["instance/precision_macro"] = float(np.mean(inst_pre_list))
            result["instance/recall_macro"]    = float(np.mean(inst_rec_list))
            result["instance/f1_macro"]        = float(np.mean(inst_f1_list))

        # ── Binary aliases (single fg class → clean top-level keys) ──────────
        if len(self._fg_classes) == 1:
            c = self._fg_classes[0]
            for metric in ("iou", "precision", "recall", "f1"):
                result[f"pixel/{metric}"] = result[f"pixel/{metric}_c{c}"]
            if not self.pixel_only:
                for metric in ("precision", "recall", "f1"):
                    result[f"instance/{metric}"] = result[f"instance/{metric}_c{c}"]

        return result

    # ── Pretty print ─────────────────────────────────────────────────────────

    def print_report(self, result: dict[str, float] | None = None) -> None:
        """Print a formatted metric report to stdout."""
        if result is None:
            result = self.compute()

        def _cls_name(c: int) -> str:
            return self.class_names.get(c, f"class_{c}")

        w = 12  # column width

        print(f"\n{'═'*70}")
        print(f"  PIXEL METRICS  (background class 0 excluded)")
        print(f"{'─'*70}")
        header = f"  {'Class':<18} {'IoU':>{w}} {'Precision':>{w}} {'Recall':>{w}} {'F1':>{w}}"
        print(header)
        print(f"  {'─'*66}")

        for c in self._fg_classes:
            tag  = f"c{c}"
            name = _cls_name(c)
            print(
                f"  {name:<18}"
                f"  {result[f'pixel/iou_{tag}']:>{w}.4f}"
                f"  {result[f'pixel/precision_{tag}']:>{w}.4f}"
                f"  {result[f'pixel/recall_{tag}']:>{w}.4f}"
                f"  {result[f'pixel/f1_{tag}']:>{w}.4f}"
            )

        print(f"  {'─'*66}")
        for avg in ("micro", "macro"):
            print(
                f"  {avg.upper():<18}"
                f"  {result[f'pixel/iou_{avg}']:>{w}.4f}"
                f"  {result[f'pixel/precision_{avg}']:>{w}.4f}"
                f"  {result[f'pixel/recall_{avg}']:>{w}.4f}"
                f"  {result[f'pixel/f1_{avg}']:>{w}.4f}"
            )

        if not self.pixel_only:
            print(f"\n{'─'*70}")
            print(
                f"  INSTANCE METRICS  "
                f"(overlap threshold={self.instance_iou_threshold:.0%}, "
                f"min pred size={self.min_pred_size_px} px)"
            )
            print(f"{'─'*70}")
            header2 = f"  {'Class':<18} {'Precision':>{w}} {'Recall':>{w}} {'F1':>{w}} {'TP':>6} {'FP':>6} {'FN':>6}"
            print(header2)
            print(f"  {'─'*66}")

            for c in self._fg_classes:
                tag  = f"c{c}"
                name = _cls_name(c)
                print(
                    f"  {name:<18}"
                    f"  {result[f'instance/precision_{tag}']:>{w}.4f}"
                    f"  {result[f'instance/recall_{tag}']:>{w}.4f}"
                    f"  {result[f'instance/f1_{tag}']:>{w}.4f}"
                    f"  {int(result[f'instance/tp_{tag}']):>6}"
                    f"  {int(result[f'instance/fp_{tag}']):>6}"
                    f"  {int(result[f'instance/fn_{tag}']):>6}"
                )

            print(f"  {'─'*66}")
            print(
                f"  {'MACRO':<18}"
                f"  {result['instance/precision_macro']:>{w}.4f}"
                f"  {result['instance/recall_macro']:>{w}.4f}"
                f"  {result['instance/f1_macro']:>{w}.4f}"
            )

        print(f"{'═'*70}\n")

    # ── Private ──────────────────────────────────────────────────────────────

    def _logits_to_mask(self, logits: np.ndarray) -> np.ndarray:
        """Convert raw model output to integer class-index mask."""
        if logits.ndim == 4:
            # (B, C, H, W)
            if logits.shape[1] == 1:
                # Binary: squeeze channel dim
                prob = 1 / (1 + np.exp(-logits[:, 0]))  # sigmoid
                return (prob >= self.threshold).astype(np.int32)
            else:
                return logits.argmax(axis=1).astype(np.int32)
        elif logits.ndim == 3:
            # (B, H, W) binary
            prob = 1 / (1 + np.exp(-logits))
            return (prob >= self.threshold).astype(np.int32)
        elif logits.ndim == 2:
            # Single image (H, W) binary
            prob = 1 / (1 + np.exp(-logits))
            return (prob >= self.threshold).astype(np.int32)
        raise ValueError(f"Unexpected logits shape: {logits.shape}")

    def _update_pixel(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """Accumulate confusion matrix for one image."""
        # Clamp to valid class range (guard against corrupt masks)
        pred = np.clip(pred, 0, self.num_classes - 1)
        gt   = np.clip(gt,   0, self.num_classes - 1)
        # Fast flat-index bincount
        flat = self.num_classes * gt.ravel() + pred.ravel()
        self._cm += np.bincount(flat, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes
        )

    def _update_instance(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """Accumulate instance TP/FP/FN for one image."""
        for c in self._fg_classes:
            gt_c   = (gt == c).astype(np.uint8)
            pred_c = (pred == c).astype(np.uint8)
            pred_c = _filter_small_blobs(pred_c, self.min_pred_size_px)

            tp, fp, fn = _match_blobs(gt_c, pred_c, self.instance_iou_threshold)
            self._inst_tp[c] += tp
            self._inst_fp[c] += fp
            self._inst_fn[c] += fn
