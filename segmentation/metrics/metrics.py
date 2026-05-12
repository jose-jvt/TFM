"""Segmentation metrics: IoU, F1, Precision, Recall."""
from __future__ import annotations

import torch
import numpy as np


class SegmentationMetrics:
    """Accumulates predictions across batches and computes final metrics.

    Usage:
        metrics = SegmentationMetrics(task="binary")
        for batch in loader:
            preds = model(batch["image"])
            metrics.update(preds, batch["mask"])
        result = metrics.compute()
    """

    def __init__(self, task: str = "binary", num_classes: int = 1, threshold: float = 0.5):
        self.task = task
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset()

    def reset(self):
        self._tp = 0.0
        self._fp = 0.0
        self._fn = 0.0
        self._tn = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        if self.task == "binary":
            preds = (torch.sigmoid(logits) > self.threshold).long()
            if preds.dim() == 4:
                preds = preds.squeeze(1)
        else:
            preds = torch.argmax(logits, dim=1)

        targets = targets.long()
        # Flatten for metric calculation
        preds_flat = preds.view(-1).cpu()
        targets_flat = targets.view(-1).cpu()

        if self.task == "binary":
            self._tp += ((preds_flat == 1) & (targets_flat == 1)).sum().item()
            self._fp += ((preds_flat == 1) & (targets_flat == 0)).sum().item()
            self._fn += ((preds_flat == 0) & (targets_flat == 1)).sum().item()
            self._tn += ((preds_flat == 0) & (targets_flat == 0)).sum().item()
        else:
            # Macro average: compute per-class and average (excluding background=0)
            for c in range(1, self.num_classes):
                self._tp += ((preds_flat == c) & (targets_flat == c)).sum().item()
                self._fp += ((preds_flat == c) & (targets_flat != c)).sum().item()
                self._fn += ((preds_flat != c) & (targets_flat == c)).sum().item()

    def compute(self) -> dict[str, float]:
        tp, fp, fn, tn = self._tp, self._fp, self._fn, self._tn
        eps = 1e-7

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)

        metrics = {
            "iou": iou,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

        if self.task == "binary":
            specificity = tn / (tn + fp + eps)
            metrics["specificity"] = specificity

        return {k: float(v) for k, v in metrics.items()}
