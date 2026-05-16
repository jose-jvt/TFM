"""Segmentation loss functions."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        spatial = list(range(1, probs.dim()))
        intersection = (probs * targets).sum(dim=spatial)
        union = probs.sum(dim=spatial) + targets.sum(dim=spatial)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class DiceBCELoss(nn.Module):
    """Combines Dice and BCE losses, weighted equally by default."""

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        focal_gamma: float = 2.0,
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        dice_loss = self.dice(logits, targets)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class MulticlassDiceLoss(nn.Module):
    """Dice loss for multiclass segmentation (softmax-based)."""

    def __init__(self, num_classes: int, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        dice_per_class = []
        for c in range(self.num_classes):
            if c == self.ignore_index:
                continue
            intersection = (probs[:, c] * targets_one_hot[:, c]).sum()
            union = probs[:, c].sum() + targets_one_hot[:, c].sum()
            dice_per_class.append((2.0 * intersection + self.smooth) / (union + self.smooth))
        return 1.0 - torch.stack(dice_per_class).mean()


def get_loss(cfg: dict, device: torch.device) -> nn.Module:
    loss_name = cfg["training"].get("loss", "dice_bce")
    task = cfg["data"].get("task", "binary")
    gamma = cfg["training"].get("focal_gamma", 2.0)

    if task == "multiclass":
        num_classes = cfg["model"]["num_classes"]
        return MulticlassDiceLoss(num_classes=num_classes).to(device)

    # Binary losses
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss().to(device)
    if loss_name == "dice":
        return DiceLoss().to(device)
    if loss_name == "focal":
        return FocalLoss(gamma=gamma).to(device)
    if loss_name == "dice_bce":
        return DiceBCELoss(focal_gamma=gamma).to(device)

    raise ValueError(f"Unknown loss: {loss_name}")
