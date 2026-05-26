"""Segmentation loss functions.

Losses can be configured in two ways:

1. **Legacy** (backward-compatible) – set ``training.loss`` in the YAML:
       training:
         loss: "dice_bce"

2. **JSON config** (recommended) – set ``training.loss_config`` to a path:
       training:
         loss_config: "configs/losses/dice_bce.json"

   The JSON defines any number of losses, each with its own weight and args:

       {
         "losses": [
           { "name": "dice",   "weight": 0.5, "args": { "smooth": 1.0 } },
           { "name": "focal",  "weight": 0.5, "args": { "gamma": 2.0  } }
         ]
       }

Available loss names
--------------------
  dice            – Dice loss (binary)
  bce             – Binary cross-entropy with logits
  focal           – Focal loss (binary)
  tversky         – Tversky loss (binary) — penalises FP/FN asymmetrically
  multiclass_dice – Softmax Dice loss (multiclass)

Args reference
--------------
  dice:
    smooth      float   (default 1.0)

  bce:
    pos_weight  float   (default null) — scalar weight for positive class

  focal:
    gamma       float   (default 2.0)
    pos_weight  float   (default null)

  tversky:
    alpha       float   (default 0.3) — FP penalty  (alpha+beta should = 1)
    beta        float   (default 0.7) — FN penalty
    smooth      float   (default 1.0)
    — alpha=beta=0.5 → Dice; beta>alpha → recall-focused (fewer missed tarps)

  multiclass_dice:
    smooth        float   (default 1.0)
    ignore_index  int     (default -1)
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Primitive losses
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Dice loss. Works with both 3-D (B,H,W) and 4-D (B,C,H,W) inputs."""

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
    """Focal loss for binary segmentation."""

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        return ((1.0 - p_t) ** self.gamma * bce).mean()


class TverskyLoss(nn.Module):
    """Tversky loss for binary segmentation.

    Generalises Dice by weighting FP (alpha) and FN (beta) independently.
    Setting beta > alpha focuses on recall — useful when missing a tarp is
    more costly than a false alarm (typical in damage assessment).

    alpha=0.5, beta=0.5  →  equivalent to Dice.
    alpha=0.3, beta=0.7  →  penalises false negatives more.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        spatial = list(range(1, probs.dim()))
        tp = (probs * targets).sum(dim=spatial)
        fp = (probs * (1.0 - targets)).sum(dim=spatial)
        fn = ((1.0 - probs) * targets).sum(dim=spatial)
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky.mean()


class DiceBCELoss(nn.Module):
    """Dice + BCE (legacy convenience class, kept for backward compatibility)."""

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.register_buffer("pos_weight", pos_weight)

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
            inter = (probs[:, c] * targets_one_hot[:, c]).sum()
            union = probs[:, c].sum() + targets_one_hot[:, c].sum()
            dice_per_class.append((2.0 * inter + self.smooth) / (union + self.smooth))
        return 1.0 - torch.stack(dice_per_class).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Combined loss
# ─────────────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """Weighted sum of multiple loss functions loaded from a JSON config.

    Parameters
    ----------
    components :
        List of ``(name, loss_module, weight)`` tuples. Weights need not sum
        to 1 — they are applied as-is to allow easy ablation.
    """

    def __init__(self, components: list[tuple[str, nn.Module, float]]):
        super().__init__()
        self._names   = [n for n, _, _ in components]
        self._weights = [w for _, _, w in components]
        # Register as ModuleList so sub-module parameters are visible
        self._modules_list = nn.ModuleList([m for _, m, _ in components])

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total = torch.tensor(0.0, device=logits.device)
        for module, weight in zip(self._modules_list, self._weights):
            total = total + weight * module(logits, targets)
        return total

    def __repr__(self) -> str:
        parts = [f"{n}(w={w})" for n, w in zip(self._names, self._weights)]
        return f"CombinedLoss([{', '.join(parts)}])"


# ─────────────────────────────────────────────────────────────────────────────
# Factory: build a single loss from a name + args dict
# ─────────────────────────────────────────────────────────────────────────────

def _make_loss(name: str, args: dict, device: torch.device) -> nn.Module:
    """Instantiate one loss module by name with the provided args dict."""
    name = name.lower().strip()

    def _pw() -> torch.Tensor | None:
        v = args.get("pos_weight")
        return torch.tensor([float(v)], device=device) if v is not None else None

    if name == "dice":
        return DiceLoss(smooth=float(args.get("smooth", 1.0)))

    if name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=_pw())

    if name == "focal":
        return FocalLoss(
            gamma=float(args.get("gamma", 2.0)),
            pos_weight=_pw(),
        )

    if name == "tversky":
        return TverskyLoss(
            alpha=float(args.get("alpha", 0.3)),
            beta=float(args.get("beta",  0.7)),
            smooth=float(args.get("smooth", 1.0)),
        )

    if name == "multiclass_dice":
        if "num_classes" not in args:
            raise ValueError("multiclass_dice requires 'num_classes' in args.")
        return MulticlassDiceLoss(
            num_classes=int(args["num_classes"]),
            smooth=float(args.get("smooth", 1.0)),
            ignore_index=int(args.get("ignore_index", -1)),
        )

    raise ValueError(
        f"Unknown loss name '{name}'. "
        f"Valid options: dice, bce, focal, tversky, multiclass_dice."
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSON loader
# ─────────────────────────────────────────────────────────────────────────────

def load_loss_from_json(json_path: str | Path, device: torch.device) -> nn.Module:
    """Build a loss (possibly combined) from a JSON config file.

    JSON schema
    -----------
    {
      "description": "optional human-readable note",
      "losses": [
        { "name": "<loss_name>", "weight": <float>, "args": { ... } },
        ...
      ]
    }

    A single-entry list is unwrapped — no CombinedLoss wrapper is added.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Loss config not found: {json_path}")

    with open(json_path) as f:
        config = json.load(f)

    entries = config.get("losses")
    if not entries:
        raise ValueError(f"JSON '{json_path}' must contain a non-empty 'losses' list.")

    components: list[tuple[str, nn.Module, float]] = []
    for entry in entries:
        name   = entry["name"]
        weight = float(entry.get("weight", 1.0))
        args   = entry.get("args", {})
        module = _make_loss(name, args, device)
        components.append((name, module, weight))

    if len(components) == 1:
        print(f"  Loss: {components[0][0]}")
        return components[0][1].to(device)

    combined = CombinedLoss(components).to(device)
    print(f"  Loss: {combined}")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def get_loss(cfg: dict, device: torch.device) -> nn.Module:
    """Return the configured loss module.

    Priority
    --------
    1. ``training.loss_config`` – path to a JSON file (recommended).
    2. ``training.loss``        – legacy string key (backward-compatible).
    """
    loss_config_path = cfg["training"].get("loss_config")
    if loss_config_path:
        return load_loss_from_json(loss_config_path, device)

    # ── Backward-compatible path ──────────────────────────────────────────────
    loss_name = cfg["training"].get("loss", "dice_bce")
    task      = cfg["data"].get("task", "binary")
    gamma     = cfg["training"].get("focal_gamma", 2.0)

    if task == "multiclass":
        num_classes = cfg["model"]["num_classes"]
        return MulticlassDiceLoss(num_classes=num_classes).to(device)

    if loss_name == "bce":
        return nn.BCEWithLogitsLoss().to(device)
    if loss_name == "dice":
        return DiceLoss().to(device)
    if loss_name == "focal":
        return FocalLoss(gamma=gamma).to(device)
    if loss_name == "dice_bce":
        return DiceBCELoss().to(device)

    raise ValueError(
        f"Unknown loss '{loss_name}'. "
        f"Use training.loss_config to load from JSON instead."
    )
