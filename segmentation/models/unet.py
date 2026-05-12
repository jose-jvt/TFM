"""U-Net model builder using segmentation-models-pytorch.

Supports any timm-compatible encoder with optional ImageNet pretrained weights.
Swap encoder_name in config to experiment with different backbones.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_model(cfg: dict) -> nn.Module:
    model_cfg = cfg["model"]
    num_classes = model_cfg["num_classes"]  # 1 = binary, N = multiclass
    pretrained = model_cfg.get("pretrained", True)

    model = smp.Unet(
        encoder_name=model_cfg.get("encoder", "resnet34"),
        encoder_weights="imagenet" if pretrained else None,
        in_channels=model_cfg.get("in_channels", 3),
        classes=num_classes,
        activation=None,  # raw logits; loss functions apply activation internally
    )
    return model


def load_checkpoint(cfg: dict, checkpoint_path: str, device: torch.device) -> nn.Module:
    model = build_model(cfg)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    return model.to(device)
