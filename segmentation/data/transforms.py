"""Albumentations-based transforms for training and validation."""
from __future__ import annotations

from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(cfg: dict) -> A.Compose:
    aug_cfg = cfg.get("augmentation", {})
    image_size: Optional[int] = cfg["data"].get("image_size")

    transforms = []

    if image_size:
        transforms.append(A.Resize(image_size, image_size))

    if aug_cfg.get("enabled", True):
        if aug_cfg.get("horizontal_flip", True):
            transforms.append(A.HorizontalFlip(p=0.5))
        if aug_cfg.get("vertical_flip", True):
            transforms.append(A.VerticalFlip(p=0.5))
        if aug_cfg.get("random_rotate_90", True):
            transforms.append(A.RandomRotate90(p=0.5))
        if aug_cfg.get("brightness_contrast", True):
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=aug_cfg.get("brightness_limit", 0.2),
                    contrast_limit=aug_cfg.get("contrast_limit", 0.2),
                    p=0.4,
                )
            )

    transforms += [
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def get_val_transforms(cfg: dict) -> A.Compose:
    image_size: Optional[int] = cfg["data"].get("image_size")

    transforms = []
    if image_size:
        transforms.append(A.Resize(image_size, image_size))

    transforms += [
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ]

    return A.Compose(transforms)
