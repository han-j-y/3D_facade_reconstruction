"""Labeled balcony crops for kind (+ optional material) training."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from balcony_train.labels import (
    CLASSES,
    MATERIAL_IGNORE_INDEX,
    MATERIALS,
    MultitaskSample,
    class_counts,
    ensure_crop_dirs,
    iter_labeled_samples,
    iter_multitask_samples,
    material_counts,
)
from balcony_train.model import make_transform

__all__ = [
    "CLASSES",
    "MATERIALS",
    "MATERIAL_IGNORE_INDEX",
    "MultitaskSample",
    "RailingCropDataset",
    "MultitaskCropDataset",
    "class_counts",
    "material_counts",
    "ensure_crop_dirs",
    "iter_labeled_samples",
    "iter_multitask_samples",
]


class RailingCropDataset(Dataset):
    """Kind-only dataset (folder labels). Kept for simple kind evals."""

    def __init__(self, samples: list[tuple[Path, int]], *, transform=None) -> None:
        self.samples = list(samples)
        self.transform = transform or make_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), int(label)


class MultitaskCropDataset(Dataset):
    """Returns (image, kind_idx, material_idx). material may be IGNORE."""

    def __init__(self, samples: list[MultitaskSample], *, transform=None) -> None:
        self.samples = list(samples)
        self.transform = transform or make_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        path, kind_idx, mat_idx = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), int(kind_idx), int(mat_idx)
