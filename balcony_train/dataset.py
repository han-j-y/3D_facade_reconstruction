"""Labeled balcony crops: folder name is the class (baluster/ | solid/)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from balcony_train.labels import CLASSES, class_counts, ensure_crop_dirs, iter_labeled_samples
from balcony_train.model import make_transform

__all__ = [
    "CLASSES",
    "RailingCropDataset",
    "class_counts",
    "ensure_crop_dirs",
    "iter_labeled_samples",
]


class RailingCropDataset(Dataset):
    """Reads labeled images under ``crops/baluster/`` and ``crops/solid/``. Ignores unlabeled/."""

    def __init__(self, samples: list[tuple[Path, int]], *, transform=None) -> None:
        self.samples = list(samples)
        self.transform = transform or make_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), int(label)
