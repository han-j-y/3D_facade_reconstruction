"""Dataset utilities for image → AST token sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from window_ast.schema import Vocab

try:
    from torchvision import transforms as T
except ImportError:  # pragma: no cover
    T = None

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_eval_transform(image_size: int = 224):
    if T is None:
        raise ImportError("torchvision is required")
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


def make_train_transform(image_size: int = 224):
    if T is None:
        raise ImportError("torchvision is required")
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


class WindowAstDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        vocab: Vocab,
        *,
        transform=None,
        max_len: int = 160,
        load_heatmaps: bool = False,
        heatmap_size: int = 56,
    ) -> None:
        self.rows = rows
        self.vocab = vocab
        self.transform = transform or make_eval_transform()
        self.max_len = max_len
        self.load_heatmaps = load_heatmaps
        self.heatmap_size = heatmap_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        img = Image.open(row["image"]).convert("RGB")
        image = self.transform(img)
        tokens = row.get("tokens") or []
        if isinstance(tokens, str):
            tokens = tokens.split()
        token_ids = self.vocab.encode(list(tokens), add_special=True)
        if len(token_ids) > self.max_len:
            token_ids = token_ids[: self.max_len - 1] + [self.vocab.eos_id]
        out: dict[str, Any] = {
            "image": image,
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "id": row["id"],
            "family_id": row.get("family_id", ""),
            "step_index": int(row.get("step_index", 0)),
            "recipe": row.get("recipe", ""),
        }
        if self.load_heatmaps:
            from window_ast.heatmaps import gt_heatmap_tensor

            out["heat_maps"] = gt_heatmap_tensor(
                row.get("ir"),
                size=self.heatmap_size,
            )
        return out


def collate_batch(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    lengths = [b["token_ids"].numel() for b in batch]
    max_len = max(lengths)
    tokens = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["token_ids"].numel()
        tokens[i, :n] = b["token_ids"]
    out: dict[str, Any] = {
        "image": images,
        "token_ids": tokens,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "ids": [b["id"] for b in batch],
        "family_ids": [b["family_id"] for b in batch],
        "step_indices": torch.tensor([b["step_index"] for b in batch], dtype=torch.long),
        "recipes": [b["recipe"] for b in batch],
    }
    if batch and "heat_maps" in batch[0]:
        out["heat_maps"] = torch.stack([b["heat_maps"] for b in batch], dim=0)
    return out
