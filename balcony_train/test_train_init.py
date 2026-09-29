"""Init-checkpoint loading for floor-head training (no GPU, no DINOv2)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from balcony_train.labels import CLASSES, FLOOR_SHAPES, MATERIALS
from balcony_train.train import load_init_heads


class _Stub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classes = CLASSES
        self.materials = MATERIALS
        self.floor_shapes = FLOOR_SHAPES
        self.kind_head = nn.Linear(4, len(CLASSES))
        self.material_head = nn.Linear(4, len(MATERIALS))
        self.floor_head = nn.Linear(4, len(FLOOR_SHAPES))


class LoadInitHeadsTests(unittest.TestCase):
    def test_loads_kind_and_material_without_floor(self) -> None:
        src = _Stub()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            torch.save(
                {
                    "classes": list(CLASSES),
                    "materials": list(MATERIALS),
                    "kind_head": src.kind_head.state_dict(),
                    "material_head": src.material_head.state_dict(),
                },
                path,
            )
            dst = _Stub()
            had_floor = load_init_heads(dst, path)
        self.assertFalse(had_floor)
        self.assertTrue(
            torch.equal(dst.kind_head.weight, src.kind_head.weight)
        )
        self.assertTrue(
            torch.equal(dst.material_head.weight, src.material_head.weight)
        )

    def test_loads_matching_floor_head(self) -> None:
        src = _Stub()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            torch.save(
                {
                    "classes": list(CLASSES),
                    "materials": list(MATERIALS),
                    "floor_shapes": list(FLOOR_SHAPES),
                    "kind_head": src.kind_head.state_dict(),
                    "material_head": src.material_head.state_dict(),
                    "floor_head": src.floor_head.state_dict(),
                },
                path,
            )
            dst = _Stub()
            had_floor = load_init_heads(dst, path)
        self.assertTrue(had_floor)
        self.assertTrue(
            torch.equal(dst.floor_head.weight, src.floor_head.weight)
        )


if __name__ == "__main__":
    unittest.main()
