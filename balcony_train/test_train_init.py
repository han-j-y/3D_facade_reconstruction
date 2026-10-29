"""Init-checkpoint loading for floor-head training (no GPU, no DINOv2)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from balcony_train.labels import CLASSES, ENCLOSURE_SKINS, FLOOR_SHAPES, MATERIALS
from balcony_train.model import apply_skin_checkpoint, apply_structure_checkpoint
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

    def test_missing_skin_head_stays_untrained(self) -> None:
        model = _Stub()
        trained = apply_skin_checkpoint(
            model,
            {"kind_head": {}, "floor_shapes": list(FLOOR_SHAPES)},
        )
        self.assertFalse(trained)
        self.assertFalse(model._skin_trained)

    def test_old_checkpoint_has_no_structure_head(self) -> None:
        ckpt_path = Path(__file__).resolve().parents[1] / "checkpoints" / "railing_best.pt"
        if not ckpt_path.is_file():
            self.skipTest("railing checkpoint missing")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = _Stub()
        self.assertFalse(apply_structure_checkpoint(model, ckpt))
        self.assertFalse(model._structure_trained)

    def test_railing_best_has_no_skin_head(self) -> None:
        ckpt_path = Path(__file__).resolve().parents[1] / "checkpoints" / "railing_best.pt"
        if not ckpt_path.is_file():
            self.skipTest("railing checkpoint missing")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.assertNotIn("skin_head", ckpt)
        model = _Stub()
        self.assertFalse(apply_skin_checkpoint(model, ckpt))
        self.assertEqual(ENCLOSURE_SKINS, ("punched_window", "curtain_wall"))


if __name__ == "__main__":
    unittest.main()
