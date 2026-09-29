"""Tests for floor-shape t-SNE row selection (no GPU / sklearn)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from balcony_train.tsne_floor_shape import iter_floor_rows


class TsneFloorShapeTests(unittest.TestCase):
    def test_keeps_labeled_real_and_synth_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels.jsonl"
            ow = root / "crops" / "open_work"
            solid = root / "crops" / "solid"
            ow.mkdir(parents=True)
            solid.mkdir()
            Image.new("RGB", (8, 8)).save(ow / "cmp_a.png")
            Image.new("RGB", (8, 8)).save(ow / "flux2_metal_000.png")
            Image.new("RGB", (8, 8)).save(ow / "cmp_nolabel.png")
            Image.new("RGB", (8, 8)).save(ow / "other.png")
            Image.new("RGB", (8, 8)).save(solid / "cmp_b.png")
            records = [
                {"path": "cmp_a.png", "kind": "open_work", "floor_shape": "circle"},
                {"path": "flux2_metal_000.png", "kind": "open_work", "floor_shape": "Hexagon"},
                {"path": "cmp_nolabel.png", "kind": "open_work", "floor_shape": None},
                {"path": "other.png", "kind": "open_work", "floor_shape": "rectangle"},
                {"path": "cmp_b.png", "kind": "solid", "floor_shape": "trapezoid"},
            ]
            labels.write_text(
                "\n".join(json.dumps(rec) for rec in records) + "\n",
                encoding="utf-8",
            )
            rows = iter_floor_rows(root / "crops", labels)
            self.assertEqual(
                [(path.name, src, kind, floor) for path, src, kind, floor in rows],
                [
                    ("cmp_a.png", "real", "open_work", "circle"),
                    ("flux2_metal_000.png", "synth", "open_work", "hexagon"),
                    ("cmp_b.png", "real", "solid", "trapezoid"),
                ],
            )


if __name__ == "__main__":
    unittest.main()
