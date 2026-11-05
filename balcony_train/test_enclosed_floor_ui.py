"""Floor-plan label helpers for enclosed cells. No window is opened."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.enclosed_floor_ui import (  # noqa: E402
    assign_photo_shape,
    crop_stem,
    enclosed_queue,
    extract_cells,
    photo_shape,
)


class EnclosedFloorLabelTests(unittest.TestCase):
    def test_crop_stem_is_unique_per_facade(self) -> None:
        self.assertEqual(crop_stem("cmp_b0001", (1, 2)), "cmp_b0001_f1b2")
        self.assertNotEqual(crop_stem("cmp_b0001", (1, 2)), crop_stem("cmp_b0002", (1, 2)))

    def test_queue_keeps_only_enclosed_photos_with_a_grid(self) -> None:
        photo = ROOT / "data" / "base"
        dsl = ROOT / "runs" / "batch"
        if not (photo / "cmp_b0001.jpg").is_file():
            self.skipTest("data/base photo missing")
        labels = {"cmp_b0001": [], "missing_photo": [(0, 0)]}
        self.assertEqual(enclosed_queue(labels, photo, dsl), [])
        labels["cmp_b0001"] = [(1, 0)]
        if (dsl / "cmp_b0001" / "facade_dsl.json").is_file():
            self.assertEqual(enclosed_queue(labels, photo, dsl), ["cmp_b0001"])

    def test_one_shape_is_written_on_every_cell(self) -> None:
        records: dict = {"other_f0b0": {"floor_shape": "circle", "enclosure": "enclosed"}}
        cells = [(1, 0), (1, 1)]
        assign_photo_shape(records, "cmp_b0008", cells, "circle")
        self.assertEqual(photo_shape(records, "cmp_b0008", cells), "circle")
        self.assertEqual(records["cmp_b0008_f1b0"]["enclosure"], "enclosed")
        self.assertIsNone(records["cmp_b0008_f1b0"]["kind"])
        assign_photo_shape(records, "cmp_b0008", [(1, 0)], "rectangle")
        self.assertNotIn("cmp_b0008_f1b1", records)
        self.assertEqual(records["other_f0b0"]["floor_shape"], "circle")
        with self.assertRaises(ValueError):
            assign_photo_shape(records, "cmp_b0008", [(1, 0)], "triangle")

    def test_extract_skips_cells_off_the_grid(self) -> None:
        photo = Image.new("RGB", (30, 20), (10, 20, 30))
        floor_y = {0: (0, 10), 1: (10, 20)}
        bay_x = {0: (0, 15), 1: (15, 30)}
        crops = extract_cells(photo, floor_y, bay_x, [(1, 1), (9, 9)])
        self.assertEqual([cell for cell, _im in crops], [(1, 1)])
        self.assertEqual(crops[0][1].size, (15, 10))


if __name__ == "__main__":
    unittest.main()
