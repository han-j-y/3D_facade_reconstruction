"""Tests for SAM3 recrop helpers (no GPU / transformers required)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.recrop_sam3 import (  # noqa: E402
    clamp_box_xyxy,
    filter_records,
    list_input_images,
    pick_max_score_record,
)


class RecropSam3HelperTests(unittest.TestCase):
    def test_pick_max_score(self) -> None:
        best = pick_max_score_record(
            [
                {"score": 0.2, "box_xyxy": [0, 0, 10, 10]},
                {"score": 0.9, "box_xyxy": [1, 1, 20, 20]},
                {"score": 0.5, "box_xyxy": [2, 2, 8, 8]},
            ]
        )
        assert best is not None
        self.assertAlmostEqual(best["score"], 0.9)
        self.assertEqual(best["box_xyxy"], [1, 1, 20, 20])
        self.assertIsNone(pick_max_score_record([]))

    def test_filter_records_min_side_and_max_frac(self) -> None:
        boxes = [
            [0, 0, 5, 5],  # too small
            [0, 0, 50, 40],  # ok
            [0, 0, 95, 90],  # too large vs max_side_frac=0.8 on 100x100
        ]
        scores = [0.9, 0.8, 0.95]
        kept = filter_records(
            boxes,
            scores,
            iw=100,
            ih=100,
            min_side=16,
            max_side_frac=0.8,
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["box_xyxy"], [0, 0, 50, 40])

    def test_clamp_and_list_images(self) -> None:
        self.assertEqual(clamp_box_xyxy([-5, -5, 50, 40], width=30, height=20), (0, 0, 30, 20))
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            Image.new("RGB", (8, 8), (1, 2, 3)).save(d / "flux2_masonry_000.png")
            Image.new("RGB", (8, 8), (4, 5, 6)).save(d / "other.png")
            Image.new("RGB", (8, 8), (7, 8, 9)).save(d / "a.jpg")
            Image.new("RGB", (8, 8), (10, 11, 12)).save(d / "b.webp")
            (d / "ignore.txt").write_text("x", encoding="utf-8")
            found = list_input_images(d, "flux2_masonry_*.png")
            self.assertEqual([p.name for p in found], ["flux2_masonry_000.png"])
            all_imgs = {p.name for p in list_input_images(d, "")}
            self.assertEqual(all_imgs, {"flux2_masonry_000.png", "other.png", "a.jpg", "b.webp"})


if __name__ == "__main__":
    unittest.main()
