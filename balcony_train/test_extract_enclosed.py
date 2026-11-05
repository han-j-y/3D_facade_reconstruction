"""Tests for enclosed extraction (no GPU / DA3 weights required)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from balcony_train.extract_enclosed import (  # noqa: E402
    draw_stacked_rel_overlay,
    estimate_wall_depth,
    extract_enclosures,
    is_interference,
    is_rectangular_silhouette,
    largest_ones_rectangle,
    near_mask,
    rel_stack_color,
    silhouette_fill,
    storey_width_reason,
    vegetation_fraction,
)


class ExtractEnclosedTests(unittest.TestCase):
    def test_irregular_silhouette_is_not_rectangle(self) -> None:
        # L-shape: area is half the bounding box.
        self.assertFalse(is_rectangular_silhouette(50, 10, 10, min_fill=0.72))
        self.assertTrue(is_rectangular_silhouette(90, 10, 10, min_fill=0.72))
        self.assertGreaterEqual(silhouette_fill(90, 10, 10), 0.72)

    def test_tree_like_blob_dropped_rectangle_kept(self) -> None:
        h, w = 80, 80
        depth = np.full((h, w), 10.0, dtype=np.float32)
        # Rectangular enclosure, 5% in front of the wall.
        depth[20:50, 20:50] = 9.5
        # Hollow near blob (tree/car/person): not a filled rectangle.
        depth[52:56, 6:42] = 8.0
        depth[74:78, 6:42] = 8.0
        depth[52:78, 6:10] = 8.0
        depth[52:78, 38:42] = 8.0

        out = extract_enclosures(depth, min_rel=0.02, min_fill=0.72)
        self.assertIsNotNone(out["wall_median"])
        self.assertGreaterEqual(len(out["kept"]), 1)
        kept_boxes = [tuple(r["box_xyxy"]) for r in out["kept"]]
        self.assertTrue(any(b[0] <= 20 and b[2] >= 50 and b[1] <= 20 and b[3] >= 50 for b in kept_boxes))
        self.assertTrue(any(r["reason"] == "not_rectangle" for r in out["rejected"]))

    def test_flush_rectangle_not_enclosure(self) -> None:
        depth = np.full((60, 60), 10.0, dtype=np.float32)
        depth[15:40, 15:40] = 9.95  # 0.5% closer, below 2%
        out = extract_enclosures(depth, min_rel=0.02, min_fill=0.72)
        self.assertEqual(out["kept"], [])

    def test_near_mask_uses_two_percent(self) -> None:
        depth = np.array([[10.0, 9.7], [10.0, 9.85]], dtype=np.float32)
        mask = near_mask(depth, 10.0, min_rel=0.02)
        self.assertTrue(mask[0, 1])  # 3% closer
        self.assertFalse(mask[1, 1])  # 1.5% closer

    def test_gallery_recovered_after_dropping_tree_core(self) -> None:
        h, w = 120, 100
        depth = np.full((h, w), 10.0, dtype=np.float32)
        depth[28:52, 35:60] = 9.5  # gallery
        depth[100:118, 15:85] = 8.5  # nearer tree at the foot of the facade
        rgb = np.full((h, w, 3), 160, dtype=np.uint8)
        rgb[100:118, 15:85] = (40, 150, 45)
        out = extract_enclosures(depth, rgb=rgb, min_rel=0.02, min_fill=0.72)
        self.assertGreaterEqual(len(out["kept"]), 1)
        box = out["kept"][0]["box_xyxy"]
        self.assertLess(box[1], 90)

    def test_green_bottom_blob_is_interference(self) -> None:
        rgb = np.zeros((40, 40, 3), dtype=np.uint8)
        rgb[:, :] = (180, 180, 180)
        rgb[34:40, 5:25] = (30, 160, 40)
        box = (5, 34, 25, 40)
        self.assertGreater(vegetation_fraction(rgb, box), 0.4)
        self.assertEqual(
            is_interference(box, height=40, rgb=rgb, rel_delta=0.12),
            "bottom_clutter",
        )
        self.assertEqual(
            is_interference((10, 8, 30, 24), height=40, rgb=rgb, rel_delta=0.5),
            "too_near",
        )

    def test_largest_ones_rectangle_inside_blob(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:12, 3:15] = True
        mask[12:18, 3:7] = True  # extra irregular tail
        box = largest_ones_rectangle(mask)
        self.assertEqual(box, (3, 2, 15, 12))

    def test_stacked_overlay_draws_each_threshold(self) -> None:
        photo = Image.new("RGB", (80, 80), (10, 10, 10))
        stacked = draw_stacked_rel_overlay(
            photo,
            [
                (0.01, [{"box_xyxy": [4, 4, 20, 30]}]),
                (0.02, [{"box_xyxy": [10, 10, 40, 50]}]),
            ],
        )
        self.assertEqual(stacked.size, (80, 80))
        self.assertEqual(rel_stack_color(0.01), (255, 60, 60))
        self.assertEqual(rel_stack_color(0.02), (255, 160, 0))
        pix = stacked.getpixel((4, 4))
        self.assertEqual(pix, (255, 60, 60))

    def test_storey_width_gate(self) -> None:
        iw, ih = 400, 1000
        self.assertEqual(
            storey_width_reason([10, 10, 50, 80], iw=iw, ih=ih),
            "too_short",
        )
        self.assertEqual(
            storey_width_reason([0, 100, 320, 250], iw=iw, ih=ih),
            "too_wide",
        )
        self.assertIsNone(
            storey_width_reason([40, 100, 200, 250], iw=iw, ih=ih)
        )

    def test_extract_drops_short_or_wide_front_rects(self) -> None:
        h, w = 100, 100
        depth = np.full((h, w), 10.0, dtype=np.float32)
        depth[20:26, 20:50] = 9.5  # 3% closer but only 6% tall
        out = extract_enclosures(depth, min_rel=0.02, min_height_frac=0.10)
        self.assertTrue(any(r["reason"] == "too_short" for r in out["rejected"]))
        self.assertEqual(out["kept"], [])
        depth = np.full((h, w), 10.0, dtype=np.float32)
        depth[10:40, 2:90] = 9.5  # 3% closer, 88% wide
        out = extract_enclosures(depth, min_rel=0.02, max_width_frac=0.75)
        self.assertTrue(any(r["reason"] == "too_wide" for r in out["rejected"]))
        self.assertEqual(out["kept"], [])

    def test_wall_ignores_nearest_clutter(self) -> None:
        depth = np.full((40, 40), 10.0, dtype=np.float32)
        depth[30:40, :] = 2.0  # ground tree strip
        wall = estimate_wall_depth(depth, bottom_frac=0.2, trim=0.15)
        self.assertIsNotNone(wall)
        self.assertGreater(wall, 8.0)


if __name__ == "__main__":
    unittest.main()
