"""Tests for DA3 enclosure helpers (no GPU / DA3 weights required)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.try_da3_enclosure import (  # noqa: E402
    ENCLOSURE_ROIS,
    box_from_norm,
    colorize_depth,
    is_closer,
    resize_depth,
    roi_median,
    score_image,
)


class TryDa3EnclosureTests(unittest.TestCase):
    def test_rois_cover_five_test_facades(self) -> None:
        stems = {"cmp_b0007", "cmp_b0008", "cmp_b0010", "cmp_b0223", "cmp_b0307"}
        self.assertEqual(set(ENCLOSURE_ROIS), stems)
        for stem, rois in ENCLOSURE_ROIS.items():
            kinds = {r["kind"] for r in rois}
            self.assertIn("enclosed", kinds, stem)
            self.assertIn("wall", kinds, stem)

    def test_box_from_norm_and_median(self) -> None:
        depth = np.ones((100, 200), dtype=np.float32) * 10.0
        depth[10:30, 20:60] = 4.0
        box = box_from_norm((0.10, 0.10, 0.30, 0.30), 200, 100)
        self.assertEqual(box, (20, 10, 60, 30))
        self.assertAlmostEqual(roi_median(depth, box), 4.0)

    def test_enclosed_closer_than_wall(self) -> None:
        self.assertTrue(is_closer(4.0, 10.0))
        self.assertFalse(is_closer(10.0, 4.0))
        self.assertFalse(is_closer(9.9, 10.0, min_rel=0.05))

    def test_score_image_flags_enclosed_when_shallower(self) -> None:
        h, w = 100, 100
        depth = np.full((h, w), 8.0, dtype=np.float32)
        depth[10:40, 10:40] = 5.0
        rois = (
            {"name": "bay", "kind": "enclosed", "box": (0.10, 0.10, 0.40, 0.40)},
            {"name": "wall", "kind": "wall", "box": (0.60, 0.60, 0.90, 0.90)},
        )
        scored = score_image(depth, width=w, height=h, rois=rois)
        self.assertTrue(scored["helps_enclosed"])
        self.assertEqual(scored["enclosed_closer"], ["bay"])

    def test_colorize_and_resize_shapes(self) -> None:
        small = np.linspace(1.0, 5.0, 12, dtype=np.float32).reshape(3, 4)
        rgb = colorize_depth(small)
        self.assertEqual(rgb.shape, (3, 4, 3))
        big = resize_depth(small, 8, 6)
        self.assertEqual(big.shape, (6, 8))


if __name__ == "__main__":
    unittest.main()
