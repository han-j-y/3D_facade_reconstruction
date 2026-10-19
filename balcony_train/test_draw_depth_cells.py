"""Tests for depth-board drawing helpers (no GPU)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.draw_depth_cells import grid_depth_mask  # noqa: E402


class DrawDepthCellsTests(unittest.TestCase):
    def test_grid_mask_covers_cells_only(self) -> None:
        mask = grid_depth_mask(10, 12, {0: (2, 8)}, {0: (1, 5), 1: (5, 9)})
        self.assertFalse(bool(mask[0, 0]))
        self.assertTrue(bool(mask[3, 2]))
        self.assertTrue(bool(mask[3, 6]))
        self.assertFalse(bool(mask[3, 10]))
        self.assertEqual(int(mask.sum()), 6 * 8)


if __name__ == "__main__":
    unittest.main()
