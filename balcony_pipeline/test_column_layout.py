"""Structural window bay columns from densest floor (same as main)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from facade_recovery.column_layout import (  # noqa: E402
    assign_floors,
    infer_window_bay_columns,
)
import numpy as np


class ColumnLayoutTests(unittest.TestCase):
    def test_densest_floor_sets_bay_count(self) -> None:
        units = [
            {"floor": 1, "box_xyxy": [10, 10, 40, 40], "kind": "window"},
            {"floor": 1, "box_xyxy": [50, 10, 80, 40], "kind": "window"},
            {"floor": 1, "box_xyxy": [90, 10, 120, 40], "kind": "window"},
            {"floor": 0, "box_xyxy": [10, 60, 80, 90], "kind": "window"},
        ]
        cols = infer_window_bay_columns(units, iw=130)
        self.assertEqual(len(cols), 3)
        self.assertEqual(units[0]["bay"], 0)
        self.assertEqual(units[1]["bay"], 1)
        self.assertEqual(units[2]["bay"], 2)
        self.assertEqual(units[0]["colspan"], 1)
        self.assertGreaterEqual(int(units[3]["colspan"]), 1)

    def test_assign_floors_splits_by_row_tol(self) -> None:
        cy = np.array([0.10, 0.12, 0.50, 0.52], dtype=np.float64)
        floors = assign_floors(cy, 0.055)
        self.assertEqual(int(floors[0]), int(floors[1]))
        self.assertEqual(int(floors[2]), int(floors[3]))
        self.assertNotEqual(int(floors[0]), int(floors[2]))


if __name__ == "__main__":
    unittest.main()
