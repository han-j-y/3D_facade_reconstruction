"""Snap balcony boxes to window floor×bay layout."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from snap import (  # noqa: E402
    _bands_from_window_instances,
    center_bay_ids,
    snap_box,
    snap_units,
)


def _windows_dsl() -> dict:
    return {
        "instances": [
            {"floor": 1, "bay": 0, "box_xyxy": [0, 50, 100, 150]},
            {"floor": 1, "bay": 1, "box_xyxy": [100, 50, 220, 150]},
            {"floor": 1, "bay": 2, "box_xyxy": [220, 50, 400, 150]},
        ]
    }


class SnapTests(unittest.TestCase):
    def test_multi_bay_span_lists_overlapping_bays(self) -> None:
        floor_y, bay_x = _bands_from_window_instances(_windows_dsl()["instances"])
        loc = snap_box([10, 60, 390, 140], floor_y, bay_x)
        self.assertEqual(loc["bay_start"], 0)
        self.assertEqual(loc["bay_end"], 2)
        self.assertEqual(loc["bays"], [0, 1, 2])

    def test_full_cover_bays_all_count_for_center(self) -> None:
        bay_x = {0: (0, 100), 1: (100, 200), 2: (200, 300)}
        # Balcony fully covers bay 0 and bay 1; partial on bay 2
        ids = center_bay_ids([0, 0, 250, 50], bay_x)
        self.assertEqual(ids, [0, 1])

    def test_partial_overlap_uses_max_overlap_bay(self) -> None:
        bay_x = {0: (0, 100), 1: (100, 200)}
        # ~5 px on bay 0, ~95 px on bay 1
        ids = center_bay_ids([95, 0, 195, 50], bay_x)
        self.assertEqual(ids, [1])

    def test_one_full_one_partial_uses_full_only(self) -> None:
        bay_x = {0: (0, 100), 1: (100, 200)}
        ids = center_bay_ids([50, 0, 200, 50], bay_x)
        self.assertEqual(ids, [1])

    def test_snap_units_propagates_bays_center(self) -> None:
        out = snap_units([[10, 60, 390, 140]], _windows_dsl())
        self.assertEqual(out[0]["bays"], [0, 1, 2])
        self.assertIn("bays_center", out[0])

    def test_layout_columns_xy_preferred_over_instance_boxes(self) -> None:
        dsl = {
            "meta": {
                "columns_xy": [[0, 50], [50, 150], [150, 400]],
                "floors_y": [[40, 160]],
            },
            "layout": {
                "floors": [{"id": 1, "h_norm": 0.3}],
                "bays": [
                    {"id": 0, "w_norm": 0.12},
                    {"id": 1, "w_norm": 0.25},
                    {"id": 2, "w_norm": 0.63},
                ],
            },
            "instances": _windows_dsl()["instances"],
        }
        # Instance bay 0 is 0–100 (would be a single bay). Layout splits at x=50.
        out = snap_units([[10, 60, 90, 140]], dsl)
        self.assertEqual(out[0]["bays"], [0, 1])
        self.assertEqual(out[0]["floor"], 1)


if __name__ == "__main__":
    unittest.main()
