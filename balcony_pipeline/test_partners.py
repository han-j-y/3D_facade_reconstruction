"""Partner window pairing for balcony horizontal center."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from partners import (  # noqa: E402
    mean_window_center_norm,
    partner_windows_for_unit,
)


class PartnerWindowTests(unittest.TestCase):
    def test_overlap_pair_center_both_windows(self) -> None:
        """Wide slab under two windows: both partners, mean center."""
        instances = [
            {"unit_id": 1, "floor": 2, "bay": 1, "box_xyxy": [100, 40, 180, 120]},
            {"unit_id": 2, "floor": 2, "bay": 2, "box_xyxy": [190, 40, 270, 120]},
            {"unit_id": 3, "floor": 1, "bay": 1, "box_xyxy": [100, 200, 180, 280]},
        ]
        unit = {"floor": 2, "box_xyxy": [95, 115, 275, 175]}
        partners = partner_windows_for_unit(
            unit,
            instances,
            center_bays=[1],
            span_bays=[1, 2],
            image_size=(400, 400),
        )
        self.assertEqual({p["unit_id"] for p in partners}, {1, 2})
        cx = mean_window_center_norm(partners, 400.0)
        self.assertAlmostEqual(cx, (140 + 230) / 2 / 400)

    def test_overlap_single_window(self) -> None:
        instances = [
            {"unit_id": 5, "floor": 0, "bay": 0, "box_xyxy": [40, 50, 90, 130]},
        ]
        unit = {"floor": 0, "box_xyxy": [35, 125, 95, 185]}
        partners = partner_windows_for_unit(
            unit,
            instances,
            center_bays=[0],
            span_bays=[0],
            image_size=(400, 400),
        )
        self.assertEqual(len(partners), 1)
        self.assertEqual(partners[0]["unit_id"], 5)

    def test_bay_span_above_when_no_overlap(self) -> None:
        instances = [
            {"unit_id": 1, "floor": 1, "bay": 0, "box_xyxy": [0, 40, 100, 100]},
        ]
        unit = {"floor": 1, "box_xyxy": [0, 130, 100, 190]}
        partners = partner_windows_for_unit(
            unit,
            instances,
            center_bays=[0],
            span_bays=[0],
            image_size=(400, 400),
        )
        self.assertEqual(len(partners), 1)
        self.assertEqual(partners[0]["unit_id"], 1)

    def test_overlap_beats_wrong_floor_same_bay(self) -> None:
        instances = [
            {"unit_id": 9, "floor": 5, "bay": 9, "box_xyxy": [50, 40, 90, 110]},
            {"unit_id": 1, "floor": 1, "bay": 0, "box_xyxy": [200, 40, 260, 110]},
        ]
        unit = {"floor": 1, "box_xyxy": [48, 105, 92, 165]}
        partners = partner_windows_for_unit(
            unit,
            instances,
            center_bays=[0],
            span_bays=[0],
            image_size=(400, 400),
        )
        self.assertEqual(len(partners), 1)
        self.assertEqual(partners[0]["unit_id"], 9)

    def test_supporting_row_drops_upper_floor(self) -> None:
        instances = [
            {"unit_id": 1, "floor": 2, "bay": 1, "box_xyxy": [100, 80, 180, 140]},
            {"unit_id": 2, "floor": 3, "bay": 1, "box_xyxy": [100, 10, 180, 60]},
        ]
        unit = {"floor": 2, "box_xyxy": [95, 135, 185, 175]}
        partners = partner_windows_for_unit(
            unit,
            instances,
            center_bays=[1],
            span_bays=[1],
            image_size=(400, 400),
        )
        self.assertEqual(len(partners), 1)
        self.assertEqual(partners[0]["unit_id"], 1)

    def test_mean_center(self) -> None:
        partners = [
            {"box_xyxy": [100, 0, 200, 50]},
            {"box_xyxy": [300, 0, 500, 50]},
        ]
        cx = mean_window_center_norm(partners, 1000.0)
        self.assertAlmostEqual(cx, (150 + 400) / 2 / 1000)


if __name__ == "__main__":
    unittest.main()
