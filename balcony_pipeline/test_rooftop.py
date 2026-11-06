"""Rooftop label: only above the top window row."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_balcony import mark_rooftop_units  # noqa: E402


def dsl() -> dict:
    return {
        "meta": {"columns_xy": [[0, 100], [100, 200]], "floors_y": [[100, 200], [200, 300]]},
        "layout": {"floors": [{"id": 0}, {"id": 1}], "bays": [{"id": 0}, {"id": 1}]},
        "instances": [
            {"floor": 0, "bay": 0, "kind": "window", "box_xyxy": [20, 120, 80, 180]},
            {"floor": 1, "bay": 0, "kind": "window", "box_xyxy": [20, 220, 80, 280]},
            {"floor": 1, "bay": 1, "kind": "window", "box_xyxy": [120, 220, 180, 280]},
        ],
    }


class RooftopTests(unittest.TestCase):
    def mark(self, box: list[int], floor: int = 0) -> bool:
        unit = {"floor": floor, "box_xyxy": box}
        mark_rooftop_units([unit], dsl(), image_height=400)
        return unit["rooftop"]

    def test_above_the_top_row_is_rooftop(self) -> None:
        self.assertTrue(self.mark([110, 50, 190, 100]))

    def test_level_with_the_top_row_in_an_empty_bay_is_a_normal_balcony(self) -> None:
        self.assertFalse(self.mark([110, 140, 190, 190]))

    def test_lower_floor_is_never_rooftop(self) -> None:
        self.assertFalse(self.mark([110, 50, 190, 100], floor=1))


if __name__ == "__main__":
    unittest.main()
