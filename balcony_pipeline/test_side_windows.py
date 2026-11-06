"""Side-wall windows from a depth ramp."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from side_windows import flag_side_windows  # noqa: E402


def cell(floor: int, bay: int) -> dict:
    return {"floor": floor, "bay_start": bay, "bay_end": bay, "box_xyxy": [0, 0, 1, 1]}


def win(uid: int, floor: int, bay: int, box: list[int]) -> dict:
    return {"unit_id": uid, "floor": floor, "bay": bay, "kind": "window", "box_xyxy": box}


class FlagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.depth = np.full((200, 200), 10.0)
        self.cells = [cell(0, 1), cell(0, 2), cell(1, 1), cell(1, 2)]
        self.wins = [
            win(0, 0, 1, [40, 20, 80, 60]),
            win(1, 0, 2, [100, 20, 140, 60]),
            win(2, 1, 1, [40, 100, 80, 140]),
            win(3, 1, 2, [100, 100, 140, 140]),
        ]

    def ramp(self, box: list[int], step: float) -> None:
        x0, y0, x1, y1 = box
        self.depth[y0:y1, x0:x1] = 10.0 + step * np.arange(x1 - x0)

    def test_outer_column_on_two_floors(self) -> None:
        self.ramp(self.wins[1]["box_xyxy"], 0.05)
        self.ramp(self.wins[3]["box_xyxy"], 0.05)
        self.assertEqual(sorted(flag_side_windows(self.wins, self.cells, self.depth)), [1, 3])
        self.assertTrue(self.wins[1]["side_face"])
        self.assertNotIn("side_face", self.wins[0])

    def test_one_floor_is_not_enough(self) -> None:
        self.ramp(self.wins[1]["box_xyxy"], 0.05)
        self.assertEqual(flag_side_windows(self.wins, self.cells, self.depth), [])

    def test_ramp_must_fall_away_outside_the_block(self) -> None:
        self.ramp(self.wins[1]["box_xyxy"], -0.05)
        self.ramp(self.wins[3]["box_xyxy"], -0.05)
        self.assertEqual(flag_side_windows(self.wins, self.cells, self.depth), [])

    def test_block_keeps_a_front_window(self) -> None:
        wins = [self.wins[1], self.wins[3]]
        self.ramp(wins[0]["box_xyxy"], 0.05)
        self.ramp(wins[1]["box_xyxy"], 0.05)
        self.assertEqual(flag_side_windows(wins, self.cells, self.depth), [])

    def test_window_outside_a_block_is_ignored(self) -> None:
        self.ramp(self.wins[1]["box_xyxy"], 0.05)
        self.ramp(self.wins[3]["box_xyxy"], 0.05)
        self.assertEqual(flag_side_windows(self.wins, [cell(0, 7)], self.depth), [])


if __name__ == "__main__":
    unittest.main()
