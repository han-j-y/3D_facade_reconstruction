"""GPU-free tests for the Stage 6 left-right repair."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.enclosed_symmetry import (  # noqa: E402
    grid_axis_x,
    symmetry_repair,
    window_boxes_from_dsl,
    window_symmetry,
)

IW, IH = 400, 400
BAY_X = {0: (0, 100), 1: (100, 200), 2: (200, 300), 3: (300, 400)}


def _rel_in(fronts: list[list[int]]):
    """Depth stub: 5% in front inside any ``fronts`` box, flat wall elsewhere."""

    def rel(box: list[int]) -> float:
        for f in fronts:
            if box[0] >= f[0] - 2 and box[2] <= f[2] + 2 and box[1] >= f[1] - 2 and box[3] <= f[3] + 2:
                return 0.05
        return 0.0

    return rel


def _run(recs, fronts, *, window_sym=0.9):
    return symmetry_repair(
        [{"box_xyxy": b} for b in recs],
        cx=grid_axis_x(BAY_X),
        window_sym=window_sym,
        iw=IW,
        ih=IH,
        rel_fn=_rel_in(fronts),
        on_grid=lambda b: True,
        min_rel=0.015,
    )


def _boxes(out):
    return sorted(r["box_xyxy"] for r in out)


class EnclosedSymmetryTests(unittest.TestCase):
    def test_axis_is_middle_of_bays_and_windows_score(self) -> None:
        cx = grid_axis_x(BAY_X)
        self.assertEqual(cx, 200.0)
        windows = [[20, 20, 80, 80], [320, 20, 380, 80]]
        self.assertAlmostEqual(window_symmetry(windows, cx), 1.0)
        self.assertLess(window_symmetry([[20, 20, 80, 80]], cx), 0.6)

    def test_low_window_symmetry_changes_nothing(self) -> None:
        out, log = _run([[10, 100, 90, 300]], [], window_sym=0.4)
        self.assertEqual(_boxes(out), [[10, 100, 90, 300]])
        self.assertIn("skip", log)

    def test_missing_mirror_is_added_when_in_front(self) -> None:
        out, log = _run([[10, 100, 90, 300]], [[10, 100, 90, 300], [310, 100, 390, 300]])
        self.assertEqual(_boxes(out), [[10, 100, 90, 300], [310, 100, 390, 300]])
        self.assertEqual(log["added"], 1)

    def test_isolated_box_is_dropped_when_mirror_is_flat(self) -> None:
        out, log = _run([[10, 100, 90, 300]], [[10, 100, 90, 300]])
        self.assertEqual(out, [])
        self.assertEqual(log["dropped"], 1)

    def test_box_on_axis_is_kept(self) -> None:
        out, log = _run([[160, 100, 240, 300]], [])
        self.assertEqual(_boxes(out), [[160, 100, 240, 300]])
        self.assertEqual(log["dropped"], 0)

    def test_pair_vertical_range_aligns_only_where_in_front(self) -> None:
        left, right = [10, 100, 90, 300], [310, 50, 390, 300]
        out, log = _run([left, right], [[10, 50, 90, 300], right])
        self.assertEqual(_boxes(out), [[10, 50, 90, 300], right])
        self.assertEqual(log["aligned"], 1)
        out_flat, log_flat = _run([left, right], [left, right])
        self.assertEqual(_boxes(out_flat), [left, right])
        self.assertEqual(log_flat["aligned"], 0)

    def test_mirror_outside_photo_keeps_box(self) -> None:
        out = symmetry_repair(
            [{"box_xyxy": [10, 100, 90, 300]}],
            cx=300.0,
            window_sym=0.9,
            iw=IW,
            ih=IH,
            rel_fn=lambda b: 0.0,
            on_grid=lambda b: True,
            min_rel=0.015,
        )[0]
        self.assertEqual(_boxes(out), [[10, 100, 90, 300]])

    def test_window_boxes_skip_doors(self) -> None:
        dsl = {
            "instances": [
                {"kind": "window", "box_xyxy": [1, 2, 3, 4]},
                {"kind": "door", "box_xyxy": [5, 6, 7, 8]},
                {"box_xyxy": [9, 10, 11, 12]},
            ]
        }
        self.assertEqual(window_boxes_from_dsl(dsl), [[1, 2, 3, 4], [9, 10, 11, 12]])


if __name__ == "__main__":
    unittest.main()
