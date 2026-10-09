"""Stacked enclosures share floors; windows move onto enclosed fronts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from balcony_layout import (  # noqa: E402
    balcony_frames,
    column_points,
    front_edge_x,
    front_window_targets,
)
from facade_spec import get_cell  # noqa: E402


def _ir(enclosure: str, shape: str = "rectangle", width: float = 2.0) -> dict:
    return {
        "type": "balcony",
        "structure": "projecting",
        "enclosure": enclosure,
        "floor": {"shape": shape, "params": {"width": width, "depth": 1.0}},
        "railing": {"kind": "open_work", "height": 1.1},
        "output": {"slab_thickness": 0.2},
    }


def _spec(placements: list[dict], library: dict) -> dict:
    return {
        "type": "facade",
        "wall": {"depth": 0.42, "base_front_y": 0.0},
        "grid": {
            "rows": [{"name": f"F{i}", "h": 3.0} for i in range(4)],
            "cols": [{"name": f"B{j}", "w": 2.6} for j in range(3)],
        },
        "balconies": library,
        "balcony_placement": placements,
        "placement_params": {"bottom_margin_ratio": 0.14, "mirror_x": True},
    }


def _at(row: int, col: int, kind: str) -> dict:
    return {"row": row, "col0": col, "col1": col, "type": kind}


class StackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lib = {
            "enc": _ir("enclosed"),
            "half": _ir("half_enclosed"),
            "open": _ir("open", width=2.4),
            "circ": _ir("enclosed", shape="circle", width=2.4),
        }

    def test_column_of_enclosures_shares_floors(self) -> None:
        spec = _spec([_at(3, 1, "enc"), _at(2, 1, "enc"), _at(1, 1, "open")], self.lib)
        bottom, middle, top = balcony_frames(spec)
        self.assertEqual(bottom["roof"], "shared")
        self.assertEqual(bottom["above"], 1)
        self.assertAlmostEqual(bottom["top_z"], middle["sill_z"])
        self.assertEqual(middle["roof"], "shared")
        self.assertEqual(middle["above"], 2)
        self.assertAlmostEqual(middle["top_z"], top["sill_z"])
        self.assertIsNone(top["roof"])
        self.assertTrue(top["aligned"])
        self.assertAlmostEqual(top["bx0"], bottom["bx0"])
        self.assertAlmostEqual(top["bx1"], bottom["bx1"])

    def test_top_of_stack_keeps_own_roof(self) -> None:
        spec = _spec([_at(3, 1, "half"), _at(2, 1, "half")], self.lib)
        lower, upper = balcony_frames(spec)
        self.assertEqual(lower["roof"], "shared")
        self.assertEqual(upper["roof"], "own")
        # Roof top lands on the next storey's sill line.
        next_sill = upper["z1"] + (upper["sill_z"] - upper["z0"])
        self.assertAlmostEqual(upper["top_z"] + upper["thick"], next_sill)

    def test_open_balcony_does_not_stack(self) -> None:
        spec = _spec([_at(3, 1, "open"), _at(2, 1, "open")], self.lib)
        lower, upper = balcony_frames(spec)
        self.assertIsNone(lower["roof"])
        self.assertFalse(upper["aligned"])

    def test_neighbor_bay_is_not_above(self) -> None:
        spec = _spec([_at(3, 1, "enc"), _at(2, 0, "open")], self.lib)
        lower, other = balcony_frames(spec)
        self.assertEqual(lower["roof"], "own")
        self.assertFalse(other["aligned"])

    def test_upper_takes_lower_plan_shape(self) -> None:
        spec = _spec([_at(3, 1, "circ"), _at(2, 1, "open")], self.lib)
        lower, upper = balcony_frames(spec)
        self.assertEqual(upper["shape"], "circle")
        self.assertEqual(upper["outline"], lower["outline"])


class FrontWindowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lib = {
            "enc": _ir("enclosed"),
            "half": _ir("half_enclosed"),
            "tri": _ir("enclosed", shape="triangle"),
        }

    def _window(self, spec: dict, row: int, col: int, w: float = 1.2, h: float = 1.4) -> dict:
        cell = get_cell(spec, row, col, mirror_x=True)
        cx = 0.5 * (cell["x0"] + cell["x1"])
        cz = cell["z0"] + 0.55 * (cell["z1"] - cell["z0"])
        return {"r": row, "c": col, "span": 1, "ox": cx - w / 2, "oz": cz - h / 2, "ww": w, "hh": h}

    def test_window_in_enclosed_cell_moves_to_front(self) -> None:
        spec = _spec([_at(2, 1, "enc")], self.lib)
        frames = balcony_frames(spec)
        wins = [self._window(spec, 2, 1), self._window(spec, 2, 0)]
        targets = front_window_targets(frames, wins)
        self.assertEqual(list(targets), [0])
        fi, (ox, oz, ww, hh) = targets[0]
        self.assertEqual(fi, 0)
        fx0, fx1 = front_edge_x(frames[0])
        self.assertGreaterEqual(ox, fx0)
        self.assertLessEqual(ox + ww, fx1)
        self.assertGreaterEqual(oz, frames[0]["sill_z"] + frames[0]["thick"])
        self.assertLessEqual(oz + hh, frames[0]["top_z"])

    def test_wide_window_is_clamped_to_front(self) -> None:
        spec = _spec([_at(2, 1, "enc")], self.lib)
        frames = balcony_frames(spec)
        targets = front_window_targets(frames, [self._window(spec, 2, 1, w=2.5)])
        _, (ox, _, ww, _) = targets[0]
        fx0, fx1 = front_edge_x(frames[0])
        self.assertAlmostEqual(ox, fx0 + 0.08)
        self.assertAlmostEqual(ox + ww, fx1 - 0.08)

    def test_half_enclosed_keeps_wall_window(self) -> None:
        spec = _spec([_at(2, 1, "half")], self.lib)
        self.assertEqual(front_window_targets(balcony_frames(spec), [self._window(spec, 2, 1)]), {})

    def test_no_flat_front_keeps_wall_window(self) -> None:
        spec = _spec([_at(2, 1, "tri")], self.lib)
        frames = balcony_frames(spec)
        self.assertIsNone(front_edge_x(frames[0]))
        self.assertEqual(front_window_targets(frames, [self._window(spec, 2, 1)]), {})

    def test_narrow_front_keeps_wall_window(self) -> None:
        lib = {"slim": _ir("enclosed", width=0.8)}
        spec = _spec([_at(2, 1, "slim")], lib)
        frames = balcony_frames(spec)
        fx0, fx1 = front_edge_x(frames[0])
        self.assertLess(fx1 - fx0, 1.0)
        self.assertEqual(front_window_targets(frames, [self._window(spec, 2, 1)]), {})

    def test_other_floor_window_stays(self) -> None:
        spec = _spec([_at(2, 1, "enc")], self.lib)
        targets = front_window_targets(balcony_frames(spec), [self._window(spec, 1, 1)])
        self.assertEqual(targets, {})


class ColumnTests(unittest.TestCase):
    def test_default_columns_follow_bends(self) -> None:
        lib = {"r": _ir("half_enclosed"), "c": _ir("half_enclosed", shape="circle", width=2.4)}
        spec = _spec([_at(2, 0, "r"), _at(2, 2, "c")], lib)
        rect, circ = balcony_frames(spec)
        self.assertEqual(len(column_points(rect)), 2)
        self.assertEqual(len(column_points(circ)), 3)
        self.assertEqual(len(column_points(rect, count=3)), 3)
        for x, y in column_points(rect):
            self.assertGreater(y, rect["y_wall"])
            self.assertTrue(rect["bx0"] < x < rect["bx1"])


if __name__ == "__main__":
    unittest.main()
