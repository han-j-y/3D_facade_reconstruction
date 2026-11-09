"""Stacked enclosures share floors; windows move onto enclosed fronts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from facade_spec import wall_margins_from_region, wall_world_extent
from balcony_layout import (
    center_on_windows,
    _outline_type_widths,
    _fit_spans,
    side_wall_targets,  # noqa: E402
    balcony_frames,
    enclosed_window_fits,
    column_points,
    front_edge_x,
    front_window_targets,
    photo_x_span,
    seat_on_window_bottoms,
    windows_behind_glass,
)
from facade_spec import get_cell, juliet_window_overlap  # noqa: E402


def _ir(
    enclosure: str,
    shape: str = "rectangle",
    width: float = 2.0,
    structure: str = "projecting",
) -> dict:
    return {
        "type": "balcony",
        "structure": structure,
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

    def test_slab_over_a_block_of_cells_takes_the_whole_block(self) -> None:
        # An oriel cut into three bay cells under one open balcony that spans all three bays.
        spec = _spec([_at(2, 0, "enc"), _at(2, 1, "enc"), _at(2, 2, "enc"), {"row": 1, "col0": 0, "col1": 2, "type": "wide"}],
                     self.lib | {"wide": _ir("open", width=7.0)})
        *cells, top = balcony_frames(spec)
        self.assertAlmostEqual(top["bx0"], min(c["bx0"] for c in cells))
        self.assertAlmostEqual(top["bx1"], max(c["bx1"] for c in cells))
        self.assertTrue(all(c["roof"] == "shared" for c in cells))

    def test_open_balcony_on_an_oriel_takes_the_oriel_width(self) -> None:
        # val30 Copenhagen way288954202: the open balcony sits over one cell of a two-cell oriel.
        # Cells fill their bays (one oriel); the balcony is one bay wide.
        spec = _spec([_at(2, 1, "full"), _at(2, 2, "full"), _at(1, 2, "open")], self.lib | {"full": _ir("enclosed", width=2.6)})
        left, right, top = balcony_frames(spec)
        self.assertAlmostEqual(top["bx0"], min(left["bx0"], right["bx0"]))
        self.assertAlmostEqual(top["bx1"], max(left["bx1"], right["bx1"]))
        self.assertEqual((left["roof"], right["roof"]), ("shared", "shared"))
        self.assertAlmostEqual(left["top_z"], top["sill_z"])

    def test_long_slab_over_a_block_keeps_its_length(self) -> None:
        spec = _spec([_at(2, 1, "enc"), {"row": 1, "col0": 0, "col1": 2, "type": "wide"}], self.lib | {"wide": _ir("open", width=7.0)})
        cell, top = balcony_frames(spec)
        self.assertLess(top["bx0"], cell["bx0"] - 0.5)  # reaches well past the block: the photo span stays
        self.assertGreater(top["bx1"], cell["bx1"] + 0.5)

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

    def test_juliet_has_no_slab_to_share(self) -> None:
        self.lib["jul"] = _ir("open", structure="juliet")
        spec = _spec([_at(3, 1, "enc"), _at(2, 1, "jul")], self.lib)
        lower, upper = balcony_frames(spec)
        self.assertIsNone(upper["outline"])
        self.assertEqual(lower["roof"], "own")
        self.assertFalse(upper["carries_stack"])

    def test_rooftop_on_a_stack_carries_its_roof(self) -> None:
        def roof_at(row: int, col: int) -> dict:
            return {**_at(row, col, "open"), "structure": "rooftop"}

        spec = _spec([_at(3, 1, "enc"), roof_at(2, 1), roof_at(1, 0)], self.lib)
        lower, on_stack, alone = balcony_frames(spec)
        self.assertEqual(lower["roof"], "shared")
        self.assertEqual(on_stack["structure"], "rooftop")
        self.assertTrue(on_stack["carries_stack"])
        self.assertFalse(alone["carries_stack"])
        self.assertEqual(self.lib["open"]["structure"], "projecting")

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

    def test_curtain_wall_drops_the_overlapping_window(self) -> None:
        lib = {
            "glass": {
                **_ir("enclosed"),
                "walls": {"infill": "glass", "curtain_wall": True},
            }
        }
        spec = _spec([_at(2, 1, "glass")], lib)
        frames = balcony_frames(spec)
        self.assertTrue(frames[0]["curtain_wall"])
        wins = [self._window(spec, 2, 1), self._window(spec, 2, 0)]
        self.assertEqual(front_window_targets(frames, wins), {})
        self.assertEqual(windows_behind_glass(frames, wins), {0})

    def test_punched_frame_is_not_curtain_wall(self) -> None:
        spec = _spec([_at(2, 1, "enc")], self.lib)
        frames = balcony_frames(spec)
        self.assertFalse(frames[0]["curtain_wall"])
        self.assertEqual(windows_behind_glass(frames, [self._window(spec, 2, 1)]), set())

    def test_other_floor_window_stays(self) -> None:
        spec = _spec([_at(2, 1, "enc")], self.lib)
        targets = front_window_targets(balcony_frames(spec), [self._window(spec, 1, 1)])
        self.assertEqual(targets, {})

    def test_fused_circle_drops_windows_on_every_floor(self) -> None:
        lib = {"c": _curtain_circle()}
        spec = _spec([_at(3, 1, "c"), _at(2, 1, "c")], lib)
        frames = balcony_frames(spec)
        wins = [
            self._window(spec, 3, 1),
            self._window(spec, 2, 1),
            self._window(spec, 2, 0),
        ]
        self.assertEqual(windows_behind_glass(frames, wins), {0, 1})


def _curtain_circle() -> dict:
    ir = _ir("enclosed", shape="circle", width=2.4)
    ir["walls"] = {"infill": "glass", "curtain_wall": True}
    return ir


class CurtainCircleTests(unittest.TestCase):
    def test_single_cell_is_a_semicircle(self) -> None:
        spec = _spec([_at(2, 1, "c")], {"c": _curtain_circle()})
        frame = balcony_frames(spec)[0]
        width = frame["bx1"] - frame["bx0"]
        self.assertTrue(frame["curtain_drum"])
        self.assertFalse(frame.get("skip_body"))
        self.assertAlmostEqual(frame["depth"], width / 2.0)
        cx = 0.5 * (frame["bx0"] + frame["bx1"])
        for x, y in frame["outline"]:
            self.assertAlmostEqual((x - cx) ** 2 + y ** 2, frame["depth"] ** 2, places=3)
        self.assertGreater(len(frame["outline"]), 2)

    def test_stack_is_one_drum(self) -> None:
        spec = _spec(
            [_at(3, 1, "c"), _at(2, 1, "c"), _at(1, 1, "c")],
            {"c": _curtain_circle()},
        )
        bottom, mid, top = balcony_frames(spec)
        self.assertTrue(bottom["curtain_drum"])
        self.assertFalse(bottom.get("skip_body"))
        self.assertTrue(mid["skip_body"])
        self.assertTrue(top["skip_body"])
        self.assertEqual(bottom["roof"], "own")
        self.assertAlmostEqual(bottom["top_z"], top["top_z"])
        self.assertEqual(bottom["row_lo"], 1)
        self.assertEqual(bottom["row_hi"], 3)
        self.assertEqual(
            bottom["floor_sills"],
            sorted(fr["sill_z"] for fr in (bottom, mid, top)),
        )
        self.assertLess(bottom["floor_sills"][0], bottom["floor_sills"][1])
        self.assertLess(bottom["floor_sills"][1], bottom["floor_sills"][2])
        self.assertAlmostEqual(bottom["depth"], (bottom["bx1"] - bottom["bx0"]) / 2.0)
        self.assertNotAlmostEqual(bottom["depth"], 1.5)

    def test_adjacent_bays_merge(self) -> None:
        spec = _spec([_at(2, 0, "c"), _at(2, 1, "c")], {"c": _curtain_circle()})
        a, b = balcony_frames(spec)
        host, other = (a, b) if a.get("curtain_drum") else (b, a)
        self.assertTrue(host["curtain_drum"])
        self.assertTrue(other["skip_body"])
        self.assertEqual(host["col0"], 0)
        self.assertEqual(host["col1"], 1)
        self.assertEqual(len(host["floor_sills"]), 1)
        self.assertGreater(host["bx1"] - host["bx0"], 4.0)
        self.assertAlmostEqual(host["depth"], (host["bx1"] - host["bx0"]) / 2.0)

    def test_diagonal_cells_stay_separate(self) -> None:
        spec = _spec([_at(3, 0, "c"), _at(2, 1, "c")], {"c": _curtain_circle()})
        a, b = balcony_frames(spec)
        self.assertTrue(a["curtain_drum"])
        self.assertTrue(b["curtain_drum"])
        self.assertFalse(a.get("skip_body"))
        self.assertFalse(b.get("skip_body"))

    def test_punched_circle_is_one_drum(self) -> None:
        spec = _spec(
            [_at(3, 0, "c"), _at(3, 1, "c"), _at(2, 0, "c")],
            {"c": _ir("enclosed", shape="circle", width=2.4)},
        )
        host, side, upper = balcony_frames(spec)
        self.assertTrue(host["drum"])
        self.assertFalse(host["curtain_drum"])
        self.assertFalse(host.get("skip_body"))
        self.assertTrue(side["skip_body"])
        self.assertTrue(upper["skip_body"])
        self.assertEqual((host["col0"], host["col1"]), (0, 1))
        self.assertEqual((host["row_lo"], host["row_hi"]), (2, 3))
        self.assertEqual(len(host["floor_sills"]), 2)
        self.assertAlmostEqual(host["depth"], (host["bx1"] - host["bx0"]) / 2.0)
        self.assertAlmostEqual(max(y for _, y in host["outline"]) - host["y_wall"], host["depth"], places=3)

    def test_narrow_middle_bay_does_not_split_a_punched_drum(self) -> None:
        spec = _spec(
            [_at(2, 0, "c"), _at(2, 1, "c"), _at(2, 2, "c")],
            {"c": _ir("enclosed", shape="circle", width=2.4)},
        )
        spec["grid"]["cols"][1]["w"] = 0.2
        outer = balcony_frames(spec)
        win = lambda col, f: {"r": 2, "c": col, "span": 1, "ox": 0.5 * (f["bx0"] + f["bx1"]) - 0.4, "ww": 0.8}
        fits = enclosed_window_fits(spec, [win(0, outer[0]), win(2, outer[0])])
        self.assertTrue(all(v is not None for v in fits.values()))
        frames = balcony_frames(spec, fits)
        self.assertEqual(sum(1 for f in frames if not f.get("skip_body")), 1)

    def test_punched_and_curtain_circles_stay_separate(self) -> None:
        spec = _spec(
            [_at(2, 0, "c"), _at(2, 1, "p")],
            {"c": _curtain_circle(), "p": _ir("enclosed", shape="circle", width=2.4)},
        )
        a, b = balcony_frames(spec)
        self.assertTrue(a["curtain_drum"])
        self.assertFalse(b["curtain_drum"])
        self.assertFalse(a.get("skip_body"))
        self.assertFalse(b.get("skip_body"))

    def test_rectangle_curtain_does_not_become_a_drum(self) -> None:
        lib = {
            "g": {**_ir("enclosed"), "walls": {"infill": "glass", "curtain_wall": True}},
        }
        spec = _spec([_at(3, 1, "g"), _at(2, 1, "g")], lib)
        lower, upper = balcony_frames(spec)
        self.assertFalse(lower.get("curtain_drum"))
        self.assertFalse(upper.get("skip_body"))
        self.assertEqual(lower["roof"], "shared")
        self.assertAlmostEqual(lower["depth"], 1.5)

    def test_open_balcony_above_is_the_lid(self) -> None:
        spec = _spec(
            [_at(3, 1, "c"), _at(2, 1, "c"), _at(1, 1, "o")],
            {"c": _curtain_circle(), "o": _ir("open")},
        )
        bottom, mid, top = balcony_frames(spec)
        self.assertTrue(bottom["curtain_drum"])
        self.assertTrue(mid["skip_body"])
        self.assertFalse(top.get("skip_body"))
        self.assertEqual(bottom["roof"], "shared")
        self.assertAlmostEqual(bottom["top_z"], top["sill_z"])


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


class CenterOnWindowsTest(unittest.TestCase):
    """Windows are the reference: an open balcony moves sideways onto its window's compiled centre."""

    def _frame(self, rec: dict, kind: str = "open") -> dict:
        spec = _spec([rec | {"type": "b"}], {"b": _ir(kind, width=2.0)})
        return balcony_frames(spec)[0]

    def test_moves_onto_its_window(self) -> None:
        f = self._frame({"row": 1, "col0": 1, "col1": 1, "window_unit_ids": [7]})
        w0 = f["bx1"] - f["bx0"]
        center_on_windows([f], [{"window_unit_ids": [7]}], [{"unit_id": 7, "ox": 0.6, "ww": 1.0}])
        self.assertAlmostEqual(0.5 * (f["bx0"] + f["bx1"]), 1.1)
        self.assertAlmostEqual(f["bx1"] - f["bx0"], w0)  # width stays
        self.assertAlmostEqual(min(x for x, _ in f["outline"]), f["bx0"])

    def test_two_windows_take_their_midpoint(self) -> None:
        f = self._frame({"row": 1, "col0": 0, "col1": 1})
        wins = [{"unit_id": 1, "ox": -2.0, "ww": 1.0}, {"unit_id": 2, "ox": 1.0, "ww": 1.0}]
        center_on_windows([f], [{"window_unit_ids": [1, 2]}], wins)
        self.assertAlmostEqual(0.5 * (f["bx0"] + f["bx1"]), 0.0)

    def test_gallery_under_several_windows_keeps_its_place(self) -> None:
        f = self._frame({"row": 1, "col0": 0, "col1": 2})
        cx = 0.5 * (f["bx0"] + f["bx1"])
        wins = [{"unit_id": 1, "r": 1, "ox": -3.0, "ww": 1.0}, {"unit_id": 2, "r": 1, "ox": 0.5, "ww": 1.0}]
        center_on_windows([f], [{"window_unit_ids": [1]}], wins)
        self.assertAlmostEqual(0.5 * (f["bx0"] + f["bx1"]), cx)

    def test_no_window_or_side_face_keeps_its_place(self) -> None:
        f = self._frame({"row": 1, "col0": 1, "col1": 1})
        cx = 0.5 * (f["bx0"] + f["bx1"])
        center_on_windows([f], [{"window_unit_ids": [3]}], [{"unit_id": 3, "ox": 2.0, "ww": 0.2, "side_face": True}])
        center_on_windows([f], [{}], [{"unit_id": 3, "ox": 2.0, "ww": 1.0}])
        self.assertAlmostEqual(0.5 * (f["bx0"] + f["bx1"]), cx)


class JulietOverlapTest(unittest.TestCase):
    """How much of a juliet box its window covers: own window first, else the same floor's windows."""

    BOX = {"box_xyxy": [0, 100, 100, 200], "floor": 2, "partner_window_unit_ids": [7]}

    def test_own_window(self) -> None:
        wins = [{"unit_id": 7, "floor": 2, "box_xyxy": [10, 20, 90, 160]}, {"unit_id": 8, "floor": 2, "box_xyxy": [200, 20, 290, 100]}]
        self.assertAlmostEqual(juliet_window_overlap(self.BOX, wins), 0.6)

    def test_same_floor_without_own_window(self) -> None:
        box = dict(self.BOX, partner_window_unit_ids=[])
        wins = [{"unit_id": 8, "floor": 2, "box_xyxy": [200, 120, 290, 200]}, {"unit_id": 9, "floor": 1, "box_xyxy": [0, 0, 9, 9]}]
        self.assertAlmostEqual(juliet_window_overlap(box, wins), 0.8)

    def test_no_window(self) -> None:
        self.assertEqual(juliet_window_overlap(dict(self.BOX, partner_window_unit_ids=[]), []), 0.0)


class OutlineTypeWidthTest(unittest.TestCase):
    """Open round balconies share one width per type: the median photo width, at the facade's scale."""

    def _frames(self, shape: str, enclosure: str = "open") -> list:
        recs = [{"row": r, "col0": 1, "col1": 1, "type": "t", "width_norm": w, "window_cx_norm": 0.5}
                for r, w in ((0, 0.2), (1, 0.22), (2, 0.2))]  # 200 / 220 / 200 px of an 800 px, 7.8 m grid
        spec = _spec(recs, {"t": _ir(enclosure, shape=shape, width=1.0)})
        spec["meta"] = {"image_size": [1000, 800]}
        spec["image_extent"] = {"x0": 100.0, "x1": 900.0}
        return balcony_frames(spec)

    def test_round_type_takes_the_median_photo_width(self) -> None:
        a, b, c = self._frames("circle")  # median 1.95 m, not the IR's 1.0 m; never wider than 98% of its own box
        self.assertAlmostEqual(b["bx1"] - b["bx0"], 0.25 * 7.8)
        self.assertAlmostEqual(a["bx1"] - a["bx0"], 0.98 * 0.25 * 7.8)

    def test_enclosed_round_types_get_no_shared_width(self) -> None:
        self.assertEqual(_outline_type_widths(self._frames("circle", "enclosed")), {})
        self.assertEqual(set(_outline_type_widths(self._frames("circle"))), {"t"})


class PhotoSpanTest(unittest.TestCase):
    """Photo norms are fractions of image width; the wall spans only the grid's extent."""

    def _span(self, spec: dict) -> tuple[float, float, float]:
        rec = {"width_norm": 0.1, "window_cx_norm": 0.18}
        return photo_x_span(spec, rec, mirror_x=True, bay_x0=0.0, bay_x1=1.0, row=0, col0=0, col1=0)

    def test_maps_through_grid_extent(self) -> None:
        spec = _spec([], {})
        spec["meta"] = {"image_size": [1000, 800]}
        spec["image_extent"] = {"x0": 100.0, "x1": 900.0}
        x0, x1, cx = self._span(spec)  # 180 px is 10% into the grid, 100 px wide is 12.5% of it
        self.assertAlmostEqual(cx, (0.5 - 0.1) * 7.8)
        self.assertAlmostEqual(x1 - x0, 0.125 * 7.8)

    def test_balcony_past_the_grid_stays_on_the_wall_margin(self) -> None:
        spec = _spec([], {})
        spec["meta"] = {"image_size": [1000, 800]}
        spec["image_extent"] = {"x0": 100.0, "x1": 900.0}
        total_w = sum(float(c["w"]) for c in spec["grid"]["cols"])
        spec["wall_margins"] = {"left": total_w / 8.0, "right": 0.0, "top": 0.0, "bottom": 0.0}
        rec = {"width_norm": 0.04, "window_cx_norm": 0.05}  # 50 px, left of the grid, inside the margin
        _, _, cx = photo_x_span(spec, rec, mirror_x=True, bay_x0=0.0, bay_x1=1.0, row=0, col0=0, col1=0)
        self.assertAlmostEqual(cx, total_w / 2.0 + total_w * 50.0 / 800.0)

    def test_without_margins_a_balcony_stays_in_the_grid(self) -> None:
        spec = _spec([], {})
        spec["meta"] = {"image_size": [1000, 800]}
        spec["image_extent"] = {"x0": 100.0, "x1": 900.0}
        total_w = sum(float(c["w"]) for c in spec["grid"]["cols"])
        rec = {"width_norm": 0.04, "window_cx_norm": 0.05}
        x0, x1, _ = photo_x_span(spec, rec, mirror_x=True, bay_x0=0.0, bay_x1=1.0, row=0, col0=0, col1=0)
        self.assertLessEqual(x1, total_w / 2.0 + 1e-9)

    def test_no_image_size_keeps_norms(self) -> None:
        x0, x1, cx = self._span(_spec([], {}))
        self.assertAlmostEqual(cx, (0.5 - 0.18) * 7.8)
        self.assertAlmostEqual(x1 - x0, 0.1 * 7.8)


class EnclosedFitTest(unittest.TestCase):
    def frame(self, i: int, row: int, col: int, x0: float, x1: float, **kw) -> dict:
        return dict(index=i, row=row, col0=col, col1=col, x0=x0, x1=x1, enclosure="enclosed", from_bays=True, curtain_wall=False, shape="rectangle") | kw

    def win(self, r: int, c: int, ox: float, ww: float = 1.0) -> dict:
        return dict(r=r, c=c, span=1, ox=ox, ww=ww)

    def test_cuts_outer_edges_to_windows(self) -> None:
        fits = _fit_spans([self.frame(0, 0, 0, 0.0, 4.0), self.frame(1, 0, 1, 4.0, 8.0)], [self.win(0, 0, 2.0), self.win(0, 1, 5.0)])
        self.assertEqual(fits, {0: (1.75, 4.0), 1: (4.0, 6.25)})

    def test_stack_keeps_one_width(self) -> None:
        fits = _fit_spans([self.frame(0, 0, 0, 0.0, 4.0), self.frame(1, 1, 0, 0.0, 4.0)], [self.win(0, 0, 1.0)])
        self.assertEqual(fits[0], fits[1])

    def test_removes_column_without_windows(self) -> None:
        fits = _fit_spans([self.frame(0, 0, 0, 0.0, 4.0), self.frame(1, 0, 1, 4.0, 8.0)], [self.win(0, 0, 1.0)])
        self.assertIsNone(fits[1])

    def test_no_windows_keeps_bays(self) -> None:
        self.assertEqual(_fit_spans([self.frame(0, 0, 0, 0.0, 4.0)], []), {})

    def test_ignores_window_centred_outside_block(self) -> None:
        fits = _fit_spans([self.frame(0, 0, 0, 0.0, 4.0)], [self.win(0, 0, 1.0), self.win(0, 0, 3.5, 5.0)])
        self.assertEqual(fits[0], (0.75, 2.25))

    def test_never_wider_than_bays(self) -> None:
        self.assertEqual(_fit_spans([self.frame(0, 0, 0, 0.0, 4.0)], [self.win(0, 0, 0.1)])[0], (0.0, 1.35))

    def test_skips_curtain_wall_and_photo_width(self) -> None:
        frames = [self.frame(0, 0, 0, 0.0, 4.0, curtain_wall=True), self.frame(1, 2, 0, 0.0, 4.0, from_bays=False)]
        self.assertEqual(_fit_spans(frames, [self.win(0, 0, 1.0), self.win(2, 0, 1.0)]), {})


class SideWallTest(unittest.TestCase):
    def frame(self, col: int, x0: float, x1: float) -> dict:
        return dict(
            index=col, row=0, col0=col, col1=col, enclosure="enclosed", curtain_wall=False, shape="rectangle",
            outline=[(x0, 0.0), (x0, 1.5), (x1, 1.5), (x1, 0.0)], y_wall=0.0, bx0=x0, bx1=x1, x0=x0, x1=x1,
            sill_z=0.0, thick=0.1, top_z=3.0, from_bays=True,
        )

    def win(self, c: int, ox: float, ww: float = 1.0, **kw) -> dict:
        return dict(r=0, c=c, span=1, ox=ox, oz=1.0, ww=ww, hh=1.2, **kw)

    def test_goes_on_the_near_side(self) -> None:
        t = side_wall_targets([self.frame(3, 0.0, 2.0)], [self.win(4, 2.5, side_face=True)])[0]
        self.assertEqual((t["sign"], t["edge_x"]), (1, 2.0))
        t = side_wall_targets([self.frame(3, 0.0, 2.0)], [self.win(2, -1.5, side_face=True)])[0]
        self.assertEqual((t["sign"], t["edge_x"]), (-1, 0.0))

    def test_mirror_on_the_far_wall(self) -> None:
        t = side_wall_targets([self.frame(3, 0.0, 2.0)], [self.win(4, 2.5, side_face=True)])[0]
        self.assertEqual((t["mirror"]["sign"], t["mirror"]["edge_x"], t["mirror"]["y0"]), (-1, 0.0, t["y0"]))

    def test_mirror_crosses_touching_cells(self) -> None:
        frames = [dict(self.frame(2, -2.0, 0.0), index=9), self.frame(3, 0.0, 2.0)]
        t = side_wall_targets(frames, [self.win(4, 2.5, side_face=True)])[0]
        self.assertEqual((t["mirror"]["frame"], t["mirror"]["edge_x"]), (9, -2.0))

    def test_width_capped_by_the_wall(self) -> None:
        t = side_wall_targets([self.frame(3, 0.0, 2.0)], [self.win(4, 2.5, ww=3.0, side_face=True)])[0]
        self.assertAlmostEqual(t["ww"], 1.5 - 0.24 - 0.12)
        self.assertGreaterEqual(t["y0"], 0.12)

    def test_takes_height_of_front_window(self) -> None:
        front = dict(self.win(3, 0.5), on_balcony=3, oz=0.8, hh=1.6)
        t = side_wall_targets([self.frame(3, 0.0, 2.0)], [front, self.win(4, 2.5, side_face=True)])[1]
        self.assertEqual((t["oz"], t["hh"]), (0.8, 1.6))

    def test_only_flagged_windows(self) -> None:
        self.assertEqual(side_wall_targets([self.frame(3, 0.0, 2.0)], [self.win(4, 2.5)]), {})

    def test_skips_curtain_wall_and_far_blocks(self) -> None:
        curtain = dict(self.frame(3, 0.0, 2.0), curtain_wall=True)
        self.assertEqual(side_wall_targets([curtain], [self.win(4, 2.5, side_face=True)]), {})
        self.assertEqual(side_wall_targets([self.frame(3, 0.0, 2.0)], [self.win(6, 9.0, side_face=True)]), {})

    def test_block_sits_between_front_and_side_windows(self) -> None:
        frames = [self.frame(0, 0.0, 4.0), self.frame(1, 4.0, 8.0)]
        front = self.win(0, 1.0)
        fits = _fit_spans(frames, [front, self.win(1, 5.0, side_face=True)])
        self.assertEqual(fits[0], (2.125, 4.875))  # centre 3.5 = middle of 0.75..6.25; width 0.75..3.5
        self.assertIsNone(fits[1])
        self.assertEqual(front["ox"], 3.0)

    def test_mirrored_side_window(self) -> None:
        frames = [self.frame(0, 0.0, 4.0), self.frame(1, 4.0, 8.0)]
        front = self.win(1, 6.0)
        fits = _fit_spans(frames, [front, self.win(0, 2.0, side_face=True)])
        self.assertEqual(fits[1], (3.125, 5.875))
        self.assertIsNone(fits[0])
        self.assertEqual(front["ox"], 4.0)

    def test_sides_on_both_flanks_keep_the_centre(self) -> None:
        frames = [self.frame(i, 4.0 * i, 4.0 * i + 4.0) for i in range(3)]
        front = self.win(1, 5.5)
        fits = _fit_spans(frames, [front, self.win(0, 2.0, side_face=True), self.win(2, 9.0, side_face=True)])
        self.assertEqual(front["ox"], 5.5)
        self.assertEqual(fits[1], (5.25, 6.75))


class OuterBayHeightTest(unittest.TestCase):
    def _frame(self, rec: dict) -> dict:
        spec = _spec([rec | {"type": "balc"}], {"balc": {"floor": {"shape": "rectangle"}}})
        spec["image_extent"] = {"x0": 0.0, "x1": 100.0, "y0": 100.0, "y1": 500.0}
        return balcony_frames(spec)[0]

    def test_keeps_its_photo_height(self) -> None:
        f = self._frame({"row": 0, "col0": 0, "col1": 0, "photo_bottom_y": 300.0})
        self.assertAlmostEqual(f["sill_z"], 6.0)  # halfway down the 12 m grid, whatever floor it was given

    def test_projecting_railing_is_always_1_1(self) -> None:
        f = self._frame({"row": 0, "col0": 0, "col1": 0, "photo_bottom_y": 300.0, "photo_top_y": 260.0})
        self.assertAlmostEqual(f["sill_z"], 6.0)
        self.assertAlmostEqual(f["photo_rail_h"], 1.1)  # not the 40 px = 1.2 m box (user, 2026-11-08)

    def test_juliet_photo_railing_height_is_clamped(self) -> None:
        def one(top):
            spec = _spec([{"row": 0, "col0": 0, "col1": 0, "type": "j", "photo_bottom_y": 300.0, "photo_top_y": top}],
                         {"j": _ir("open", structure="juliet")})
            spec["image_extent"] = {"x0": 0.0, "x1": 100.0, "y0": 100.0, "y1": 500.0}
            return balcony_frames(spec)[0]["photo_rail_h"]
        self.assertEqual((one(298.0), one(200.0)), (0.3, 1.1))

    def test_juliets_of_a_type_share_the_median_height(self) -> None:
        recs = [{"row": 0, "col0": c, "col1": c, "type": "j", "photo_bottom_y": 300.0, "photo_top_y": t}
                for c, t in ((0, 290.0), (1, 280.0), (2, 270.0))]  # 0.3, 0.6, 0.9 m
        spec = _spec(recs, {"j": _ir("open", structure="juliet")})
        spec["image_extent"] = {"x0": 0.0, "x1": 100.0, "y0": 100.0, "y1": 500.0}
        self.assertEqual([round(f["photo_rail_h"], 6) for f in balcony_frames(spec)], [0.6, 0.6, 0.6])

    def test_tall_box_hangs_from_its_top(self) -> None:
        f = self._frame({"row": 0, "col0": 0, "col1": 0, "photo_bottom_y": 300.0, "photo_top_y": 200.0})
        self.assertAlmostEqual(f["photo_rail_h"], 1.1)  # the type's railing, not the 3 m box
        self.assertAlmostEqual(f["sill_z"] + f["thick"] + f["photo_rail_h"], 9.0)  # railing top on the box top

    def _juliet(self, overlap=None) -> dict:
        rec = {"row": 0, "col0": 0, "col1": 0, "type": "j", "photo_bottom_y": 300.0, "photo_top_y": 200.0}
        if overlap is not None:
            rec["window_overlap"] = overlap
        spec = _spec([rec], {"j": _ir("open", structure="juliet")})
        spec["image_extent"] = {"x0": 0.0, "x1": 100.0, "y0": 100.0, "y1": 500.0}
        return balcony_frames(spec)[0]

    def test_tall_juliet_with_its_window_in_the_box_stands_on_the_bottom(self) -> None:
        f = self._juliet(overlap=0.6)
        self.assertAlmostEqual(f["sill_z"], 6.0)
        self.assertAlmostEqual(f["photo_rail_h"], 1.1)

    def test_tall_juliet_below_its_window_hangs_from_the_top(self) -> None:
        for ov in (0.1, None):
            f = self._juliet(overlap=ov)
            self.assertAlmostEqual(f["sill_z"] + f["photo_rail_h"], 9.0)

    def test_tall_box_floating_above_its_window_drops_to_it(self) -> None:
        f = self._frame({"row": 0, "col0": 0, "col1": 0, "photo_bottom_y": 300.0, "photo_top_y": 200.0})
        win = {"r": 0, "oz": 7.0, "ox": f["bx0"], "ww": f["bx1"] - f["bx0"]}
        seat_on_window_bottoms([f], [win])
        self.assertAlmostEqual(f["sill_z"] + f["thick"], 7.0)  # slab top on the window bottom
        self.assertAlmostEqual(f["photo_rail_h"], 1.1)  # the whole balcony moves down; railing stays 1.1 m

    def test_tall_box_below_its_window_stays_on_top(self) -> None:
        f = self._frame({"row": 0, "col0": 0, "col1": 0, "photo_bottom_y": 300.0, "photo_top_y": 200.0})
        win = {"r": 0, "oz": 8.5, "ox": f["bx0"], "ww": f["bx1"] - f["bx0"]}
        seat_on_window_bottoms([f], [win])
        self.assertAlmostEqual(f["sill_z"] + f["thick"] + f["photo_rail_h"], 9.0)

    def test_photo_placed_balcony_is_not_seated_on_a_window(self) -> None:
        f = self._frame({"row": 0, "col0": 0, "col1": 0, "photo_bottom_y": 300.0, "photo_top_y": 260.0})
        win = {"r": 0, "oz": 7.5, "ox": f["bx0"], "ww": f["bx1"] - f["bx0"]}
        seat_on_window_bottoms([f], [win])
        self.assertAlmostEqual(f["sill_z"], 6.0)

    def test_others_keep_their_floor(self) -> None:
        f = self._frame({"row": 0, "col0": 0, "col1": 0})
        self.assertAlmostEqual(f["sill_z"], 9.0 + 3.0 * 0.14)


class WallMarginTest(unittest.TestCase):
    EXT = {"x0": 100.0, "x1": 900.0, "y0": 50.0, "y1": 450.0}

    def test_margins_use_the_grid_scale(self) -> None:
        m = wall_margins_from_region([0, 0, 1000, 500], self.EXT, 8.0, 4.0)
        self.assertEqual(m, {"left": 1.0, "right": 1.0, "top": 0.5, "bottom": 0.5})

    def test_region_inside_the_grid_adds_nothing(self) -> None:
        m = wall_margins_from_region([200, 100, 800, 400], self.EXT, 8.0, 4.0)
        self.assertEqual(set(m.values()), {0.0})

    def test_no_region_no_margins(self) -> None:
        self.assertIsNone(wall_margins_from_region(None, self.EXT, 8.0, 4.0))

    def test_photo_left_is_plus_x_when_mirrored(self) -> None:
        spec = {"grid": {"rows": [{"h": 4.0}], "cols": [{"w": 8.0}]}, "wall_margins": {"left": 1.0, "right": 2.0, "top": 0.5, "bottom": 0.0}}
        self.assertEqual(wall_world_extent(spec, mirror_x=True), (-6.0, 5.0, 0.0, 4.5))
        self.assertEqual(wall_world_extent(spec, mirror_x=False), (-5.0, 6.0, 0.0, 4.5))


class SeatTest(unittest.TestCase):
    def _frame(self, **kw) -> dict:
        f = {"index": 0, "row": 1, "bx0": 0.0, "bx1": 2.0, "thick": 0.2, "sill_z": 5.0,
             "enclosure": "open", "structure": "projecting"}
        return {**f, **kw}

    def _window(self, **kw) -> dict:
        return {"r": 1, "ox": 0.5, "ww": 1.0, "oz": 4.0, **kw}

    def test_slab_top_on_window_bottom(self) -> None:
        f = self._frame()
        seat_on_window_bottoms([f], [self._window()])
        self.assertAlmostEqual(f["sill_z"] + f["thick"], 4.0)

    def test_slab_below_window_bottom_stays(self) -> None:
        f = self._frame(sill_z=3.0)
        seat_on_window_bottoms([f], [self._window()])
        self.assertEqual(f["sill_z"], 3.0)

    def test_juliet_starts_at_window_bottom(self) -> None:
        f = self._frame(structure="juliet")
        seat_on_window_bottoms([f], [self._window()])
        self.assertAlmostEqual(f["sill_z"], 4.0)

    def test_lowest_of_several_windows(self) -> None:
        f = self._frame()
        seat_on_window_bottoms([f], [self._window(), self._window(ox=1.5, ww=0.4, oz=3.5)])
        self.assertAlmostEqual(f["sill_z"] + f["thick"], 3.5)

    def test_keeps_height_without_a_window(self) -> None:
        f, g = self._frame(), self._frame(structure="rooftop")
        seat_on_window_bottoms([f, g], [self._window(r=2), self._window(ox=3.0)])
        self.assertEqual(f["sill_z"], 5.0)
        seat_on_window_bottoms([g], [self._window()])
        self.assertEqual(g["sill_z"], 5.0)

    def test_skips_window_moved_onto_a_balcony(self) -> None:
        f = self._frame()
        seat_on_window_bottoms([f], [self._window(on_balcony=0)])
        self.assertEqual(f["sill_z"], 5.0)

    def test_enclosure_below_follows_the_slab(self) -> None:
        f = self._frame()
        below = self._frame(index=1, row=2, enclosure="enclosed", above=0, top_z=5.0)
        seat_on_window_bottoms([f, below], [self._window()])
        self.assertAlmostEqual(below["top_z"], f["sill_z"])


if __name__ == "__main__":
    unittest.main()
