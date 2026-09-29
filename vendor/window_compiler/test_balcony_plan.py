"""Surface-panel layout and plan outlines (no Blender)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from balcony_plan import (  # noqa: E402
    HEXAGON_CUT_ANGLE_DEG,
    HEXAGON_CUT_RUN_M,
    MULLION_WIDTH_M,
    SURFACE_PANEL_M,
    TRAPEZOID_WALL_ANGLE_DEG,
    _panel_divisions,
    outer_edges,
    slab_outline,
    surface_panel_parts,
)


class SurfacePanelLayoutTests(unittest.TestCase):
    def test_divisions_floor_not_round(self) -> None:
        self.assertEqual(_panel_divisions(2.0, 1.0), 2)
        self.assertEqual(_panel_divisions(1.5, 1.0), 1)

    def test_2m_by_1_5m_matches_catalog(self) -> None:
        parts = surface_panel_parts(
            x0=0.0,
            x1=2.0,
            y_wall=0.0,
            depth=1.5,
            z_lo=0.0,
            rail_h=1.1,
        )
        names = [str(p["name"]) for p in parts]
        self.assertEqual(sum(n.startswith("mull_f_") for n in names), 3)
        self.assertEqual(sum(n.startswith("panel_f_") for n in names), 2)
        self.assertEqual(sum(n.startswith("panel_L_") for n in names), 1)
        self.assertEqual(sum(n.startswith("panel_R_") for n in names), 1)
        self.assertTrue(any(n.startswith("mull_L_") for n in names))
        self.assertTrue(any(n.startswith("mull_R_") for n in names))
        panels = [p for p in parts if str(p["name"]).startswith("panel_")]
        self.assertTrue(all(abs(float(p["sy"]) - SURFACE_PANEL_M) < 1e-9 or
                            abs(float(p["sx"]) - SURFACE_PANEL_M) < 1e-9 for p in panels))
        mulls = [p for p in parts if str(p["name"]).startswith("mull_")]
        for p in mulls:
            self.assertAlmostEqual(float(p["sx"]), MULLION_WIDTH_M)
            self.assertAlmostEqual(float(p["sy"]), MULLION_WIDTH_M)
            self.assertAlmostEqual(float(p["sz"]), 1.1)

    def test_outer_face_flush_with_slab_front(self) -> None:
        parts = surface_panel_parts(
            x0=0.0,
            x1=2.0,
            y_wall=1.0,
            depth=1.5,
            z_lo=0.2,
            rail_h=1.1,
        )
        front = next(p for p in parts if p["name"] == "mull_f_0")
        y_front = 1.0 + 1.5
        self.assertAlmostEqual(float(front["cy"]) + MULLION_WIDTH_M / 2.0, y_front)

    def test_front_only_has_no_sides(self) -> None:
        parts = surface_panel_parts(
            x0=0.0,
            x1=2.0,
            y_wall=0.0,
            depth=0.06,
            z_lo=0.0,
            rail_h=1.1,
            front_only=True,
        )
        names = [str(p["name"]) for p in parts]
        self.assertFalse(any("_L_" in n or "_R_" in n for n in names))
        self.assertTrue(any(n.startswith("panel_f_") for n in names))


class TrapezoidOutlineTests(unittest.TestCase):
    def test_sides_meet_wall_at_75_degrees(self) -> None:
        outline = slab_outline("trapezoid", x0=0.0, x1=4.0, y_wall=0.0, depth=1.5)
        self.assertEqual(len(outline), 4)
        x0, y0 = outline[0]
        x1, y1 = outline[1]
        angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
        self.assertAlmostEqual(angle, TRAPEZOID_WALL_ANGLE_DEG, places=5)
        wall = outline[3][0] - outline[0][0]
        front = outline[2][0] - outline[1][0]
        self.assertLess(front, wall)
        self.assertAlmostEqual(outline[1][0] - outline[0][0], outline[3][0] - outline[2][0])

    def test_outer_edges_skip_the_wall(self) -> None:
        outline = slab_outline("trapezoid", x0=0.0, x1=4.0, y_wall=0.0, depth=1.5)
        edges = outer_edges(outline, 0.0)
        self.assertEqual(len(edges), 3)
        for p0, p1 in edges:
            self.assertFalse(abs(p0[1]) < 1e-6 and abs(p1[1]) < 1e-6)


class HexagonOutlineTests(unittest.TestCase):
    def test_front_corners_cut_1m_at_30_degrees(self) -> None:
        outline = slab_outline("hexagon", x0=0.0, x1=3.0, y_wall=0.0, depth=1.5)
        self.assertEqual(len(outline), 6)
        self.assertEqual(outline[0], (0.0, 0.0))
        self.assertEqual(outline[5], (3.0, 0.0))
        run = HEXAGON_CUT_RUN_M
        rise = run * math.tan(math.radians(HEXAGON_CUT_ANGLE_DEG))
        self.assertAlmostEqual(outline[2][0], run)
        self.assertAlmostEqual(outline[3][0], 3.0 - run)
        self.assertAlmostEqual(outline[2][1], 1.5)
        self.assertAlmostEqual(outline[1][1], 1.5 - rise)
        dx, dy = outline[1][0] - outline[2][0], outline[1][1] - outline[2][1]
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        self.assertAlmostEqual(angle, HEXAGON_CUT_ANGLE_DEG, places=5)
        edges = outer_edges(outline, 0.0)
        self.assertEqual(len(edges), 5)


if __name__ == "__main__":
    unittest.main()
