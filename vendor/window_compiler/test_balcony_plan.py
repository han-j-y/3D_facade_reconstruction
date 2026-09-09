"""Surface-panel layout (no Blender)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from balcony_plan import (  # noqa: E402
    MULLION_WIDTH_M,
    SURFACE_PANEL_M,
    _panel_divisions,
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


if __name__ == "__main__":
    unittest.main()
