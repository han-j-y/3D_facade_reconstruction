"""Normalize recovery DSL → compiler façade (placement_fit + balconies)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPILER = ROOT / "vendor" / "window_compiler"
for p in (ROOT, COMPILER):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from facade_spec import (  # noqa: E402
    normalize_facade_spec,
    placement_fit_for_instance,
    world_placement_in_cell,
    get_cell,
)


class FacadeSpecPlacementTests(unittest.TestCase):
    def _dsl(self) -> dict:
        return {
            "schema": "facade_recovery_dsl_v1",
            "meta": {
                "facade_id": "t",
                "image": "x.jpg",
                "image_size": [400, 400],
                "columns_xy": [[0, 100], [100, 200], [200, 300]],
                "floors_y": [[50, 150], [150, 250]],
            },
            "layout": {
                "floors": [
                    {"id": 0, "name": "F0", "h_norm": 0.25},
                    {"id": 1, "name": "F1", "h_norm": 0.25},
                ],
                "bays": [
                    {"id": 0, "name": "B0", "w_norm": 0.25},
                    {"id": 1, "name": "B1", "w_norm": 0.25},
                    {"id": 2, "name": "B2", "w_norm": 0.25},
                ],
                "placement": [
                    ["win_type_00", None, "win_type_00"],
                    [None, "win_type_00", None],
                ],
                "placement_spans": [[1, 0, 1], [0, 1, 0]],
                "balconies": [
                    {
                        "floor": 1,
                        "bay_start": 1,
                        "bay_end": 1,
                        "type": "balc_type_00",
                        "width_norm": 0.2,
                        "window_cx_norm": 0.4,
                    }
                ],
            },
            "window_types": [
                {
                    "type_id": 0,
                    "name": "win_type_00",
                    "structure_ir": {
                        "type": "window",
                        "boundary": {
                            "id": "root",
                            "shape": "rectangle",
                            "params": {"width": 1.0, "height": 1.2},
                        },
                        "layout": [],
                    },
                }
            ],
            "balcony_types": [
                {
                    "type_id": 0,
                    "name": "balc_type_00",
                    "structure_ir": {
                        "structure": "projecting",
                        "enclosure": "open",
                        "railing": {"kind": "baluster", "height": 1.1},
                        "floor": {"shape": "rectangle", "params": {"depth": 0.8}},
                    },
                }
            ],
            "instances": [
                {
                    "unit_id": 0,
                    "kind": "window",
                    "type_id": 0,
                    "floor": 0,
                    "bay": 0,
                    "colspan": 1,
                    "box_xyxy": [10, 60, 90, 140],
                },
                {
                    "unit_id": 1,
                    "kind": "window",
                    "type_id": 0,
                    "floor": 0,
                    "bay": 2,
                    "colspan": 1,
                    "box_xyxy": [210, 60, 290, 140],
                },
                {
                    "unit_id": 2,
                    "kind": "window",
                    "type_id": 0,
                    "floor": 1,
                    "bay": 1,
                    "colspan": 1,
                    "box_xyxy": [120, 160, 180, 240],
                },
            ],
        }

    def test_normalize_writes_placement_fit_and_balconies(self) -> None:
        facade = normalize_facade_spec(self._dsl())
        self.assertIn("placement_fit", facade)
        self.assertTrue(facade["placement_fit"][0][0])
        self.assertEqual(len(facade["balcony_placement"]), 1)
        self.assertIn("balc_type_00", facade["balconies"])
        self.assertEqual(facade["meta"].get("columns_xy")[0], [0, 100])

    def test_world_placement_from_fit(self) -> None:
        facade = normalize_facade_spec(self._dsl())
        cell = get_cell(facade, 0, 0, mirror_x=True)
        fit = facade["placement_fit"][0][0]
        ox, oz, ww, hh = world_placement_in_cell(cell, fit)
        self.assertGreater(ww, 0.1)
        self.assertGreater(hh, 0.1)
        self.assertGreaterEqual(ox, min(cell["x0"], cell["x1"]) - 1e-6)


if __name__ == "__main__":
    unittest.main()
