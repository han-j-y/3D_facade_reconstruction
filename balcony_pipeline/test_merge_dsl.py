"""Photo-norm merge fields for balcony slab width."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from merge_dsl import _photo_norms, merge_balcony_into_windows_dsl  # noqa: E402


class PhotoNormTests(unittest.TestCase):
    def test_norms(self) -> None:
        w, cx = _photo_norms([100, 0, 300, 50], (1000, 800))
        self.assertAlmostEqual(w, 0.2)
        self.assertAlmostEqual(cx, 0.2)

    def test_merge_writes_norms(self) -> None:
        dsl = {
            "schema": "facade_recovery_dsl_v1",
            "meta": {"image_size": [1000, 800]},
            "layout": {"floors": [], "bays": [], "placement": []},
            "window_types": [],
            "instances": [],
        }
        units = [
            {
                "unit_id": 0,
                "type_id": 0,
                "floor": 1,
                "bay_start": 0,
                "bay_end": 2,
                "bay": 1,
                "box_xyxy": [100, 10, 300, 80],
            }
        ]
        out = merge_balcony_into_windows_dsl(
            dsl, balcony_types=[{"type_id": 0, "name": "balc_type_00", "structure_ir": {}}], units=units
        )
        b = out["layout"]["balconies"][0]
        self.assertAlmostEqual(b["width_norm"], 0.2)
        self.assertAlmostEqual(b["cx_norm"], 0.2)
        self.assertEqual(b["bay_start"], 0)
        self.assertEqual(b["bay_end"], 2)

    def test_merge_per_unit_railing_type(self) -> None:
        dsl = {
            "schema": "facade_recovery_dsl_v1",
            "meta": {"image_size": [1000, 800]},
            "layout": {"floors": [], "bays": [], "placement": []},
            "window_types": [],
            "instances": [],
        }
        baluster_ir = {
            "structure": "projecting",
            "enclosure": "open",
            "railing": {"kind": "baluster", "height": 1.1},
        }
        solid_ir = {
            "structure": "projecting",
            "enclosure": "open",
            "railing": {"kind": "solid", "height": 1.1},
        }
        units = [
            {
                "unit_id": 0,
                "type_id": 0,
                "floor": 1,
                "bay_start": 0,
                "bay_end": 0,
                "bay": 0,
                "box_xyxy": [100, 10, 200, 80],
                "structure_ir": baluster_ir,
            },
            {
                "unit_id": 1,
                "type_id": 0,
                "floor": 2,
                "bay_start": 0,
                "bay_end": 0,
                "bay": 0,
                "box_xyxy": [100, 100, 200, 170],
                "structure_ir": solid_ir,
            },
        ]
        types = [
            {"name": "balc_baluster", "structure_ir": baluster_ir},
            {"name": "balc_solid", "structure_ir": solid_ir},
        ]
        out = merge_balcony_into_windows_dsl(
            dsl,
            balcony_types=types,
            units=units,
            per_unit_railing=True,
        )
        self.assertEqual(out["layout"]["balconies"][0]["type"], "balc_baluster")
        self.assertEqual(out["layout"]["balconies"][1]["type"], "balc_solid")


if __name__ == "__main__":
    unittest.main()
