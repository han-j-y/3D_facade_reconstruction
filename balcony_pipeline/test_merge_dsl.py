"""Photo-norm merge fields for balcony slab width."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from merge_dsl import (  # noqa: E402
    _mean_bay_center_norm,
    _photo_norms,
    merge_balcony_into_windows_dsl,
)


class PhotoNormTests(unittest.TestCase):
    def test_norms(self) -> None:
        w, cx = _photo_norms([100, 0, 300, 50], (1000, 800))
        self.assertAlmostEqual(w, 0.2)
        self.assertAlmostEqual(cx, 0.2)

    def test_merge_writes_norms_photo_center(self) -> None:
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
            dsl,
            balcony_types=[{"type_id": 0, "name": "balc_type_00", "structure_ir": {}}],
            units=units,
            center_mode="photo",
        )
        b = out["layout"]["balconies"][0]
        self.assertAlmostEqual(b["width_norm"], 0.2)
        self.assertAlmostEqual(b["cx_norm"], 0.2)
        self.assertEqual(b["bay_start"], 0)
        self.assertEqual(b["bay_end"], 2)

    def test_merge_bay_center_omits_cx_norm(self) -> None:
        dsl = {
            "schema": "facade_recovery_dsl_v1",
            "meta": {"image_size": [1000, 800]},
            "layout": {"floors": [], "bays": [], "placement": []},
            "window_types": [],
            "instances": [
                {"floor": 1, "bay": 0, "box_xyxy": [0, 50, 100, 150]},
                {"floor": 1, "bay": 1, "box_xyxy": [100, 50, 200, 150]},
                {"floor": 1, "bay": 2, "box_xyxy": [200, 50, 300, 150]},
            ],
        }
        units = [
            {
                "unit_id": 0,
                "type_id": 0,
                "floor": 1,
                "bay_start": 0,
                "bay_end": 2,
                "bays": [0, 1, 2],
                "bays_center": [0, 1],
                "bay": 1,
                "box_xyxy": [100, 10, 300, 80],
            }
        ]
        out = merge_balcony_into_windows_dsl(
            dsl,
            balcony_types=[{"type_id": 0, "name": "balc_type_00", "structure_ir": {}}],
            units=units,
            center_mode="bay",
        )
        b = out["layout"]["balconies"][0]
        self.assertAlmostEqual(b["width_norm"], 0.2)
        self.assertNotIn("cx_norm", b)
        self.assertEqual(b["bays"], [0, 1, 2])
        self.assertEqual(b["bays_center"], [0, 1])
        # mean of bay 0 center 50 and bay 1 center 150 -> 100 / 1000
        self.assertAlmostEqual(b["bay_cx_norm"], 0.1)
        self.assertEqual(out["meta"]["balcony_center_mode"], "bay")

    def test_merge_window_center_from_overlap_partners(self) -> None:
        dsl = {
            "schema": "facade_recovery_dsl_v1",
            "meta": {"image_size": [400, 400]},
            "layout": {"floors": [], "bays": [], "placement": []},
            "window_types": [],
            "instances": [
                {"unit_id": 1, "floor": 2, "bay": 1, "box_xyxy": [100, 40, 180, 120]},
                {"unit_id": 2, "floor": 2, "bay": 2, "box_xyxy": [190, 40, 270, 120]},
            ],
        }
        units = [
            {
                "unit_id": 0,
                "type_id": 0,
                "floor": 2,
                "bay_start": 1,
                "bay_end": 2,
                "bays": [1, 2],
                "bays_center": [1],
                "bay": 1,
                "box_xyxy": [95, 115, 275, 175],
            }
        ]
        out = merge_balcony_into_windows_dsl(
            dsl,
            balcony_types=[{"type_id": 0, "name": "balc_type_00", "structure_ir": {}}],
            units=units,
            center_mode="window",
        )
        b = out["layout"]["balconies"][0]
        self.assertEqual(b["partner_window_unit_ids"], [1, 2])
        self.assertAlmostEqual(b["window_cx_norm"], (140 + 230) / 2 / 400)

    def test_merge_window_default_mode(self) -> None:
        dsl = {
            "schema": "facade_recovery_dsl_v1",
            "meta": {"image_size": [1000, 800]},
            "layout": {"floors": [], "bays": [], "placement": []},
            "window_types": [],
            "instances": [
                {"unit_id": 5, "floor": 0, "bay": 0, "box_xyxy": [40, 50, 60, 150]},
            ],
        }
        units = [
            {
                "unit_id": 0,
                "type_id": 0,
                "floor": 0,
                "bay_start": 0,
                "bay_end": 0,
                "bays_center": [0],
                "bay": 0,
                "box_xyxy": [30, 200, 70, 280],
            }
        ]
        out = merge_balcony_into_windows_dsl(
            dsl,
            balcony_types=[{"type_id": 0, "name": "balc_type_00", "structure_ir": {}}],
            units=units,
        )
        b = out["layout"]["balconies"][0]
        self.assertAlmostEqual(b["window_cx_norm"], 0.05)
        self.assertEqual(out["meta"]["balcony_center_mode"], "window")

    def test_mean_bay_center_norm(self) -> None:
        bay_x = {0: (0, 100), 1: (100, 300), 2: (300, 400)}
        cx = _mean_bay_center_norm([0, 1, 2], bay_x, 1000.0)
        self.assertAlmostEqual(cx, (50 + 200 + 350) / 3 / 1000)

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
        self.assertEqual(out["layout"]["balconies"][0]["type"], "balc_open_work")
        self.assertEqual(out["layout"]["balconies"][1]["type"], "balc_solid")


if __name__ == "__main__":
    unittest.main()
