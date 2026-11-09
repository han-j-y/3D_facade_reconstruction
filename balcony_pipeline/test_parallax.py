"""Straddling-balcony parallax fix (parallax.py) on a synthetic depth map."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parallax import fix_straddling_balconies  # noqa: E402


def _scene(tilt_m: float, box: list[int]) -> tuple[dict, np.ndarray]:
    """400 x 200 wall at 10 m, right side ``tilt_m`` farther; one projecting balcony 1 m in front of it whose
    right 20 px ramp back to the wall (its side face, camera on the right)."""
    h, w = 200, 400
    depth = 10.0 + tilt_m * np.tile(np.arange(w) / w, (h, 1))
    x0, y0, x1, y1 = box
    depth[y0:y1, x0:x1 - 20] -= 1.0
    depth[y0:y1, x1 - 20:x1] -= np.linspace(1.0, 0.0, 20)
    windows = [{"unit_id": 1, "bay": 1, "floor": 1, "box_xyxy": [90, 60, 120, 110]},
               {"unit_id": 2, "bay": 2, "floor": 1, "box_xyxy": [170, 60, 200, 110]}]
    dsl = {
        "meta": {"image_size": [w, h]},
        "instances": windows,
        "layout": {"floors": [], "bays": [], "balconies": [
            {"floor": 1, "bays": [1, 2], "type": "b", "box_xyxy": box, "width_norm": (x1 - x0) / w,
             "partner_window_unit_ids": [1, 2], "window_cx_norm": 145 / w}]},
        "balcony_types": [{"name": "b", "structure_ir": {"enclosure": "open", "structure": "projecting"}}],
    }
    return dsl, depth


class StraddleTests(unittest.TestCase):
    def test_camera_right_keeps_the_right_window_and_drops_the_side_face(self) -> None:
        dsl, depth = _scene(-1.0, [120, 110, 210, 140])  # nearer on the right; box left of window 2's centre
        cam = fix_straddling_balconies(dsl, depth)
        self.assertEqual(cam["side"], "right")
        rec = dsl["layout"]["balconies"][0]
        self.assertEqual(rec["partner_window_unit_ids"], [2])
        self.assertEqual(rec["box_xyxy"][2], 210)  # camera-side edge kept
        self.assertGreater(rec["box_xyxy"][0], 120)
        self.assertAlmostEqual(rec["window_cx_norm"], 185 / 400)

    def test_no_fix_when_tilt_and_windows_disagree(self) -> None:
        dsl, depth = _scene(+1.0, [120, 110, 210, 140])  # tilt says left, windows say right
        cam = fix_straddling_balconies(dsl, depth)
        self.assertIsNone(cam["side"])
        self.assertEqual(dsl["layout"]["balconies"][0]["partner_window_unit_ids"], [1, 2])


if __name__ == "__main__":
    unittest.main()
