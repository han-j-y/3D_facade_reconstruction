"""Recovery profile: railing_only vote fingerprint (no Blender)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from heuristic_ir import (  # noqa: E402
    RAILING_VERTICAL_EDGE_THRESHOLD,
    _infer_rail_kind,
    _railing_vertical_edge_score,
    balcony_view,
    infer_balcony_ir,
    ir_to_tokens,
)
from recovery_profile import FIXED_DEFAULTS, resolve_profile  # noqa: E402
import numpy as np
from PIL import Image


class RailingOnlyProfileTests(unittest.TestCase):
    def test_view_only_railing(self) -> None:
        view = balcony_view(
            {
                "structure": "inset",
                "enclosure": "enclosed",
                "floor": {"shape": "triangle"},
                "railing": {"kind": "metal"},
                "supports": {"count": 4},
            },
            profile_name="railing_only",
        )
        self.assertEqual(view, {"railing_kind": "baluster"})

    def test_infer_fixed_axes(self) -> None:
        crop = Image.new("RGB", (64, 48), (200, 200, 200))
        ir = infer_balcony_ir(
            crop,
            box=[10, 10, 74, 58],
            image_size=(400, 300),
            profile_name="railing_only",
        )
        self.assertEqual(ir["structure"], FIXED_DEFAULTS["structure"])
        self.assertEqual(ir["enclosure"], FIXED_DEFAULTS["enclosure"])
        self.assertEqual(ir["floor"]["shape"], FIXED_DEFAULTS["floor_shape"])
        self.assertEqual(ir["supports"]["count"], FIXED_DEFAULTS["supports_count"])
        self.assertIn(ir["railing"]["kind"], ("baluster", "solid"))
        self.assertNotEqual(ir["railing"]["kind"], "glass")
        toks = ir_to_tokens(ir, profile_name="railing_only")
        self.assertEqual(toks[0], "BALCONY")
        self.assertTrue(any(t.startswith("railing=") for t in toks))
        self.assertFalse(any(t.startswith("structure=") for t in toks))

    def test_railing_vertical_edge_baluster_vs_solid(self) -> None:
        h, w = 48, 64
        solid_gray = np.full((h, w), 200.0, dtype=np.float32)
        self.assertLess(
            _railing_vertical_edge_score(solid_gray), RAILING_VERTICAL_EDGE_THRESHOLD
        )
        self.assertEqual(_infer_rail_kind(vertical_edge=2.0), "solid")

        baluster_gray = solid_gray.copy()
        top = h // 2
        for x in range(0, w, 8):
            baluster_gray[:top, x : min(x + 3, w)] = 40.0
        edge = _railing_vertical_edge_score(baluster_gray)
        self.assertGreaterEqual(edge, RAILING_VERTICAL_EDGE_THRESHOLD)
        self.assertEqual(_infer_rail_kind(vertical_edge=edge), "baluster")

        def _rgb(gray: np.ndarray) -> Image.Image:
            u8 = np.clip(gray, 0, 255).astype(np.uint8)
            return Image.fromarray(np.stack([u8, u8, u8], axis=2))

        solid_crop = _rgb(solid_gray)
        baluster_crop = _rgb(baluster_gray)
        box = [0, 0, w, h]
        self.assertEqual(
            infer_balcony_ir(
                solid_crop, box=box, image_size=(400, 300), profile_name="railing_only"
            )["railing"]["kind"],
            "solid",
        )
        self.assertEqual(
            infer_balcony_ir(
                baluster_crop, box=box, image_size=(400, 300), profile_name="railing_only"
            )["railing"]["kind"],
            "baluster",
        )

    def test_glass_maps_to_solid_in_view(self) -> None:
        view = balcony_view(
            {"railing": {"kind": "glass"}, "floor": {}, "supports": {}},
            profile_name="railing_only",
        )
        self.assertEqual(view["railing_kind"], "solid")

    def test_full_profile_includes_structure(self) -> None:
        p = resolve_profile("full")
        self.assertTrue(p["structure"])
        view = balcony_view(
            {
                "structure": "projecting",
                "enclosure": "open",
                "floor": {"shape": "rectangle"},
                "railing": {"kind": "glass"},
                "supports": {"count": 0},
            },
            profile_name="full",
        )
        self.assertIn("structure", view)
        self.assertIn("railing_kind", view)


if __name__ == "__main__":
    unittest.main()
