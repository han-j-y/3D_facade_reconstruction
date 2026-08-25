"""Recovery profile: railing_only vote fingerprint (no Blender)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from heuristic_ir import balcony_view, infer_balcony_ir, ir_to_tokens  # noqa: E402
from recovery_profile import FIXED_DEFAULTS, resolve_profile  # noqa: E402
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
