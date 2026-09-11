"""Recovery profile: railing_only vote fingerprint (no Blender)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from heuristic_ir import (  # noqa: E402
    RAILING_MAX_OPAQUE_RUN,
    RAILING_MIN_OPAQUE_VERTICAL_THICKNESS,
    RAILING_OPAQUE_MEDIAN_DEV,
    RAILING_SOLID_UNIFORM_RANGE,
    _infer_rail_kind,
    _railing_has_solid_patch,
    _railing_max_opaque_run,
    balcony_view,
    infer_balcony_ir,
    ir_to_tokens,
)
from recovery_profile import FIXED_DEFAULTS, resolve_profile  # noqa: E402


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
        self.assertEqual(
            view, {"railing_kind": "open_work", "railing_material": "metal"}
        )

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
        self.assertIn(ir["railing"]["kind"], ("open_work", "solid"))
        self.assertNotEqual(ir["railing"]["kind"], "glass")
        toks = ir_to_tokens(ir, profile_name="railing_only")
        self.assertEqual(toks[0], "BALCONY")
        self.assertTrue(any(t.startswith("railing=") for t in toks))
        self.assertFalse(any(t.startswith("structure=") for t in toks))

    def test_thick_uniform_face_solid_vs_baluster(self) -> None:
        h, w = 48, 64
        top = h // 2

        def _rgb(gray: np.ndarray) -> Image.Image:
            u8 = np.clip(gray, 0, 255).astype(np.uint8)
            return Image.fromarray(np.stack([u8, u8, u8], axis=2))

        uniform = np.full((h, w), 200.0, dtype=np.float32)
        band = uniform[:top]

        # Flat band with no opaque texture → open_work.
        self.assertFalse(_railing_has_solid_patch(band))
        self.assertEqual(
            _infer_rail_kind(has_qualifying_solid_patch=False),
            "open_work",
        )

        # Thin vertical bars → short horizontal runs → open_work.
        baluster_gray = uniform.copy()
        for x in range(0, w, 8):
            baluster_gray[:top, x : min(x + 3, w)] = 40.0
        band_b = baluster_gray[:top]
        self.assertLess(_railing_max_opaque_run(band_b), RAILING_MAX_OPAQUE_RUN)
        self.assertFalse(_railing_has_solid_patch(band_b))
        self.assertEqual(
            infer_balcony_ir(
                _rgb(baluster_gray),
                box=[0, 0, w, h],
                image_size=(400, 300),
                profile_name="railing_only",
            )["railing"]["kind"],
            "open_work",
        )

        # Thin horizontal handrail: long run but vertical thickness < 5 → baluster.
        handrail_gray = uniform.copy()
        handrail_gray[top // 2, 8:30] = 40.0
        self.assertGreaterEqual(
            _railing_max_opaque_run(handrail_gray[:top]),
            RAILING_MAX_OPAQUE_RUN,
        )
        self.assertFalse(_railing_has_solid_patch(handrail_gray[:top]))

        # Thick uniform opaque block → solid.
        solid_gray = uniform.copy()
        thick = RAILING_MIN_OPAQUE_VERTICAL_THICKNESS + 2
        solid_gray[:thick, 8:30] = 80.0
        band_solid = solid_gray[:top]
        self.assertTrue(_railing_has_solid_patch(band_solid))
        self.assertEqual(
            infer_balcony_ir(
                _rgb(solid_gray),
                box=[0, 0, w, h],
                image_size=(400, 300),
                profile_name="railing_only",
            )["railing"]["kind"],
            "solid",
        )

        # Thick but textured face (range > uniform threshold) → baluster.
        textured_gray = uniform.copy()
        textured_gray[:thick, 8:20] = 80.0
        textured_gray[:thick, 20:30] = 80.0 + RAILING_SOLID_UNIFORM_RANGE + 5.0
        self.assertFalse(_railing_has_solid_patch(textured_gray[:top]))

    def test_opaque_uses_median_deviation_not_dark_only(self) -> None:
        h, w = 24, 64
        band = np.full((h, w), 100.0, dtype=np.float32)
        band[:, 10:30] = 100.0 + RAILING_OPAQUE_MEDIAN_DEV + 5.0
        run = _railing_max_opaque_run(band)
        self.assertGreaterEqual(run, 20)

    def test_glass_maps_to_surface_panel_in_view(self) -> None:
        view = balcony_view(
            {"railing": {"kind": "glass"}, "floor": {}, "supports": {}},
            profile_name="railing_only",
        )
        self.assertEqual(view["railing_kind"], "surface_panel")

    def test_three_rail_kinds_pass_through_view(self) -> None:
        for kind in ("open_work", "surface_panel", "solid"):
            view = balcony_view(
                {"railing": {"kind": kind}, "floor": {}, "supports": {}},
                profile_name="railing_only",
            )
            self.assertEqual(view["railing_kind"], kind, msg=kind)
            if kind == "open_work":
                self.assertEqual(view.get("railing_material"), "metal", msg=kind)
            else:
                self.assertNotIn("railing_material", view)
        self.assertEqual(
            balcony_view(
                {"railing": {"kind": "baluster"}, "floor": {}, "supports": {}},
                profile_name="railing_only",
            )["railing_kind"],
            "open_work",
        )
        self.assertEqual(
            balcony_view(
                {"railing": {"kind": "lined_panel"}, "floor": {}, "supports": {}},
                profile_name="railing_only",
            )["railing_kind"],
            "open_work",
        )

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

    def test_rail_kind_override_skips_heuristic(self) -> None:
        crop = Image.new("RGB", (64, 48), (200, 200, 200))
        kwargs = dict(
            box=[0, 0, 64, 48],
            image_size=(400, 300),
            profile_name="railing_only",
        )
        heuristic = infer_balcony_ir(crop, **kwargs)
        self.assertEqual(heuristic["railing"]["kind"], "open_work")
        self.assertEqual(heuristic["railing"]["material"], "metal")
        forced = infer_balcony_ir(crop, rail_kind_override="solid", **kwargs)
        self.assertEqual(forced["railing"]["kind"], "solid")
        self.assertNotIn("material", forced["railing"])
        self.assertEqual(forced["output"]["railing_thickness"], 0.20)
        panel = infer_balcony_ir(crop, rail_kind_override="surface_panel", **kwargs)
        self.assertEqual(panel["railing"]["kind"], "surface_panel")
        self.assertEqual(panel["output"]["railing_thickness"], 0.025)
        masonry = infer_balcony_ir(
            crop,
            rail_kind_override="open_work",
            rail_material_override="masonry",
            **kwargs,
        )
        self.assertEqual(masonry["railing"]["kind"], "open_work")
        self.assertEqual(masonry["railing"]["material"], "masonry")


if __name__ == "__main__":
    unittest.main()
