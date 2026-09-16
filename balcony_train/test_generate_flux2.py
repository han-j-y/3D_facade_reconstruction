"""Tests for FLUX.2 generator helpers (no GPU / diffusers required)."""

from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train import generate_flux2 as gen  # noqa: E402
from balcony_train.generate_flux2 import DEFAULT_PREFIX, PROMPT_SETS, _next_index  # noqa: E402
from balcony_train.prompts_masonry import (  # noqa: E402
    MASONRY_OPENWORK_PROMPTS,
    NEGATIVE_PROMPT,
)
from balcony_train.prompts_metal import METAL_OPENWORK_PROMPTS  # noqa: E402
from balcony_train.prompts_solid import SOLID_PROMPTS  # noqa: E402
from balcony_train.prompts_surface_panel import SURFACE_PANEL_PROMPTS  # noqa: E402


class GenerateFlux2HelperTests(unittest.TestCase):
    def test_prompts_non_empty(self) -> None:
        self.assertEqual(len(MASONRY_OPENWORK_PROMPTS), 100)
        joined = " ".join(MASONRY_OPENWORK_PROMPTS).lower()
        self.assertTrue(
            any(
                k in joined
                for k in ("looking up", "looking upward", "upward perspective")
            )
        )
        for floor in ("2nd-floor", "5th-floor", "10th-floor"):
            self.assertIn(floor, joined)
        for p in MASONRY_OPENWORK_PROMPTS:
            pl = p.lower()
            self.assertTrue(
                "street" in pl or "sidewalk" in pl,
                msg=f"missing street-side cue: {p}",
            )
            self.assertTrue(
                any(
                    k in pl
                    for k in (
                        "open",
                        "opening",
                        "openings",
                        "gap",
                        "gaps",
                        "pierced",
                        "perforated",
                        "grille",
                        "baluster",
                    )
                ),
                msg=f"missing openings cue: {p}",
            )
            self.assertTrue(
                any(
                    k in pl
                    for k in (
                        "full window",
                        "entire window",
                        "whole window",
                        "complete window",
                        "window visible",
                        "window in frame",
                        "window unit",
                        "medium",
                    )
                ),
                msg=f"missing wider framing / full-window cue: {p}",
            )
            self.assertTrue(
                any(
                    k in pl
                    for k in (
                        "2nd",
                        "second",
                        "third",
                        "upper",
                        "above the ground",
                        "above ground",
                    )
                ),
                msg=f"missing upper-story cue: {p}",
            )
        self.assertTrue(
            any(k in joined for k in ("concrete", "stone", "masonry", "limestone"))
        )
        self.assertTrue("post" in joined or "pier" in joined or "baluster" in joined)
        for shape in (
            "rectangular",
            "elliptical",
            "semicircular",
            "triangular",
            "hexagonal",
            "trapezoid",
        ):
            self.assertIn(shape, joined, msg=f"missing slab shape: {shape}")
        self.assertTrue("classical" in joined)
        self.assertTrue("cornice" in joined or "bracket" in joined)
        self.assertLess(len(NEGATIVE_PROMPT), 500)

    def test_surface_panel_and_solid_prompt_sets(self) -> None:
        self.assertEqual(
            set(PROMPT_SETS), {"masonry", "metal", "surface_panel", "solid"}
        )
        self.assertEqual(len(SURFACE_PANEL_PROMPTS), 100)
        self.assertEqual(len(SOLID_PROMPTS), 100)
        self.assertEqual(len(METAL_OPENWORK_PROMPTS), 100)
        surf = " ".join(SURFACE_PANEL_PROMPTS).lower()
        solid = " ".join(SOLID_PROMPTS).lower()
        metal = " ".join(METAL_OPENWORK_PROMPTS).lower()
        for p in SURFACE_PANEL_PROMPTS:
            pl = p.lower()
            self.assertTrue("street" in pl or "sidewalk" in pl, msg=p)
            self.assertTrue(
                any(k in pl for k in ("glass", "frosted", "metal", "privacy", "panel")),
                msg=p,
            )
            self.assertTrue(
                "baluster" not in pl
                or "no baluster" in pl
                or "not baluster" in pl
                or "no open baluster" in pl,
                msg=p,
            )
        self.assertTrue("frame" in surf)
        self.assertTrue("panel" in surf)
        for p in SOLID_PROMPTS:
            pl = p.lower()
            self.assertTrue("street" in pl or "sidewalk" in pl, msg=p)
            self.assertTrue(
                any(k in pl for k in ("solid", "parapet", "waist wall", "mass", "opaque")),
                msg=p,
            )
            self.assertTrue(
                any(
                    k in pl
                    for k in (
                        "no opening",
                        "no openings",
                        "without gaps",
                        "opaque",
                        "mass wall",
                        "wall-like",
                        "waist wall",
                        "solid",
                    )
                ),
                msg=p,
            )
        self.assertTrue(
            any(k in solid for k in ("concrete", "stone", "plaster", "stucco", "mortar"))
        )
        for p in METAL_OPENWORK_PROMPTS:
            pl = p.lower()
            self.assertTrue("street" in pl or "sidewalk" in pl, msg=p)
            self.assertTrue(
                any(
                    k in pl
                    for k in (
                        "metal",
                        "iron",
                        "steel",
                        "wrought",
                        "filigree",
                        "lattice",
                        "picket",
                    )
                ),
                msg=p,
            )
            self.assertTrue(
                any(
                    k in pl
                    for k in ("open", "opening", "openings", "gap", "gaps", "see-through")
                ),
                msg=p,
            )
        self.assertTrue("open" in metal or "gap" in metal)
        self.assertEqual(PROMPT_SETS["surface_panel"]["prefix"], "flux2_surface_")
        self.assertEqual(PROMPT_SETS["solid"]["prefix"], "flux2_solid_")
        self.assertEqual(PROMPT_SETS["metal"]["prefix"], "flux2_metal_")

    def test_next_index_skips_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / f"{DEFAULT_PREFIX}000.png").touch()
            (d / f"{DEFAULT_PREFIX}001.png").touch()
            self.assertEqual(_next_index(d, DEFAULT_PREFIX), 2)

    def test_cpu_first_load_avoids_immediate_cuda(self) -> None:
        src = inspect.getsource(gen._load_pipeline_cpu_first)
        self.assertIn('device_map="cpu"', src)
        load_src = inspect.getsource(gen._load_pipeline)
        self.assertIn("_load_pipeline_cpu_first", load_src)
        self.assertIn("enable_model_cpu_offload", load_src)


if __name__ == "__main__":
    unittest.main()
