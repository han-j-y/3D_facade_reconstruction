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
from balcony_train.generate_flux2 import (  # noqa: E402
    DEFAULT_PREFIX,
    PROMPT_SETS,
    _next_index,
    iter_prompt_outputs,
)
from balcony_train.prompts_enclosure import (  # noqa: E402
    ENCLOSED_PROMPTS,
    HALF_ENCLOSED_PROMPTS,
)
from balcony_train.prompts_floor_shape import (  # noqa: E402
    HEXAGON_PROMPTS,
    TRAPEZOID_PROMPTS,
    TRIANGLE_PROMPTS,
    prefixes_for,
)
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
            set(PROMPT_SETS),
            {
                "masonry",
                "metal",
                "surface_panel",
                "solid",
                "triangle",
                "trapezoid",
                "hexagon",
                "half_enclosed",
                "enclosed",
            },
        )
        self.assertEqual(len(SURFACE_PANEL_PROMPTS), 50)
        self.assertEqual(len(set(SURFACE_PANEL_PROMPTS)), 50)
        self.assertEqual(len(SOLID_PROMPTS), 50)
        self.assertEqual(len(METAL_OPENWORK_PROMPTS), 100)
        surf = " ".join(SURFACE_PANEL_PROMPTS).lower()
        solid = " ".join(SOLID_PROMPTS).lower()
        metal = " ".join(METAL_OPENWORK_PROMPTS).lower()
        self.assertIn("103,104,99", surf)
        self.assertTrue(any(k in surf for k in ("white", "grey", "glass", "dark red", "light blue")))
        self.assertTrue("plant" in surf or "pedestrian" in surf or "tree" in surf)
        weak = sum(1 for p in SURFACE_PANEL_PROMPTS if "weak seams" in p.lower())
        self.assertEqual(weak, 17)
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

    def test_floor_plan_prompt_sets_rotate_kind_prefixes(self) -> None:
        banks = {
            "triangle": TRIANGLE_PROMPTS,
            "trapezoid": TRAPEZOID_PROMPTS,
            "hexagon": HEXAGON_PROMPTS,
        }
        cues = {
            "triangle": ("triangular", "apex", "point"),
            "trapezoid": ("trapezoid", "75"),
            "hexagon": ("hexagon", "chamfer", "30"),
        }
        for shape, prompts in banks.items():
            prefixes = prefixes_for(shape)
            self.assertEqual(PROMPT_SETS[shape]["prompts"], prompts)
            self.assertEqual(PROMPT_SETS[shape]["prefixes"], prefixes)
            self.assertEqual(len(prompts), len(prefixes))
            self.assertEqual(len(set(prompts)), len(prefixes))
            joined = " ".join(prompts).lower()
            for cue in cues[shape]:
                self.assertIn(cue, joined, msg=shape)
            self.assertEqual(len(prompts), 50)
            if shape in ("triangle", "trapezoid"):
                self.assertEqual(prefixes[0], f"flux2_{shape}_metal_")
                self.assertEqual(prefixes[1], f"flux2_{shape}_masonry_")
                self.assertEqual(prefixes[2], f"flux2_{shape}_solid_")
                self.assertFalse(any("surface" in name for name in prefixes))
                self.assertIn("floor", joined)
                self.assertNotIn("privacy panel", joined)
            if shape == "hexagon":
                self.assertEqual(prefixes[3], "flux2_hexagon_surface_")
            if shape == "triangle":
                self.assertIn("the floor shape is a triangle", joined)
            if shape == "trapezoid":
                self.assertIn("the floor shape is a trapezoid", joined)
                self.assertIn("floor plan", joined)
                self.assertIn("only the", joined)
            for prompt in prompts:
                lower = prompt.lower()
                self.assertTrue(lower.startswith("a long flat facade"), msg=prompt)
                self.assertNotIn("corner", lower, msg=prompt)
                self.assertIn("street", lower, msg=prompt)
                self.assertIn("top of the floor is visible", lower, msg=prompt)
            with tempfile.TemporaryDirectory() as tmp:
                planned = iter_prompt_outputs(
                    Path(tmp), len(prefixes), prompts, f"flux2_{shape}_", prefixes
                )
            names = [path.name for path, _prompt in planned]
            counters: dict[str, int] = {}
            expected: list[str] = []
            for prefix in prefixes:
                index = counters.get(prefix, 0)
                expected.append(f"{prefix}{index:03d}.png")
                counters[prefix] = index + 1
            self.assertEqual(names, expected)

    def test_enclosure_prompt_sets_have_two_each(self) -> None:
        self.assertEqual(PROMPT_SETS["half_enclosed"]["prompts"], HALF_ENCLOSED_PROMPTS)
        self.assertEqual(PROMPT_SETS["enclosed"]["prompts"], ENCLOSED_PROMPTS)
        self.assertEqual(len(HALF_ENCLOSED_PROMPTS), 50)
        self.assertEqual(len(ENCLOSED_PROMPTS), 50)
        self.assertEqual(len(set(HALF_ENCLOSED_PROMPTS)), 50)
        self.assertEqual(len(set(ENCLOSED_PROMPTS)), 50)
        half = " ".join(HALF_ENCLOSED_PROMPTS).lower()
        closed = " ".join(ENCLOSED_PROMPTS).lower()
        self.assertIn("column", half)
        self.assertIn("open to the outside air", half)
        self.assertIn("no glass walls", half)
        self.assertTrue("window" in closed or "windows" in closed)
        self.assertIn("not open to the outside air", closed)
        for prompt in (*HALF_ENCLOSED_PROMPTS, *ENCLOSED_PROMPTS):
            lower = prompt.lower()
            self.assertTrue(lower.startswith("a long flat facade"), msg=prompt)
            self.assertNotIn("corner", lower)
            self.assertIn("beyond the exterior wall line", lower)
            self.assertIn("not recessed", lower)
        self.assertNotIn("loggia", half)
        self.assertNotIn("curved bay", closed)
        self.assertTrue(PROMPT_SETS["triangle"]["stay_unlabeled"])
        self.assertTrue(PROMPT_SETS["enclosed"]["stay_unlabeled"])

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
