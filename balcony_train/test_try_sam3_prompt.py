"""Tests for SAM3 prompt-set helpers (no GPU / transformers required)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.try_sam3_prompt import (  # noqa: E402
    PROMPT_SETS,
    keep_enclosure_box,
    merge_boxes,
    resolve_prompts,
)


class TrySam3PromptTests(unittest.TestCase):
    def test_enclosure_set_targets_volume_not_pane(self) -> None:
        self.assertEqual(
            PROMPT_SETS["enclosure"],
            (
                "glass enclosed balcony",
                "bay window",
                "colonnade",
            ),
        )

    def test_projecting_set_avoids_window_and_balcony(self) -> None:
        self.assertEqual(
            PROMPT_SETS["projecting"],
            (
                "projecting bay",
                "oriel",
                "facade projection",
                "protruding wall",
                "canted bay",
            ),
        )
        for phrase in PROMPT_SETS["projecting"]:
            self.assertNotIn("window", phrase)
            self.assertNotIn("balcony", phrase)
            self.assertNotIn("glass", phrase)
        self.assertIn("bay window", PROMPT_SETS["enclosed_search"])
        self.assertIn("glass enclosed balcony", PROMPT_SETS["enclosed_search"])

    def test_enclosed_search2_covers_gallery_and_projection(self) -> None:
        phrases = PROMPT_SETS["enclosed_search2"]
        self.assertIn("enclosed gallery", phrases)
        self.assertIn("glazed gallery", phrases)
        self.assertIn("closed balcony", phrases)
        self.assertIn("oriel window", phrases)
        self.assertEqual(len(phrases), 21)

    def test_one_floor_set_uses_volume_phrases(self) -> None:
        self.assertEqual(
            PROMPT_SETS["one_floor"],
            ("enclosed balcony", "closed balcony", "glazed balcony"),
        )
        iw, ih = 800, 1000
        # Shorter than one storey (~10% of height).
        self.assertFalse(
            keep_enclosure_box([40, 100, 200, 170], iw=iw, ih=ih, min_height_frac=0.10)
        )
        # One-storey gallery / bay (taller than 10%, not a tiny square).
        self.assertTrue(
            keep_enclosure_box([40, 80, 280, 280], iw=iw, ih=ih, min_height_frac=0.10)
        )

    def test_resolve_prompts_defaults_to_enclosure(self) -> None:
        self.assertEqual(resolve_prompts(None, None), list(PROMPT_SETS["enclosure"]))

    def test_resolve_prompts_combines_set_and_extra(self) -> None:
        self.assertEqual(
            resolve_prompts("glass", ["loggia"]),
            ["glass enclosed balcony", "loggia"],
        )

    def test_merge_keeps_larger_box(self) -> None:
        small = {"score": 0.9, "box_xyxy": [10, 10, 30, 30], "prompt": 0}
        large = {"score": 0.4, "box_xyxy": [0, 0, 80, 80], "prompt": 1}
        other = {"score": 0.5, "box_xyxy": [200, 200, 240, 240], "prompt": 2}
        merged = merge_boxes([small, large, other], overlap=0.6)
        boxes = [tuple(r["box_xyxy"]) for r in merged]
        self.assertEqual(boxes, [(0, 0, 80, 80), (200, 200, 240, 240)])

    def test_keep_enclosure_box_drops_sashes(self) -> None:
        iw, ih = 800, 1200
        # Ordinary tall sash.
        self.assertFalse(keep_enclosure_box([10, 10, 50, 180], iw=iw, ih=ih))
        # Narrow relative to the facade (below 8% width).
        self.assertFalse(keep_enclosure_box([10, 100, 60, 220], iw=iw, ih=ih))
        # Small compact window.
        self.assertFalse(keep_enclosure_box([10, 10, 120, 130], iw=iw, ih=ih))
        # Handrail / cornice strip.
        self.assertFalse(keep_enclosure_box([40, 500, 500, 540], iw=iw, ih=ih))
        # Wide one-storey gallery / loggia (was dropped when max_flat_aspect=2.8).
        self.assertTrue(keep_enclosure_box([40, 500, 520, 640], iw=iw, ih=ih))
        # Stacked oriel (taller than a sash, wide enough to be a bay).
        self.assertTrue(keep_enclosure_box([20, 100, 180, 700], iw=iw, ih=ih))
        # Wide gallery / oriel (one storey, not a rail).
        self.assertTrue(keep_enclosure_box([40, 400, 400, 620], iw=iw, ih=ih))


if __name__ == "__main__":
    unittest.main()
