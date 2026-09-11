"""Tests for FLUX.2 generator helpers (no GPU / diffusers required)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.generate_flux2 import DEFAULT_PREFIX, _next_index  # noqa: E402
from balcony_train.prompts_masonry import (  # noqa: E402
    MASONRY_OPENWORK_PROMPTS,
    NEGATIVE_PROMPT,
)


class GenerateFlux2HelperTests(unittest.TestCase):
    def test_prompts_non_empty(self) -> None:
        self.assertGreaterEqual(len(MASONRY_OPENWORK_PROMPTS), 8)
        self.assertTrue("photo" in MASONRY_OPENWORK_PROMPTS[0].lower())
        self.assertTrue(
            any("masonry" in p.lower() or "brick" in p.lower() for p in MASONRY_OPENWORK_PROMPTS)
        )
        self.assertLess(len(NEGATIVE_PROMPT), 120)

    def test_next_index_skips_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / f"{DEFAULT_PREFIX}000.png").touch()
            (d / f"{DEFAULT_PREFIX}001.png").touch()
            self.assertEqual(_next_index(d, DEFAULT_PREFIX), 2)


if __name__ == "__main__":
    unittest.main()
