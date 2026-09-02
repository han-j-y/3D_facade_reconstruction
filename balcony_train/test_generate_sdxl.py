"""Tests for SDXL generator helpers (no GPU / diffusers required)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.generate_sdxl import DEFAULT_PREFIX, _next_index  # noqa: E402
from balcony_train.prompts_balcony import BALCONY_PROMPTS, NEGATIVE_PROMPT  # noqa: E402


class GenerateSdxlHelperTests(unittest.TestCase):
    def test_prompts_non_empty(self) -> None:
        self.assertGreaterEqual(len(BALCONY_PROMPTS), 5)
        self.assertTrue(BALCONY_PROMPTS[0].startswith("a photo of"))
        self.assertLess(len(BALCONY_PROMPTS[0]), 80)
        self.assertLess(len(NEGATIVE_PROMPT), 50)

    def test_next_index_skips_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / f"{DEFAULT_PREFIX}000.png").touch()
            (d / f"{DEFAULT_PREFIX}001.png").touch()
            self.assertEqual(_next_index(d, DEFAULT_PREFIX), 2)


if __name__ == "__main__":
    unittest.main()
