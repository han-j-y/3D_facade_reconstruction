"""Discrete balcony_view matches the catalog vote axes (no Blender)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from heuristic_ir import balcony_view, ir_to_tokens  # noqa: E402


class BalconyViewTests(unittest.TestCase):
    def test_metal_alias_and_no_opening(self) -> None:
        view = balcony_view(
            {
                "structure": "projecting",
                "enclosure": "open",
                "floor": {"shape": "rectangle"},
                "railing": {"kind": "metal", "height": 1.1},
                "supports": {"count": 0},
                "opening": "door",
            }
        )
        self.assertEqual(view["railing_kind"], "baluster")
        self.assertNotIn("opening", view)
        self.assertNotIn("opening=", " ".join(ir_to_tokens({"enclosure": "open", "railing": {"kind": "metal"}, "floor": {}})))

    def test_enclosed_omits_railing(self) -> None:
        view = balcony_view(
            {
                "structure": "projecting",
                "enclosure": "enclosed",
                "floor": {"shape": "rectangle"},
                "railing": {"kind": "glass"},
                "supports": {"count": 0},
            }
        )
        self.assertNotIn("railing_kind", view)
        self.assertEqual(view["enclosure"], "enclosed")


if __name__ == "__main__":
    unittest.main()
