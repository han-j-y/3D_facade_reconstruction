"""Structure from DA3 depth: protrusion against the wall beside the box."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from structure_depth import protrusion, structure_from_depth  # noqa: E402


def wall(depth=10.0, h=100, w=200):
    return np.full((h, w), depth, dtype=np.float32)


class StructureDepthTests(unittest.TestCase):
    def test_box_in_front_of_wall_is_projecting(self) -> None:
        d = wall()
        d[40:60, 80:120] = 9.0  # 1 m in front
        self.assertEqual(structure_from_depth(d, [80, 40, 120, 60], 0.10), "projecting")
        self.assertAlmostEqual(protrusion(d, [80, 40, 120, 60])["prot_m"], 1.0)

    def test_flush_or_recessed_is_juliet(self) -> None:
        d = wall()
        self.assertEqual(structure_from_depth(d, [80, 40, 120, 60], 0.10), "juliet")
        d[40:60, 80:120] = 10.8  # inset: behind the wall
        self.assertEqual(structure_from_depth(d, [80, 40, 120, 60], 0.10), "juliet")

    def test_recess_above_and_below_is_ignored(self) -> None:
        d = wall()
        d[:, 80:120] = 11.0  # deep recess column, box at the recess wall depth
        self.assertEqual(structure_from_depth(d, [80, 40, 120, 60], 0.10), "juliet")

    def test_larger_side_wins_and_edge_uses_one_side(self) -> None:
        d = wall()
        d[:, 125:] = 9.0  # right neighbour sticks out
        d[40:60, 80:120] = 9.5
        self.assertEqual(protrusion(d, [80, 40, 120, 60])["side"], "left")
        self.assertEqual(protrusion(d, [0, 40, 40, 60])["side"], "right")

    def test_off_or_missing(self) -> None:
        self.assertIsNone(structure_from_depth(None, [0, 0, 10, 10], 0.10))
        self.assertIsNone(structure_from_depth(wall(), [80, 40, 120, 60], 0.0))
        self.assertIsNone(structure_from_depth(wall(w=40), [0, 40, 40, 60], 0.10))


if __name__ == "__main__":
    unittest.main()
