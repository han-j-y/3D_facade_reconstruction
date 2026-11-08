"""Soft type vote for kind / material / floor (vote.soft_type_labels)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from vote import apply_soft_labels, soft_type_labels  # noqa: E402


def probs(kind, material=(0.5, 0.5), floor=(1.0, 0.0, 0.0)):
    return {"kind": list(kind), "material": list(material), "floor": list(floor), "structure": [0.5, 0.5]}


class SoftVoteTests(unittest.TestCase):
    def test_confident_minority_wins_over_unsure_majority(self) -> None:
        ps = [probs((0.45, 0.55, 0)), probs((0.40, 0.60, 0)), probs((0.95, 0.05, 0)), probs((0.90, 0.10, 0))]
        self.assertEqual(soft_type_labels(ps, [0, 1, 2, 3])["kind"], "open_work")

    def test_material_weighted_by_open_work(self) -> None:
        # open work wins on the mean; the two panel members' masonry votes barely count for the material
        ps = [probs((0.9, 0.1, 0), material=(0.8, 0.2)), probs((0.4, 0.6, 0), material=(0.0, 1.0)),
              probs((0.4, 0.6, 0), material=(0.0, 1.0)), probs((0.9, 0.1, 0), material=(0.7, 0.3))]
        out = soft_type_labels(ps, [])
        self.assertEqual(out["kind"], "open_work")
        self.assertEqual(out["material"], "metal")

    def test_no_material_off_open_work(self) -> None:
        out = soft_type_labels([probs((0.1, 0.1, 0.8))], [0])
        self.assertEqual(out["kind"], "solid")
        self.assertIsNone(out["material"])

    def test_floor_over_voters_only_triangle_as_rectangle(self) -> None:
        ps = [probs((1, 0, 0), floor=(0.0, 0.0, 1.0)), probs((1, 0, 0), floor=(0.3, 0.4, 0.3))]
        self.assertEqual(soft_type_labels(ps, [1])["floor"], "rectangle")
        self.assertEqual(soft_type_labels(ps, [0])["floor"], "circle")
        self.assertIsNone(soft_type_labels(ps, [])["floor"])

    def test_missing_probs_falls_back(self) -> None:
        self.assertIsNone(soft_type_labels([probs((1, 0, 0)), None], [0]))

    def test_apply_keeps_other_fields(self) -> None:
        ir = {"structure": "juliet", "railing": {"kind": "solid", "height": 1.1}, "floor": {"shape": "circle", "params": {"w": 1}}}
        out = apply_soft_labels(ir, {"kind": "open_work", "material": "metal", "floor": None})
        self.assertEqual(out["railing"], {"kind": "open_work", "height": 1.1, "material": "metal"})
        self.assertEqual(out["floor"]["shape"], "circle")
        self.assertEqual(ir["railing"]["kind"], "solid")


if __name__ == "__main__":
    unittest.main()
