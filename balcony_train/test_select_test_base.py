"""Tests for the Base test-set photo selection (no images needed)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.select_test_base import CLASSES, crop_classes, photo_counts, select  # noqa: E402


def _rec(path, kind="open_work", material="metal", floor=None, structure=None, enclosure="open"):
    return {"path": path, "kind": kind, "material": material, "floor_shape": floor,
            "enclosure": enclosure, "skin": None, "structure": structure}


class CountTests(unittest.TestCase):
    def test_triangle_counts_as_rectangle(self):
        self.assertIn("rectangle", crop_classes(_rec("a.png", floor="triangle")))

    def test_material_only_on_open_work(self):
        self.assertEqual(crop_classes(_rec("a.png", kind="solid", material="metal")), ["solid"])

    def test_enclosed_skipped(self):
        self.assertEqual(crop_classes(_rec("a.png", enclosure="enclosed")), [])

    def test_juliet_without_floor_is_not_unlabeled_floor(self):
        recs = {"x": _rec("cmp_b0001_pbal_000.png", structure="juliet"),
                "y": _rec("cmp_b0001_pbal_001.png")}
        p = photo_counts(recs)["cmp_b0001"]
        self.assertEqual(p["no_floor"], 1)
        self.assertEqual(p["no_structure"], 1)
        self.assertEqual(p["counts"]["juliet"], 1)

    def test_crops_without_stem_skipped(self):
        self.assertEqual(photo_counts({"x": _rec("0003.png")}), {})


def _photos(n=40):
    """Synthetic photos: every fourth carries the rare classes."""
    recs = {}
    for i in range(1, n + 1):
        stem = f"cmp_b{i:04d}"
        for j in range(3):
            if i % 4 == 0:
                kinds = [("surface_panel", None, "circle", "projecting"),
                         ("solid", None, "rectangle", "juliet"),
                         ("open_work", "masonry", "circle", "projecting")]
            else:
                kinds = [("open_work", "metal", "rectangle", "projecting")] * 3
            k, m, f, s = kinds[j]
            name = f"{stem}_pbal_{j:03d}.png"
            recs[name] = _rec(name, kind=k, material=m, floor=f, structure=s)
    return photo_counts(recs)


class SelectTests(unittest.TestCase):
    def setUp(self):
        self.photos = _photos()
        self.oriel = {f"cmp_b{i:04d}" for i in range(1, 15)}

    def test_oriel_count_and_targets(self):
        chosen = select(self.photos, self.oriel, n_oriel=5, target=6)
        self.assertEqual(sum(s in self.oriel for s in chosen), 5)
        tot = {c: sum(self.photos[s]["counts"][c] for s in chosen) for c in CLASSES}
        for c in CLASSES:
            self.assertGreaterEqual(tot[c], 6, c)

    def test_deterministic(self):
        a = select(self.photos, self.oriel, n_oriel=5, target=6)
        b = select(self.photos, self.oriel, n_oriel=5, target=6)
        self.assertEqual(a, b)

    def test_too_few_oriels(self):
        with self.assertRaises(ValueError):
            select(self.photos, {"cmp_b0001"}, n_oriel=2, target=6)


if __name__ == "__main__":
    unittest.main()
