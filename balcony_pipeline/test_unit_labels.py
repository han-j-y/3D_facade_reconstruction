import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from unit_labels import ExternalUnitLabels, UnitLabelPredictor, normalize_label  # noqa: E402

ROWS = [
    {"unit_index": 0, "box_xyxy": [0, 0, 10, 10], "kind": "open_work",
     "material": "metal", "floor_shape": "rectangle", "structure": "juliet"},
    {"unit_index": 1, "box_xyxy": [20, 0, 30, 10], "kind": "solid",
     "material": "masonry", "floor_shape": "circle", "structure": "projecting"},
]


class TestUnitLabels(unittest.TestCase):
    def test_lookup_by_index(self):
        lab = ExternalUnitLabels(ROWS)
        p = UnitLabelPredictor(lab, 0, [0, 0, 10, 10]).predict_full(None)
        self.assertEqual(p["kind"], "open_work")
        self.assertEqual(p["material"], "metal")
        self.assertEqual(p["structure"], "juliet")
        self.assertIsNone(p["enclosure"])

    def test_material_cleared_unless_open_work(self):
        p = ExternalUnitLabels(ROWS).predict_for(1, [20, 0, 30, 10])
        self.assertIsNone(p["material"])

    def test_box_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ExternalUnitLabels(ROWS).predict_for(0, [0, 0, 11, 10])

    def test_check_units_counts(self):
        lab = ExternalUnitLabels(ROWS)
        lab.check_units([[0, 0, 10, 10], [20, 0, 30, 10]])
        with self.assertRaises(ValueError):
            lab.check_units([[0, 0, 10, 10]])

    def test_vocabulary(self):
        bad = dict(ROWS[0], kind="glass")
        with self.assertRaises(ValueError):
            normalize_label(bad)
        with self.assertRaises(ValueError):
            normalize_label(dict(ROWS[0], material=None))
        with self.assertRaises(ValueError):
            normalize_label(dict(ROWS[0], structure="rooftop"))

    def test_juliet_floor(self):
        jul = dict(ROWS[0], floor_shape=None)
        self.assertIsNone(normalize_label(jul)["floor_shape"])
        self.assertIsNone(normalize_label(jul, strict=True)["floor_shape"])
        with self.assertRaises(ValueError):
            normalize_label(ROWS[0], strict=True)  # juliet with a floor
        with self.assertRaises(ValueError):
            normalize_label(dict(ROWS[1], floor_shape=None))  # projecting needs a floor

    def test_strict_no_triangle(self):
        tri = dict(ROWS[1], floor_shape="triangle")
        self.assertEqual(normalize_label(tri)["floor_shape"], "triangle")
        with self.assertRaises(ValueError):
            normalize_label(tri, strict=True)


if __name__ == "__main__":
    unittest.main()
