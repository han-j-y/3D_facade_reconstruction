"""Connected enclosed cells and their union crop, without reading photos."""

from __future__ import annotations

import unittest

from balcony_train.export_enclosed_crops import (
    connected_components,
    is_eval_stem,
    union_box,
)


class ExportEnclosedCropTests(unittest.TestCase):
    def test_components_join_sides_and_keep_diagonals_apart(self) -> None:
        groups = connected_components([(0, 0), (0, 1), (1, 1), (3, 3), (2, 0)])
        self.assertEqual(groups[0], [(0, 0), (0, 1), (1, 1)])
        self.assertEqual(groups[1], [(2, 0)])
        self.assertEqual(groups[2], [(3, 3)])

    def test_union_box_adds_margin_and_clamps(self) -> None:
        floor_y = {0: (0, 100), 1: (100, 200)}
        bay_x = {0: (0, 50), 1: (50, 150)}
        box = union_box([(0, 0), (0, 1)], floor_y, bay_x, 160, 210, margin=0.10)
        self.assertEqual(box, (0, 0, 160, 110))
        inner = union_box([(1, 1)], floor_y, bay_x, 400, 400, margin=0.0)
        self.assertEqual(inner, (50, 100, 150, 200))

    def test_eval_stems_are_refused(self) -> None:
        self.assertTrue(is_eval_stem("cmp_x0004"))
        self.assertFalse(is_eval_stem("cmp_b0007"))


if __name__ == "__main__":
    unittest.main()
