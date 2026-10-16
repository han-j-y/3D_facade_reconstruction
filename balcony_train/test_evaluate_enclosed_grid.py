"""GPU-free tests for enclosed floor×bay scoring."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from balcony_train.evaluate_enclosed_grid import (  # noqa: E402
    cells_from_boxes,
    format_report,
    render_compare,
    score_photo,
    select_labels,
    summarize,
)


class EnclosedGridEvalTests(unittest.TestCase):
    def test_box_snaps_to_covered_floor_and_bay(self) -> None:
        floor_y = {0: (0, 100), 1: (100, 200)}
        bay_x = {0: (0, 100), 1: (100, 200)}
        cells = cells_from_boxes([[10, 5, 90, 180]], floor_y, bay_x)
        self.assertEqual(cells, {(0, 0), (1, 0)})

    def test_empty_label_is_correct_only_when_prediction_is_empty(self) -> None:
        none = score_photo([], set())
        self.assertFalse(none["has_enclosure"])
        self.assertTrue(none["detected"])
        self.assertTrue(none["exact"])
        false = score_photo([], {(1, 0)})
        self.assertFalse(false["detected"])
        self.assertFalse(false["exact"])

    def test_detect_allows_a_partial_cell_hit(self) -> None:
        row = score_photo([(1, 0), (2, 0)], {(1, 0)})
        self.assertTrue(row["detected"])
        self.assertFalse(row["exact"])
        self.assertEqual(row["fn"], 1)

    def test_report_counts_detect_and_exact_cells(self) -> None:
        rows = [
            score_photo([(1, 0)], {(1, 0)}),
            score_photo([(2, 1), (3, 1)], {(2, 1)}),
            score_photo([], set()),
        ]
        for i, row in enumerate(rows):
            row["stem"] = f"cmp_{i}"
        summary = summarize(rows)
        self.assertEqual(summary["enclosure"]["detect"], 2)
        self.assertEqual(summary["enclosure"]["bay_floor"], 1)
        self.assertEqual(summary["none"]["detect"], 1)
        self.assertEqual(summary["none"]["bay_floor"], 1)
        text = format_report(summary, rows)
        self.assertIn("enclosure n=2  detect=2/2  detect_acc=1.000  bay_floor=1/2  bay_floor_acc=0.500", text)
        self.assertIn("none n=1  detect=1/1  detect_acc=1.000  bay_floor=1/1  bay_floor_acc=1.000", text)
        self.assertIn("cmp_1", text)
        self.assertNotIn("cmp_0", text)

    def test_stems_keeps_named_labels_and_splits_commas(self) -> None:
        labels = {"cmp_x0005": [(1, 0)], "cmp_x0134": [], "cmp_x0006": [(2, 1)]}
        picked, missing = select_labels(labels, ["cmp_x0005,", "cmp_x0134"])
        self.assertEqual(set(picked), {"cmp_x0005", "cmp_x0134"})
        self.assertEqual(picked["cmp_x0134"], [])
        self.assertEqual(missing, [])
        picked, missing = select_labels(labels, ["cmp_x0005, cmp_x9999"])
        self.assertEqual(set(picked), {"cmp_x0005"})
        self.assertEqual(missing, ["cmp_x9999"])
        untouched, missing = select_labels(labels, None)
        self.assertEqual(set(untouched), set(labels))
        self.assertEqual(missing, [])

    def test_compare_colors_label_prediction_and_overlap(self) -> None:
        photo = Image.new("RGB", (200, 100), (180, 180, 180))
        floor_y = {0: (0, 100)}
        bay_x = {0: (0, 100), 1: (100, 200)}
        painted = render_compare(photo, floor_y, bay_x, {(0, 0)}, {(0, 1)})
        blue = painted.getpixel((50, 50))
        red = painted.getpixel((150, 50))
        self.assertGreater(blue[2], blue[0])
        self.assertGreater(red[0], red[2])
        both = render_compare(photo, floor_y, bay_x, {(0, 0)}, {(0, 0)})
        purple = both.getpixel((50, 50))
        self.assertGreater(purple[0], 80)
        self.assertGreater(purple[2], 80)


if __name__ == "__main__":
    unittest.main()
