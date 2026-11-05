"""Tests for all-balcony SAM3 crop helpers (no GPU)."""

from __future__ import annotations

import unittest

from balcony_train.crop_balconies_sam3 import crop_output_name, sort_records


class CropBalconiesSam3HelperTests(unittest.TestCase):
    def test_crop_output_name(self) -> None:
        self.assertEqual(crop_output_name("cmp_b0001", 0), "cmp_b0001_bal_000.png")
        self.assertEqual(crop_output_name("cmp_b0001", 12), "cmp_b0001_bal_012.png")

    def test_sort_records_score_then_position(self) -> None:
        records = [
            {"score": 0.5, "box_xyxy": [10, 20, 30, 40]},
            {"score": 0.9, "box_xyxy": [0, 50, 10, 60]},
            {"score": 0.5, "box_xyxy": [1, 10, 5, 15]},
        ]
        ordered = sort_records(records)
        self.assertAlmostEqual(ordered[0]["score"], 0.9)
        self.assertEqual(ordered[1]["box_xyxy"], [1, 10, 5, 15])
        self.assertEqual(ordered[2]["box_xyxy"], [10, 20, 30, 40])


if __name__ == "__main__":
    unittest.main()
