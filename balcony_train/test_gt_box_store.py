import tempfile
import unittest
from pathlib import Path

from balcony_train.gt_box_store import clean_box, load_gt, make_row, prefill_boxes, save_gt


class GtBoxStoreTest(unittest.TestCase):
    def test_clean_box_sorts_clips_and_drops_tiny(self):
        self.assertEqual(clean_box([50, 40, -5, 10.4], 100, 100), [0, 10, 50, 40])
        self.assertEqual(clean_box([90, 90, 130, 140], 100, 100), [90, 90, 100, 100])
        self.assertIsNone(clean_box([10, 10, 11, 50], 100, 100))

    def test_lsaa_prefill_uses_pipeline_boxes(self):
        cache = {"France_x": {"raw": [[0, 0, 5, 5]], "filtered": [[10, 10, 40, 30]]}}
        self.assertEqual(prefill_boxes("France_x", (100, 100), cache), ("pipeline", [[10, 10, 40, 30]]))
        self.assertEqual(prefill_boxes("UK_missing", (100, 100), cache), ("pipeline", []))

    def test_save_load_round_trip_and_edited_flag(self):
        same = make_row("a", "a.jpg", (100, 80), [[5, 5, 20, 20]], "pipeline", [[5, 5, 20, 20]])
        changed = make_row("b", "b.jpg", (100, 80), [], "cmp_label", [[1, 1, 9, 9]])
        self.assertFalse(same["edited"])
        self.assertTrue(changed["edited"])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "gt.jsonl"
            save_gt(path, {"b": changed, "a": same})
            rows = load_gt(path)
            self.assertEqual(list(rows), ["a", "b"])
            self.assertEqual(rows["b"]["boxes"], [])
            self.assertEqual(rows["a"]["boxes"], [[5, 5, 20, 20]])
            self.assertEqual(rows["a"]["ignore"], [])

    def test_ignore_regions_saved_apart_and_mark_edited(self):
        row = make_row("c", "c.jpg", (100, 80), [[5, 5, 20, 20]], "pipeline", [[5, 5, 20, 20]],
                       ignore=[[90, 0, 100, 10]])
        self.assertEqual(row["boxes"], [[5, 5, 20, 20]])
        self.assertEqual(row["ignore"], [[90, 0, 100, 10]])
        self.assertTrue(row["edited"])


if __name__ == "__main__":
    unittest.main()
