"""GPU-free tests for enclosed floor×bay scoring."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from balcony_train.score_enclosed_plan import copy_plan_mistakes  # noqa: E402
from balcony_train.evaluate_enclosed_grid import (  # noqa: E402
    apply_plan_score,
    cells_from_boxes,
    format_report,
    load_pred_cells,
    render_compare,
    score_photo,
    select_labels,
    summarize,
    vote_plan,
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
        self.assertIn("total n=3  detect=3/3  detect_acc=1.000  bay_floor=2/3  bay_floor_acc=0.667", text)
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

    def test_vote_needs_a_predicted_cell_and_a_strict_majority(self) -> None:
        image = Image.new("RGB", (100, 100), (20, 20, 20))
        floor_y = {0: (0, 50), 1: (50, 100)}
        bay_x = {0: (0, 100)}
        calls: list[tuple[int, int]] = []

        def predict(crop: Image.Image) -> dict[str, str]:
            calls.append(crop.size)
            if len(calls) == 1:
                return {"floor_shape": "circle", "skin": "curtain_wall"}
            return {"floor_shape": "triangle", "skin": "curtain_wall"}

        shape, skin = vote_plan(image, {(1, 0), (0, 0)}, floor_y, bay_x, predict)
        self.assertEqual((shape, skin), ("rectangle", "curtain_wall"))
        self.assertEqual(len(calls), 2)
        self.assertEqual(vote_plan(image, set(), floor_y, bay_x, predict), (None, None))

    def test_triangle_matches_rectangle_and_a_miss_is_wrong(self) -> None:
        hit = score_photo([(1, 0)], {(1, 0)})
        apply_plan_score(hit, "triangle", "punched_window", "rectangle", "punched_window")
        self.assertTrue(hit["shape_ok"])
        self.assertTrue(hit["skin_ok"])
        missed = score_photo([(1, 0)], set())
        apply_plan_score(missed, "rectangle", "curtain_wall", None, None)
        self.assertFalse(missed["shape_ok"])
        self.assertFalse(missed["skin_ok"])
        pending = score_photo([(1, 0)], {(1, 0)})
        apply_plan_score(pending, None, None, "rectangle", "punched_window")
        self.assertFalse(pending["plan_labeled"])
        for i, row in enumerate((hit, missed)):
            row["stem"] = f"cmp_{i}"
            row["plan_labeled"] = True
        text = format_report(summarize([hit, missed]), [hit, missed])
        self.assertIn("shape 1/1  shape_acc=1.000  skin 1/1  skin_acc=1.000  no_cells=1", text)
        self.assertNotIn("rectangle  None", text)

    def test_pred_json_accepts_flat_and_nested_cells(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "preds.json"
            path.write_text(json.dumps({"cmp_x0005": [[1, 0]]}), encoding="utf-8")
            self.assertEqual(load_pred_cells(path), {"cmp_x0005": {(1, 0)}})
            path.write_text(
                json.dumps({"eval": {"preds": {"cmp_x0006": [[2, 1]]}}}),
                encoding="utf-8",
            )
            self.assertEqual(load_pred_cells(path), {"cmp_x0006": {(2, 1)}})

    def test_plan_mistakes_copy_floor_and_skin_separately(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            circle = root / "cmp_circle.jpg"
            triangle = root / "cmp_triangle.jpg"
            skin = root / "cmp_skin.jpg"
            for path in (circle, triangle, skin):
                Image.new("RGB", (8, 8), (10, 10, 10)).save(path)
            rows = []
            for photo, gt_shape, pred_shape, gt_skin, pred_skin in (
                (circle, "circle", "rectangle", "curtain_wall", "curtain_wall"),
                (triangle, "triangle", "rectangle", "punched_window", "punched_window"),
                (skin, "rectangle", "rectangle", "punched_window", "curtain_wall"),
            ):
                row = score_photo([(1, 0)], {(1, 0)})
                row["stem"] = photo.stem
                row["photo"] = str(photo)
                apply_plan_score(row, gt_shape, gt_skin, pred_shape, pred_skin)
                rows.append(row)
            n = copy_plan_mistakes(rows, root / "wrong", stamp="2026-10-19_155200")
            self.assertEqual(n, 2)
            run = root / "wrong" / "2026-10-19_155200"
            self.assertTrue((run / "floor" / "true-circle__pred-rectangle" / "cmp_circle.jpg").is_file())
            self.assertTrue(
                (run / "skin" / "true-punched_window__pred-curtain_wall" / "cmp_skin.jpg").is_file()
            )
            self.assertFalse(any(run.rglob("cmp_triangle.jpg")))
            text = (run / "mistakes.txt").read_text(encoding="utf-8")
            self.assertIn("floor\tcircle\trectangle\tcmp_circle.jpg", text)
            self.assertIn("skin\tpunched_window\tcurtain_wall\tcmp_skin.jpg", text)


if __name__ == "__main__":
    unittest.main()
