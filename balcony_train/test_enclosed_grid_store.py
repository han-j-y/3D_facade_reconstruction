"""Tests for enclosed floor×bay ground truth (no GUI)."""

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

from balcony_train.enclosed_grid_store import (  # noqa: E402
    GRID_FAILURE,
    GridRecord,
    bay_cells,
    cell_at,
    facade_stems_from_crop_labels,
    list_grid_stems,
    load_floor_bays,
    load_ignore,
    load_labels,
    load_records,
    normalize_cells,
    partial_bays,
    record_for_save,
    write_labels,
    write_records,
)


class EnclosedGridStoreTests(unittest.TestCase):
    def test_empty_cells_mean_saved_without_enclosure(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.jsonl"
            write_labels(path, {"cmp_b0008": []})
            loaded = load_labels(path)
            self.assertIn("cmp_b0008", loaded)
            self.assertEqual(loaded["cmp_b0008"], [])
            self.assertNotIn("cmp_b0007", loaded)

    def test_roundtrip_selected_cells(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.jsonl"
            write_labels(path, {"cmp_b0007": [(2, 0), (1, 0), (1, 0)]})
            loaded = load_labels(path)
            self.assertEqual(loaded["cmp_b0007"], [(1, 0), (2, 0)])
            row = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(row["cells"], [[1, 0], [2, 0]])

    def test_cell_at_uses_floor_and_bay_bands(self) -> None:
        floor_y = {0: (0, 100), 1: (100, 200)}
        bay_x = {0: (0, 50), 1: (80, 140)}
        self.assertEqual(cell_at(10, 20, floor_y, bay_x), (0, 0))
        self.assertEqual(cell_at(100, 150, floor_y, bay_x), (1, 1))
        self.assertIsNone(cell_at(60, 20, floor_y, bay_x))
        self.assertIsNone(cell_at(10, 250, floor_y, bay_x))

    def test_shared_edge_prefers_the_band_that_starts_there(self) -> None:
        floor_y = {0: (0, 100), 1: (100, 200)}
        bay_x = {0: (0, 40)}
        self.assertEqual(cell_at(10, 100, floor_y, bay_x), (1, 0))

    def test_list_grid_stems_skips_photos_without_dsl(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            photos = root / "photos"
            dsl = root / "dsl"
            photos.mkdir()
            (photos / "cmp_a.jpg").write_bytes(b"x")
            (photos / "cmp_b.jpg").write_bytes(b"x")
            (dsl / "cmp_a").mkdir(parents=True)
            (dsl / "cmp_a" / "facade_dsl.json").write_text("{}", encoding="utf-8")
            ready, missing = list_grid_stems(photos, dsl)
            self.assertEqual(ready, ["cmp_a"])
            self.assertEqual(missing, ["cmp_b"])

    def test_crop_labels_collapse_to_facade_stems(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"path": "cmp_x0002_bal_002.png", "kind": "open_work"}',
                        '{"path": "cmp_x0002_bal_004.png", "kind": "solid"}',
                        '{"path": "cmp_x0005_bal_001.png", "kind": "open_work"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                facade_stems_from_crop_labels(path),
                ["cmp_x0002", "cmp_x0005"],
            )

    def test_stem_filter_keeps_only_requested_facades(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            photos = root / "photos"
            dsl = root / "dsl"
            photos.mkdir()
            (photos / "cmp_a.jpg").write_bytes(b"x")
            (photos / "cmp_b.jpg").write_bytes(b"x")
            for stem in ("cmp_a", "cmp_b"):
                (dsl / stem).mkdir(parents=True)
                (dsl / stem / "facade_dsl.json").write_text("{}", encoding="utf-8")
            ready, missing = list_grid_stems(photos, dsl, stems=["cmp_b", "cmp_c"])
            self.assertEqual(ready, ["cmp_b"])
            self.assertEqual(missing, ["cmp_c"])

    def test_normalize_rejects_a_bare_number(self) -> None:
        with self.assertRaises(ValueError):
            normalize_cells([1])

    def test_normalize_accepts_a_set_of_cells(self) -> None:
        self.assertEqual(normalize_cells({(2, 0), (1, 0), (1, 0)}), [(1, 0), (2, 0)])
        saved = record_for_save({(1, 0), (2, 0)}, "circle", "curtain_wall", plan=True)
        self.assertEqual(saved.cells, [(1, 0), (2, 0)])
        self.assertEqual(saved.floor_shape, "circle")
        self.assertEqual(saved.skin, "curtain_wall")

    def test_plan_roundtrip_and_cells_only_view(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.jsonl"
            write_records(
                path,
                {
                    "cmp_x0005": GridRecord([(1, 0), (2, 0)], "triangle", "punched_window"),
                    "cmp_x0009": GridRecord([], "circle", "curtain_wall"),
                },
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn('"floor_shape": "triangle"', text)
            self.assertIn('"skin": "punched_window"', text)
            rows = [json.loads(line) for line in text.splitlines()]
            empty = next(row for row in rows if row["stem"] == "cmp_x0009")
            self.assertEqual(set(empty), {"stem", "cells"})
            loaded = load_records(path)
            self.assertEqual(loaded["cmp_x0005"].floor_shape, "triangle")
            self.assertEqual(loaded["cmp_x0005"].skin, "punched_window")
            self.assertEqual(loaded["cmp_x0009"].cells, [])
            self.assertIsNone(loaded["cmp_x0009"].floor_shape)
            self.assertEqual(load_labels(path)["cmp_x0005"], [(1, 0), (2, 0)])

    def test_save_requires_both_plan_fields_and_keeps_them_when_plan_is_off(self) -> None:
        with self.assertRaises(ValueError):
            record_for_save([(1, 0)], "circle", None, plan=True)
        previous = GridRecord([(1, 0)], "circle", "curtain_wall")
        kept = record_for_save([(1, 0), (2, 0)], None, None, plan=False, previous=previous)
        self.assertEqual(kept.floor_shape, "circle")
        self.assertEqual(kept.skin, "curtain_wall")
        cleared = record_for_save([], "circle", "curtain_wall", plan=True)
        self.assertEqual(cleared.cells, [])
        self.assertIsNone(cleared.skin)

    def test_grid_failure_is_not_a_photo_with_no_enclosure(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.jsonl"
            write_records(
                path,
                {
                    "cmp_x0045": GridRecord([], status=GRID_FAILURE),
                    "cmp_x0009": GridRecord([]),
                },
            )
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            row = next(item for item in rows if item["stem"] == "cmp_x0045")
            self.assertEqual(row, {"stem": "cmp_x0045", "status": "grid_failure"})
            self.assertNotIn("cells", row)
            loaded = load_records(path)
            self.assertEqual(loaded["cmp_x0045"].status, GRID_FAILURE)
            self.assertEqual(loaded["cmp_x0045"].cells, [])
            self.assertNotIn("cmp_x0045", load_labels(path))
            self.assertEqual(load_labels(path)["cmp_x0009"], [])
            kept = record_for_save([], None, None, previous=loaded["cmp_x0045"])
            self.assertEqual(kept.status, GRID_FAILURE)
            labeled = record_for_save([(1, 0)], "rectangle", "punched_window", plan=True, previous=kept)
            self.assertIsNone(labeled.status)
            self.assertEqual(labeled.cells, [(1, 0)])

    def test_cell_only_write_does_not_invent_a_plan(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.jsonl"
            write_labels(path, {"cmp_b0007": [(1, 0)]})
            row = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(set(row), {"stem", "cells"})

    def test_ignore_cells_roundtrip_apart_from_cells(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.jsonl"
            rec = record_for_save(
                {(1, 2)}, "rectangle", "punched_window", plan=True, ignore={(1, 3), (0, 0)}
            )
            empty = record_for_save(set(), ignore={(2, 2)})
            plain = record_for_save({(0, 1)}, "circle", "curtain_wall", plan=True)
            write_records(path, {"a": rec, "b": empty, "c": plain})
            rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
            loaded = load_records(path)
            labels = load_labels(path)
            ignore = load_ignore(path)
        self.assertEqual(loaded["a"].cells, [(1, 2)])
        self.assertEqual(loaded["a"].ignore, [(0, 0), (1, 3)])
        self.assertEqual(loaded["b"].cells, [])
        self.assertEqual(loaded["b"].ignore, [(2, 2)])
        self.assertNotIn("ignore", rows[2])
        self.assertEqual(labels["b"], [])
        self.assertEqual(ignore, {"a": [(0, 0), (1, 3)], "b": [(2, 2)]})

    def test_floor_bays_group_union_columns(self) -> None:
        floor_y = {0: (0, 10), 1: (10, 20)}
        bay_x = {0: (0, 10), 1: (10, 20), 2: (20, 30)}
        meta = {"floor_bays": [
            [{"c0": 0, "c1": 1}, {"c0": 1, "c1": 3}],
            [{"c0": 0, "c1": 2}, {"c0": 2, "c1": 3}],
        ]}
        with TemporaryDirectory() as tmp:
            dsl = Path(tmp) / "facade_dsl.json"
            dsl.write_text(json.dumps({"meta": meta}), encoding="utf-8")
            fb = load_floor_bays(dsl, floor_y, bay_x)
            dsl.write_text(json.dumps({"meta": {}}), encoding="utf-8")
            plain = load_floor_bays(dsl, floor_y, bay_x)
        self.assertEqual(fb, {0: [(0, 1), (1, 3)], 1: [(0, 2), (2, 3)]})
        self.assertEqual(plain[0], [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(bay_cells((0, 2), fb), [(0, 1), (0, 2)])
        self.assertEqual(bay_cells((1, 0), fb), [(1, 0), (1, 1)])
        self.assertEqual(partial_bays({(0, 1), (1, 0), (1, 1)}, fb), [(0, 1, 3)])
        self.assertEqual(partial_bays({(0, 1), (0, 2)}, fb), [])


if __name__ == "__main__":
    unittest.main()
