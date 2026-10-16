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
    cell_at,
    facade_stems_from_crop_labels,
    list_grid_stems,
    load_labels,
    normalize_cells,
    write_labels,
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


if __name__ == "__main__":
    unittest.main()
