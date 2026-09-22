"""Tests for evaluate filename filters and list helpers (no GPU)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from balcony_train.evaluate import (
    ScoredCrop,
    copy_eval_images,
    copy_misclassified_images,
    error_run_dir,
    error_run_stamp,
    eval_image_paths,
    filter_indices_by_name_prefix,
    misclass_folder_name,
    write_eval_list,
)
from balcony_train.labels import MATERIAL_IGNORE_INDEX


class EvaluateFilterTests(unittest.TestCase):
    def test_filter_keeps_cmp_only(self) -> None:
        samples = [
            (Path("open_work/cmp_b0001_bal_000.png"), 0, 0),
            (Path("open_work/flux2_masonry_001.png"), 0, 1),
            (Path("solid/cmp_b0002_bal_000.png"), 2, MATERIAL_IGNORE_INDEX),
            (Path("solid/flux2_solid_000.png"), 2, MATERIAL_IGNORE_INDEX),
        ]
        kept = filter_indices_by_name_prefix(samples, [0, 1, 2, 3], "cmp")
        self.assertEqual(kept, [0, 2])

    def test_empty_prefix_keeps_all(self) -> None:
        samples = [
            (Path("a/flux2_metal_000.png"), 0, 0),
            (Path("a/cmp_b0001_bal_000.png"), 0, 1),
        ]
        kept = filter_indices_by_name_prefix(samples, [0, 1], "")
        self.assertEqual(kept, [0, 1])

    def test_write_eval_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "cmp_a.png"
            b = root / "cmp_b.png"
            a.write_bytes(b"x")
            b.write_bytes(b"x")
            samples = [(a, 0, 0), (b, 1, MATERIAL_IGNORE_INDEX)]
            paths = eval_image_paths(samples, [0, 1])
            out = root / "eval_list.txt"
            write_eval_list(paths, list_out=out, also_print=False)
            text = out.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(text), 2)
            self.assertTrue(text[0].endswith("cmp_a.png"))
            self.assertTrue(text[1].endswith("cmp_b.png"))

    def test_copy_eval_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_dir = root / "open_work"
            src_dir.mkdir()
            a = src_dir / "cmp_a.png"
            a.write_bytes(b"img")
            dest = root / "view"
            n = copy_eval_images([a], dest)
            self.assertEqual(n, 1)
            self.assertTrue((dest / "cmp_a.png").is_file())
            self.assertEqual((dest / "cmp_a.png").read_bytes(), b"img")

    def test_copy_misclassified_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            solid = root / "src" / "solid_wrong.png"
            both = root / "src" / "both_wrong.png"
            mat_only = root / "src" / "mat_wrong.png"
            ok = root / "src" / "ok.png"
            for path in (solid, both, mat_only, ok):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(path.name.encode())
            # kind: 0 open_work, 1 surface_panel, 2 solid
            # material: 0 metal, 1 masonry
            rows = [
                ScoredCrop(solid, 2, 0, MATERIAL_IGNORE_INDEX, 0),
                ScoredCrop(both, 0, 1, 1, 0),
                ScoredCrop(mat_only, 0, 0, 1, 0),
                ScoredCrop(ok, 0, 0, 0, 0),
            ]
            dest = root / "Wrong"
            stamp = "2026-09-21_152435"
            n = copy_misclassified_images(rows, dest, stamp=stamp)
            self.assertEqual(n, 4)
            run = dest / stamp
            kind_solid = (
                run
                / "kind"
                / misclass_folder_name("solid", "open_work")
                / "solid_wrong.png"
            )
            kind_both = (
                run
                / "kind"
                / misclass_folder_name("open_work", "surface_panel")
                / "both_wrong.png"
            )
            mat_both = (
                run
                / "material"
                / misclass_folder_name("masonry", "metal")
                / "both_wrong.png"
            )
            mat_only_dest = (
                run
                / "material"
                / misclass_folder_name("masonry", "metal")
                / "mat_wrong.png"
            )
            self.assertTrue(kind_solid.is_file())
            self.assertTrue(kind_both.is_file())
            self.assertTrue(mat_both.is_file())
            self.assertTrue(mat_only_dest.is_file())
            self.assertFalse(any(dest.rglob("ok.png")))

    def test_error_run_dir_adds_suffix_when_stamp_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = error_run_dir(root, "2026-09-21_152435")
            first.mkdir()
            second = error_run_dir(root, "2026-09-21_152435")
            self.assertEqual(first.name, "2026-09-21_152435")
            self.assertEqual(second.name, "2026-09-21_152435_2")
            self.assertEqual(len(error_run_stamp()), 17)


if __name__ == "__main__":
    unittest.main()
