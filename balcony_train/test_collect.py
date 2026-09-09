"""Collect unique balcony unit crops into unlabeled/."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.collect import _dest_name, collect_crops  # noqa: E402
from balcony_train.labels import (  # noqa: E402
    class_counts,
    class_index,
    ensure_crop_dirs,
    iter_labeled_samples,
)


def _png(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


class CollectTests(unittest.TestCase):
    def test_dest_name_uses_facade_and_type(self) -> None:
        src = Path("runs/batch/cmp_b0082/balcony/assets/cmp_b0082/types/type_00/unit_003.png")
        self.assertEqual(_dest_name(src), "cmp_b0082_type_00_unit_003.png")

    def test_collect_copies_balcony_units_skips_dups_and_window_crops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "batch"
            balc = src / "cmp_b0082" / "balcony" / "assets" / "cmp_b0082" / "types" / "type_00"
            _png(balc / "unit_000.png", (1, 2, 3))
            _png(balc / "unit_001.png", (4, 5, 6))
            _png(src / "cmp_b0082" / "assets" / "types" / "type_00" / "unit_000.png", (9, 9, 9))
            crops = root / "crops"
            stats = collect_crops(src, crops)
            self.assertEqual(stats["copied"], 2)
            unlabeled = list((crops / "unlabeled").glob("*.png"))
            self.assertEqual(len(unlabeled), 2)
            again = collect_crops(src, crops)
            self.assertEqual(again["copied"], 0)
            self.assertEqual(again["skipped"], 2)


class DatasetFolderTests(unittest.TestCase):
    def test_iter_labeled_ignores_unlabeled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            crops = Path(tmp)
            ensure_crop_dirs(crops)
            _png(crops / "unlabeled" / "x.png")
            _png(crops / "solid" / "a.png", (20, 20, 20))
            _png(crops / "open_work" / "b.png", (30, 30, 30))
            samples = iter_labeled_samples(crops)
            self.assertEqual(
                class_counts(samples),
                {
                    "open_work": 1,
                    "surface_panel": 0,
                    "solid": 1,
                },
            )
            labels = {p.name: i for p, i in samples}
            self.assertEqual(labels["b.png"], class_index("open_work"))
            self.assertEqual(labels["a.png"], class_index("solid"))

    def test_iter_labeled_includes_jpeg_and_webp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            crops = Path(tmp)
            ensure_crop_dirs(crops)
            _png(crops / "solid" / "a.png")
            Image.new("RGB", (8, 8), (1, 2, 3)).save(crops / "solid" / "b.jpg")
            Image.new("RGB", (8, 8), (4, 5, 6)).save(crops / "open_work" / "c.jpeg")
            Image.new("RGB", (8, 8), (7, 8, 9)).save(crops / "open_work" / "d.webp")
            self.assertEqual(
                class_counts(iter_labeled_samples(crops)),
                {
                    "open_work": 2,
                    "surface_panel": 0,
                    "solid": 2,
                },
            )


if __name__ == "__main__":
    unittest.main()
