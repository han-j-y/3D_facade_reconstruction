"""Tests for CMP XML balcony crops and the open/enclosed queues."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.crop_balconies_xml import (  # noqa: E402
    crop_xml_balconies,
    parse_balcony_norm_boxes,
)
from balcony_train.label_store import merge_label_sources  # noqa: E402
from balcony_train.xml_balcony_labels import (  # noqa: E402
    distribution_counts,
    enclosed_queue,
    open_queue,
    send_back_to_open,
    send_to_enclosed,
)
from balcony_train.label_store import find_image, save_annotation  # noqa: E402

_XML = """<object><labelname>window</labelname><x>0.1</x><x>0.2</x><y>0.1</y><y>0.2</y></object>
<object><labelname>balcony</labelname><x>0.40</x><x>0.10</x><y>0.50</y><y>0.20</y></object>
<object><labelname>balcony</labelname><x>0.05</x><x>0.15</x><y>0.05</y><y>0.12</y></object>
"""


def _jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), (180, 180, 180)).save(path, format="JPEG")


def _label_png(path: Path, blobs: list[tuple[int, int, int, int]]) -> None:
    """Palette PNG. Index 7 is balcony. ``blobs`` are (x0, y0, x1, y1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("P", (40, 30), 2)
    pal = [0] * 768
    pal[7 * 3 : 7 * 3 + 3] = [170, 255, 85]
    im.putpalette(pal)
    px = im.load()
    for x0, y0, x1, y1 in blobs:
        for y in range(y0, y1):
            for x in range(x0, x1):
                px[x, y] = 7
    im.save(path)


class XmlCropTests(unittest.TestCase):
    def test_parse_ignores_windows_and_sorts_top_then_left(self) -> None:
        boxes = parse_balcony_norm_boxes(_XML.replace("\n", "\r"))
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0], (0.05, 0.05, 0.15, 0.12))
        self.assertEqual(boxes[1], (0.10, 0.20, 0.40, 0.50))

    def test_crop_follows_png_balcony_not_xml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "extended"
            image_dir.mkdir(parents=True)
            Image.new("RGB", (40, 30), (180, 180, 180)).save(image_dir / "cmp_x0001.jpg", format="JPEG")
            Image.new("RGB", (40, 30), (180, 180, 180)).save(image_dir / "cmp_x0002.jpg", format="JPEG")
            _label_png(
                image_dir / "cmp_x0001.png",
                [(4, 5, 10, 12), (20, 18, 35, 28)],
            )
            (image_dir / "cmp_x0001.xml").write_text(_XML, encoding="utf-8")
            (image_dir / "cmp_x0002.xml").write_text(
                "<object><labelname>balcony</labelname><x>0</x><x>1</x><y>0</y><y>1</y></object>",
                encoding="utf-8",
            )
            out = root / "crops" / "unlabeled"
            stats = crop_xml_balconies(
                image_dir,
                out_dir=out,
                manifest_path=root / "manifest.jsonl",
                none_path=root / "none.jsonl",
                pad_frac=0.0,
                min_side=1,
            )
            self.assertEqual(stats["n_photos"], 2)
            self.assertEqual(stats["n_crops"], 2)
            self.assertEqual(stats["n_none"], 1)
            with Image.open(out / "cmp_x0001_bal_000.png") as first:
                self.assertEqual(first.size, (6, 7))
            with Image.open(out / "cmp_x0001_bal_001.png") as second:
                self.assertEqual(second.size, (15, 10))
            none_text = (root / "none.jsonl").read_text(encoding="utf-8")
            self.assertIn("cmp_x0002.jpg", none_text)


class XmlQueueTests(unittest.TestCase):
    def test_open_then_enclosed_queues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crops = root / "crops"
            unlabeled = crops / "unlabeled"
            unlabeled.mkdir(parents=True)
            Image.new("RGB", (8, 8), (10, 10, 10)).save(unlabeled / "a_bal_000.png")
            Image.new("RGB", (8, 8), (10, 10, 10)).save(unlabeled / "b_bal_000.png")
            jsonl = root / "labels.jsonl"
            labels: dict = {}
            self.assertEqual(open_queue(crops, labels), ["a_bal_000", "b_bal_000"])
            self.assertEqual(enclosed_queue(crops, labels), [])

            path_b = find_image(crops, "b_bal_000")
            assert path_b is not None
            labels = send_to_enclosed(
                crops_dir=crops, jsonl_path=jsonl, image_path=path_b, labels=labels
            )
            self.assertEqual(open_queue(crops, labels), ["a_bal_000"])
            self.assertEqual(enclosed_queue(crops, labels), ["b_bal_000"])

            path_b = find_image(crops, "b_bal_000")
            assert path_b is not None
            labels = save_annotation(
                crops_dir=crops,
                jsonl_path=jsonl,
                image_path=path_b,
                kind=None,
                material=None,
                floor_shape="circle",
                enclosure="enclosed",
                skin="curtain_wall",
                labels=labels,
            )
            self.assertEqual(enclosed_queue(crops, labels), [])
            counts = distribution_counts(merge_label_sources(crops, jsonl))
            self.assertEqual(counts["enclosure"]["enclosed"], 1)
            self.assertEqual(counts["skin"]["curtain_wall"], 1)
            self.assertEqual(counts["floor_shape"]["circle"], 1)

            path_a = find_image(crops, "a_bal_000")
            assert path_a is not None
            labels = save_annotation(
                crops_dir=crops,
                jsonl_path=jsonl,
                image_path=path_a,
                kind="open_work",
                material="metal",
                floor_shape="rectangle",
                enclosure="open",
                structure="juliet",
                labels=labels,
            )
            counts = distribution_counts(merge_label_sources(crops, jsonl))
            self.assertEqual(counts["structure"]["juliet"], 1)
            self.assertEqual(counts["structure"]["projecting"], 0)
            self.assertEqual(open_queue(crops, labels), [])

            path_b = find_image(crops, "b_bal_000")
            assert path_b is not None
            labels = send_back_to_open(
                crops_dir=crops, jsonl_path=jsonl, image_path=path_b, labels=labels
            )
            self.assertIn("b_bal_000", open_queue(crops, labels))
            self.assertEqual(enclosed_queue(crops, labels), [])


if __name__ == "__main__":
    unittest.main()
