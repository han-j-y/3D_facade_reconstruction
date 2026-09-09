"""Tests for JSONL label store (no UI)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.label_store import (  # noqa: E402
    merge_label_sources,
    normalize_kind,
    normalize_material,
    queue_stems,
    save_annotation,
)


def _touch_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (120, 120, 120)).save(path)


class LabelStoreTests(unittest.TestCase):
    def test_normalize(self) -> None:
        self.assertEqual(normalize_kind("baluster"), "open_work")
        self.assertEqual(normalize_material("masonry", kind="open_work"), "masonry")
        self.assertIsNone(normalize_material("masonry", kind="solid"))

    def test_save_open_work_requires_material(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crops = root / "crops"
            jsonl = root / "labels.jsonl"
            src = crops / "unlabeled" / "a.png"
            _touch_png(src)
            with self.assertRaises(ValueError):
                save_annotation(
                    crops_dir=crops,
                    jsonl_path=jsonl,
                    image_path=src,
                    kind="open_work",
                    material=None,
                )

    def test_save_and_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crops = root / "crops"
            jsonl = root / "labels.jsonl"
            a = crops / "unlabeled" / "a.png"
            b = crops / "unlabeled" / "b.png"
            _touch_png(a)
            _touch_png(b)
            labels = save_annotation(
                crops_dir=crops,
                jsonl_path=jsonl,
                image_path=a,
                kind="open_work",
                material="masonry",
            )
            self.assertEqual(labels["a"]["material"], "masonry")
            self.assertTrue((crops / "open_work" / "a.png").is_file())
            labels = save_annotation(
                crops_dir=crops,
                jsonl_path=jsonl,
                image_path=b,
                kind="solid",
                material=None,
                labels=labels,
            )
            self.assertIsNone(labels["b"]["material"])
            q = queue_stems(crops, labels, unlabeled_only=True, need_material=True)
            self.assertEqual(q, [])

            # Folder-only open_work without material stays in queue.
            c = crops / "open_work" / "c.png"
            _touch_png(c)
            merged = merge_label_sources(crops, jsonl)
            q2 = queue_stems(crops, merged, unlabeled_only=True, need_material=True)
            self.assertIn("c", q2)


if __name__ == "__main__":
    unittest.main()
