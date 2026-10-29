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
    load_labels,
    merge_label_sources,
    normalize_floor_shape,
    normalize_kind,
    normalize_material,
    normalize_structure,
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
        self.assertEqual(normalize_floor_shape("Triangle"), "triangle")
        self.assertEqual(normalize_floor_shape("Hexagon"), "rectangle")
        self.assertEqual(normalize_floor_shape("trapezoid"), "rectangle")
        self.assertIsNone(normalize_floor_shape(None))
        self.assertEqual(normalize_structure("projecting"), "projecting")
        self.assertEqual(normalize_structure("Juliet"), "juliet")
        self.assertIsNone(normalize_structure(""))
        self.assertIsNone(normalize_structure(None))
        with self.assertRaises(ValueError):
            normalize_floor_shape("pentagon")
        with self.assertRaises(ValueError):
            normalize_structure("inset")
        with self.assertRaises(ValueError):
            normalize_structure("rooftop")

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

    def test_floor_shape_roundtrip_and_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crops = root / "crops"
            jsonl = root / "labels.jsonl"
            src = crops / "unlabeled" / "hex.png"
            _touch_png(src)
            labels = save_annotation(
                crops_dir=crops,
                jsonl_path=jsonl,
                image_path=src,
                kind="solid",
                material=None,
                floor_shape="circle",
            )
            self.assertEqual(labels["hex"]["floor_shape"], "circle")
            loaded = load_labels(jsonl)
            self.assertEqual(loaded["hex"]["floor_shape"], "circle")
            q = queue_stems(crops, labels, unlabeled_only=True, need_floor_shape=True)
            self.assertEqual(q, [])

            trap = crops / "solid" / "trap.png"
            _touch_png(trap)
            labels = save_annotation(
                crops_dir=crops,
                jsonl_path=jsonl,
                image_path=trap,
                kind="solid",
                material=None,
                floor_shape="triangle",
                labels=labels,
            )
            plain = crops / "solid" / "plain.png"
            _touch_png(plain)
            merged = merge_label_sources(crops, jsonl)
            self.assertEqual(merged["trap"]["floor_shape"], "triangle")
            self.assertIsNone(merged["plain"]["floor_shape"])
            pending = queue_stems(
                crops, merged, unlabeled_only=True, need_floor_shape=True
            )
            self.assertEqual(pending, ["plain"])

    def test_structure_roundtrip_keeps_projecting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crops = root / "crops"
            jsonl = root / "labels.jsonl"
            src = crops / "unlabeled" / "rail.png"
            _touch_png(src)
            labels = save_annotation(
                crops_dir=crops,
                jsonl_path=jsonl,
                image_path=src,
                kind="open_work",
                material="metal",
                floor_shape="rectangle",
                enclosure="open",
                structure="juliet",
            )
            self.assertEqual(labels["rail"]["structure"], "juliet")
            loaded = load_labels(jsonl)
            self.assertEqual(loaded["rail"]["structure"], "juliet")
            self.assertIn('"structure": "juliet"', jsonl.read_text(encoding="utf-8"))

            placed = crops / "open_work" / "rail.png"
            labels = save_annotation(
                crops_dir=crops,
                jsonl_path=jsonl,
                image_path=placed,
                kind="open_work",
                material="metal",
                floor_shape="rectangle",
                enclosure="open",
                structure="projecting",
                labels=labels,
            )
            self.assertEqual(labels["rail"]["structure"], "projecting")
            self.assertEqual(load_labels(jsonl)["rail"]["structure"], "projecting")

            labels = save_annotation(
                crops_dir=crops,
                jsonl_path=jsonl,
                image_path=placed,
                kind="open_work",
                material="metal",
                floor_shape="rectangle",
                enclosure="open",
                structure="juliet",
                labels=labels,
            )
            self.assertEqual(load_labels(jsonl)["rail"]["structure"], "juliet")

    def test_juliet_needs_no_floor_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crops = root / "crops"
            jsonl = root / "labels.jsonl"
            for stem in ("jul", "proj"):
                _touch_png(crops / "unlabeled" / f"{stem}.png")
            labels = {}
            for stem, structure in (("jul", "juliet"), ("proj", "projecting")):
                labels = save_annotation(
                    crops_dir=crops,
                    jsonl_path=jsonl,
                    image_path=crops / "unlabeled" / f"{stem}.png",
                    kind="open_work",
                    material="metal",
                    floor_shape=None,
                    enclosure="open",
                    structure=structure,
                    labels=labels,
                )
            q = queue_stems(crops, labels, unlabeled_only=True, need_floor_shape=True)
            self.assertEqual(q, ["proj"])

    def test_missing_structure_loads_as_unlabeled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "labels.jsonl"
            jsonl.write_text(
                '{"path": "a.png", "kind": "solid", "material": null, '
                '"floor_shape": "rectangle", "enclosure": "open", "skin": null}\n',
                encoding="utf-8",
            )
            self.assertIsNone(load_labels(jsonl)["a"]["structure"])


if __name__ == "__main__":
    unittest.main()
