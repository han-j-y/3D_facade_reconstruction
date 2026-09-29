"""Tests for filename-prefix bulk labeling."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from balcony_train.label_from_prefix import label_from_prefixes, match_prefix
from balcony_train.label_store import load_labels


class LabelFromPrefixTests(unittest.TestCase):
    def test_match_prefix(self) -> None:
        self.assertEqual(
            match_prefix("flux2_masonry_012.png"),
            ("flux2_masonry_", "open_work", "masonry", None),
        )
        self.assertEqual(
            match_prefix("flux2_metal_001.png"),
            ("flux2_metal_", "open_work", "metal", None),
        )
        self.assertEqual(
            match_prefix("flux2_surface_000.png"),
            ("flux2_surface_", "surface_panel", None, None),
        )
        self.assertEqual(
            match_prefix("flux2_solid_003.png"),
            ("flux2_solid_", "solid", None, None),
        )
        self.assertEqual(
            match_prefix("flux2_triangle_metal_000.png"),
            ("flux2_triangle_metal_", "open_work", "metal", "triangle"),
        )
        self.assertEqual(
            match_prefix("flux2_hexagon_solid_002.png"),
            ("flux2_hexagon_solid_", "solid", None, "hexagon"),
        )
        self.assertEqual(
            match_prefix("flux2_trapezoid_surface_001.png"),
            ("flux2_trapezoid_surface_", "surface_panel", None, "trapezoid"),
        )
        self.assertIsNone(match_prefix("other_001.png"))

    def test_label_moves_and_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crops = root / "crops"
            unlab = crops / "unlabeled"
            unlab.mkdir(parents=True)
            jsonl = root / "labels.jsonl"
            (unlab / "flux2_masonry_000.png").write_bytes(b"x")
            (unlab / "flux2_metal_000.png").write_bytes(b"x")
            (unlab / "flux2_surface_000.png").write_bytes(b"x")
            (unlab / "flux2_solid_000.png").write_bytes(b"x")
            (unlab / "flux2_hexagon_masonry_000.png").write_bytes(b"x")
            (unlab / "ignore_me.png").write_bytes(b"x")

            stats = label_from_prefixes(
                crops_dir=crops,
                unlabeled_dir=unlab,
                jsonl_path=jsonl,
            )
            self.assertEqual(stats["labeled"], 5)
            self.assertEqual(stats["skipped_no_prefix"], 1)
            self.assertTrue((crops / "open_work" / "flux2_masonry_000.png").is_file())
            self.assertTrue((crops / "open_work" / "flux2_metal_000.png").is_file())
            self.assertTrue(
                (crops / "surface_panel" / "flux2_surface_000.png").is_file()
            )
            self.assertTrue((crops / "solid" / "flux2_solid_000.png").is_file())
            self.assertTrue((unlab / "ignore_me.png").is_file())

            labels = load_labels(jsonl)
            self.assertEqual(labels["flux2_masonry_000"]["material"], "masonry")
            self.assertEqual(labels["flux2_metal_000"]["material"], "metal")
            self.assertIsNone(labels["flux2_surface_000"]["material"])
            self.assertEqual(labels["flux2_solid_000"]["kind"], "solid")
            self.assertIsNone(labels["flux2_metal_000"]["floor_shape"])
            self.assertEqual(labels["flux2_hexagon_masonry_000"]["kind"], "open_work")
            self.assertEqual(labels["flux2_hexagon_masonry_000"]["material"], "masonry")
            self.assertEqual(labels["flux2_hexagon_masonry_000"]["floor_shape"], "hexagon")
            self.assertTrue(
                (crops / "open_work" / "flux2_hexagon_masonry_000.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
