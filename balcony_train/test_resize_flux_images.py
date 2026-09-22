"""Tests for flux2 crop resize (no GPU)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from balcony_train.resize_flux_images import (
    FLUX_ORIGINAL_DIRNAME,
    horizontal_band_box,
    iter_flux_images,
    match_flux_kind_to_real,
    match_rgb,
    replace_flux_with_resized,
)


class ResizeFluxImagesTests(unittest.TestCase):
    def test_horizontal_band_box_crops_top_and_bottom(self) -> None:
        # 639/372 is taller than 120/44, so the crop keeps the full width.
        left, top, right, bottom = horizontal_band_box((639, 372), (120, 44))
        self.assertEqual(left, 0)
        self.assertEqual(right, 639)
        self.assertGreater(top, 0)
        self.assertLess(bottom, 372)
        self.assertEqual(bottom - top, int(round(639 * 44 / 120)))

    def test_horizontal_band_box_crops_sides_when_already_wide(self) -> None:
        left, top, right, bottom = horizontal_band_box((400, 40), (120, 44))
        self.assertEqual(top, 0)
        self.assertEqual(bottom, 40)
        self.assertGreater(left, 0)
        self.assertLess(right, 400)

    def test_archives_original_and_replaces_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            crops = Path(tmp) / "crops"
            unlabeled = crops / "unlabeled"
            unlabeled.mkdir(parents=True)
            flux = unlabeled / "flux2_metal_000.png"
            real = unlabeled / "cmp_b0001_bal_000.png"
            Image.new("RGB", (100, 40), (10, 20, 30)).save(flux)
            Image.new("RGB", (80, 80), (1, 2, 3)).save(real)
            kind = crops / "open_work"
            kind.mkdir()
            labeled = kind / "flux2_masonry_001.png"
            Image.new("RGB", (50, 20), (4, 5, 6)).save(labeled)

            found = iter_flux_images(crops)
            self.assertEqual(
                [p.name for p in found],
                ["flux2_metal_000.png", "flux2_masonry_001.png"],
            )

            stats = replace_flux_with_resized(crops, target=(120, 44), jpeg_quality=60)
            self.assertEqual(stats["saved"], 2)
            archive = crops / FLUX_ORIGINAL_DIRNAME
            with Image.open(archive / "unlabeled" / "flux2_metal_000.png") as image:
                self.assertEqual(image.size, (100, 40))
            with Image.open(flux) as image:
                self.assertEqual(image.size, (120, 44))
            with Image.open(archive / "open_work" / "flux2_masonry_001.png") as image:
                self.assertEqual(image.size, (50, 20))
            with Image.open(labeled) as image:
                self.assertEqual(image.size, (120, 44))
            with Image.open(real) as image:
                self.assertEqual(image.size, (80, 80))

            replace_flux_with_resized(crops, target=(120, 44), jpeg_quality=60)
            with Image.open(archive / "unlabeled" / "flux2_metal_000.png") as image:
                self.assertEqual(image.size, (100, 40))
            with Image.open(flux) as image:
                self.assertEqual(image.size, (120, 44))

    def test_match_rgb_moves_mean_onto_target(self) -> None:
        pixels = np.full((4, 4, 3), 200, dtype=np.float32)
        matched = match_rgb(
            pixels,
            pixels.mean(axis=(0, 1)),
            pixels.std(axis=(0, 1)),
            np.array([80, 90, 100], dtype=np.float32),
            np.array([10, 10, 10], dtype=np.float32),
        )
        self.assertEqual(matched.shape, (4, 4, 3))
        self.assertTrue(np.all(matched.mean(axis=(0, 1)) == np.array([80, 90, 100])))

    def test_match_flux_kind_uses_real_cmp_and_keeps_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            crops = Path(tmp) / "crops"
            folder = crops / "surface_panel"
            folder.mkdir(parents=True)
            Image.new("RGB", (30, 20), (40, 50, 60)).save(folder / "cmp_a.png")
            Image.new("RGB", (30, 20), (80, 90, 100)).save(folder / "cmp_b.png")
            flux = folder / "flux2_surface_000.png"
            Image.new("RGB", (200, 120), (220, 220, 220)).save(flux)
            stats = match_flux_kind_to_real(crops, "surface_panel", target=(20, 10))
            self.assertEqual(stats["saved"], 1)
            self.assertEqual(stats["real_n"], 2)
            archive = crops / FLUX_ORIGINAL_DIRNAME / "surface_panel" / flux.name
            with Image.open(archive) as image:
                self.assertEqual(image.size, (200, 120))
            with Image.open(flux) as image:
                self.assertEqual(image.size, (20, 10))
                self.assertLess(image.getpixel((0, 0))[0], 180)


if __name__ == "__main__":
    unittest.main()
