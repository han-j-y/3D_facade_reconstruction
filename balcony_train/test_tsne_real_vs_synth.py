"""Tests for t-SNE real-vs-synth helpers (no GPU / sklearn)."""

from __future__ import annotations

import unittest
from pathlib import Path

from balcony_train.tsne_real_vs_synth import (
    iter_crop_paths,
    source_label,
    subsample_balanced,
)


class TsneRealVsSynthHelperTests(unittest.TestCase):
    def test_source_label(self) -> None:
        self.assertEqual(source_label(Path("cmp_b0001_bal_000.png")), "real")
        self.assertEqual(source_label(Path("FLUX2_masonry_001.PNG")), "synth")
        self.assertIsNone(source_label(Path("other_001.png")))

    def test_subsample_balanced(self) -> None:
        rows = [
            (Path(f"r{i}.png"), "real", "open_work") for i in range(10)
        ] + [(Path(f"s{i}.png"), "synth", "solid") for i in range(10)]
        out = subsample_balanced(rows, max_per_source=3, seed=0)
        self.assertEqual(sum(1 for _, s, _ in out if s == "real"), 3)
        self.assertEqual(sum(1 for _, s, _ in out if s == "synth"), 3)

    def test_iter_crop_paths(self) -> None:
        import tempfile

        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ow = root / "open_work"
            ow.mkdir()
            Image.new("RGB", (8, 8)).save(ow / "cmp_b0001_bal_000.png")
            Image.new("RGB", (8, 8)).save(ow / "flux2_metal_000.png")
            Image.new("RGB", (8, 8)).save(ow / "ignore_me.png")
            rows = iter_crop_paths(root)
            names = {p.name for p, _, _ in rows}
            self.assertEqual(names, {"cmp_b0001_bal_000.png", "flux2_metal_000.png"})
            srcs = {s for _, s, _ in rows}
            self.assertEqual(srcs, {"real", "synth"})


if __name__ == "__main__":
    unittest.main()
