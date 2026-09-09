"""Tests for stratified train/val/test splits."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.splits import (  # noqa: E402
    build_split_record,
    indices_from_record,
    load_split,
    save_split,
    stratified_split_indices,
)


class StratifiedSplitTests(unittest.TestCase):
    def _fake_samples(self, n_open_work: int, n_solid: int):
        rows = []
        for i in range(n_open_work):
            rows.append((Path(f"open_work/b{i:03d}.png"), 0))
        for i in range(n_solid):
            rows.append((Path(f"solid/s{i:03d}.png"), 2))
        return rows

    def test_8_1_1_counts(self) -> None:
        samples = self._fake_samples(180, 70)
        fold = stratified_split_indices(samples, seed=42)
        self.assertEqual(len(fold["train"]), 200)
        self.assertEqual(len(fold["val"]), 25)
        self.assertEqual(len(fold["test"]), 25)
        self.assertEqual(len(fold["train"]) + len(fold["val"]) + len(fold["test"]), len(samples))

    def test_stratified_each_fold_has_both_classes(self) -> None:
        samples = self._fake_samples(180, 70)
        fold = stratified_split_indices(samples, seed=7)
        for name in ("train", "val", "test"):
            labels = {samples[i][1] for i in fold[name]}
            self.assertEqual(labels, {0, 2}, msg=name)

    def test_save_load_roundtrip(self) -> None:
        samples = self._fake_samples(18, 7)
        fold = stratified_split_indices(samples, seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            samples_abs = [(base / p, lb) for p, lb in samples]
            record = build_split_record(
                samples_abs,
                fold,
                data_dir=base,
                seed=1,
                train_frac=0.8,
                val_frac=0.1,
                test_frac=0.1,
            )
            path = save_split(record, base / "split.json")
            loaded = load_split(path)
            self.assertEqual(loaded["split_counts"], record["split_counts"])
            back = indices_from_record(loaded, samples_abs)
            self.assertEqual(set(back["train"]), set(fold["train"]))


if __name__ == "__main__":
    unittest.main()
