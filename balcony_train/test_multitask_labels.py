"""Multitask sample loading from folders + labels.jsonl."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.labels import (  # noqa: E402
    MATERIAL_IGNORE_INDEX,
    iter_multitask_samples,
    material_counts,
    strat_key,
)


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (90, 90, 90)).save(path)


class MultitaskLabelsTests(unittest.TestCase):
    def test_iter_multitask_defaults_metal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crops = root / "crops"
            jsonl = root / "labels.jsonl"
            _png(crops / "open_work" / "a.png")
            _png(crops / "solid" / "b.png")
            jsonl.write_text(
                json.dumps(
                    {"path": "c.png", "kind": "open_work", "material": "masonry"}
                )
                + "\n",
                encoding="utf-8",
            )
            _png(crops / "open_work" / "c.png")
            rows = iter_multitask_samples(crops, jsonl)
            by_stem = {p.stem: (k, m) for p, k, m in rows}
            self.assertEqual(by_stem["a"][0], 0)
            self.assertEqual(by_stem["a"][1], 0)  # default metal
            self.assertEqual(by_stem["b"][1], MATERIAL_IGNORE_INDEX)
            self.assertEqual(by_stem["c"][1], 1)  # masonry
            self.assertEqual(material_counts(rows)["masonry"], 1)
            self.assertEqual(strat_key(0, 1), "open_work|masonry")


if __name__ == "__main__":
    unittest.main()
