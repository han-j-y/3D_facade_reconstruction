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
    FLOOR_IGNORE_INDEX,
    KIND_IGNORE_INDEX,
    MATERIAL_IGNORE_INDEX,
    enclosure_counts,
    floor_counts,
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
            by_stem = {p.stem: (k, m, f) for p, k, m, f, _e in rows}
            self.assertEqual(by_stem["a"][0], 0)
            self.assertEqual(by_stem["a"][1], 0)  # default metal
            self.assertEqual(by_stem["a"][2], FLOOR_IGNORE_INDEX)
            self.assertEqual(by_stem["b"][1], MATERIAL_IGNORE_INDEX)
            self.assertEqual(by_stem["c"][1], 1)  # masonry
            self.assertEqual(material_counts(rows)["masonry"], 1)
            self.assertEqual(strat_key(0, 1), "open_work|masonry")

            _png(crops / "solid" / "hex.png")
            jsonl.write_text(
                jsonl.read_text(encoding="utf-8")
                + json.dumps(
                    {
                        "path": "hex.png",
                        "kind": "solid",
                        "material": None,
                        "floor_shape": "hexagon",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = iter_multitask_samples(crops, jsonl)
            by_stem = {p.stem: (k, m, f) for p, k, m, f, _e in rows}
            self.assertEqual(by_stem["hex"][2], 3)
            self.assertEqual(by_stem["b"][2], FLOOR_IGNORE_INDEX)
            self.assertEqual(floor_counts(rows)["hexagon"], 1)
            self.assertEqual(sum(floor_counts(rows).values()), 1)

    def test_enclosure_only_rows_train_without_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crops = root / "crops"
            jsonl = root / "labels.jsonl"
            _png(crops / "solid" / "s.png")
            _png(crops / "unlabeled" / "glass.png")
            _png(crops / "unlabeled" / "todo.png")
            lines = [
                {"path": "s.png", "kind": "solid", "enclosure": "half_enclosed"},
                {"path": "glass.png", "kind": None, "enclosure": "enclosed"},
            ]
            jsonl.write_text(
                "".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8"
            )
            rows = {p.stem: (k, e) for p, k, _m, _f, e in iter_multitask_samples(crops, jsonl)}
            self.assertEqual(set(rows), {"s", "glass"})
            self.assertEqual(rows["s"], (2, 1))
            self.assertEqual(rows["glass"], (KIND_IGNORE_INDEX, 2))
            counts = enclosure_counts(list(iter_multitask_samples(crops, jsonl)))
            self.assertEqual(counts, {"open": 0, "half_enclosed": 1, "enclosed": 1})
            self.assertEqual(strat_key(2, MATERIAL_IGNORE_INDEX, 1), "solid+half_enclosed")
            self.assertEqual(strat_key(KIND_IGNORE_INDEX, MATERIAL_IGNORE_INDEX, 2), "no_kind+enclosed")
            self.assertEqual(strat_key(0, 0, 0), "open_work|metal")


if __name__ == "__main__":
    unittest.main()
