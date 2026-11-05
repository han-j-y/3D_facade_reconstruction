"""Tests for box-detector helpers (no GPU / weights required)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.try_box_detectors import (  # noqa: E402
    BACKENDS,
    DEFAULT_LABELS,
    drop_full_frame,
    grounding_text,
    parse_qwen_boxes,
    scale_box,
    smart_resize,
)


class TryBoxDetectorsTests(unittest.TestCase):
    def test_backends_cover_three_models(self) -> None:
        self.assertEqual(
            set(BACKENDS),
            {"grounding_dino", "llmdet", "qwen25vl"},
        )
        self.assertIn("enclosed balcony", DEFAULT_LABELS)
        self.assertIn("glazed gallery", DEFAULT_LABELS)

    def test_grounding_text_is_lowercase_dot_separated(self) -> None:
        text = grounding_text(["Enclosed Balcony", "Oriel Window"])
        self.assertEqual(text, "enclosed balcony . oriel window .")

    def test_smart_resize_multiples_of_28(self) -> None:
        h, w = smart_resize(1024, 636)
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)
        self.assertLessEqual(h * w, 768 * 28 * 28)

    def test_drop_full_frame_keeps_local_box(self) -> None:
        recs = [
            {"box_xyxy": [0, 0, 100, 100], "score": 0.5, "label": "glazed gallery"},
            {"box_xyxy": [20, 30, 40, 70], "score": 0.4, "label": "oriel window"},
        ]
        kept = drop_full_frame(recs, iw=100, ih=100, max_area_frac=0.55)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["label"], "oriel window")

    def test_scale_box_roundtrip_identity(self) -> None:
        box = scale_box([10, 20, 110, 220], src_w=200, src_h=400, dst_w=200, dst_h=400)
        self.assertEqual(box, [10, 20, 110, 220])
        doubled = scale_box([10, 20, 110, 220], src_w=100, src_h=200, dst_w=200, dst_h=400)
        self.assertEqual(doubled, [20, 40, 220, 440])

    def test_parse_qwen_json_array(self) -> None:
        text = (
            '[{"bbox_2d": [12, 34, 56, 78], "label": "glazed gallery"},'
            ' {"bbox_2d": [1, 2, 3, 4], "label": "oriel"}]'
        )
        recs = parse_qwen_boxes(text)
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0]["label"], "glazed gallery")
        self.assertEqual(recs[0]["box_xyxy"], [12.0, 34.0, 56.0, 78.0])

    def test_parse_qwen_fenced_and_messy(self) -> None:
        text = (
            "Here you go:\n```json\n"
            '{"bbox_2d": [5, 6, 15, 40], "label": "bay"}\n'
            "```\n"
        )
        recs = parse_qwen_boxes(text)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["label"], "bay")
        self.assertEqual(recs[0]["box_xyxy"][2], 15.0)


if __name__ == "__main__":
    unittest.main()
