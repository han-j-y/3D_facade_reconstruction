"""GPU-free tests for SAM3+DA3 volume helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from balcony_train.try_sam3_da3_volumes import (  # noqa: E402
    RED_GT,
    STEMS,
    box_iou,
    detect_volumes,
    box_rel_delta,
    clip_shop_span,
    column_storey_reason,
    grow_horizontal,
    grow_vertical,
    local_wall_median,
    merge_adjacent_bays,
    merge_vertical_columns,
    nms_columns,
    robust_median,
    same_column,
    seed_ok,
    trim_to_projecting,
    volume_stages,
    x_overlap_frac,
)


class Sam3Da3VolumeTests(unittest.TestCase):
    def test_volume_stages_has_five_steps(self) -> None:
        img = Image.new("RGB", (80, 100), (170, 150, 130))
        depth = np.full((100, 80), 10.0, dtype=np.float32)
        depth[30:75, 22:42] = 9.5
        seeds = [{"box_xyxy": [22, 38, 42, 58], "score": 0.6, "prompt": "oriel window"}]
        stages = volume_stages(img, depth, seeds)
        self.assertEqual(len(stages), 5)
        self.assertEqual([s[0] for s in stages], [
            "1_sam3", "2_depth", "3_merge", "4_grow", "5_clip",
        ])
        self.assertEqual(detect_volumes(img, depth, seeds), stages[-1][2])

    def test_four_target_photos(self) -> None:
        self.assertEqual(set(STEMS), set(RED_GT))
        self.assertEqual(len(RED_GT["cmp_b0007"]), 3)
        self.assertEqual(len(RED_GT["cmp_b0008"]), 1)
        self.assertEqual(len(RED_GT["cmp_b0010"]), 1)
        self.assertEqual(len(RED_GT["cmp_b0223"]), 2)

    def test_filter_stems_keeps_named_subset(self) -> None:
        from balcony_train.try_sam3_da3_volumes import filter_stems

        paths = [Path("cmp_b0007.jpg"), Path("cmp_b0330.jpg")]
        kept = filter_stems(paths, ("cmp_b0007",))
        self.assertEqual([p.stem for p in kept], ["cmp_b0007"])

    def test_iou_and_x_overlap(self) -> None:
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
        self.assertGreater(x_overlap_frac([10, 0, 30, 50], [12, 60, 28, 90]), 0.7)

    def test_local_wall_sees_projecting_box(self) -> None:
        depth = np.full((80, 80), 10.0, dtype=np.float32)
        depth[20:60, 20:40] = 9.5  # 5% closer column
        wall = local_wall_median(depth, [20, 20, 40, 60])
        rel = box_rel_delta(depth, [20, 20, 40, 60])
        self.assertIsNotNone(wall)
        self.assertGreater(wall, 9.7)
        self.assertIsNotNone(rel)
        self.assertGreater(rel, 0.03)

    def test_wide_gallery_does_not_merge_with_oriel(self) -> None:
        gallery = [100, 500, 400, 800]
        oriel = [380, 250, 500, 480]
        self.assertFalse(same_column(gallery, oriel, min_each=0.40))
        self.assertTrue(same_column([10, 10, 40, 40], [12, 50, 38, 80], min_each=0.40))

    def test_vertical_merge_joins_same_column(self) -> None:
        recs = [
            {"box_xyxy": [10, 10, 30, 30], "score": 0.5},
            {"box_xyxy": [12, 34, 28, 55], "score": 0.4},
            {"box_xyxy": [50, 10, 70, 40], "score": 0.3},
        ]
        merged = merge_vertical_columns(recs, ih=80, max_y_gap_frac=0.2)
        self.assertEqual(len(merged), 2)
        col = next(r for r in merged if r["box_xyxy"][0] < 20)
        self.assertEqual(col["box_xyxy"][1], 10)
        self.assertEqual(col["box_xyxy"][3], 55)

    def test_oriel_does_not_merge_with_ground_portal(self) -> None:
        recs = [
            {"box_xyxy": [50, 40, 80, 200], "score": 0.5},
            {"box_xyxy": [52, 540, 82, 900], "score": 0.4},
        ]
        merged = merge_vertical_columns(recs, ih=1024, max_y_gap_frac=0.4)
        self.assertEqual(len(merged), 2)

    def test_gallery_does_not_merge_with_shop(self) -> None:
        recs = [
            {"box_xyxy": [180, 547, 460, 698], "score": 0.7},
            {"box_xyxy": [176, 706, 466, 860], "score": 0.6},
        ]
        merged = merge_vertical_columns(recs, ih=1024, max_y_gap_frac=0.2)
        self.assertEqual(len(merged), 2)

    def test_lowest_gallery_floor_still_merges(self) -> None:
        recs = [
            {"box_xyxy": [150, 320, 410, 450], "score": 0.5},
            {"box_xyxy": [155, 479, 412, 625], "score": 0.4},
            {"box_xyxy": [147, 655, 413, 816], "score": 0.4},
        ]
        merged = merge_vertical_columns(recs, ih=1024, max_y_gap_frac=0.10)
        self.assertEqual(len(merged), 1)

    def test_column_nms_drops_portal_under_oriel(self) -> None:
        recs = [
            {"box_xyxy": [500, 30, 640, 470], "score": 0.5},
            {"box_xyxy": [490, 450, 650, 800], "score": 0.4},
        ]
        kept = nms_columns(recs, iw=768, ih=1024)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["box_xyxy"][1], 30)

    def test_column_nms_keeps_gallery_floors(self) -> None:
        recs = [
            {"box_xyxy": [150, 300, 410, 670], "score": 0.5},
            {"box_xyxy": [147, 654, 413, 816], "score": 0.4},
        ]
        kept = nms_columns(recs, iw=589, ih=1024)
        self.assertEqual(len(kept), 2)

    def test_clip_shop_span_keeps_mid_facade(self) -> None:
        clipped = clip_shop_span([176, 538, 471, 993], 1024)
        self.assertEqual(clipped[3], int(round(0.78 * 1024)))
        bay = clip_shop_span([143, 679, 284, 811], 1024)
        self.assertEqual(bay[3], 811)
        gallery = clip_shop_span([146, 301, 413, 791], 1024)
        self.assertEqual(gallery[3], 791)

    def test_grow_extends_closer_column(self) -> None:
        depth = np.full((100, 60), 10.0, dtype=np.float32)
        depth[15:80, 20:40] = 9.6
        grown = grow_vertical(
            depth, [20, 40, 40, 55], veg=None, min_rel=0.01, step=5
        )
        self.assertLessEqual(grown[1], 20)
        self.assertGreaterEqual(grown[3], 75)

    def test_grow_does_not_jump_tree_belt(self) -> None:
        depth = np.full((120, 60), 10.0, dtype=np.float32)
        depth[10:50, 20:40] = 9.6
        depth[50:70, 20:40] = 4.0  # trees in front of the shop
        depth[70:110, 20:40] = 9.5
        veg = np.zeros((120, 60), dtype=bool)
        veg[50:70, 20:40] = True
        grown = grow_vertical(
            depth, [20, 20, 40, 45], veg=veg, min_rel=0.01, step=5
        )
        self.assertLess(grown[3], 70)

    def test_horizontal_grow_widens_projecting_bay(self) -> None:
        depth = np.full((80, 200), 10.0, dtype=np.float32)
        depth[20:60, 40:140] = 9.5
        grown = grow_horizontal(
            depth, [40, 20, 80, 60], veg=None, min_rel=0.01, step=4
        )
        self.assertGreaterEqual(grown[2], 95)

    def test_trim_drops_padding_edge(self) -> None:
        depth = np.full((80, 80), 10.0, dtype=np.float32)
        depth[20:60, 20:50] = 9.5
        depth[0:12, 20:50] = 2.0
        trimmed = trim_to_projecting(
            depth, [20, 0, 50, 60], veg=None, clutter=None, min_rel=0.01, step=4,
            seed_med=9.5,
        )
        self.assertGreaterEqual(trimmed[1], 12)

    def test_adjacent_bay_joins_turret_and_inner_window(self) -> None:
        recs = [
            {"box_xyxy": [10, 40, 40, 120], "score": 0.5},
            {"box_xyxy": [48, 50, 80, 110], "score": 0.4},
            {"box_xyxy": [200, 40, 230, 120], "score": 0.3},
        ]
        merged = merge_adjacent_bays(recs, iw=240)
        self.assertEqual(len(merged), 2)
        bay = next(r for r in merged if r["box_xyxy"][0] < 20)
        self.assertEqual(bay["box_xyxy"], [10, 40, 80, 120])

    def test_adjacent_bay_skips_wide_gallery(self) -> None:
        recs = [
            {"box_xyxy": [10, 20, 40, 80], "score": 0.5},
            {"box_xyxy": [45, 50, 160, 90], "score": 0.4},
        ]
        merged = merge_adjacent_bays(recs, iw=200)
        self.assertEqual(len(merged), 2)

    def test_column_storey_drops_fragment_keeps_oriel(self) -> None:
        self.assertEqual(
            column_storey_reason([10, 10, 18, 40], iw=100, ih=100), "too_narrow"
        )
        self.assertEqual(
            column_storey_reason([10, 40, 40, 48], iw=100, ih=100), "too_short"
        )
        self.assertIsNone(column_storey_reason([20, 20, 40, 50], iw=100, ih=100))

    def test_seed_rejects_sash_and_full_width(self) -> None:
        self.assertFalse(seed_ok([0, 10, 4, 40], iw=100, ih=100))
        self.assertFalse(seed_ok([0, 10, 90, 40], iw=100, ih=100))
        self.assertFalse(seed_ok([10, 40, 70, 70], iw=100, ih=100))  # 60% wide fused box
        self.assertFalse(seed_ok([20, 5, 40, 95], iw=100, ih=100))  # full-height seed
        self.assertTrue(seed_ok([20, 20, 45, 40], iw=100, ih=100))

    def test_robust_median_ignores_tree_pixels(self) -> None:
        depth = np.full((20, 20), 10.0, dtype=np.float32)
        depth[0:4, :] = 2.0
        med = robust_median(depth, [0, 0, 20, 20])
        self.assertGreater(med, 8.0)


if __name__ == "__main__":
    unittest.main()
