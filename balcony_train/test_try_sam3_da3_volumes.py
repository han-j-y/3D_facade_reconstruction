"""GPU-free tests for SAM3+DA3 volume helpers."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from balcony_train.extract_enclosed import foliage_mask  # noqa: E402
from balcony_train.try_sam3_da3_volumes import (  # noqa: E402
    RED_GT,
    STEMS,
    box_cover_frac,
    box_iou,
    detect_volumes,
    box_rel_delta,
    clip_shop_span,
    column_storey_reason,
    covered_by_exclude,
    drop_narrower_than_below,
    front_fill_frac,
    grow_horizontal,
    grow_vertical,
    local_wall_median,
    merge_adjacent_bays,
    merge_vertical_columns,
    nms_columns,
    on_window_floor_bay,
    resolve_floor_h,
    similar_box_width,
    robust_median,
    same_column,
    seed_ok,
    split_seed_records,
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

    def test_stage3_merges_one_storey_gap_and_stage4_does_not_move(self) -> None:
        img = Image.new("RGB", (120, 200), (170, 150, 130))
        depth = np.full((200, 120), 10.0, dtype=np.float32)
        depth[20:120, 30:70] = 9.5
        seeds = [
            {"box_xyxy": [30, 20, 70, 50], "score": 0.6, "prompt": "oriel window"},
            {"box_xyxy": [30, 90, 70, 120], "score": 0.5, "prompt": "oriel window"},
        ]
        stages = volume_stages(img, depth, seeds)
        merged = stages[2][2]
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["box_xyxy"], [30, 20, 70, 120])
        self.assertEqual(stages[3][2][0]["box_xyxy"], merged[0]["box_xyxy"])

    def test_stage4_grows_down_not_up(self) -> None:
        img = Image.new("RGB", (120, 200), (170, 150, 130))
        depth = np.full((200, 120), 10.0, dtype=np.float32)
        depth[40:140, 30:80] = 9.5
        seeds = [
            {"box_xyxy": [30, 40, 80, 90], "score": 0.8, "prompt": "oriel window"},
        ]
        stages = volume_stages(img, depth, seeds, grow_rel=0.006)
        stage3 = stages[2][2][0]["box_xyxy"]
        stage4 = stages[3][2][0]["box_xyxy"]
        self.assertEqual(stage3[1], 40)
        self.assertEqual(stage4[0], stage3[0])
        self.assertEqual(stage4[1], stage3[1])
        self.assertEqual(stage4[2], stage3[2])
        self.assertGreater(stage4[3], stage3[3])

    def test_stage3_width_uses_stage2_boxes(self) -> None:
        img = Image.new("RGB", (280, 220), (170, 150, 130))
        depth = np.full((220, 280), 10.0, dtype=np.float32)
        depth[10:180, 25:110] = 9.5
        bay = [30, 90, 100, 170]
        upper = [45, 20, 80, 55]
        hidden = [48, 100, 78, 155]
        seeds = [
            {"box_xyxy": bay, "score": 0.7, "prompt": "oriel window"},
            {"box_xyxy": upper, "score": 0.6, "prompt": "oriel window"},
            {"box_xyxy": hidden, "score": 0.5, "prompt": "oriel window"},
        ]
        stages = volume_stages(img, depth, seeds)
        stage2 = stages[1][2]
        stage3 = stages[2][2]
        self.assertEqual(len(stage2), 2)
        for box in stage3:
            x0, y0, x1, y1 = box["box_xyxy"]
            self.assertFalse(y0 <= 20 and y1 >= 170)

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

    def test_front_fill_solid_vs_sparse(self) -> None:
        depth = np.full((80, 80), 10.0, dtype=np.float32)
        depth[20:60, 20:50] = 9.5
        self.assertGreater(front_fill_frac(depth, [20, 20, 50, 60]), 0.9)
        thin = np.full((80, 80), 10.0, dtype=np.float32)
        thin[20:60, 20:24] = 9.5
        fill = front_fill_frac(thin, [20, 20, 50, 60])
        self.assertIsNotNone(fill)
        self.assertLess(fill, 0.30)

    def test_stage2_drops_sparse_front(self) -> None:
        img = Image.new("RGB", (80, 100), (170, 150, 130))
        depth = np.full((100, 80), 10.0, dtype=np.float32)
        depth[20:80, 22:26] = 9.4
        seeds = [{"box_xyxy": [22, 25, 50, 75], "score": 0.6, "prompt": "oriel window"}]
        stages = volume_stages(img, depth, seeds, floor_h=40.0)
        self.assertEqual(len(stages[0][2]), 1)
        self.assertEqual(stages[1][2], [])

    def test_box_rel_ignores_tree_and_lamp(self) -> None:
        depth = np.full((80, 80), 10.0, dtype=np.float32)
        depth[20:60, 20:35] = 4.0
        depth[20:60, 35:50] = 9.5
        rgb = np.full((80, 80, 3), 170, dtype=np.uint8)
        rgb[20:60, 20:35] = (30, 180, 40)
        veg = foliage_mask(rgb)
        rel = box_rel_delta(depth, [20, 20, 50, 60], veg=veg)
        self.assertIsNotNone(rel)
        self.assertGreater(rel, 0.03)
        self.assertLess(rel, 0.12)

    def test_tree_only_box_has_no_rel(self) -> None:
        depth = np.full((80, 80), 10.0, dtype=np.float32)
        depth[20:60, 20:50] = 4.0
        rgb = np.full((80, 80, 3), 170, dtype=np.uint8)
        rgb[20:60, 20:50] = (30, 180, 40)
        veg = foliage_mask(rgb)
        self.assertIsNone(box_rel_delta(depth, [20, 20, 50, 60], veg=veg))

    def test_fire_escape_exclude_drops_overlap(self) -> None:
        img = Image.new("RGB", (80, 100), (170, 150, 130))
        depth = np.full((100, 80), 10.0, dtype=np.float32)
        depth[20:80, 20:55] = 9.5
        seeds = [
            {"box_xyxy": [22, 25, 50, 75], "score": 0.6, "prompt": "oriel window"},
            {"box_xyxy": [20, 20, 55, 80], "score": 0.5, "prompt": "fire escape"},
        ]
        stages = volume_stages(img, depth, seeds, floor_h=40.0)
        self.assertEqual(stages[0][2], [])
        self.assertEqual(stages[-1][2], [])
        self.assertTrue(covered_by_exclude([22, 25, 50, 75], [[20, 20, 55, 80]]))
        seeds_only, excl = split_seed_records(seeds)
        self.assertEqual(len(seeds_only), 1)
        self.assertEqual(len(excl), 1)
        self.assertAlmostEqual(box_cover_frac([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

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

    def test_vertical_merge_skips_width_mismatch(self) -> None:
        recs = [
            {"box_xyxy": [10, 10, 30, 40], "score": 0.5},
            {"box_xyxy": [8, 44, 50, 70], "score": 0.4},
        ]
        self.assertFalse(similar_box_width(recs[0]["box_xyxy"], recs[1]["box_xyxy"]))
        merged = merge_vertical_columns(recs, ih=80, max_y_gap_frac=0.2)
        self.assertEqual(len(merged), 2)

    def test_drop_narrower_than_enclosed_below(self) -> None:
        recs = [
            {"box_xyxy": [10, 10, 30, 40], "score": 0.5},
            {"box_xyxy": [8, 44, 50, 70], "score": 0.4},
        ]
        kept = drop_narrower_than_below(recs)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["box_xyxy"], [8, 44, 50, 70])

    def test_keep_similar_width_above_enclosed(self) -> None:
        recs = [
            {"box_xyxy": [10, 10, 30, 40], "score": 0.5},
            {"box_xyxy": [12, 44, 28, 70], "score": 0.4},
        ]
        kept = drop_narrower_than_below(recs)
        self.assertEqual(len(kept), 2)

    def test_keep_narrow_box_below_enclosed(self) -> None:
        recs = [
            {"box_xyxy": [8, 10, 50, 40], "score": 0.5},
            {"box_xyxy": [10, 44, 30, 70], "score": 0.4},
        ]
        kept = drop_narrower_than_below(recs)
        self.assertEqual(len(kept), 2)

    def test_keep_narrow_box_in_other_column(self) -> None:
        recs = [
            {"box_xyxy": [10, 10, 30, 40], "score": 0.5},
            {"box_xyxy": [50, 44, 90, 70], "score": 0.4},
        ]
        kept = drop_narrower_than_below(recs)
        self.assertEqual(len(kept), 2)

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

    def test_stacked_oriel_merges_when_centers_are_far(self) -> None:
        recs = [
            {"box_xyxy": [100, 30, 290, 380], "score": 0.5},
            {"box_xyxy": [110, 500, 290, 780], "score": 0.4},
        ]
        merged = merge_vertical_columns(recs, ih=1000, max_y_gap_frac=0.22)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["box_xyxy"], [100, 30, 290, 780])

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

    def test_seed_drops_at_or_below_floor_fraction(self) -> None:
        # 20px box is ok without floor_h (5.5% of image) but dies at 0.85 * storey.
        self.assertTrue(seed_ok([20, 20, 45, 40], iw=100, ih=100))
        self.assertFalse(
            seed_ok([20, 20, 45, 40], iw=100, ih=100, min_height_px=50.0)
        )
        self.assertFalse(
            seed_ok([20, 20, 45, 70], iw=100, ih=100, min_height_px=50.0)
        )
        self.assertTrue(
            seed_ok([20, 20, 45, 71], iw=100, ih=100, min_height_px=50.0)
        )

    def test_stage1_drops_short_crop_when_floor_h_given(self) -> None:
        img = Image.new("RGB", (80, 100), (170, 150, 130))
        depth = np.full((100, 80), 10.0, dtype=np.float32)
        depth[20:90, 22:50] = 9.5
        short = {"box_xyxy": [22, 30, 50, 50], "score": 0.6, "prompt": "oriel window"}
        tall = {"box_xyxy": [22, 25, 50, 75], "score": 0.6, "prompt": "oriel window"}
        dropped = volume_stages(img, depth, [short], floor_h=40.0)
        kept = volume_stages(img, depth, [tall], floor_h=40.0)
        self.assertEqual(dropped[0][2], [])
        self.assertEqual(len(kept[0][2]), 1)
        self.assertIn("0.85 floor", dropped[0][1])

    def test_on_window_floor_bay_requires_storey_and_column(self) -> None:
        floor_y = {0: (20, 100), 1: (100, 180)}
        bay_x = {0: (10, 80), 1: (80, 150)}
        self.assertTrue(on_window_floor_bay([20, 30, 70, 95], floor_y, bay_x))
        self.assertFalse(on_window_floor_bay([20, 185, 70, 230], floor_y, bay_x))
        self.assertFalse(on_window_floor_bay([160, 30, 190, 95], floor_y, bay_x))
        self.assertFalse(on_window_floor_bay([20, 30, 70, 95], None, bay_x))

    def test_stage5_keeps_grid_box_drops_same_box_off_grid(self) -> None:
        img = Image.new("RGB", (200, 200), (170, 150, 130))
        depth = np.full((200, 200), 10.0, dtype=np.float32)
        depth[40:130, 170:198] = 9.7
        seed = {"box_xyxy": [170, 40, 198, 130], "score": 0.7, "prompt": "oriel window"}
        floor_y = {0: (35, 110), 1: (110, 190)}
        bay_x = {1: (155, 200)}
        dropped = volume_stages(img, depth, [seed])
        kept = volume_stages(img, depth, [seed], floor_y=floor_y, bay_x=bay_x)
        self.assertEqual(dropped[-1][2], [])
        self.assertEqual(len(kept[-1][2]), 1)
        self.assertIn("keep floor", kept[-1][1])

    def test_resolve_floor_h_reads_pipeline_dsl(self) -> None:
        dsl = {
            "meta": {
                "columns_xy": [[0, 100]],
                "floors_y": [[0, 80], [80, 200]],
            },
            "layout": {
                "floors": [{"id": 0}, {"id": 1}],
                "bays": [{"id": 0}],
            },
            "instances": [],
        }
        with TemporaryDirectory() as tmp:
            stem_dir = Path(tmp) / "cmp_x0005"
            stem_dir.mkdir()
            (stem_dir / "facade_dsl.json").write_text(
                json.dumps(dsl), encoding="utf-8"
            )
            self.assertEqual(resolve_floor_h("cmp_x0005", Path(tmp)), 100.0)
            self.assertIsNone(resolve_floor_h("missing", Path(tmp)))
            self.assertIsNone(resolve_floor_h("cmp_x0005", None))

    def test_robust_median_ignores_tree_pixels(self) -> None:
        depth = np.full((20, 20), 10.0, dtype=np.float32)
        depth[0:4, :] = 2.0
        med = robust_median(depth, [0, 0, 20, 20])
        self.assertGreater(med, 8.0)


if __name__ == "__main__":
    unittest.main()
