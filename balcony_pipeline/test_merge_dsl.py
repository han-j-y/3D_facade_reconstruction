"""Photo-norm merge fields for balcony slab width."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from merge_dsl import (  # noqa: E402
    attach_to_window,
    floor_of_top_edge,
    outer_bay_center,
    _move_stacks_together,
    _mean_bay_center_norm,
    _photo_norms,
    merge_balcony_into_windows_dsl,
)


class PhotoNormTests(unittest.TestCase):
    def test_norms(self) -> None:
        w, cx = _photo_norms([100, 0, 300, 50], (1000, 800))
        self.assertAlmostEqual(w, 0.2)
        self.assertAlmostEqual(cx, 0.2)

    def test_merge_writes_norms_photo_center(self) -> None:
        dsl = {
            "schema": "facade_recovery_dsl_v1",
            "meta": {"image_size": [1000, 800]},
            "layout": {"floors": [], "bays": [], "placement": []},
            "window_types": [],
            "instances": [],
        }
        units = [
            {
                "unit_id": 0,
                "type_id": 0,
                "floor": 1,
                "bay_start": 0,
                "bay_end": 2,
                "bay": 1,
                "box_xyxy": [100, 10, 300, 80],
            }
        ]
        out = merge_balcony_into_windows_dsl(
            dsl,
            balcony_types=[{"type_id": 0, "name": "balc_type_00", "structure_ir": {}}],
            units=units,
            center_mode="photo",
        )
        b = out["layout"]["balconies"][0]
        self.assertAlmostEqual(b["width_norm"], 0.2)
        self.assertAlmostEqual(b["cx_norm"], 0.2)
        self.assertEqual(b["bay_start"], 0)
        self.assertEqual(b["bay_end"], 2)
        self.assertNotIn("structure", b)

    def test_merge_marks_rooftop_on_the_placement(self) -> None:
        dsl = {
            "schema": "facade_recovery_dsl_v1",
            "meta": {"image_size": [1000, 800]},
            "layout": {"floors": [], "bays": [], "placement": []},
            "window_types": [],
            "instances": [],
        }
        base = {"type_id": 0, "floor": 0, "bay_start": 0, "bay_end": 0, "bay": 0}
        units = [
            {**base, "unit_id": 0, "box_xyxy": [100, 10, 300, 80], "rooftop": True},
            {**base, "unit_id": 1, "floor": 1, "box_xyxy": [100, 210, 300, 280]},
        ]
        out = merge_balcony_into_windows_dsl(
            dsl,
            balcony_types=[{"type_id": 0, "name": "balc_type_00", "structure_ir": {}}],
            units=units,
            center_mode="photo",
        )
        roof, below = out["layout"]["balconies"]
        self.assertEqual(roof["type"], below["type"])
        self.assertEqual(roof["structure"], "rooftop")
        self.assertNotIn("structure", below)

    def test_merge_bay_center_omits_cx_norm(self) -> None:
        dsl = {
            "schema": "facade_recovery_dsl_v1",
            "meta": {"image_size": [1000, 800]},
            "layout": {"floors": [], "bays": [], "placement": []},
            "window_types": [],
            "instances": [
                {"floor": 1, "bay": 0, "box_xyxy": [0, 50, 100, 150]},
                {"floor": 1, "bay": 1, "box_xyxy": [100, 50, 200, 150]},
                {"floor": 1, "bay": 2, "box_xyxy": [200, 50, 300, 150]},
            ],
        }
        units = [
            {
                "unit_id": 0,
                "type_id": 0,
                "floor": 1,
                "bay_start": 0,
                "bay_end": 2,
                "bays": [0, 1, 2],
                "bays_center": [0, 1],
                "bay": 1,
                "box_xyxy": [100, 10, 300, 80],
            }
        ]
        out = merge_balcony_into_windows_dsl(
            dsl,
            balcony_types=[{"type_id": 0, "name": "balc_type_00", "structure_ir": {}}],
            units=units,
            center_mode="bay",
        )
        b = out["layout"]["balconies"][0]
        self.assertAlmostEqual(b["width_norm"], 0.2)
        self.assertNotIn("cx_norm", b)
        self.assertEqual(b["bays"], [0, 1, 2])
        self.assertEqual(b["bays_center"], [0, 1])
        # mean of bay 0 center 50 and bay 1 center 150 -> 100 / 1000
        self.assertAlmostEqual(b["bay_cx_norm"], 0.1)
        self.assertEqual(out["meta"]["balcony_center_mode"], "bay")

    def test_merge_window_center_from_overlap_partners(self) -> None:
        dsl = {
            "schema": "facade_recovery_dsl_v1",
            "meta": {"image_size": [400, 400]},
            "layout": {"floors": [], "bays": [], "placement": []},
            "window_types": [],
            "instances": [
                {"unit_id": 1, "floor": 2, "bay": 1, "box_xyxy": [100, 40, 180, 120]},
                {"unit_id": 2, "floor": 2, "bay": 2, "box_xyxy": [190, 40, 270, 120]},
            ],
        }
        units = [
            {
                "unit_id": 0,
                "type_id": 0,
                "floor": 2,
                "bay_start": 1,
                "bay_end": 2,
                "bays": [1, 2],
                "bays_center": [1],
                "bay": 1,
                "box_xyxy": [95, 115, 275, 175],
            }
        ]
        out = merge_balcony_into_windows_dsl(
            dsl,
            balcony_types=[{"type_id": 0, "name": "balc_type_00", "structure_ir": {}}],
            units=units,
            center_mode="window",
        )
        b = out["layout"]["balconies"][0]
        self.assertEqual(b["partner_window_unit_ids"], [1, 2])
        self.assertAlmostEqual(b["window_cx_norm"], (140 + 230) / 2 / 400)

    def test_merge_window_default_mode(self) -> None:
        dsl = {
            "schema": "facade_recovery_dsl_v1",
            "meta": {"image_size": [1000, 800]},
            "layout": {"floors": [], "bays": [], "placement": []},
            "window_types": [],
            "instances": [
                {"unit_id": 5, "floor": 0, "bay": 0, "box_xyxy": [40, 50, 60, 150]},
            ],
        }
        units = [
            {
                "unit_id": 0,
                "type_id": 0,
                "floor": 0,
                "bay_start": 0,
                "bay_end": 0,
                "bays_center": [0],
                "bay": 0,
                "box_xyxy": [30, 150, 70, 230],  # hangs right under its window
            }
        ]
        out = merge_balcony_into_windows_dsl(
            dsl,
            balcony_types=[{"type_id": 0, "name": "balc_type_00", "structure_ir": {}}],
            units=units,
        )
        b = out["layout"]["balconies"][0]
        self.assertAlmostEqual(b["window_cx_norm"], 0.05)
        self.assertEqual(out["meta"]["balcony_center_mode"], "window")

    def test_window_id_and_bay_index_do_not_collide(self) -> None:
        # Balcony A has no window behind it (keyed by its bay, 1); balcony B hangs under window unit 1.
        dsl = {
            "schema": "facade_recovery_dsl_v1",
            "meta": {"image_size": [1000, 800]},
            "layout": {"floors": [], "bays": [], "placement": []},
            "window_types": [],
            "instances": [{"unit_id": 1, "floor": 0, "bay": 2, "box_xyxy": [440, 50, 460, 150]}],
        }
        units = [
            {"unit_id": 0, "type_id": 0, "floor": 0, "bay_start": 1, "bay_end": 1, "bays_center": [1], "bay": 1,
             "box_xyxy": [130, 150, 170, 230]},
            {"unit_id": 1, "type_id": 0, "floor": 0, "bay_start": 2, "bay_end": 2, "bays_center": [2], "bay": 2,
             "box_xyxy": [430, 150, 470, 230]},
        ]
        out = merge_balcony_into_windows_dsl(
            dsl, balcony_types=[{"type_id": 0, "name": "balc_type_00", "structure_ir": {}}], units=units
        )
        self.assertEqual(len(out["layout"]["balconies"]), 2)

    def test_merged_dsl_carries_the_order_lists(self) -> None:
        dsl = {"meta": {"image_size": [100, 100]}, "window_types": [], "instances": [],
               "layout": {"floors": [{"id": 0}, {"id": 1}], "bays": [{"id": 0}, {"id": 1}, {"id": 2}], "placement": []}}
        out = merge_balcony_into_windows_dsl(dsl, balcony_types=[], units=[])
        self.assertEqual((out["layout"]["floor_order"], out["layout"]["bay_order"]), ([0, 1], [0, 1, 2]))

    def _enclosed_case(self, instances: list[dict]) -> dict:
        units = [
            {"type_id": 0, "floor": 0, "bay_start": 1, "bay_end": 1, "bay": 1, "bays": [1], "bays_center": [1],
             "box_xyxy": [150, 60, 250, 100], "structure_ir": {"enclosure": "open"}},
            {"type_id": 0, "floor": 0, "bay_start": 3, "bay_end": 3, "bay": 3, "bays": [3], "bays_center": [3],
             "box_xyxy": [300, 0, 400, 100], "source": "enclosed_volume", "fit_box_xyxy": [300, 0, 400, 100],
             "structure_ir": {"enclosure": "enclosed"}},
        ]
        dsl = {
            "meta": {"image_size": [1000, 500], "columns_xy": [[0, 100], [100, 300], [300, 400], [400, 500]],
                     "floors_y": [[0, 100]]},
            "layout": {"floors": [{"name": "F0", "id": 0, "h_norm": 1.0}],
                       "bays": [{"name": f"B{i}", "id": i, "w_norm": 0.25} for i in range(4)],
                       "placement": [[None] * 4]},
            "window_types": [], "instances": instances,
        }
        return merge_balcony_into_windows_dsl(
            dsl, balcony_types=[{"name": "balc_type_00", "structure_ir": {}}], units=units, center_mode="window"
        )

    def test_enclosed_volume_does_not_change_the_open_balcony_center(self) -> None:
        win = {"unit_id": 0, "floor": 0, "bay": 1, "box_xyxy": [170, 10, 230, 60], "type_id": 0}
        rec = self._enclosed_case([win])["layout"]["balconies"][0]
        self.assertAlmostEqual(rec["window_cx_norm"], 0.2)
        self.assertNotIn("bay_cx_norm", rec)

    def test_side_wall_window_is_no_partner(self) -> None:
        front = {"unit_id": 0, "floor": 0, "bay": 1, "box_xyxy": [150, 10, 190, 60], "type_id": 0}
        side = {"unit_id": 1, "floor": 0, "bay": 1, "box_xyxy": [210, 10, 240, 60], "type_id": 0, "side_face": True}
        rec = self._enclosed_case([front, side])["layout"]["balconies"][0]
        self.assertEqual(rec["partner_window_unit_ids"], [0])

    def _rec(self, floor: int, box: list[int], **kw) -> dict:
        return {"floor": floor, "type": "balc_type_00", "box_xyxy": box, "width_norm": 0.1} | kw

    def test_stack_of_one_type_shares_the_median_window_center(self) -> None:
        recs = [self._rec(0, [100, 0, 200, 9], window_cx_norm=0.14), self._rec(1, [102, 0, 200, 9], window_cx_norm=0.16),
                self._rec(2, [101, 0, 200, 9], window_cx_norm=0.15)]
        _move_stacks_together(recs, 1000)
        self.assertEqual([r["window_cx_norm"] for r in recs], [0.15, 0.15, 0.15])

    def test_floor_without_a_window_follows_its_stack(self) -> None:
        recs = [self._rec(0, [100, 0, 200, 9], window_cx_norm=0.14), self._rec(1, [102, 0, 200, 9], bay_cx_norm=0.2)]
        _move_stacks_together(recs, 1000)
        self.assertEqual(recs[1]["window_cx_norm"], 0.14)
        self.assertNotIn("bay_cx_norm", recs[1])

    def test_stack_without_any_window_stays_at_its_bay(self) -> None:
        recs = [self._rec(0, [100, 0, 200, 9], bay_cx_norm=0.2), self._rec(1, [102, 0, 200, 9], bay_cx_norm=0.2)]
        _move_stacks_together(recs, 1000)
        self.assertEqual([r["bay_cx_norm"] for r in recs], [0.2, 0.2])

    def test_other_type_and_other_column_do_not_move_together(self) -> None:
        recs = [self._rec(0, [100, 0, 200, 9], window_cx_norm=0.14),
                self._rec(1, [100, 0, 200, 9], window_cx_norm=0.2, type="balc_type_01"),
                self._rec(2, [700, 0, 800, 9], window_cx_norm=0.75)]
        _move_stacks_together(recs, 1000)
        self.assertEqual([r["window_cx_norm"] for r in recs], [0.14, 0.2, 0.75])

    def test_balcony_at_its_bay_overlapping_another_uses_its_box(self) -> None:
        recs = [self._rec(0, [100, 0, 200, 9], bay_cx_norm=0.5), self._rec(0, [500, 0, 600, 9], window_cx_norm=0.5, type="balc_type_01")]
        _move_stacks_together(recs, 1000)
        self.assertEqual(recs[0]["cx_norm"], 0.15)
        self.assertNotIn("bay_cx_norm", recs[0])

    def test_long_gallery_does_not_join_the_small_balconies_it_spans(self) -> None:
        recs = [self._rec(0, [100, 0, 200, 9], window_cx_norm=0.15), self._rec(1, [100, 0, 800, 9], window_cx_norm=0.45),
                self._rec(0, [600, 0, 700, 9], window_cx_norm=0.65), self._rec(2, [100, 0, 200, 9], window_cx_norm=0.16),
                self._rec(2, [600, 0, 700, 9], window_cx_norm=0.66)]
        _move_stacks_together(recs, 1000)
        self.assertEqual(recs[1]["window_cx_norm"], 0.45)
        self.assertEqual(recs[0]["window_cx_norm"], recs[3]["window_cx_norm"])
        self.assertEqual(recs[2]["window_cx_norm"], recs[4]["window_cx_norm"])
        self.assertLess(recs[0]["window_cx_norm"], 0.2)
        self.assertGreater(recs[2]["window_cx_norm"], 0.6)

    def test_one_floor_never_stacks_with_itself(self) -> None:
        recs = [self._rec(0, [100, 0, 200, 9], window_cx_norm=0.14), self._rec(0, [120, 0, 220, 9], window_cx_norm=0.18)]
        _move_stacks_together(recs, 1000)
        self.assertEqual([r["window_cx_norm"] for r in recs], [0.14, 0.18])

    def test_balcony_on_its_own_box_is_left_alone(self) -> None:
        recs = [self._rec(0, [100, 0, 200, 9], cx_norm=0.15), self._rec(1, [100, 0, 200, 9], window_cx_norm=0.2)]
        _move_stacks_together(recs, 1000)
        self.assertEqual(recs[0]["cx_norm"], 0.15)

    def _outer_meta(self) -> dict:
        return {"columns_xy": [[100, 300], [300, 500]], "facade_region_xyxy": [0, 0, 1000, 400]}

    def _win(self) -> list[dict]:
        return [{"box_xyxy": [120, 10, 160, 60], "kind": "window"}]

    def test_box_past_the_grid_is_in_the_outer_bay(self) -> None:
        self.assertEqual(outer_bay_center([700, 0, 900, 50], self._outer_meta(), self._win()), ("right", 750.0))
        self.assertEqual(outer_bay_center([0, 0, 60, 50], self._outer_meta(), self._win()), ("left", 50.0))

    def test_box_on_the_grid_has_no_outer_bay(self) -> None:
        self.assertIsNone(outer_bay_center([200, 0, 380, 50], self._outer_meta(), self._win()))

    def test_no_region_no_outer_bay(self) -> None:
        self.assertIsNone(outer_bay_center([700, 0, 900, 50], {"columns_xy": [[100, 500]]}, self._win()))

    def test_thin_margin_is_no_bay(self) -> None:
        meta = {"columns_xy": [[100, 500]], "facade_region_xyxy": [90, 0, 510, 400]}
        self.assertIsNone(outer_bay_center([505, 0, 509, 50], meta, self._win()))

    def test_merge_places_balcony_in_its_outer_bay_not_on_a_window(self) -> None:
        dsl = {
            "meta": {"image_size": [1000, 400], **self._outer_meta()},
            "layout": {"floors": [{"name": "F0", "id": 0, "h_norm": 1.0}],
                       "bays": [{"name": "B0", "id": 0, "w_norm": 0.5}, {"name": "B1", "id": 1, "w_norm": 0.5}],
                       "placement": [[None, None]]},
            "window_types": [],
            "instances": [{"unit_id": 0, "floor": 0, "bay": 1, "box_xyxy": [320, 10, 360, 60], "type_id": 0}],
        }
        unit = {"type_id": 0, "floor": 0, "bay_start": 1, "bay_end": 1, "bay": 1, "bays": [1], "bays_center": [1],
                "box_xyxy": [700, 40, 900, 90], "structure_ir": {}}
        rec = merge_balcony_into_windows_dsl(dsl, balcony_types=[], units=[unit], center_mode="window")["layout"]["balconies"][0]
        self.assertEqual(rec["outer_bay"], "right")
        self.assertAlmostEqual(rec["bay_cx_norm"], 0.75)
        self.assertNotIn("window_cx_norm", rec)

    def test_floor_follows_the_top_edge(self) -> None:
        floors = {0: (0, 100), 1: (100, 200)}
        self.assertEqual(floor_of_top_edge([0, 80, 9, 130], floors, 1), 0)  # most of it hangs below, into floor 1
        self.assertEqual(floor_of_top_edge([0, 150, 9, 180], floors, 0), 1)
        self.assertEqual(floor_of_top_edge([0, 500, 9, 520], floors, 1), 1)

    def test_two_outer_balconies_on_one_floor_both_stay(self) -> None:
        recs = [self._rec(2, [770, 637, 843, 708], bay_cx_norm=0.8, outer_bay="right"),
                self._rec(2, [768, 784, 843, 854], bay_cx_norm=0.8, outer_bay="right")]
        _move_stacks_together(recs, 1000)
        self.assertEqual([r["bay_cx_norm"] for r in recs], [0.8, 0.8])

    def test_balcony_takes_the_floor_of_the_window_above_it(self) -> None:
        floors = {0: (0, 100), 1: (100, 200), 2: (200, 300)}
        win = {"unit_id": 7, "floor": 0, "box_xyxy": [10, 40, 60, 95]}
        # corner-bay balcony hanging under a floor-0 window, its centre in floor 1's band
        partners, floor, far = attach_to_window([0, 100, 80, 150], [win], 1, floors)
        self.assertEqual((floor, far, partners), (0, [], [win]))

    def test_window_reaching_down_behind_the_railing_is_attached(self) -> None:
        floors = {0: (0, 100), 1: (100, 200)}
        door = {"unit_id": 3, "floor": 1, "box_xyxy": [10, 110, 60, 195]}
        self.assertEqual(attach_to_window([0, 160, 80, 200], [door], 1, floors)[1:], (1, []))

    def test_window_far_above_is_not_the_balconys(self) -> None:
        floors = {0: (0, 100), 1: (100, 200), 2: (200, 300)}
        win = {"unit_id": 5, "floor": 0, "box_xyxy": [10, 20, 60, 60]}  # 0.9 floor heights above the balcony
        partners, floor, far = attach_to_window([0, 150, 80, 190], [win], 1, floors)
        self.assertEqual((partners, floor, far), ([], 1, [5]))

    def test_window_below_the_balcony_is_not_the_balconys(self) -> None:
        floors = {0: (0, 100), 1: (100, 200), 2: (200, 300)}
        win = {"unit_id": 9, "floor": 2, "box_xyxy": [10, 220, 60, 290]}
        self.assertEqual(attach_to_window([0, 120, 80, 160], [win], 1, floors), ([], 1, [9]))

    def test_mean_bay_center_norm(self) -> None:
        bay_x = {0: (0, 100), 1: (100, 300), 2: (300, 400)}
        cx = _mean_bay_center_norm([0, 1, 2], bay_x, 1000.0)
        self.assertAlmostEqual(cx, (50 + 200 + 350) / 3 / 1000)

    def test_merge_per_unit_railing_type(self) -> None:
        dsl = {
            "schema": "facade_recovery_dsl_v1",
            "meta": {"image_size": [1000, 800]},
            "layout": {"floors": [], "bays": [], "placement": []},
            "window_types": [],
            "instances": [],
        }
        baluster_ir = {
            "structure": "projecting",
            "enclosure": "open",
            "railing": {"kind": "baluster", "height": 1.1},
        }
        solid_ir = {
            "structure": "projecting",
            "enclosure": "open",
            "railing": {"kind": "solid", "height": 1.1},
        }
        units = [
            {
                "unit_id": 0,
                "type_id": 0,
                "floor": 1,
                "bay_start": 0,
                "bay_end": 0,
                "bay": 0,
                "box_xyxy": [100, 10, 200, 80],
                "structure_ir": baluster_ir,
            },
            {
                "unit_id": 1,
                "type_id": 0,
                "floor": 2,
                "bay_start": 0,
                "bay_end": 0,
                "bay": 0,
                "box_xyxy": [100, 100, 200, 170],
                "structure_ir": solid_ir,
            },
        ]
        types = [
            {"name": "balc_baluster", "structure_ir": baluster_ir},
            {"name": "balc_solid", "structure_ir": solid_ir},
        ]
        out = merge_balcony_into_windows_dsl(
            dsl,
            balcony_types=types,
            units=units,
            per_unit_railing=True,
        )
        self.assertEqual(out["layout"]["balconies"][0]["type"], "balc_open_work_metal")
        self.assertEqual(out["layout"]["balconies"][1]["type"], "balc_solid")

    def test_merge_per_unit_floor_shape(self) -> None:
        dsl = {
            "schema": "facade_recovery_dsl_v1",
            "meta": {"image_size": [1000, 800]},
            "layout": {"floors": [], "bays": [], "placement": []},
            "window_types": [],
            "instances": [],
        }
        circle_ir = {
            "railing": {"kind": "open_work", "material": "metal"},
            "floor": {"shape": "circle"},
        }
        trap_ir = {
            "railing": {"kind": "open_work", "material": "metal"},
            "floor": {"shape": "triangle"},
        }
        units = [
            {
                "unit_id": 0,
                "type_id": 0,
                "floor": 1,
                "bay_start": 0,
                "bay_end": 0,
                "bay": 0,
                "box_xyxy": [100, 10, 200, 80],
                "structure_ir": circle_ir,
            },
            {
                "unit_id": 1,
                "type_id": 0,
                "floor": 2,
                "bay_start": 0,
                "bay_end": 0,
                "bay": 0,
                "box_xyxy": [100, 100, 200, 170],
                "structure_ir": trap_ir,
            },
        ]
        out = merge_balcony_into_windows_dsl(
            dsl,
            balcony_types=[
                {"name": "balc_open_work_metal_circle", "structure_ir": circle_ir},
                {"name": "balc_open_work_metal_triangle", "structure_ir": trap_ir},
            ],
            units=units,
            per_unit_floor=True,
        )
        self.assertEqual(
            out["layout"]["balconies"][0]["type"], "balc_open_work_metal_circle"
        )
        self.assertEqual(
            out["layout"]["balconies"][1]["type"], "balc_open_work_metal_triangle"
        )


class SharedBayTests(unittest.TestCase):
    """val30 London way289953461: a juliet over bays 3-4 linked only to window 14, a balcony over bays 2-3."""

    def test_shared_bay_goes_to_wider_cover_and_partners_follow(self) -> None:
        from merge_dsl import assign_bays_to_balconies

        bay_x = {2: (380, 500), 3: (505, 600), 4: (630, 700)}
        inst = [
            {"unit_id": 12, "floor": 2, "bay": 2, "box_xyxy": [387, 554, 437, 683]},
            {"unit_id": 13, "floor": 2, "bay": 3, "box_xyxy": [514, 556, 561, 687]},
            {"unit_id": 14, "floor": 2, "bay": 4, "box_xyxy": [644, 560, 686, 719]},
        ]
        a = {"floor": 2, "bay_start": 2, "bay_end": 3, "bays": [2, 3], "bays_center": [2], "bay": 2,
             "box_xyxy": [384, 672, 509, 770], "partner_window_unit_ids": [12], "window_cx_norm": 0.515, "_window_mode": True}
        b = {"floor": 2, "bay_start": 3, "bay_end": 4, "bays": [3, 4], "bays_center": [4], "bay": 4,
             "box_xyxy": [510, 673, 796, 772], "partner_window_unit_ids": [14], "window_cx_norm": 0.83, "_window_mode": True}
        assign_bays_to_balconies([a, b], inst, bay_x, 800.0)
        self.assertEqual(a["bays"], [2])
        self.assertEqual((a["bay_start"], a["bay_end"]), (2, 2))
        self.assertEqual(a["partner_window_unit_ids"], [12])
        self.assertEqual(b["bays"], [3, 4])
        self.assertEqual(b["partner_window_unit_ids"], [13, 14])
        self.assertAlmostEqual(b["window_cx_norm"], (537.5 + 665) / 2 / 800, places=5)


class TypeColumnTests(unittest.TestCase):
    def test_same_type_same_bays_share_median_centre_and_width(self) -> None:
        from merge_dsl import align_type_columns

        def rec(floor, bays, cx, w, key="cx_norm", typ="balc_type_02"):
            return {"floor": floor, "bays": bays, "type": typ, key: cx, "width_norm": w}

        col = [rec(0, [0, 1], 0.24, 0.25), rec(1, [0, 1], 0.26, 0.20, "window_cx_norm"), rec(3, [0, 1], 0.27, 0.21)]
        other_bays = rec(2, [0], 0.10, 0.08)
        wide = rec(4, [0, 1], 0.40, 0.60)  # 2x wider: not this column
        other_type = rec(2, [0, 1], 0.5, 0.2, typ="balc_type_01")
        align_type_columns(col + [other_bays, wide, other_type], 1000.0)
        for r in col:
            self.assertAlmostEqual(r.get("cx_norm", r.get("window_cx_norm")), 0.26)
            self.assertAlmostEqual(r["width_norm"], 0.21)
            self.assertTrue(r["column_aligned"])
        self.assertIn("window_cx_norm", col[1])  # keeps its own centre key
        for r in (other_bays, wide, other_type):
            self.assertNotIn("column_aligned", r)


if __name__ == "__main__":
    unittest.main()
