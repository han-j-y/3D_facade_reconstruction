import copy
import unittest

from grid_order import apply_grid_order


def _dsl() -> dict:
    # 3 floors (100 px each) x 2 bays (200 px each); one window per cell, one balcony under window 2 (floor 1, bay 0)
    insts = []
    uid = 0
    for f in range(3):
        for b in range(2):
            insts.append({"unit_id": uid, "floor": f, "bay": b, "type_id": f, "box_xyxy": [b * 200 + 50, f * 100 + 20, b * 200 + 150, f * 100 + 70]})
            uid += 1
    return {
        "meta": {"image_size": [400, 300], "floors_y": [[0, 100], [100, 200], [200, 300]], "columns_xy": [[0, 200], [200, 400]],
                 "facade_region_xyxy": [0, 0, 400, 300]},
        "layout": {
            "floors": [{"name": f"F{i}", "id": i, "h_norm": 1 / 3} for i in range(3)],
            "bays": [{"name": f"B{i}", "id": i, "w_norm": 0.5} for i in range(2)],
            "placement": [[f"win_type_{f:02d}"] * 2 for f in range(3)],
            "placement_spans": [[1, 1] for _ in range(3)],
            "balconies": [{"floor": 1, "bay_start": 0, "bay_end": 0, "bay": 0, "bays": [0], "type": "balc_type_00",
                           "box_xyxy": [40, 170, 160, 200], "width_norm": 0.3, "window_cx_norm": 0.25, "partner_window_unit_ids": [2]}],
        },
        "instances": insts,
    }


class GridOrderTest(unittest.TestCase):
    def test_shared_balcony_grows_over_copy_of_its_first_bay(self) -> None:
        d = _dsl()
        d["layout"]["balconies"] = [{"floor": 1, "bay_start": 0, "bay_end": 1, "bay": 0, "bays": [0, 1], "type": "balc_type_00",
                                     "box_xyxy": [40, 170, 360, 200], "width_norm": 0.8, "window_cx_norm": 0.5,
                                     "partner_window_unit_ids": [2, 3]}]
        d["layout"]["bay_order"] = [0, 0, 1]
        (b,) = apply_grid_order(d)["layout"]["balconies"]  # one balcony, not two overlapping copies
        self.assertEqual((b["bay_start"], b["bay_end"], b["bays"], b["box_xyxy"][2]), (0, 1, [0, 2, 1], 560))

    def test_single_window_balcony_is_copied_with_its_window(self) -> None:
        d = _dsl()
        d["layout"]["balconies"][0].update({"bay_end": 1, "bays": [0, 1]})  # one window, box over two bays
        d["layout"]["bay_order"] = [0, 0, 1]
        bs = [b for b in apply_grid_order(d)["layout"]["balconies"]]
        self.assertEqual(sorted(b["box_xyxy"][0] for b in bs), [40, 240])

    def _spanned(self) -> dict:
        # floor 0: one wide window spanning both bays (box 20..380); floor 1: a narrow window spanning both bays (box 50..150)
        return {
            "meta": {"image_size": [400, 300], "floors_y": [[0, 100], [100, 200], [200, 300]], "columns_xy": [[0, 200], [200, 400]]},
            "layout": {
                "floors": [{"name": f"F{i}", "id": i, "h_norm": 1 / 3} for i in range(3)],
                "bays": [{"name": f"B{i}", "id": i, "w_norm": 0.5} for i in range(2)],
                "placement": [["win_type_00", None], ["win_type_01", None], [None, None]],
                "placement_spans": [[2, 0], [2, 0], [1, 1]],
                "balconies": [],
            },
            "instances": [{"unit_id": 0, "floor": 0, "bay": 0, "colspan": 2, "type_id": 0, "box_xyxy": [20, 20, 380, 70]},
                          {"unit_id": 1, "floor": 1, "bay": 0, "colspan": 2, "type_id": 1, "box_xyxy": [50, 120, 150, 170]}],
        }

    def test_copy_of_first_bay_stretches_an_element_wider_than_it(self) -> None:
        d = self._spanned()
        d["layout"]["bay_order"] = [0, 0, 1]
        out = apply_grid_order(d)
        top = [u for u in out["instances"] if u["floor"] == 0]
        self.assertEqual([u["box_xyxy"][0] for u in top], [20])          # one window, not an overlapping duplicate
        self.assertEqual(top[0]["box_xyxy"][2], 580)                       # stretched by the copied 200 px
        self.assertEqual(out["layout"]["placement_spans"][0], [3, 0, 0])
        mid = sorted(u["box_xyxy"][0] for u in out["instances"] if u["floor"] == 1)
        self.assertEqual(mid, [50, 250])                                   # narrow one: duplicated with its bay

    def test_copy_of_inner_bay_after_the_span_is_wall(self) -> None:
        d = self._spanned()
        d["layout"]["bay_order"] = [0, 1, 1]
        out = apply_grid_order(d)
        mid = [u for u in out["instances"] if u["floor"] == 1]
        self.assertEqual([u["box_xyxy"] for u in mid], [[50, 120, 150, 170]])  # stays where it was, same size
        self.assertEqual(out["layout"]["placement_spans"][1][:2], [2, 0])

    def test_long_balcony_grows_over_copied_inner_bay(self) -> None:
        # balcony across both bays of floor 1; copying bay 1 after itself puts a copy inside its run
        d = _dsl()
        d["layout"]["balconies"] = [{"floor": 1, "bay_start": 0, "bay_end": 1, "bay": 0, "bays": [0, 1], "type": "balc_type_00",
                                     "box_xyxy": [40, 170, 360, 200], "width_norm": 0.8, "window_cx_norm": 0.5,
                                     "partner_window_unit_ids": [2, 3]}]
        d["layout"]["bay_order"] = [0, 1, 1]
        out = apply_grid_order(d)
        (b,) = out["layout"]["balconies"]
        self.assertEqual((b["bay_start"], b["bay_end"], b["bays"]), (0, 2, [0, 1, 2]))
        self.assertEqual(b["box_xyxy"][2], 560)  # 200 px wider
        self.assertAlmostEqual(b["width_norm"], 520 / 600, places=5)
        self.assertAlmostEqual(b["window_cx_norm"], 300 / 600, places=5)
        partners = {(u["floor"], u["bay"]) for u in out["instances"] if u["unit_id"] in b["partner_window_unit_ids"]}
        self.assertEqual(partners, {(1, 0), (1, 1), (1, 2)})

    def test_no_lists_and_identity_lists_change_nothing(self) -> None:
        d = _dsl()
        self.assertEqual(apply_grid_order(d), d)
        e = copy.deepcopy(d)
        e["layout"]["floor_order"], e["layout"]["bay_order"] = [0, 1, 2], [0, 1]
        self.assertIs(apply_grid_order(e), e)  # the recovered order: returned as is, no copy

    def test_swap_floors_moves_their_elements(self) -> None:
        d = _dsl()
        d["layout"]["floor_order"] = [1, 0, 2]
        out = apply_grid_order(d)
        self.assertEqual([f["id"] for f in out["layout"]["floors"]], [1, 0, 2])
        self.assertEqual(out["layout"]["placement"][0], ["win_type_01"] * 2)
        w2 = next(u for u in out["instances"] if u["unit_id"] == 2)
        self.assertEqual(w2["box_xyxy"][1], 20)  # floor 1 now on top
        self.assertEqual(out["layout"]["balconies"][0]["box_xyxy"][1], 70)
        self.assertEqual(out["meta"]["image_size"], [400, 300])

    def test_copied_floor_gets_new_ids_and_its_balcony_its_window(self) -> None:
        d = _dsl()
        d["layout"]["floor_order"] = [0, 1, 1, 2]  # one more copy of floor 1
        out = apply_grid_order(d)
        self.assertEqual([f["id"] for f in out["layout"]["floors"]], [0, 1, 3, 2])
        self.assertEqual(out["meta"]["image_size"], [400, 400])
        self.assertEqual(len(out["instances"]), 8)
        copy_b = [b for b in out["layout"]["balconies"] if b["floor"] == 3][0]
        partner = next(u for u in out["instances"] if u["unit_id"] == copy_b["partner_window_unit_ids"][0])
        self.assertEqual((partner["floor"], partner["bay"]), (3, 0))
        self.assertEqual(copy_b["box_xyxy"][1], 270)
        self.assertEqual(out["meta"]["facade_region_xyxy"][3], 400)

    def test_deleted_bay_drops_its_elements_and_width(self) -> None:
        d = _dsl()
        d["layout"]["bay_order"] = [1]
        out = apply_grid_order(d)
        self.assertEqual(out["meta"]["image_size"], [200, 300])
        self.assertTrue(all(u["bay"] == 1 for u in out["instances"]))
        self.assertEqual(out["layout"]["balconies"], [])
        w = next(u for u in out["instances"] if u["unit_id"] == 1)
        self.assertEqual(w["box_xyxy"][0], 50)  # moved left with its bay

    def test_norms_follow_the_new_image_width(self) -> None:
        d = _dsl()
        d["layout"]["bay_order"] = [1, 0]
        b = apply_grid_order(d)["layout"]["balconies"][0]
        self.assertAlmostEqual(b["window_cx_norm"], (0.25 * 400 + 200) / 400)

    def test_unknown_id_is_an_error(self) -> None:
        d = _dsl()
        d["layout"]["floor_order"] = [0, 9]
        with self.assertRaises(ValueError):
            apply_grid_order(d)


    def _wide(self) -> dict:
        # 1 floor x 4 bays (100 px each); a door spans bays 0-3, a window spans bays 2-3 on a second floor
        return {
            "meta": {"image_size": [400, 200], "floors_y": [[0, 100], [100, 200]],
                     "columns_xy": [[0, 100], [100, 200], [200, 300], [300, 400]]},
            "layout": {
                "floors": [{"name": "F0", "id": 0, "h_norm": 0.5}, {"name": "F1", "id": 1, "h_norm": 0.5}],
                "bays": [{"name": f"B{i}", "id": i, "w_norm": 0.25} for i in range(4)],
                "placement": [[None, None, "win_type_00", None], ["d00", None, None, None]],
                "placement_spans": [[1, 1, 2, 0], [4, 0, 0, 0]],
            },
            "instances": [
                {"unit_id": 0, "floor": 0, "bay": 2, "colspan": 2, "type_id": 0, "box_xyxy": [220, 20, 280, 70]},
                {"unit_id": 1, "floor": 1, "bay": 0, "colspan": 4, "kind": "door", "type_id": 0, "box_xyxy": [150, 120, 250, 200]},
            ],
        }

    def test_span_grows_over_copies_inserted_inside_it(self) -> None:
        d = self._wide()
        d["layout"]["bay_order"] = [0, 1, 2, 1, 2, 3]  # copies of bays 1, 2 inside the door's span
        out = apply_grid_order(d)
        self.assertEqual(out["layout"]["placement_spans"][1][0], 6)
        door = next(u for u in out["instances"] if u.get("kind") == "door")
        self.assertEqual(door["colspan"], 6)
        self.assertEqual(door["box_xyxy"][2] - door["box_xyxy"][0], 100)  # keeps its width
        self.assertEqual(0.5 * (door["box_xyxy"][0] + door["box_xyxy"][2]), 300)  # still centred in its span

    def test_broken_span_covers_its_element_with_empty_slots(self) -> None:
        d = self._wide()
        d["layout"]["bay_order"] = [0, 1, 2, 1, 2, 3]  # the window's bay 2 is now followed by bay 1's copy
        out = apply_grid_order(d)
        row = out["layout"]["placement_spans"][0]
        self.assertEqual(row[2], 1)  # 60 px window from x=220 fits in its own 100 px bay
        self.assertEqual(row[4], 2)  # the copy of bay 2 is still followed by bay 3

if __name__ == "__main__":
    unittest.main()
