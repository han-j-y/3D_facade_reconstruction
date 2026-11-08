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


if __name__ == "__main__":
    unittest.main()
