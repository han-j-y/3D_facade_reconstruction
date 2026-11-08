"""val30 type GT UI: completeness rules and label edits."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.type_gt_ui import complete, empty_unit, label_text, set_label, state_for, unit_complete  # noqa: E402


def unit(**kw):
    return {**empty_unit(), **kw}


class CompletenessTests(unittest.TestCase):
    def test_not_balcony_is_complete(self) -> None:
        self.assertTrue(unit_complete(unit(not_balcony=True)))

    def test_needs_group_kind_structure(self) -> None:
        self.assertFalse(unit_complete(unit(group=1, kind="solid")))
        self.assertTrue(unit_complete(unit(group=1, kind="solid", structure="juliet")))
        self.assertFalse(unit_complete(unit(kind="solid", structure="juliet")))

    def test_open_work_needs_material_projecting_needs_floor(self) -> None:
        self.assertFalse(unit_complete(unit(group=1, kind="open_work", structure="juliet")))
        self.assertTrue(unit_complete(unit(group=1, kind="open_work", material="metal", structure="juliet")))
        self.assertFalse(unit_complete(unit(group=1, kind="solid", structure="projecting")))
        self.assertTrue(unit_complete(unit(group=1, kind="solid", structure="projecting", floor_shape="circle")))

    def test_photo_complete(self) -> None:
        self.assertTrue(complete([unit(not_balcony=True), unit(group=2, kind="solid", structure="juliet")]))
        self.assertFalse(complete([unit(group=2)]))


class LabelEditTests(unittest.TestCase):
    def test_kind_change_clears_material(self) -> None:
        u = unit(kind="open_work", material="metal")
        set_label(u, "kind", "solid")
        self.assertIsNone(u["material"])

    def test_juliet_clears_floor(self) -> None:
        u = unit(structure="projecting", floor_shape="rectangle")
        set_label(u, "structure", "juliet")
        self.assertIsNone(u["floor_shape"])

    def test_label_text(self) -> None:
        self.assertEqual(label_text(unit(kind="open_work", material="metal", structure="projecting",
                                         floor_shape="circle")), "OW me P C")
        self.assertEqual(label_text(unit(kind="solid", structure="juliet")), "SO J")


class StateTests(unittest.TestCase):
    def test_saved_state_used_only_when_boxes_match(self) -> None:
        photo = {"id": "v00", "stem": "s", "boxes": [[0, 0, 10, 10]]}
        gt = {"s": {"units": [{"box": [0, 0, 10, 10], "group": 3, "not_balcony": False, "kind": "solid",
                               "material": None, "floor_shape": None, "structure": "juliet"}]}}
        self.assertEqual(state_for(photo, gt)[0]["group"], 3)
        gt["s"]["units"][0]["box"] = [0, 0, 11, 10]
        self.assertIsNone(state_for(photo, gt)[0]["group"])



class FillGroupTests(unittest.TestCase):
    def test_unlabeled_members_take_group_labels(self) -> None:
        from balcony_train.type_gt_ui import fill_group_labels

        st = [unit(group=1), unit(group=1, kind="open_work", material="metal", structure="juliet"), unit(group=1),
              unit(group=2), unit(not_balcony=True)]
        self.assertEqual(fill_group_labels(st), 2)
        self.assertEqual([u["kind"] for u in st[:3]], ["open_work"] * 3)
        self.assertEqual(st[0]["structure"], "juliet")
        self.assertIsNone(st[3]["kind"])  # group without labels stays empty
        self.assertIsNone(st[4]["kind"])

    def test_exceptions_kept_and_majority_wins(self) -> None:
        from balcony_train.type_gt_ui import fill_group_labels

        a = dict(kind="solid", structure="juliet")
        b = dict(kind="solid", structure="projecting", floor_shape="rectangle")
        st = [unit(group=1, **a), unit(group=1, **a), unit(group=1, **b), unit(group=1)]
        fill_group_labels(st)
        self.assertEqual(st[3]["structure"], "juliet")
        self.assertEqual(st[2]["structure"], "projecting")


class FillFieldsTests(unittest.TestCase):
    def test_partial_members_filled_field_by_field(self) -> None:
        from balcony_train.type_gt_ui import fill_group_labels

        st = [unit(group=1, structure="projecting"), unit(group=1, kind="open_work"),
              unit(group=1, kind="open_work", material="masonry", structure="projecting", floor_shape="rectangle")]
        fill_group_labels(st)
        for u in st:
            self.assertTrue(unit_complete(u))
            self.assertEqual((u["kind"], u["material"], u["structure"], u["floor_shape"]),
                             ("open_work", "masonry", "projecting", "rectangle"))

    def test_without_complete_member_uses_field_majority(self) -> None:
        from balcony_train.type_gt_ui import fill_group_labels

        st = [unit(group=1, kind="solid", structure="projecting"), unit(group=1, kind="solid")]
        fill_group_labels(st)
        self.assertEqual(st[1]["structure"], "projecting")
        self.assertFalse(unit_complete(st[1]))  # floor still missing: left for the user


class SetLabelGuardTests(unittest.TestCase):
    def test_no_floor_on_juliet_no_material_off_open_work(self) -> None:
        u = unit(kind="surface_panel", structure="juliet")
        set_label(u, "floor_shape", "rectangle")
        set_label(u, "material", "masonry")
        self.assertIsNone(u["floor_shape"])
        self.assertIsNone(u["material"])


if __name__ == "__main__":
    unittest.main()
