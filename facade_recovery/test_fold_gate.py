import unittest

import numpy as np

from facade_recovery.fold_gate import adopt_singletons, vertical_pairs


def grid(widths):
    """Boxes on a floors x 2 grid (rows top to bottom), left bay at x 100, right bay mirrored about x 300."""
    boxes, floor = [], {}
    for f, (wl, wr) in enumerate(widths):
        y = 100 * f
        boxes.append([150 - wl / 2, y, 150 + wl / 2, y + 60])
        boxes.append([450 - wr / 2, y, 450 + wr / 2, y + 60])
        floor[2 * f], floor[2 * f + 1] = f, f
    return boxes, floor


AXES = [{"floor": f, "x": 300.0, "pairs": 1} for f in range(3)]


class AdoptSingletonsTests(unittest.TestCase):
    def test_singleton_joins_when_mirror_above_and_below_agree(self):
        boxes, floor = grid([(60, 60), (40, 60), (60, 60)])
        labels = np.array([0, 0, 1, 0, 0, 0])
        outline = {i: "rect" for i in floor}
        out, changes = adopt_singletons(boxes, floor, outline, labels, AXES)
        self.assertEqual(out.tolist(), [0, 0, 0, 0, 0, 0])
        self.assertEqual(changes, [(2, 1, 0)])

    def test_no_change_when_a_witness_disagrees(self):
        boxes, floor = grid([(60, 60), (40, 60), (60, 60)])
        labels = np.array([2, 0, 1, 0, 0, 0])  # the window above is another type
        outline = {i: "rect" for i in floor}
        out, changes = adopt_singletons(boxes, floor, outline, labels, AXES)
        self.assertEqual(out.tolist(), labels.tolist())
        self.assertEqual(changes, [])

    def test_no_change_when_outline_differs(self):
        boxes, floor = grid([(60, 60), (40, 60), (60, 60)])
        labels = np.array([0, 0, 1, 0, 0, 0])
        outline = {i: "rect" for i in floor}
        outline[2] = "arched"
        out, changes = adopt_singletons(boxes, floor, outline, labels, AXES)
        self.assertEqual(changes, [])

    def test_top_floor_singleton_has_too_few_witnesses(self):
        boxes, floor = grid([(40, 60), (60, 60), (60, 60)])
        labels = np.array([1, 0, 0, 0, 0, 0])  # nothing above floor 0
        out, changes = adopt_singletons(boxes, floor, {i: "rect" for i in floor}, labels, AXES)
        self.assertEqual(changes, [])


class VerticalRatioTests(unittest.TestCase):
    def test_wider_ratio_links_boxes_1_2x_apart(self):
        boxes, floor = grid([(60, 60), (72, 72)])
        occ = [set() for _ in boxes]
        self.assertEqual(vertical_pairs(boxes, floor, occ), [])
        self.assertEqual(sorted(vertical_pairs(boxes, floor, occ, 1.3, 1.3)), [(0, 2), (1, 3)])


if __name__ == "__main__":
    unittest.main()
