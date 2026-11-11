import unittest

import numpy as np

from facade_recovery.joint_vote import column_weights, g, joint_vote, shutter_width_uncertain, symmetry_weights


def column_grid(widths, x=150.0, step=100.0, h=60.0):
    """One window per floor, stacked in a column; floor f at y = step * f."""
    boxes = [[x - w / 2, step * f, x + w / 2, step * f + h] for f, w in enumerate(widths)]
    return boxes, {i: i for i in range(len(boxes))}


class GTests(unittest.TestCase):
    def test_g_zero_below_half_and_one_at_full(self):
        self.assertEqual(g(0.5), 0.0)
        self.assertEqual(g(0.3), 0.0)
        self.assertAlmostEqual(g(1.0), 1.0)
        self.assertAlmostEqual(g(0.75), 0.25)


class ShutterTests(unittest.TestCase):
    def test_side_shutter_flags_and_blind_does_not(self):
        mask = np.zeros((200, 400), bool)
        boxes = [[50, 50, 110, 150], [250, 50, 310, 150]]
        mask[50:150, 40:50] = True  # folded shutter just left of window 0
        mask[50:150, 250:310] = True  # window 1 fully covered: blind
        self.assertEqual(shutter_width_uncertain(boxes, mask), [True, False])
        self.assertEqual(shutter_width_uncertain(boxes, None), [False, False])


class WeightTests(unittest.TestCase):
    def test_column_weight_ignores_width_when_uncertain(self):
        boxes, floor = column_grid([60, 40])  # 1.5x apart in width
        occ = [set(), set()]
        self.assertAlmostEqual(column_weights(boxes, floor, occ, [False, False])[0, 1], g(40 / 60))  # x IoU 0.67
        self.assertAlmostEqual(column_weights(boxes, floor, occ, [False, True])[0, 1], 1.0)

    def test_symmetry_weight_for_a_mirror_pair(self):
        boxes = [[100, 0, 160, 60], [240, 0, 300, 60]]
        W = symmetry_weights(boxes, {0: 0, 1: 0}, [set(), set()], [False, False], [{"floor": 0, "x": 200.0}])
        self.assertAlmostEqual(W[0, 1], 1.0)


class VoteTests(unittest.TestCase):
    def test_narrow_shuttered_window_joins_its_column(self):
        boxes, floor = column_grid([60, 60, 40, 60, 60])  # window 2 narrowed by a shutter
        labels = np.array([0, 0, 1, 0, 0])
        outline = {i: "rect" for i in floor}
        feat = np.ones((5, 4))
        occ = [set()] * 5
        same, _ = joint_vote(boxes, labels, floor, occ, outline, feat, [], [False] * 5)
        self.assertEqual(same.tolist(), [0, 0, 1, 0, 0])  # width 1.5x apart: not compatible
        new, rep = joint_vote(boxes, labels, floor, occ, outline, feat, [], [False, False, True, False, False])
        self.assertEqual(new.tolist(), [0, 0, 0, 0, 0])
        self.assertEqual(rep["moves"], [[2, 1, 0]])

    def test_different_outline_never_joins(self):
        boxes, floor = column_grid([60, 60, 60])
        outline = {0: "rect", 1: "arched", 2: "rect"}
        new, _ = joint_vote(boxes, np.array([0, 1, 0]), floor, [set()] * 3, outline, np.ones((3, 4)), [], [False] * 3)
        self.assertEqual(new.tolist(), [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
