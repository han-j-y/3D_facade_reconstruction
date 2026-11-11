import unittest

from facade_recovery.fold_gate import vertical_pairs


def grid(widths):
    """Boxes on a floors x 2 grid (rows top to bottom), left bay at x 100, right bay mirrored about x 300."""
    boxes, floor = [], {}
    for f, (wl, wr) in enumerate(widths):
        y = 100 * f
        boxes.append([150 - wl / 2, y, 150 + wl / 2, y + 60])
        boxes.append([450 - wr / 2, y, 450 + wr / 2, y + 60])
        floor[2 * f], floor[2 * f + 1] = f, f
    return boxes, floor


class VerticalRatioTests(unittest.TestCase):
    def test_wider_ratio_links_boxes_1_2x_apart(self):
        boxes, floor = grid([(60, 60), (72, 72)])
        occ = [set() for _ in boxes]
        self.assertEqual(vertical_pairs(boxes, floor, occ), [])
        self.assertEqual(sorted(vertical_pairs(boxes, floor, occ, 1.3, 1.3)), [(0, 2), (1, 3)])


if __name__ == "__main__":
    unittest.main()
