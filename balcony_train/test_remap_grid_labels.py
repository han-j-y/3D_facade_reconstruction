import unittest

from balcony_train.remap_grid_labels import remap

FLOORS = {0: (0, 100), 1: (100, 200)}


class RemapTests(unittest.TestCase):
    def test_same_grid_keeps_cells(self):
        bays = {0: (0, 100), 1: (100, 200), 2: (200, 300)}
        cells, ignore, reasons = remap([(1, 1)], [(0, 2)], (FLOORS, bays), (FLOORS, bays))
        self.assertEqual(cells, [(1, 1)])
        self.assertEqual(ignore, [(0, 2)])
        self.assertEqual(reasons, [])

    def test_merged_columns_take_majority_cover(self):
        old = {0: (0, 100), 1: (100, 200), 2: (200, 300)}
        new = {0: (0, 200), 1: (200, 300)}
        # both old columns under new column 0 are labeled: clean
        cells, _ig, reasons = remap([(1, 0), (1, 1)], [], (FLOORS, old), (FLOORS, new))
        self.assertEqual(cells, [(1, 0)])
        self.assertEqual(reasons, [])
        # only one of them: new column 0 is half covered, so it is labeled and flagged
        cells, _ig, reasons = remap([(1, 1)], [], (FLOORS, old), (FLOORS, new))
        self.assertEqual(cells, [(1, 0)])
        self.assertTrue(any(r.startswith("partial f1b0") for r in reasons))

    def test_lost_cell_is_flagged(self):
        old = {0: (0, 100), 1: (100, 140), 2: (140, 300)}
        new = {0: (0, 300)}
        cells, _ig, reasons = remap([(0, 1)], [], (FLOORS, old), (FLOORS, new))
        self.assertEqual(cells, [])
        self.assertTrue(any(r.startswith("lost old f0b1") for r in reasons))


if __name__ == "__main__":
    unittest.main()
