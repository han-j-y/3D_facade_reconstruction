"""Facade region: SAM3's box grown around the elements, or the whole image."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from facade_region import choose_facade_region  # noqa: E402

WINDOWS = [[100, 100, 150, 180], [300, 100, 350, 180]]


class ChooseTests(unittest.TestCase):
    def test_takes_the_box_holding_the_windows(self) -> None:
        box, source = choose_facade_region([([50, 40, 400, 300], 0.9), ([700, 0, 800, 300], 0.95)], WINDOWS, (900, 400))
        self.assertEqual((box, source), ([50.0, 40.0, 400.0, 300.0], "sam3"))

    def test_grows_to_hold_every_element(self) -> None:
        box, _ = choose_facade_region([([50, 40, 320, 150], 0.9)], WINDOWS, (900, 400))
        self.assertEqual(box, [50.0, 40.0, 350.0, 180.0])

    def test_clips_to_the_image(self) -> None:
        box, _ = choose_facade_region([([-3, -2, 905, 402], 0.9)], WINDOWS, (900, 400))
        self.assertEqual(box, [0.0, 0.0, 900.0, 400.0])

    def test_no_detection_is_the_whole_image(self) -> None:
        self.assertEqual(choose_facade_region([], WINDOWS, (900, 400)), ([0.0, 0.0, 900.0, 400.0], "image"))

    def test_low_score_is_the_whole_image(self) -> None:
        self.assertEqual(choose_facade_region([([50, 40, 400, 300], 0.4)], WINDOWS, (900, 400))[1], "image")

    def test_box_without_the_windows_is_the_whole_image(self) -> None:
        self.assertEqual(choose_facade_region([([700, 0, 800, 300], 0.9)], WINDOWS, (900, 400))[1], "image")


if __name__ == "__main__":
    unittest.main()
