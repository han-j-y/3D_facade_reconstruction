"""Unit tests for balcony post-SAM3 filters."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from filter import (  # noqa: E402
    clearly_below_windows,
    filter_balcony_boxes,
    has_window_center_above,
    no_window_above,
    wider_than_partner_window,
)


class FilterTests(unittest.TestCase):
    def test_below_windows(self) -> None:
        # Window 50–150; expanded y1=155; shrunk sill=145
        win = [[100, 50, 200, 150]]
        # Below sill, overlaps expanded window in Y
        balc = [110, 148, 190, 260]
        self.assertTrue(
            clearly_below_windows(balc, win, iw=400, ih=400, shrink_frac=0.10)
        )
        balc_ok = [110, 120, 190, 180]
        self.assertFalse(
            clearly_below_windows(balc_ok, win, iw=400, ih=400, shrink_frac=0.10)
        )
        # Far below: no Y overlap even after expand → rule skipped
        far = [110, 200, 190, 260]
        self.assertFalse(
            clearly_below_windows(far, win, iw=400, ih=400, shrink_frac=0.10)
        )

    def test_top_not_center(self) -> None:
        # Thick box: cy below sill but y0 still inside shrunk band → keep
        win = [[100, 50, 200, 150]]
        thick = [110, 140, 190, 200]  # cy=170, y0=140 <= 145
        self.assertFalse(
            clearly_below_windows(thick, win, iw=400, ih=400, shrink_frac=0.10)
        )

    def test_shrink_drops_sill_nicks(self) -> None:
        win = [[100, 50, 200, 150]]
        balc_nick = [110, 146, 190, 200]
        self.assertTrue(
            clearly_below_windows(balc_nick, win, iw=400, ih=400, shrink_frac=0.10)
        )

    def test_expand_allows_near_miss(self) -> None:
        win = [[100, 50, 200, 150]]
        # slightly right of window; 10% expand should still overlap in X+Y
        balc = [201, 80, 250, 140]
        self.assertFalse(
            clearly_below_windows(balc, win, iw=400, ih=400, expand_frac=0.10)
        )

    def test_y_expand_links_decoration_gap(self) -> None:
        win = [[100, 50, 200, 150]]  # expanded y1=155; shrunk sill=145
        # No raw Y overlap with window (starts at 152) but overlaps expanded box
        deco = [110, 152, 190, 200]
        self.assertTrue(
            clearly_below_windows(deco, win, iw=400, ih=400, expand_frac=0.10)
        )
        # Gap too large for 10% Y expand → rule does not bind → keep
        far_deco = [110, 170, 190, 220]
        self.assertFalse(
            clearly_below_windows(far_deco, win, iw=400, ih=400, expand_frac=0.10)
        )

    def test_no_x_overlap_skips_rule(self) -> None:
        win = [[100, 50, 200, 150]]
        balc = [260, 200, 320, 260]
        self.assertFalse(
            clearly_below_windows(balc, win, iw=400, ih=400, shrink_frac=0.10)
        )

    def test_no_window_above(self) -> None:
        wins = [[100, 50, 200, 150], [100, 200, 200, 300]]
        rooftop = [110, 10, 190, 40]
        self.assertTrue(no_window_above(rooftop, wins, ih=400))
        balcony = [110, 220, 190, 280]
        self.assertFalse(no_window_above(balcony, wins, ih=400))

    def test_top_side_balcony_kept_if_any_window_higher(self) -> None:
        # Upper windows elsewhere in the photo (e.g. centre attic row).
        wins = [
            [200, 30, 280, 90],  # cy=60
            [20, 400, 90, 480],
        ]
        top_left_balcony = [10, 240, 90, 290]  # cy=265
        self.assertTrue(
            has_window_center_above(top_left_balcony, wins, ih=1024)
        )
        self.assertFalse(no_window_above(top_left_balcony, wins, ih=1024))

    def test_wide_exempts_below_windows(self) -> None:
        win = [[100, 50, 200, 150]]  # w=100
        im = Image.new("RGB", (400, 400), (200, 200, 200))
        instances = [{"box_xyxy": win[0]}]
        narrow_deco = [110, 152, 190, 200]  # w=80, would drop
        wide_balc = [70, 152, 190, 200]  # w=120 >= 1.2 * 100
        self.assertTrue(
            clearly_below_windows(narrow_deco, win, iw=400, ih=400, expand_frac=0.10)
        )
        self.assertTrue(wider_than_partner_window(wide_balc, win, iw=400, ih=400))
        kept, log = filter_balcony_boxes(
            [narrow_deco, wide_balc], im, instances
        )
        self.assertEqual(kept, [wide_balc])
        reasons = {tuple(e["box"]): e["reasons"] for e in log}
        self.assertIn("below_windows", reasons[tuple(narrow_deco)])
        self.assertIn("wide_vs_window", reasons[tuple(wide_balc)])

    def test_filter(self) -> None:
        im = Image.new("RGB", (400, 400), (200, 200, 200))
        instances = [{"box_xyxy": [100, 50, 200, 150]}]
        good = [110, 120, 190, 180]
        thick_ok = [110, 140, 190, 200]
        bad_below = [110, 148, 190, 320]
        bad_rooftop = [110, 5, 190, 35]
        kept, log = filter_balcony_boxes(
            [good, thick_ok, bad_below, bad_rooftop], im, instances
        )
        self.assertEqual(kept, [good, thick_ok])
        reasons = {tuple(e["box"]): e["reasons"] for e in log}
        self.assertIn("below_windows", reasons[tuple(bad_below)])
        self.assertIn("no_window_above", reasons[tuple(bad_rooftop)])


if __name__ == "__main__":
    unittest.main()
