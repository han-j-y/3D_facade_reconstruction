"""Pure geometry helpers for masonry open_work layout (no Blender)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

COMPILER = Path(__file__).resolve().parent
sys.path.insert(0, str(COMPILER))

from balcony_plan import (  # noqa: E402
    MASONRY_BALUSTER_DIAMETER_M,
    MASONRY_BALUSTER_GAP_M,
    MASONRY_BALUSTER_SPACING_M,
    MASONRY_RAIL_INSET_M,
    extend_segment_ends,
    is_front_edge,
    masonry_baluster_centers,
    masonry_inset_rail_frame,
    shorten_segment_end,
)


class MasonryLayoutTests(unittest.TestCase):
    def test_pitch_is_diameter_plus_gap(self) -> None:
        self.assertAlmostEqual(
            MASONRY_BALUSTER_SPACING_M,
            MASONRY_BALUSTER_DIAMETER_M + MASONRY_BALUSTER_GAP_M,
        )
        self.assertAlmostEqual(MASONRY_BALUSTER_GAP_M, 0.10)

    def test_baluster_centers_use_fixed_pitch_and_center(self) -> None:
        pts = masonry_baluster_centers(
            (0.0, 0.0),
            (3.0, 0.0),
            end_clear=0.2,
            pitch=0.25,
        )
        self.assertGreaterEqual(len(pts), 2)
        for a, b in zip(pts, pts[1:]):
            self.assertAlmostEqual(b[0] - a[0], 0.25, places=6)
        # Group centered in clear span [0.2, 2.8]
        mid = 0.5 * (pts[0][0] + pts[-1][0])
        self.assertAlmostEqual(mid, 1.5, places=5)

    def test_shorten_front_end_of_side(self) -> None:
        # Side edge from wall (y=0) to front (y=1.5); trim front by 0.15
        out = shorten_segment_end((0.0, 0.0), (0.0, 1.5), trim0=0.0, trim1=0.15)
        assert out is not None
        self.assertAlmostEqual(out[0][1], 0.0)
        self.assertAlmostEqual(out[1][1], 1.35)

    def test_extend_front_by_half_post(self) -> None:
        a, b = extend_segment_ends((1.0, 1.5), (3.0, 1.5), ext0=0.125, ext1=0.125)
        self.assertAlmostEqual(a[0], 0.875)
        self.assertAlmostEqual(b[0], 3.125)

    def test_rail_inset_constant(self) -> None:
        self.assertAlmostEqual(MASONRY_RAIL_INSET_M, 0.15)

    def test_inset_frame_shared_corners_no_dupes(self) -> None:
        outline = [(0.0, 0.0), (0.0, 1.5), (4.0, 1.5), (4.0, 0.0)]
        corners, edges = masonry_inset_rail_frame(outline, 0.0, 0.15)
        self.assertEqual(len(corners), 4)
        self.assertEqual(len(edges), 3)
        # Front span = slab width − 2·inset
        fl, fr = edges[1]
        self.assertAlmostEqual(fr[0] - fl[0], 4.0 - 0.30)
        # Wall-end posts sit on the wall; front stays inset.
        self.assertAlmostEqual(corners[0][1], 0.0)
        self.assertAlmostEqual(corners[1][1], 1.5 - 0.15)
        self.assertEqual(edges[0][1], edges[1][0])
        self.assertEqual(edges[1][1], edges[2][0])

    def test_front_edge_detect(self) -> None:
        self.assertTrue(is_front_edge((0.0, 1.5), (2.0, 1.5), y_wall=0.0))
        self.assertFalse(is_front_edge((0.0, 0.0), (0.0, 1.5), y_wall=0.0))
        # Inset side: both ends off the wall, but still a side (different Y).
        self.assertFalse(is_front_edge((0.15, 0.15), (0.15, 1.35), y_wall=0.0))
        self.assertTrue(is_front_edge((0.15, 1.35), (3.85, 1.35), y_wall=0.0))


if __name__ == "__main__":
    unittest.main()
