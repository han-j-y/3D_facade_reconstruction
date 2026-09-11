"""2D balcony slab outlines (no Blender).

Coordinates match BDSL: +X along the façade, +Y outward from the wall, +Z up.
A triangle is isosceles: wall-edge base, apex at the front center.
A circle is a semi-ellipse: wall diameter, outward depth (true semicircle when depth = diameter/2).
"""

from __future__ import annotations

import math
from typing import Sequence

BALUSTER_SPACING_M = 0.10
BALUSTER_DIAMETER_M = 0.02
TOP_RAIL_DIAMETER_M = 0.05

# Default projecting slab depth for all balcony types (meters).
DEFAULT_SLAB_DEPTH_M = 1.5

# open_work + material=masonry catalog proportions (meters).
MASONRY_POST_M = 0.25
MASONRY_BALUSTER_DIAMETER_M = 0.15
# Clear gap between baluster faces; center pitch = diameter + gap.
MASONRY_BALUSTER_GAP_M = 0.10
MASONRY_BALUSTER_SPACING_M = MASONRY_BALUSTER_DIAMETER_M + MASONRY_BALUSTER_GAP_M
MASONRY_TOP_RAIL_WIDTH_M = 0.30
MASONRY_TOP_RAIL_HEIGHT_M = 0.15
# Inset entire masonry railing (posts + balusters + top rail) from slab edge.
MASONRY_RAIL_INSET_M = 0.15

CIRCLE_ARC_STEPS = 24

Point = tuple[float, float]


def masonry_inset_rail_frame(
    outline: Sequence[Point],
    y_wall: float,
    inset: float,
    *,
    front_only: bool = False,
) -> tuple[list[Point], list[tuple[Point, Point]]]:
    """Inset open-U rail frame with shared corners (one post per corner).

    Returns ``(corners, edges)``. Corners are unique; edges share those
    points so front/side meet at a single post. Front/side are inset from the
    slab perimeter; side runs reach the wall (no wall-side gap).
    """
    inset = max(0.0, float(inset))
    if len(outline) < 2:
        return [], []
    xs = [float(p[0]) for p in outline]
    ys = [float(p[1]) for p in outline]
    x0, x1 = min(xs), max(xs)
    yw = float(y_wall)
    y_front = max(ys, key=lambda y: abs(y - yw))
    outward = 1.0 if y_front >= yw else -1.0
    depth = abs(y_front - yw)
    if x1 - x0 < 2.0 * inset + 0.05 or depth < inset + 0.05:
        inset = 0.0
    xi0 = x0 + inset
    xi1 = x1 - inset
    yf = y_front - outward * inset
    # Sides keep the lateral inset but run all the way to the wall (no wall gap).
    yw_end = yw

    if front_only:
        corners = [(xi0, yf), (xi1, yf)]
        edges = [((xi0, yf), (xi1, yf))]
        return corners, edges

    fl, fr = (xi0, yf), (xi1, yf)
    bl, br = (xi0, yw_end), (xi1, yw_end)
    corners = [bl, fl, fr, br]
    edges = [
        (bl, fl),
        (fl, fr),
        (fr, br),
    ]
    return corners, edges


def slab_outline(
    shape: str,
    *,
    x0: float,
    x1: float,
    y_wall: float,
    depth: float,
    structure: str = "projecting",
) -> list[Point]:
    """Closed polygon, wall-left → … → wall-right (CCW when depth is +Y)."""
    depth = max(0.05, float(depth))
    if x1 < x0:
        x0, x1 = x1, x0
    cx = 0.5 * (x0 + x1)
    inward = str(structure or "projecting") in {"inset", "composite"}
    y_out = y_wall - depth if inward else y_wall + depth
    name = (shape or "rectangle").lower()
    if name == "triangle":
        return [(x0, y_wall), (cx, y_out), (x1, y_wall)]
    if name == "circle":
        rx = 0.5 * (x1 - x0)
        dy = y_out - y_wall
        pts: list[Point] = []
        for i in range(CIRCLE_ARC_STEPS + 1):
            t = math.pi * (1.0 - i / CIRCLE_ARC_STEPS)
            pts.append((cx + rx * math.cos(t), y_wall + dy * math.sin(t)))
        return pts
    return [(x0, y_wall), (x0, y_out), (x1, y_out), (x1, y_wall)]


def centroid(verts: Sequence[Point]) -> Point:
    if not verts:
        return (0.0, 0.0)
    sx = sum(p[0] for p in verts)
    sy = sum(p[1] for p in verts)
    n = float(len(verts))
    return (sx / n, sy / n)


def outer_edges(
    outline: Sequence[Point],
    y_wall: float,
    *,
    atol: float = 1e-4,
) -> list[tuple[Point, Point]]:
    """Edges that are not the wall-side base."""
    n = len(outline)
    edges: list[tuple[Point, Point]] = []
    for i in range(n):
        a = outline[i]
        b = outline[(i + 1) % n]
        if abs(a[1] - y_wall) < atol and abs(b[1] - y_wall) < atol:
            continue
        edges.append((a, b))
    return edges


def support_polyline(shape: str, outline: Sequence[Point], y_wall: float) -> list[Point]:
    """Where free-standing posts go: V for triangle, front edge for a box."""
    if (shape or "rectangle").lower() == "triangle":
        return list(outline)
    front = [p for p in outline if abs(p[1] - y_wall) > 1e-4]
    return sorted(front, key=lambda p: p[0])


def rect_corner_posts(
    x0: float,
    x1: float,
    y_back: float,
    y_front: float,
    n: int,
    *,
    inset: float = 0.12,
) -> list[Point]:
    """Post XY for a rectangular slab. n=4 is the four corners (wall + front)."""
    if x1 < x0:
        x0, x1 = x1, x0
    if y_front < y_back:
        y_back, y_front = y_front, y_back
    x0 += inset
    x1 -= inset
    y_back += inset
    y_front -= inset
    bl, br = (x0, y_back), (x1, y_back)
    fl, fr = (x0, y_front), (x1, y_front)
    n = max(1, int(n))
    if n == 1:
        return [((x0 + x1) / 2.0, (y_back + y_front) / 2.0)]
    if n == 2:
        return [fl, fr]
    if n == 3:
        return [bl, br, ((x0 + x1) / 2.0, y_front)]
    pts = [bl, br, fl, fr]
    extra = n - 4
    if extra > 0:
        mids = even_along_polyline([fl, fr], extra + 2)
        pts.extend(mids[1:-1])
    return pts


def inset_edge(p0: Point, p1: Point, toward: Point, dist: float) -> tuple[Point, Point]:
    """Shift a segment toward ``toward`` by ``dist`` (keep it on the slab)."""
    ex, ey = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(ex, ey)
    if length < 1e-12:
        return p0, p1
    nx, ny = -ey / length, ex / length
    mx, my = 0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])
    if nx * (toward[0] - mx) + ny * (toward[1] - my) < 0:
        nx, ny = -nx, -ny
    ox, oy = nx * dist, ny * dist
    return (p0[0] + ox, p0[1] + oy), (p1[0] + ox, p1[1] + oy)


def inset_point(p: Point, toward: Point, dist: float) -> Point:
    dx, dy = toward[0] - p[0], toward[1] - p[1]
    length = math.hypot(dx, dy)
    if length < 1e-12:
        return p
    return (p[0] + dist * dx / length, p[1] + dist * dy / length)


def polyline_length(points: Sequence[Point]) -> float:
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += math.hypot(b[0] - a[0], b[1] - a[1])
    return total


def spaced_count(length: float, spacing: float = BALUSTER_SPACING_M) -> int:
    """How many posts along ``length`` at about ``spacing``, including both ends."""
    if length < 1e-6:
        return 0
    n_gaps = max(1, int(round(length / max(1e-6, spacing))))
    return n_gaps + 1


def shorten_segment_end(
    p0: Point,
    p1: Point,
    *,
    trim0: float = 0.0,
    trim1: float = 0.0,
) -> tuple[Point, Point] | None:
    """Trim ends of segment along its length. None if nothing remains."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return None
    ux, uy = dx / length, dy / length
    t0 = max(0.0, float(trim0))
    t1 = max(0.0, float(trim1))
    if t0 + t1 >= length - 1e-9:
        return None
    a = (p0[0] + ux * t0, p0[1] + uy * t0)
    b = (p1[0] - ux * t1, p1[1] - uy * t1)
    return a, b


def extend_segment_ends(
    p0: Point,
    p1: Point,
    *,
    ext0: float = 0.0,
    ext1: float = 0.0,
) -> tuple[Point, Point]:
    """Grow both ends of a segment along its direction."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return p0, p1
    ux, uy = dx / length, dy / length
    e0 = max(0.0, float(ext0))
    e1 = max(0.0, float(ext1))
    return (p0[0] - ux * e0, p0[1] - uy * e0), (p1[0] + ux * e1, p1[1] + uy * e1)


def masonry_baluster_centers(
    p0: Point,
    p1: Point,
    *,
    end_clear: float,
    pitch: float,
) -> list[Point]:
    """Centers on the rail centerline, fixed pitch, group centered in the clear span."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1e-9 or pitch <= 1e-9:
        return []
    clear = length - 2.0 * float(end_clear)
    if clear < 1e-6:
        return []
    # n centers with (n-1)*pitch <= clear, centered in the clear span.
    n = max(1, int(math.floor(clear / pitch + 1e-9)) + 1)
    while n > 1 and (n - 1) * pitch > clear + 1e-9:
        n -= 1
    span = 0.0 if n == 1 else (n - 1) * pitch
    start = float(end_clear) + 0.5 * (clear - span)
    ux, uy = dx / length, dy / length
    return [
        (p0[0] + ux * (start + i * pitch), p0[1] + uy * (start + i * pitch))
        for i in range(n)
    ]


def is_front_edge(
    p0: Point,
    p1: Point,
    y_wall: float,
    *,
    atol: float = 1e-4,
) -> bool:
    """True for the outward front run (both ends share front Y, off the wall).

    Side edges run toward the wall (different Y) and must not match.
    """
    if abs(p0[1] - p1[1]) > max(atol, 1e-3):
        return False
    return abs(p0[1] - y_wall) > atol and abs(p1[1] - y_wall) > atol


def even_along_polyline(points: Sequence[Point], n: int) -> list[Point]:
    """``n`` points equally spaced by arc length, including both ends when n>1."""
    n = max(0, int(n))
    if n == 0 or not points:
        return []
    if n == 1:
        if len(points) == 1:
            return [points[0]]
        return [even_along_polyline(points, 3)[1]]
    if len(points) == 1:
        return [points[0] for _ in range(n)]

    cum = [0.0]
    for a, b in zip(points, points[1:]):
        cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    total = cum[-1]
    if total < 1e-12:
        return [points[0] for _ in range(n)]

    out: list[Point] = []
    for i in range(n):
        s = 0.0 if n == 1 else total * i / (n - 1)
        placed = False
        for k in range(len(points) - 1):
            if cum[k + 1] + 1e-12 >= s or k == len(points) - 2:
                span = cum[k + 1] - cum[k]
                u = 0.0 if span < 1e-12 else (s - cum[k]) / span
                u = min(1.0, max(0.0, u))
                a, b = points[k], points[k + 1]
                out.append((a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])))
                placed = True
                break
        if not placed:
            out.append(points[-1])
    return out


MULLION_SPACING_M = 1.0
MULLION_WIDTH_M = 0.06
SURFACE_PANEL_M = 0.025
ENCLOSED_CLEAR_H_M = 2.5


def grid_stations(
    length: float,
    spacing: float = MULLION_SPACING_M,
    *,
    include_start: bool = True,
    include_end: bool = True,
) -> list[float]:
    """1 m (or ``spacing``) stations along a run of ``length``."""
    spacing = max(0.2, float(spacing))
    length = max(spacing, float(length))
    n = max(1, int(round(length / spacing)))
    step = length / n
    out: list[float] = []
    for i in range(n + 1):
        if i == 0 and not include_start:
            continue
        if i == n and not include_end:
            continue
        out.append(i * step)
    return out


def glass_bays(
    length: float,
    spacing: float = MULLION_SPACING_M,
    mull_w: float = MULLION_WIDTH_M,
    *,
    start_post: bool = True,
    end_post: bool = True,
) -> list[tuple[float, float]]:
    """Glass spans between mullion posts along a run of ``length``."""
    spacing = max(0.2, float(spacing))
    length = max(spacing, float(length))
    n = max(1, int(round(length / spacing)))
    step = length / n
    half = max(0.0, float(mull_w) / 2.0)
    bays: list[tuple[float, float]] = []
    for i in range(n):
        a = i * step
        b = length if i == n - 1 else (i + 1) * step
        has_lo = (i == 0 and start_post) or i > 0
        has_hi = (i == n - 1 and end_post) or i < n - 1
        lo = a + (half if has_lo else 0.0)
        hi = b - (half if has_hi else 0.0)
        if hi - lo > 0.02:
            bays.append((lo, hi))
    return bays


def _panel_divisions(length: float, spacing: float = MULLION_SPACING_M) -> int:
    """How many panel bays fit on ``length`` (floor, at least 1)."""
    spacing = max(0.2, float(spacing))
    length = max(0.2, float(length))
    return max(1, int(length / spacing))


def _even_stations(
    length: float,
    n: int,
    *,
    include_start: bool = True,
    include_end: bool = True,
) -> list[float]:
    n = max(1, int(n))
    length = max(1e-6, float(length))
    step = length / n
    out: list[float] = []
    for i in range(n + 1):
        if i == 0 and not include_start:
            continue
        if i == n and not include_end:
            continue
        out.append(i * step)
    return out


def _even_bays(
    length: float,
    n: int,
    mull_w: float,
    *,
    start_post: bool = True,
    end_post: bool = True,
) -> list[tuple[float, float]]:
    n = max(1, int(n))
    length = max(1e-6, float(length))
    step = length / n
    half = max(0.0, float(mull_w) / 2.0)
    bays: list[tuple[float, float]] = []
    for i in range(n):
        a = i * step
        b = length if i == n - 1 else (i + 1) * step
        has_lo = (i == 0 and start_post) or i > 0
        has_hi = (i == n - 1 and end_post) or i < n - 1
        lo = a + (half if has_lo else 0.0)
        hi = b - (half if has_hi else 0.0)
        if hi - lo > 0.02:
            bays.append((lo, hi))
    return bays


def _mullion_center(s: float, length: float, lo: float, hi: float, mull_w: float) -> float:
    if s <= 1e-9:
        return lo + mull_w / 2.0
    if s >= float(length) - 1e-9:
        return hi - mull_w / 2.0
    return lo + s


def surface_panel_parts(
    *,
    x0: float,
    x1: float,
    y_wall: float,
    depth: float,
    z_lo: float,
    rail_h: float,
    mull_w: float = MULLION_WIDTH_M,
    panel_t: float = SURFACE_PANEL_M,
    spacing: float = MULLION_SPACING_M,
    front_only: bool = False,
) -> list[dict[str, float | str]]:
    """Axis-aligned cubes for a framed surface-panel guard (60 mm mullions, 25 mm panels).

    Outer faces flush with the slab. U-shape open at the wall unless ``front_only``.
    Each dict: name, cx, cy, cz, sx, sy, sz.
    """
    if x1 < x0:
        x0, x1 = x1, x0
    width = max(0.3, float(x1) - float(x0))
    x0 = float(x0)
    x1 = x0 + width
    depth = max(mull_w, float(depth))
    y_wall = float(y_wall)
    y_front = y_wall + depth
    mull_w = max(0.02, float(mull_w))
    panel_t = max(0.008, min(float(panel_t), mull_w - 0.004))
    rail_h = max(2.0 * mull_w + 0.05, float(rail_h))
    z_lo = float(z_lo)
    z_mid = z_lo + rail_h / 2.0
    panel_h = max(0.04, rail_h - 2.0 * mull_w)
    z_panel = z_lo + mull_w + panel_h / 2.0
    z_bot = z_lo + mull_w / 2.0
    z_top = z_lo + rail_h - mull_w / 2.0
    cy_f = y_front - mull_w / 2.0
    n_front = _panel_divisions(width, spacing)

    parts: list[dict[str, float | str]] = []

    def _add(
        name: str,
        cx: float,
        cy: float,
        cz: float,
        sx: float,
        sy: float,
        sz: float,
    ) -> None:
        parts.append(
            {
                "name": name,
                "cx": cx,
                "cy": cy,
                "cz": cz,
                "sx": sx,
                "sy": sy,
                "sz": sz,
            }
        )

    for i, s in enumerate(
        _even_stations(width, n_front, include_start=True, include_end=True)
    ):
        _add(
            f"mull_f_{i}",
            _mullion_center(s, width, x0, x1, mull_w),
            cy_f,
            z_mid,
            mull_w,
            mull_w,
            rail_h,
        )
    _add("rail_f_bot", 0.5 * (x0 + x1), cy_f, z_bot, width, mull_w, mull_w)
    _add("rail_f_top", 0.5 * (x0 + x1), cy_f, z_top, width, mull_w, mull_w)
    for i, (a, b) in enumerate(
        _even_bays(width, n_front, mull_w, start_post=True, end_post=True)
    ):
        _add(
            f"panel_f_{i}",
            x0 + 0.5 * (a + b),
            cy_f,
            z_panel,
            b - a,
            panel_t,
            panel_h,
        )

    if front_only:
        return parts

    n_side = _panel_divisions(depth, spacing)
    for side, x_face, inward in (("L", x0, 1.0), ("R", x1, -1.0)):
        cx_m = x_face + inward * mull_w / 2.0
        for i, s in enumerate(
            _even_stations(depth, n_side, include_start=True, include_end=False)
        ):
            _add(
                f"mull_{side}_{i}",
                cx_m,
                _mullion_center(s, depth, y_wall, y_front, mull_w),
                z_mid,
                mull_w,
                mull_w,
                rail_h,
            )
        side_len = max(0.08, depth - mull_w)
        side_cy = y_wall + side_len / 2.0
        _add(f"rail_{side}_bot", cx_m, side_cy, z_bot, mull_w, side_len, mull_w)
        _add(f"rail_{side}_top", cx_m, side_cy, z_top, mull_w, side_len, mull_w)
        for i, (a, b) in enumerate(
            _even_bays(depth, n_side, mull_w, start_post=True, end_post=True)
        ):
            _add(
                f"panel_{side}_{i}",
                cx_m,
                y_wall + 0.5 * (a + b),
                z_panel,
                panel_t,
                b - a,
                panel_h,
            )
    return parts
