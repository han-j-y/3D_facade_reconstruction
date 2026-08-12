"""2D boundary contours for all window shapes."""

from __future__ import annotations

import math
from typing import Any

from geom2d import Region, Segment, Vec2

_EPS = 1e-9


def contour_for_region(region: Region, outer: Region | None = None) -> list[Vec2]:
    if "contour" in region.params:
        return [Vec2(float(p[0]), float(p[1])) for p in region.params["contour"]]

    shape = region.shape
    p = region.params

    if shape == "rectangle":
        return _rect_contour(region.min_x, region.min_y, region.max_x, region.max_y)
    if shape == "circle":
        c = region.center
        r = float(p.get("radius", region.min_side / 2.0))
        return _circle_contour(c.x, c.y, r)
    if shape == "ellipse":
        c = region.center
        return _ellipse_contour(c.x, c.y, float(p.get("rx", region.width / 2.0)), float(p.get("ry", region.height / 2.0)))
    if shape == "semicircle":
        c = region.center
        r = float(p.get("radius", region.width / 2.0))
        flat = str(p.get("flat", "bottom"))
        return _semicircle_contour(c.x, c.y, r, flat=flat)
    if shape in {"eyebrow", "segmental_arch"}:
        w = float(p.get("width", region.width))
        rise = float(p.get("rise", p.get("arch_height", region.height)))
        return _eyebrow_contour(w, rise)
    if shape in {"trapezoid", "trapezoid_head"}:
        w = float(p.get("width", region.width))
        h = float(p.get("height", p.get("rise", region.height)))
        top_w = _trapezoid_head_top_width(p, w)
        return _trapezoid_contour(w, h, top_w)
    if shape in {"arch_head", "rect_eyebrow", "springline_arch"}:
        w = float(p.get("width", region.width))
        body_h = float(p.get("body_height", region.height - float(p.get("rise", 0.2))))
        rise = float(p.get("rise", p.get("arch_height", w * 0.15)))
        return _arch_head_contour(w, body_h, rise)
    if shape == "quadrant":
        r = float(p.get("radius", region.min_side))
        return _quadrant_contour(r, str(p.get("corner", "bl")))
    if shape == "triangle":
        w = float(p.get("width", region.width))
        h = float(p.get("height", region.height))
        return _trapezoid_contour(w, h, 0.0)
    if shape == "polygon":
        sides = int(p.get("sides", 6))
        c = region.center
        r = float(p.get("radius", min(region.width, region.height) / 2.0))
        rot = polygon_rotation(p)
        return _regular_polygon(c.x, c.y, r, sides, rot)
    if shape == "head_body":
        return _head_body_contour(outer or region, region)
    if shape == "composite":
        return _composite_contour(p)
    if shape == "spline" and "points" in p:
        return [Vec2(float(x), float(y)) for x, y in p["points"]]

    return _rect_contour(region.min_x, region.min_y, region.max_x, region.max_y)


def bounds_of_contour(pts: list[Vec2]) -> tuple[float, float, float, float]:
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return min(xs), max(xs), min(ys), max(ys)


def polygon_rotation(params: dict[str, Any]) -> float:
    """Default polygon orientation: pointy top (-π/2) or flat top (π/sides)."""
    if "rotation" in params:
        return float(params["rotation"])
    orient = str(params.get("orient", "pointy")).lower()
    sides = int(params.get("sides", 6))
    if orient in {"flat", "edge"}:
        return math.pi / sides
    return -math.pi / 2.0


def point_in_contour(p: Vec2, contour: list[Vec2]) -> bool:
    if len(contour) < 3:
        return False
    inside = False
    n = len(contour)
    for i in range(n):
        a = contour[i]
        b = contour[(i + 1) % n]
        dy = b.y - a.y
        if abs(dy) < _EPS:
            continue
        if (a.y > p.y) != (b.y > p.y):
            x_cross = a.x + (b.x - a.x) * (p.y - a.y) / dy
            if p.x < x_cross:
                inside = not inside
    return inside


def inset_contour(contour: list[Vec2], amount: float) -> list[Vec2]:
    if amount <= _EPS or len(contour) < 3:
        return list(contour)
    semicircle = _semicircle_params_from_contour(contour)
    if semicircle is not None:
        cx, cy, r, flat = semicircle
        return inset_semicircle_contour(cx, cy, r, amount, flat=flat, segments=len(contour) - 1)
    eyebrow = _eyebrow_params_from_contour(contour)
    if eyebrow is not None:
        left_x, flat_y, width, rise = eyebrow
        return inset_eyebrow_contour(width, rise, amount, flat_y=flat_y, left_x=left_x, segments=len(contour) - 1)
    if _is_near_circular(contour):
        c = _centroid(contour)
        r = _avg_radius(contour, c)
        return _circle_contour(c.x, c.y, max(r - amount, 1e-4))
    if _is_convex(contour):
        return _offset_convex_polygon(contour, amount)
    return _offset_polygon_per_edge(contour, amount)


def inset_semicircle_contour(
    cx: float, cy: float, r: float, amount: float, *, flat: str = "bottom", segments: int = 32
) -> list[Vec2]:
    """Inset a semicircle while keeping the flat chord horizontal."""
    d = amount
    r2 = max(r - d, 1e-4)
    if d >= r - _EPS:
        return [Vec2(cx, cy + (r2 if flat == "bottom" else -r2))]
    chord_half = math.sqrt(max(r2 * r2 - d * d, 0.0))
    if flat == "bottom":
        y_flat = cy + d
    else:
        y_flat = cy - d
    left = Vec2(cx - chord_half, y_flat)
    right = Vec2(cx + chord_half, y_flat)
    a0 = math.atan2(left.y - cy, left.x - cx)
    a1 = math.atan2(right.y - cy, right.x - cx)
    return [
        Vec2(
            cx + r2 * math.cos(a0 + (a1 - a0) * i / segments),
            cy + r2 * math.sin(a0 + (a1 - a0) * i / segments),
        )
        for i in range(segments + 1)
    ]


def _segmental_arch_radius(width: float, rise: float) -> tuple[float, float, float]:
    """Circle (cx, cy, R) for a chord of `width` with sagitta `rise` (flat chord at y=0).

    Rise is clamped to just under a semicircle (width/2). Larger rises would place the
    circle center above the chord; the short-arc interpolation then traces *below* the
    chord and the arch appears inverted / lens-shaped.
    """
    width = max(float(width), 1e-6)
    rise = max(float(rise), 1e-6)
    max_rise = 0.5 * width * (1.0 - 1e-4)
    if rise > max_rise:
        rise = max_rise
    r = (width * width + 4.0 * rise * rise) / (8.0 * rise)
    return width / 2.0, rise - r, r


def _eyebrow_contour(width: float, rise: float, *, segments: int = 32) -> list[Vec2]:
    """Segmental arch (eyebrow): flat bottom, shallow circular top."""
    cx, cy, r = _segmental_arch_radius(width, rise)
    a0 = math.atan2(-cy, -cx)
    a1 = math.atan2(-cy, width - cx)
    return [
        Vec2(cx + r * math.cos(a0 + (a1 - a0) * i / segments), cy + r * math.sin(a0 + (a1 - a0) * i / segments))
        for i in range(segments + 1)
    ]


def _trapezoid_head_top_width(params: dict[str, Any], width: float) -> float:
    return float(params.get("top_width", params.get("head_top_width", width * 0.65)))


def _trapezoid_contour(width: float, height: float, top_width: float) -> list[Vec2]:
    """Trapezoid head: flat bottom (springline), narrows toward top. top_width=0 → pointed apex."""
    cx = width / 2.0
    if top_width <= _EPS:
        return [Vec2(0.0, 0.0), Vec2(width, 0.0), Vec2(cx, height)]
    return [
        Vec2(0.0, 0.0),
        Vec2(width, 0.0),
        Vec2(cx + top_width / 2.0, height),
        Vec2(cx - top_width / 2.0, height),
    ]


def _trapezoid_top_points(cx: float, y: float, top_width: float) -> list[Vec2]:
    if top_width <= _EPS:
        return [Vec2(cx, y)]
    return [Vec2(cx + top_width / 2.0, y), Vec2(cx - top_width / 2.0, y)]


def _normalize_head_body_head(params: dict[str, Any]) -> None:
    head = str(params.get("head", "semicircle"))
    if head in {"triangle", "pointed", "pointed_arch"}:
        params["head"] = "trapezoid"
        params["top_width"] = 0.0
    elif head == "trapezoid":
        w = float(params.get("width", 1.0))
        params.setdefault("top_width", params.get("head_top_width", w * 0.65))


def _arch_head_contour(width: float, body_height: float, rise: float, *, segments: int = 48) -> list[Vec2]:
    """Rectangle body + segmental-arch head (springline at y=body_height).

    The arch polyline includes its springline endpoints so the jambs meet the curve
    without duplicate near-corner vertices (those caused shoulder holes in Blender).
    """
    arch = _eyebrow_contour(width, rise, segments=segments)
    # arch is left→right along the curve; reverse so we continue from right jamb over the top.
    top = [Vec2(p.x, p.y + body_height) for p in reversed(arch)]
    return [Vec2(0.0, 0.0), Vec2(width, 0.0)] + top


def inset_arch_head_contour(params: dict[str, Any], amount: float) -> list[Vec2]:
    """Inset rect + eyebrow-head boundary by frame thickness.

    Important: the springline is *not* an outer edge, so we must not raise the arch
    chord by `amount` (that left triangular shoulder gaps in the glass mesh). Keep the
    chord at body_height and shrink width/rise instead.
    """
    w = float(params.get("width", 1.0))
    body_h = float(params.get("body_height", 1.0))
    rise = float(params.get("rise", w * 0.15))
    t = float(amount)
    inner_w = max(w - 2.0 * t, 1e-4)
    inner_rise = min(max(rise - t, 1e-4), 0.5 * inner_w * (1.0 - 1e-4))
    arch = _eyebrow_contour(inner_w, inner_rise, segments=48)
    top = [Vec2(p.x + t, p.y + body_h) for p in reversed(arch)]
    return [Vec2(t, t), Vec2(w - t, t)] + top


def inset_eyebrow_contour(
    width: float,
    rise: float,
    amount: float,
    *,
    flat_y: float = 0.0,
    left_x: float = 0.0,
    segments: int = 32,
) -> list[Vec2]:
    """Inset a segmental arch while keeping the flat chord horizontal."""
    d = amount
    cx, cy, r = _segmental_arch_radius(width, rise)
    cx += left_x
    cy += flat_y
    r2 = max(r - d, 1e-4)
    if d >= rise - _EPS:
        return [Vec2(cx, flat_y + rise - d)]
    y_flat = flat_y + d
    chord_half = math.sqrt(max(r2 * r2 - (y_flat - cy) ** 2, 0.0))
    left = Vec2(cx - chord_half, y_flat)
    right = Vec2(cx + chord_half, y_flat)
    a0 = math.atan2(left.y - cy, left.x - cx)
    a1 = math.atan2(right.y - cy, right.x - cx)
    return [
        Vec2(
            cx + r2 * math.cos(a0 + (a1 - a0) * i / segments),
            cy + r2 * math.sin(a0 + (a1 - a0) * i / segments),
        )
        for i in range(segments + 1)
    ]


def _eyebrow_params_from_contour(contour: list[Vec2]) -> tuple[float, float, float, float] | None:
    """Return (left_x, flat_y, width, rise) for a flat-bottom segmental arch."""
    if len(contour) < 8 or _semicircle_params_from_contour(contour) is not None:
        return None
    left, right = contour[0], contour[-1]
    if abs(left.y - right.y) > 1e-4:
        return None
    flat_y = (left.y + right.y) / 2.0
    width = right.x - left.x
    if width < _EPS:
        return None
    rise = max(p.y for p in contour) - flat_y
    if rise < 1e-4 or rise > width * 0.55:
        return None
    cx, cy, r = _segmental_arch_radius(width, rise)
    cx += left.x
    cy += flat_y
    for p in contour:
        if abs(math.hypot(p.x - cx, p.y - cy) - r) > max(0.03 * r, 0.01):
            return None
    return left.x, flat_y, width, rise


def inset_head_body_contour(params: dict[str, Any], amount: float) -> list[Vec2]:
    """Inset a head_body boundary by frame thickness (handles concave arch)."""
    _normalize_head_body_head(params)
    w = float(params.get("width", 1.0))
    body_h = float(params.get("body_height", w * 0.65))
    head = str(params.get("head", "semicircle"))
    head_h = float(params.get("head_height", w * 0.5))
    cx = w / 2.0
    t = amount

    base = [Vec2(t, t), Vec2(w - t, t), Vec2(w - t, body_h), Vec2(t, body_h)]
    if head in {"semicircle", "arch", "segmental", "eyebrow"}:
        # Must match _head_body_contour: head_height is rise (full semi when ≈ width/2).
        max_semi = w / 2.0
        rise = float(head_h) if head_h > _EPS else max_semi
        rise = min(max(rise, w * 0.25), max_semi)
        if rise >= max_semi - 1e-4:
            arch = _semicircle_contour(cx, body_h, max(max_semi - t, 1e-4), flat="bottom")
            return base[:3] + arch[-2:0:-1] + [base[3]]
        inner_w = max(w - 2.0 * t, 1e-4)
        inner_rise = max(rise - t, 1e-4)
        arch = inset_eyebrow_contour(
            inner_w, inner_rise, t, flat_y=body_h, left_x=t, segments=32
        )
        return base[:3] + arch[-2:0:-1] + [base[3]]
    if head == "quadrant":
        return base[:2] + [Vec2(w - t, body_h + w / 2.0 - t), Vec2(cx, body_h + w / 2.0 - t), Vec2(t, body_h)]
    if head == "trapezoid":
        top_w = _trapezoid_head_top_width(params, w)
        y_top = body_h + max(head_h - t, 1e-4)
        top_w_i = 0.0 if top_w <= _EPS else max(top_w - 2.0 * t, 1e-4)
        top_pts = _trapezoid_top_points(cx, y_top, top_w_i)
        return base[:3] + top_pts + [Vec2(t, body_h)]
    arch = _semicircle_contour(cx, body_h, max(w / 2.0 - t, 1e-4), flat="bottom")
    return base[:3] + arch[1:-1] + [base[3]]


def clip_segment_to_rect(
    seg: Segment, x0: float, y0: float, x1: float, y1: float
) -> Segment | None:
    xmin, xmax = min(x0, x1), max(x0, x1)
    ymin, ymax = min(y0, y1), max(y0, y1)
    dx = seg.b.x - seg.a.x
    dy = seg.b.y - seg.a.y
    ts: set[float] = {0.0, 1.0}
    if abs(dx) > _EPS:
        for x in (xmin, xmax):
            t = (x - seg.a.x) / dx
            if 0.0 <= t <= 1.0:
                y = seg.a.y + t * dy
                if ymin - _EPS <= y <= ymax + _EPS:
                    ts.add(t)
    if abs(dy) > _EPS:
        for y in (ymin, ymax):
            t = (y - seg.a.y) / dy
            if 0.0 <= t <= 1.0:
                x = seg.a.x + t * dx
                if xmin - _EPS <= x <= xmax + _EPS:
                    ts.add(t)
    ordered = sorted(ts)
    best: Segment | None = None
    best_len = -1.0
    for i in range(len(ordered) - 1):
        ta, tb = ordered[i], ordered[i + 1]
        if tb - ta < 1e-8:
            continue
        mid = _lerp_seg(seg, (ta + tb) * 0.5)
        if not (xmin - _EPS <= mid.x <= xmax + _EPS and ymin - _EPS <= mid.y <= ymax + _EPS):
            continue
        a = _lerp_seg(seg, ta)
        b = _lerp_seg(seg, tb)
        length = math.hypot(b.x - a.x, b.y - a.y)
        if length > best_len:
            best_len = length
            best = Segment(seg.id, a, b, seg.layer)
    return best


def boundary_ts_for_region(seg: Segment, region: Region, *, infinite: bool = False) -> set[float]:
    ts = boundary_ts_for_contour(seg, contour_for_region(region), infinite=infinite)
    if "crop" in region.params:
        crop = _rect_contour(region.min_x, region.min_y, region.max_x, region.max_y)
        ts.update(boundary_ts_for_contour(seg, crop, infinite=infinite))
    return ts


def point_in_region(p: Vec2, region: Region) -> bool:
    return region.contains(p)


def clip_segment_to_contour(seg: Segment, contour: list[Vec2]) -> Segment | None:
    ts = {0.0, 1.0}
    n = len(contour)
    for i in range(n):
        a = contour[i]
        b = contour[(i + 1) % n]
        t = _segment_segment_intersection_t(seg, Segment("_", a, b))
        if t is not None:
            ts.add(t)
    ordered = sorted(ts)
    best: Segment | None = None
    best_len = -1.0
    for i in range(len(ordered) - 1):
        ta, tb = ordered[i], ordered[i + 1]
        if tb - ta < 1e-8:
            continue
        mid = _lerp_seg(seg, (ta + tb) * 0.5)
        if not point_in_contour(mid, contour):
            continue
        a = _lerp_seg(seg, ta)
        b = _lerp_seg(seg, tb)
        length = math.hypot(b.x - a.x, b.y - a.y)
        if length > best_len:
            best_len = length
            best = Segment(seg.id, a, b, seg.layer)
    return best


def boundary_ts_for_contour(seg: Segment, contour: list[Vec2], *, infinite: bool = False) -> set[float]:
    ts: set[float] = set()
    n = len(contour)
    for i in range(n):
        a = contour[i]
        b = contour[(i + 1) % n]
        t = _segment_segment_intersection_t(seg, Segment("_", a, b))
        if t is not None and (infinite or 0.0 <= t <= 1.0):
            ts.add(t if infinite else max(0.0, min(1.0, t)))
    return ts


def build_shape_region(spec: dict[str, Any]) -> Region:
    """Build a Region with contour stored in params."""
    shape = spec.get("shape", "rectangle")
    params = dict(spec.get("params") or {})
    rid = spec.get("id", "root")

    if shape == "rectangle":
        w = float(params.get("width", 1.0))
        h = float(params.get("height", 1.0))
        contour = _rect_contour(0.0, 0.0, w, h)
    elif shape == "circle":
        d = float(params.get("diameter", params.get("size", 1.0)))
        r = d / 2.0
        contour = _circle_contour(r, r, r)
        params["radius"] = r
    elif shape == "ellipse":
        w = float(params.get("width", 1.0))
        h = float(params.get("height", 1.0))
        contour = _ellipse_contour(w / 2.0, h / 2.0, w / 2.0, h / 2.0)
        params["rx"], params["ry"] = w / 2.0, h / 2.0
    elif shape == "semicircle":
        d = float(params.get("diameter", params.get("width", 1.0)))
        r = d / 2.0
        flat = str(params.get("flat", "bottom"))
        if flat == "bottom":
            contour = _semicircle_contour(r, 0.0, r, flat="bottom")
        else:
            contour = _semicircle_contour(r, r, r, flat="top")
        params["radius"] = r
    elif shape in {"eyebrow", "segmental_arch"}:
        w = float(params.get("width", params.get("size", 1.0)))
        rise = float(params.get("rise", params.get("arch_height", params.get("height", w * 0.2))))
        contour = _eyebrow_contour(w, rise)
        params["width"] = w
        params["rise"] = rise
    elif shape in {"trapezoid", "trapezoid_head"}:
        w = float(params.get("width", params.get("size", 1.0)))
        h = float(params.get("height", params.get("rise", w * 0.35)))
        top_w = _trapezoid_head_top_width(params, w)
        params["width"] = w
        params["height"] = h
        params["top_width"] = top_w
        contour = _trapezoid_contour(w, h, top_w)
    elif shape in {"arch_head", "rect_eyebrow", "springline_arch"}:
        w = float(params.get("width", params.get("size", 1.0)))
        body_h = float(params.get("body_height", params.get("body", 1.0)))
        rise = float(params.get("rise", params.get("arch_height", w * 0.18)))
        params["width"] = w
        params["body_height"] = body_h
        params["rise"] = rise
        contour = _arch_head_contour(w, body_h, rise)
    elif shape == "quadrant":
        r = float(params.get("radius", params.get("size", 1.0)))
        corner = str(params.get("corner", "bl"))
        contour = _quadrant_contour(r, corner)
        params["radius"] = r
        params["corner"] = corner
        ax, ay = _quadrant_right_angle(corner, r)
        params["origin_x"] = ax
        params["origin_y"] = ay
    elif shape == "triangle":
        w = float(params.get("width", 1.0))
        h = float(params.get("height", 1.0))
        kind = str(params.get("kind", "iso"))
        if kind == "right":
            contour = [Vec2(0, 0), Vec2(w, 0), Vec2(0, h)]
        else:
            contour = [Vec2(0, 0), Vec2(w, 0), Vec2(w / 2.0, h)]
    elif shape == "polygon":
        sides = int(params.get("sides", 6))
        w = float(params.get("width", 1.0))
        h = float(params.get("height", w))
        cx, cy = w / 2.0, h / 2.0
        r = float(params.get("radius", min(w, h) / 2.0))
        rot = polygon_rotation(params)
        contour = _regular_polygon(cx, cy, r, sides, rot)
        params["sides"] = sides
        params["radius"] = r
    elif shape == "head_body":
        w = float(params.get("width", 1.0))
        body_h = float(params.get("body_height", 1.0))
        head_h = float(params.get("head_height", w * 0.5))
        params.setdefault("body_height", body_h)
        params.setdefault("head_height", head_h)
        params.setdefault("head", "semicircle")
        _normalize_head_body_head(params)
        tmp = Region(rid, "head_body", 0, 0, w, body_h + head_h, params)
        contour = _head_body_contour(tmp, tmp)
    elif shape == "composite":
        contour = _composite_contour(params)
    elif shape == "spline":
        pts = params.get("points") or [[0, 0], [1, 0], [1, 1], [0, 1]]
        contour = [Vec2(float(x), float(y)) for x, y in pts]
    else:
        raise ValueError(f"unsupported boundary shape: {shape!r}")

    xmin, xmax, ymin, ymax = bounds_of_contour(contour)
    params["contour"] = [[p.x, p.y] for p in contour]
    return Region(rid, shape, xmin, ymin, xmax, ymax, params)


def _rect_contour(x0: float, y0: float, x1: float, y1: float) -> list[Vec2]:
    return [Vec2(x0, y0), Vec2(x1, y0), Vec2(x1, y1), Vec2(x0, y1)]


def _circle_contour(cx: float, cy: float, r: float, segments: int = 64) -> list[Vec2]:
    return [
        Vec2(cx + r * math.cos(2.0 * math.pi * i / segments), cy + r * math.sin(2.0 * math.pi * i / segments))
        for i in range(segments)
    ]


def _ellipse_contour(cx: float, cy: float, rx: float, ry: float, segments: int = 64) -> list[Vec2]:
    return [
        Vec2(cx + rx * math.cos(2.0 * math.pi * i / segments), cy + ry * math.sin(2.0 * math.pi * i / segments))
        for i in range(segments)
    ]


def _semicircle_params_from_contour(contour: list[Vec2]) -> tuple[float, float, float, str] | None:
    """Return (cx, cy, r, flat) when contour is a flat-bottom or flat-top semicircle."""
    if len(contour) < 8:
        return None
    left, right = contour[0], contour[-1]
    if abs(left.y - right.y) > 1e-4:
        return None
    flat_y = (left.y + right.y) / 2.0
    cx = (left.x + right.x) / 2.0
    r = abs(right.x - left.x) / 2.0
    if r < _EPS:
        return None
    cy = flat_y
    mids = contour[1:-1]
    if not mids:
        return None
    if max(p.y for p in mids) > flat_y + _EPS:
        flat = "bottom"
        if any(p.y + _EPS < flat_y for p in mids):
            return None
    elif min(p.y for p in mids) < flat_y - _EPS:
        flat = "top"
        if any(p.y - _EPS > flat_y for p in mids):
            return None
    else:
        return None
    for p in contour:
        if abs(math.hypot(p.x - cx, p.y - cy) - r) > max(0.02 * r, 0.01):
            return None
    return cx, cy, r, flat


def _semicircle_contour(cx: float, cy: float, r: float, *, flat: str = "bottom", segments: int = 32) -> list[Vec2]:
    """Upper/lower half-disk; the flat chord is the closing edge (not duplicated in the arc)."""
    if flat == "bottom":
        t0, t1 = math.pi, 0.0
    else:
        t0, t1 = math.pi, 2.0 * math.pi
    return [
        Vec2(cx + r * math.cos(t0 + (t1 - t0) * i / segments), cy + r * math.sin(t0 + (t1 - t0) * i / segments))
        for i in range(segments + 1)
    ]


def _quadrant_right_angle(corner: str, r: float) -> tuple[float, float]:
    """Return the square-corner vertex for a quadrant in [0, r]²."""
    corner = corner.lower()
    if corner == "bl":
        return 0.0, 0.0
    if corner == "br":
        return r, 0.0
    if corner == "tl":
        return 0.0, r
    if corner == "tr":
        return r, r
    raise ValueError(f"unsupported quadrant corner: {corner!r}")


def _quadrant_contour(r: float, corner: str = "bl", segments: int = 48) -> list[Vec2]:
    """Quarter-circle in [0, r]². `corner` is the right-angle vertex; arc is opposite."""
    corner = corner.lower()
    cx, cy = _quadrant_right_angle(corner, r)

    if corner == "bl":
        pts = [Vec2(cx, cy), Vec2(cx + r, cy)]
        angles = [(math.pi / 2.0) * i / segments for i in range(1, segments + 1)]
    elif corner == "br":
        pts = [Vec2(cx, cy), Vec2(cx - r, cy)]
        angles = [math.pi - (math.pi / 2.0) * i / segments for i in range(1, segments + 1)]
    elif corner == "tl":
        pts = [Vec2(cx, cy), Vec2(cx + r, cy)]
        angles = [-(math.pi / 2.0) * i / segments for i in range(1, segments + 1)]
    elif corner == "tr":
        pts = [Vec2(cx, cy), Vec2(cx - r, cy)]
        angles = [math.pi + (math.pi / 2.0) * i / segments for i in range(1, segments + 1)]
    else:
        raise ValueError(f"unsupported quadrant corner: {corner!r}")

    for t in angles:
        pts.append(Vec2(cx + r * math.cos(t), cy + r * math.sin(t)))
    return pts


def _triangle_contour(region: Region) -> list[Vec2]:
    p = region.params
    w, h = region.width, region.height
    kind = str(p.get("kind", "iso"))
    if kind == "right":
        return [Vec2(region.min_x, region.min_y), Vec2(region.max_x, region.min_y), Vec2(region.min_x, region.max_y)]
    return [
        Vec2(region.min_x, region.min_y),
        Vec2(region.max_x, region.min_y),
        Vec2((region.min_x + region.max_x) / 2.0, region.max_y),
    ]


def _regular_polygon(cx: float, cy: float, r: float, sides: int, rotation: float = 0.0) -> list[Vec2]:
    return [
        Vec2(cx + r * math.cos(rotation + 2.0 * math.pi * i / sides), cy + r * math.sin(rotation + 2.0 * math.pi * i / sides))
        for i in range(sides)
    ]


def _head_body_contour(outer: Region, inner: Region | None) -> list[Vec2]:
    p = outer.params
    _normalize_head_body_head(p)
    w = float(p.get("width", outer.width))
    body_h = float(p.get("body_height", outer.height * 0.65))
    head = str(p.get("head", "semicircle"))
    head_h = float(p.get("head_height", w * 0.5))
    cx = w / 2.0

    base = [Vec2(0, 0), Vec2(w, 0), Vec2(w, body_h), Vec2(0, body_h)]
    if head in {"semicircle", "arch", "segmental", "eyebrow"}:
        # Unified arch head: head_height is the rise above the springline.
        # rise ≈ width/2 → full semicircle; smaller rise → segmental (arch_head-like).
        max_semi = w / 2.0
        rise = float(head_h) if head_h > _EPS else max_semi
        # Keep arches clearly arched (avoid near-flat heads).
        rise = min(max(rise, w * 0.25), max_semi)
        if rise >= max_semi - 1e-4:
            arch = _semicircle_contour(cx, body_h, max_semi, flat="bottom")
            return base[:3] + arch[-2:0:-1] + [base[3]]
        arch = _eyebrow_contour(w, rise)
        top = [Vec2(p.x, p.y + body_h) for p in arch[-2:0:-1]]
        return base[:3] + top + [base[3]]
    if head == "quadrant":
        return base[:2] + [Vec2(w, body_h + w / 2.0), Vec2(cx, body_h + w / 2.0), Vec2(0, body_h)]
    if head == "trapezoid":
        top_w = _trapezoid_head_top_width(p, w)
        top_pts = _trapezoid_top_points(cx, body_h + head_h, top_w)
        return base[:3] + top_pts + [base[3]]
    return base[:3] + _semicircle_contour(cx, body_h, w / 2.0, flat="bottom")[1:-1] + [base[3]]


def _merge_contours(a: list[Vec2], b: list[Vec2]) -> list[Vec2]:
    out = list(a)
    for p in b:
        if not out or math.hypot(p.x - out[-1].x, p.y - out[-1].y) > 1e-4:
            if not any(math.hypot(p.x - q.x, p.y - q.y) < 1e-4 for q in out):
                out.append(p)
    return out


def _composite_contour(params: dict[str, Any]) -> list[Vec2]:
    pts = params.get("points") or params.get("contour")
    if pts:
        return [Vec2(float(x), float(y)) for x, y in pts]
    parts = params.get("parts") or []
    out: list[Vec2] = []
    for part in parts:
        sub = build_shape_region({"id": "_", **part})
        out = _merge_contours(out, contour_for_region(sub))
    return out or _rect_contour(0, 0, 1, 1)


def _centroid(pts: list[Vec2]) -> Vec2:
    x = sum(p.x for p in pts) / len(pts)
    y = sum(p.y for p in pts) / len(pts)
    return Vec2(x, y)


def _avg_radius(pts: list[Vec2], c: Vec2) -> float:
    return sum(math.hypot(p.x - c.x, p.y - c.y) for p in pts) / len(pts)


def _is_near_circular(pts: list[Vec2], tol: float = 0.08) -> bool:
    if len(pts) < 12:
        return False
    c = _centroid(pts)
    rs = [math.hypot(p.x - c.x, p.y - c.y) for p in pts]
    return (max(rs) - min(rs)) / max(sum(rs) / len(rs), 1e-6) < tol


def _is_convex(pts: list[Vec2]) -> bool:
    n = len(pts)
    if n < 3:
        return True
    sign = 0
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        c = pts[(i + 2) % n]
        cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
        if abs(cross) < _EPS:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def _offset_polygon_per_edge(pts: list[Vec2], d: float) -> list[Vec2]:
    """Approximate inset for simple concave polygons (head_body, etc.)."""
    n = len(pts)
    if n < 3:
        return list(pts)
    out: list[Vec2] = []
    for i in range(n):
        prev = pts[(i - 1) % n]
        cur = pts[i]
        nxt = pts[(i + 1) % n]
        e0 = _norm(Vec2(cur.x - prev.x, cur.y - prev.y))
        e1 = _norm(Vec2(nxt.x - cur.x, nxt.y - cur.y))
        n0 = Vec2(-e0.y, e0.x)
        n1 = Vec2(-e1.y, e1.x)
        bis = _norm(Vec2(n0.x + n1.x, n0.y + n1.y))
        dot = max(n0.x * bis.x + n0.y * bis.y, 0.3)
        scale = d / dot
        out.append(Vec2(cur.x + bis.x * scale, cur.y + bis.y * scale))
    return out


def _offset_convex_polygon(pts: list[Vec2], d: float) -> list[Vec2]:
    n = len(pts)
    if n < 3:
        return list(pts)
    out: list[Vec2] = []
    for i in range(n):
        prev = pts[(i - 1) % n]
        cur = pts[i]
        nxt = pts[(i + 1) % n]
        e0 = _norm(Vec2(cur.x - prev.x, cur.y - prev.y))
        e1 = _norm(Vec2(nxt.x - cur.x, nxt.y - cur.y))
        n0 = Vec2(-e0.y, e0.x)
        n1 = Vec2(-e1.y, e1.x)
        bis = _norm(Vec2(n0.x + n1.x, n0.y + n1.y))
        dot = max(n0.x * bis.x + n0.y * bis.y, 0.3)
        scale = d / dot
        out.append(Vec2(cur.x + bis.x * scale, cur.y + bis.y * scale))
    return out


def _norm(v: Vec2) -> Vec2:
    l = math.hypot(v.x, v.y)
    if l < _EPS:
        return Vec2(0.0, 0.0)
    return Vec2(v.x / l, v.y / l)


def _lerp_seg(seg: Segment, t: float) -> Vec2:
    return Vec2(seg.a.x + t * (seg.b.x - seg.a.x), seg.a.y + t * (seg.b.y - seg.a.y))


def _segment_segment_intersection_t(seg: Segment, edge: Segment) -> float | None:
    ax, ay = seg.a.x, seg.a.y
    bx, by = seg.b.x, seg.b.y
    cx, cy = edge.a.x, edge.a.y
    dx, dy = edge.b.x, edge.b.y
    denom = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx)
    if abs(denom) < _EPS:
        return None
    t = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / denom
    u = ((cx - ax) * (by - ay) - (cy - ay) * (bx - ax)) / denom
    if -1e-6 <= t <= 1.0 + 1e-6 and -1e-6 <= u <= 1.0 + 1e-6:
        return max(0.0, min(1.0, t))
    return None
