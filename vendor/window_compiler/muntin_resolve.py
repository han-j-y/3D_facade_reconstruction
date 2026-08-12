"""Trim muntin segments so every endpoint lies on the frame or another muntin."""

from __future__ import annotations

import math

from geom2d import Region, Segment, Vec2
from shapes import boundary_ts_for_region, contour_for_region, point_in_region

_EPS = 1e-7
_MIN_LEN = 1e-5


def resolve_muntins(
    segments: list[Segment],
    inner: Region,
    outer: Region | None,
    regions: dict[str, Region] | None = None,
) -> list[Segment]:
    if not segments:
        return []

    regions = regions or {}

    def _zone(seg: Segment) -> Region:
        if seg.clip_region_id and seg.clip_region_id in regions:
            return regions[seg.clip_region_id]
        return inner

    pieces = _split_all(
        segments,
        inner,
        outer,
        lambda seg, inn, out, infinite=False: boundary_ts_for_region(seg, _zone(seg), infinite=infinite),
        lambda p, seg: point_in_region(p, _zone(seg)),
        zone_fn=_zone,
    )
    return _relabel(pieces)


def _split_all(
    segments: list[Segment],
    inner: Region,
    outer: Region | None,
    boundary_ts_fn,
    contains_fn,
    zone_fn=None,
) -> list[Segment]:
    zone_fn = zone_fn or (lambda seg: inner)
    work = [
        _extend_segment_to_region(seg, zone_fn(seg), outer, boundary_ts_fn, contains_fn, seg)
        for seg in segments
    ]

    split_ts: list[set[float]] = []
    for seg in work:
        ts = {0.0, 1.0}
        ts.update(boundary_ts_fn(seg, inner, outer, infinite=False))
        for other in work:
            if other is seg:
                continue
            t = _segment_intersection_t(seg, other)
            if t is not None:
                ts.add(t)
            t2 = _segment_intersection_t(other, seg)
            if t2 is not None:
                u = _map_other_t_to_self(seg, other, t2)
                if u is not None:
                    ts.add(u)
        split_ts.append(ts)

    out: list[Segment] = []
    for seg, ts in zip(work, split_ts):
        ordered = sorted(ts)
        for i in range(len(ordered) - 1):
            ta, tb = ordered[i], ordered[i + 1]
            if tb - ta < _MIN_LEN / max(_seg_len(seg), _MIN_LEN):
                continue
            mid = _point_at(seg, (ta + tb) * 0.5)
            if not contains_fn(mid, seg):
                continue
            a = _point_at(seg, ta)
            b = _point_at(seg, tb)
            if _dist(a, b) < _MIN_LEN:
                continue
            out.append(Segment(id=seg.id, a=a, b=b, layer=seg.layer, clip_region_id=seg.clip_region_id))
    return out


def _relabel(segments: list[Segment]) -> list[Segment]:
    counts: dict[str, int] = {}
    out: list[Segment] = []
    for seg in segments:
        n = counts.get(seg.id, 0)
        counts[seg.id] = n + 1
        sid = seg.id if n == 0 else f"{seg.id}_{n}"
        out.append(Segment(id=sid, a=seg.a, b=seg.b, layer=seg.layer, clip_region_id=seg.clip_region_id))
    return out


def _seg_len(seg: Segment) -> float:
    return _dist(seg.a, seg.b)


def _dist(a: Vec2, b: Vec2) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _lerp(seg: Segment, t: float) -> Vec2:
    return Vec2(seg.a.x + t * (seg.b.x - seg.a.x), seg.a.y + t * (seg.b.y - seg.a.y))


def _point_at(seg: Segment, t: float) -> Vec2:
    return _lerp(seg, t)


def _extend_segment_to_region(
    seg: Segment,
    zone: Region,
    outer: Region | None,
    boundary_ts_fn,
    contains_fn,
    orig: Segment,
) -> Segment:
    dx = seg.b.x - seg.a.x
    dy = seg.b.y - seg.a.y
    if abs(dx) < _EPS and abs(dy) < _EPS:
        return seg

    ts = boundary_ts_fn(seg, zone, outer, infinite=True)
    if not ts:
        return seg
    ordered = sorted(ts)

    inside: list[tuple[float, float]] = []
    for i in range(len(ordered) - 1):
        ta, tb = ordered[i], ordered[i + 1]
        if tb - ta < _EPS:
            continue
        mid = _point_at(seg, (ta + tb) * 0.5)
        if not contains_fn(mid, orig):
            continue
        inside.append((ta, tb))

    if not inside:
        return seg

    merged: list[tuple[float, float]] = [inside[0]]
    for ta, tb in inside[1:]:
        la, lb = merged[-1]
        if ta <= lb + _EPS:
            merged[-1] = (la, max(lb, tb))
        else:
            merged.append((ta, tb))

    t_mid = 0.5
    chosen: tuple[float, float] | None = None
    for ta, tb in merged:
        if ta - _EPS <= t_mid <= tb + _EPS:
            chosen = (ta, tb)
            break
    if chosen is None:
        chosen = max(merged, key=lambda iv: min(iv[1], 1.0) - max(iv[0], 0.0))

    return Segment(
        id=seg.id,
        a=_point_at(seg, chosen[0]),
        b=_point_at(seg, chosen[1]),
        layer=seg.layer,
        clip_region_id=orig.clip_region_id,
    )


def _segment_intersection_t(seg: Segment, other: Segment) -> float | None:
    ax, ay = seg.a.x, seg.a.y
    bx, by = seg.b.x, seg.b.y
    cx, cy = other.a.x, other.a.y
    dx, dy = other.b.x, other.b.y
    denom = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx)
    if abs(denom) < _EPS:
        return None
    t = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / denom
    u = ((cx - ax) * (by - ay) - (cy - ay) * (bx - ax)) / denom
    if -_EPS <= t <= 1.0 + _EPS and -_EPS <= u <= 1.0 + _EPS:
        return max(0.0, min(1.0, t))
    return None


def _map_other_t_to_self(seg: Segment, other: Segment, u: float) -> float | None:
    px = other.a.x + u * (other.b.x - other.a.x)
    py = other.a.y + u * (other.b.y - other.a.y)
    dx = seg.b.x - seg.a.x
    dy = seg.b.y - seg.a.y
    if abs(dx) > abs(dy):
        if abs(dx) < _EPS:
            return None
        t = (px - seg.a.x) / dx
    else:
        if abs(dy) < _EPS:
            return None
        t = (py - seg.a.y) / dy
    if -_EPS <= t <= 1.0 + _EPS:
        return max(0.0, min(1.0, t))
    return None
