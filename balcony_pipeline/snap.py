"""Snap balcony boxes onto the window recovery floor×bay layout."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _window_instances(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for u in instances:
        if str(u.get("kind") or "window") == "door":
            continue
        if not u.get("box_xyxy"):
            continue
        out.append(u)
    return out


def _bands_from_window_instances(
    instances: list[dict[str, Any]],
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    """floor id → (y0, y1), bay id → (x0, x1) from window unit boxes."""
    floors: dict[int, list[list[int]]] = defaultdict(list)
    bays: dict[int, list[list[int]]] = defaultdict(list)
    for u in _window_instances(instances):
        box = [int(v) for v in u["box_xyxy"]]
        floors[int(u["floor"])].append(box)
        bays[int(u["bay"])].append(box)
    floor_y = {}
    for fid, boxes in floors.items():
        floor_y[fid] = (min(b[1] for b in boxes), max(b[3] for b in boxes))
    bay_x = {}
    for bid, boxes in bays.items():
        bay_x[bid] = (min(b[0] for b in boxes), max(b[2] for b in boxes))
    return floor_y, bay_x


def _bands_from_layout(
    windows_dsl: dict[str, Any],
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    """Prefer meta.columns_xy / meta.floors_y when present (main structural grid)."""
    meta = windows_dsl.get("meta") or {}
    layout = windows_dsl.get("layout") or {}
    floors = list(layout.get("floors") or [])
    bays = list(layout.get("bays") or [])
    columns_xy = meta.get("columns_xy")
    floors_y = meta.get("floors_y")

    bay_x: dict[int, tuple[int, int]] = {}
    if (
        isinstance(columns_xy, list)
        and bays
        and len(columns_xy) == len(bays)
    ):
        for j, rec in enumerate(bays):
            pair = columns_xy[j]
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                bay_x = {}
                break
            bay_x[int(rec["id"])] = (int(round(float(pair[0]))), int(round(float(pair[1]))))

    floor_y: dict[int, tuple[int, int]] = {}
    if isinstance(floors_y, list) and floors and len(floors_y) == len(floors):
        for i, rec in enumerate(floors):
            pair = floors_y[i]
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                floor_y = {}
                break
            floor_y[int(rec["id"])] = (
                int(round(float(pair[0]))),
                int(round(float(pair[1]))),
            )

    return floor_y, bay_x


def _bands_from_windows_dsl(
    windows_dsl: dict[str, Any],
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    """Layout bands if authored; else min/max of window instance boxes."""
    floor_y, bay_x = _bands_from_window_instances(windows_dsl.get("instances") or [])
    ly, lx = _bands_from_layout(windows_dsl)
    if ly:
        floor_y = ly
    if lx:
        bay_x = lx
    return floor_y, bay_x


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    lo, hi = max(a0, b0), min(a1, b1)
    return max(0.0, hi - lo)


def _bay_overlap_stats(
    x0: float,
    x1: float,
    bx0: float,
    bx1: float,
) -> tuple[float, float]:
    """Return (overlap_px, cover_frac) where cover_frac = overlap / bay_width."""
    overlap_px = _overlap_1d(x0, x1, bx0, bx1)
    bay_w = max(1.0, bx1 - bx0)
    return overlap_px, overlap_px / bay_w


def center_bay_ids(
    box: list[int] | list[float],
    bay_x: dict[int, tuple[int, int]],
    *,
    full_cover_frac: float = 1.0,
) -> list[int]:
    """Pick bay(s) that define balcony horizontal center.

    1. If any bay has cover_frac >= full_cover_frac, use all such bays.
    2. Else use the single bay with largest overlap_px (tie: nearer balcony cx, lower id).
    3. Mean of their X centers is applied later in merge_dsl.
    """
    x0, x1 = float(box[0]), float(box[2])
    cx = 0.5 * (x0 + x1)
    stats: list[tuple[int, float, float]] = []
    for bid, (bx0, bx1) in bay_x.items():
        ov_px, cover = _bay_overlap_stats(x0, x1, float(bx0), float(bx1))
        if ov_px <= 0.0:
            continue
        stats.append((int(bid), ov_px, cover))

    if not stats:
        return [0]

    thr = float(full_cover_frac)
    fully = sorted(bid for bid, _, cover in stats if cover >= thr - 1e-9)
    if fully:
        return fully

    def _rank(item: tuple[int, float, float]) -> tuple[float, float, int]:
        bid, ov_px, _ = item
        bx0, bx1 = bay_x[bid]
        bay_cx = 0.5 * (float(bx0) + float(bx1))
        dist = abs(cx - bay_cx)
        return (ov_px, -dist, -bid)

    return [max(stats, key=_rank)[0]]


def snap_box(
    box: list[int],
    floor_y: dict[int, tuple[int, int]],
    bay_x: dict[int, tuple[int, int]],
    *,
    min_frac: float = 0.15,
) -> dict[str, Any]:
    """Assign floor + inclusive bay span by 1D overlap with window bands."""
    x0, y0, x1, y1 = [float(v) for v in box]
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)

    floor_hits: list[tuple[float, int]] = []
    for fid, (fy0, fy1) in floor_y.items():
        ov = _overlap_1d(y0, y1, fy0, fy1) / bh
        if ov >= min_frac:
            floor_hits.append((ov, fid))
    floor_id = max(floor_hits)[1] if floor_hits else (
        min(floor_y, key=lambda f: abs(0.5 * (y0 + y1) - 0.5 * sum(floor_y[f])))
        if floor_y
        else 0
    )

    bay_hits: list[tuple[int, float]] = []
    for bid, (bx0, bx1) in sorted(bay_x.items()):
        ov = _overlap_1d(x0, x1, bx0, bx1) / bw
        if ov >= min_frac:
            bay_hits.append((bid, ov))
    if not bay_hits and bay_x:
        cx = 0.5 * (x0 + x1)
        nearest = min(bay_x, key=lambda b: abs(cx - 0.5 * sum(bay_x[b])))
        bay_hits = [(nearest, 1.0)]

    ids = sorted({int(b) for b, _ in bay_hits}) or [0]
    bays_center = center_bay_ids(box, bay_x)
    return {
        "floor": int(floor_id),
        "bay_start": int(min(ids)),
        "bay_end": int(max(ids)),
        "bays": ids,
        "bays_center": bays_center,
        "bay": int(max(ids, key=lambda b: dict(bay_hits).get(b, 0.0))) if bay_hits else 0,
    }


def snap_units(
    boxes: list[list[int]],
    windows_dsl: dict[str, Any],
) -> list[dict[str, Any]]:
    floor_y, bay_x = _bands_from_windows_dsl(windows_dsl)
    out = []
    for i, box in enumerate(boxes):
        loc = snap_box(box, floor_y, bay_x)
        out.append({"unit_id": i, "box_xyxy": [int(v) for v in box], **loc})
    return out
