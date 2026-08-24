"""Snap balcony boxes onto the window recovery floor×bay layout."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _bands_from_window_instances(
    instances: list[dict[str, Any]],
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    """floor id → (y0, y1), bay id → (x0, x1) from window unit boxes."""
    floors: dict[int, list[list[int]]] = defaultdict(list)
    bays: dict[int, list[list[int]]] = defaultdict(list)
    for u in instances:
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


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    lo, hi = max(a0, b0), min(a1, b1)
    return max(0.0, hi - lo)


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

    ids = [b for b, _ in bay_hits] or [0]
    return {
        "floor": int(floor_id),
        "bay_start": int(min(ids)),
        "bay_end": int(max(ids)),
        "bay": int(max(ids, key=lambda b: dict(bay_hits).get(b, 0.0))) if bay_hits else 0,
    }


def snap_units(
    boxes: list[list[int]],
    windows_dsl: dict[str, Any],
) -> list[dict[str, Any]]:
    instances = windows_dsl.get("instances") or []
    floor_y, bay_x = _bands_from_window_instances(instances)
    out = []
    for i, box in enumerate(boxes):
        loc = snap_box(box, floor_y, bay_x)
        out.append({"unit_id": i, "box_xyxy": [int(v) for v in box], **loc})
    return out
