"""Left-right symmetry repair for enclosed-volume boxes (Stage 6).

Adapted from ``walkthrough_symmetry_repair.py`` in the main repo: the axis is
the middle of the window grid, a box is mirrored about it and matched by IoU
or a soft center distance. Here the repair edits boxes instead of window types.

1. Axis = middle of the window bay extent. If the windows themselves are not
   mirror-symmetric about it (score < ``SYM_GATE``), nothing changes.
2. A box with no mirror partner gets its mirror added when that spot is in
   front of the wall by the depth threshold and lands on the window grid.
3. Mirror pairs share one vertical range. Each box extends to the pair's
   union only where the added strip is in front of the wall.
4. A box whose mirror spot fails the depth or grid check is dropped, unless
   it sits on the axis or its mirror falls outside the photo.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from window_ast.symmetry import facade_bilateral_symmetry  # noqa: E402

SYM_GATE = 0.75
WINDOW_MIRROR_IOU = 0.25
MATCH_IOU = 0.15
MATCH_CENTER = 0.35
# A box overlapping its own mirror this much sits on the axis.
SELF_MIRROR_IOU = 0.50
# Mirror spots with less of their width inside the photo are not judged.
MIN_IN_FRAME = 0.50
MAX_REL = 0.22
DUPLICATE_IOU = 0.30

Box = list[int]


def window_boxes_from_dsl(dsl: dict[str, Any]) -> list[Box]:
    """Window (not door) instance boxes from a window ``facade_dsl.json``."""
    out: list[Box] = []
    for u in dsl.get("instances") or []:
        if str(u.get("kind") or "window") == "door":
            continue
        box = u.get("box_xyxy")
        if box:
            out.append([int(v) for v in box[:4]])
    return out


def grid_axis_x(bay_x: dict[int, tuple[int, int]]) -> float | None:
    if not bay_x:
        return None
    x0 = min(float(a) for a, _ in bay_x.values())
    x1 = max(float(b) for _, b in bay_x.values())
    return 0.5 * (x0 + x1)


def window_symmetry(windows: list[Box], cx: float) -> float:
    if not windows:
        return 0.0
    return float(facade_bilateral_symmetry(windows, cx, iou_thresh=WINDOW_MIRROR_IOU))


def reflect_x(box: Box, cx: float) -> list[float]:
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    return [2.0 * cx - x1, y0, 2.0 * cx - x0, y1]


def box_iou(a: list[float], b: list[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ba = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / max(aa + ba - inter, 1e-8))


def match_reflected(
    target: list[float],
    boxes: list[Box],
    *,
    iou_thr: float = MATCH_IOU,
    center_thr: float = MATCH_CENTER,
    exclude: int | None = None,
) -> tuple[int | None, float]:
    """Best match to ``target``; score = IoU + soft center (center over diagonal)."""
    tcx = 0.5 * (target[0] + target[2])
    tcy = 0.5 * (target[1] + target[3])
    diag = max(1.0, float(np.hypot(target[2] - target[0], target[3] - target[1])))
    best_i, best_s = None, -1.0
    for i, b in enumerate(boxes):
        if exclude is not None and i == exclude:
            continue
        fb = [float(v) for v in b[:4]]
        iou = box_iou(target, fb)
        dist = float(np.hypot(tcx - 0.5 * (fb[0] + fb[2]), tcy - 0.5 * (fb[1] + fb[3]))) / diag
        if iou < iou_thr and dist > center_thr:
            continue
        s = iou + max(0.0, 1.0 - dist)
        if s > best_s:
            best_s, best_i = s, i
    return best_i, best_s


def _in_frame_frac(box: list[float], iw: int) -> float:
    w = box[2] - box[0]
    if w <= 0:
        return 0.0
    return max(0.0, min(box[2], float(iw)) - max(box[0], 0.0)) / w


def _clamp(box: list[float], iw: int, ih: int) -> Box:
    x0 = int(round(max(0.0, min(box[0], iw))))
    x1 = int(round(max(0.0, min(box[2], iw))))
    y0 = int(round(max(0.0, min(box[1], ih))))
    y1 = int(round(max(0.0, min(box[3], ih))))
    return [x0, y0, x1, y1]


def symmetry_repair(
    recs: list[dict[str, Any]],
    *,
    cx: float | None,
    window_sym: float,
    iw: int,
    ih: int,
    rel_fn: Callable[[Box], float | None],
    on_grid: Callable[[Box], bool],
    min_rel: float,
    gate: float = SYM_GATE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (boxes, log). ``rel_fn`` is the box-vs-local-wall depth ratio."""
    log: dict[str, Any] = {"axis_x": cx, "window_sym": window_sym, "added": 0, "aligned": 0, "dropped": 0}
    if cx is None:
        log["skip"] = "no grid"
        return [dict(r) for r in recs], log
    if window_sym < gate:
        log["skip"] = f"window sym {window_sym:.2f} < {gate:.2f}"
        return [dict(r) for r in recs], log

    items = [dict(r, box_xyxy=[int(v) for v in r["box_xyxy"][:4]]) for r in recs]
    boxes = [it["box_xyxy"] for it in items]
    n0 = len(items)

    partner: list[int | None] = [None] * n0
    centered = [False] * n0
    for i, b in enumerate(boxes):
        rb = reflect_x(b, cx)
        if box_iou(rb, [float(v) for v in b]) >= SELF_MIRROR_IOU:
            centered[i] = True
            continue
        j, _ = match_reflected(rb, boxes, exclude=i)
        partner[i] = j

    pairs: set[tuple[int, int]] = set()
    for i, j in enumerate(partner):
        if j is not None and not centered[j]:
            pairs.add((min(i, j), max(i, j)))

    dropped: set[int] = set()
    for i in range(n0):
        if centered[i] or partner[i] is not None:
            continue
        rb = reflect_x(boxes[i], cx)
        if _in_frame_frac(rb, iw) < MIN_IN_FRAME:
            continue
        mb = _clamp(rb, iw, ih)
        if mb[2] - mb[0] < 2 or mb[3] - mb[1] < 2:
            continue
        if any(box_iou([float(v) for v in mb], [float(v) for v in it["box_xyxy"]]) > DUPLICATE_IOU for it in items):
            continue
        rel = rel_fn(mb)
        if rel is not None and min_rel <= rel <= MAX_REL and on_grid(mb):
            items.append({"box_xyxy": mb, "rel_delta": rel, "sym": "mirror", "mirror_of": i})
            pairs.add((i, len(items) - 1))
            log["added"] += 1
        else:
            dropped.add(i)

    for a, b in sorted(pairs):
        if a in dropped or b in dropped:
            continue
        y0 = min(items[a]["box_xyxy"][1], items[b]["box_xyxy"][1])
        y1 = max(items[a]["box_xyxy"][3], items[b]["box_xyxy"][3])
        for k in (a, b):
            bx0, by0, bx1, by1 = items[k]["box_xyxy"]
            new_y0, new_y1 = by0, by1
            if y0 < by0 and _strip_ok(rel_fn, [bx0, y0, bx1, by0], min_rel):
                new_y0 = y0
            if y1 > by1 and _strip_ok(rel_fn, [bx0, by1, bx1, y1], min_rel):
                new_y1 = y1
            if (new_y0, new_y1) != (by0, by1):
                items[k] = dict(items[k], box_xyxy=[bx0, new_y0, bx1, new_y1], sym_aligned=True)
                log["aligned"] += 1

    log["dropped"] = len(dropped)
    out = [it for k, it in enumerate(items) if k not in dropped]
    return out, log


def _strip_ok(rel_fn: Callable[[Box], float | None], strip: Box, min_rel: float) -> bool:
    if strip[3] - strip[1] < 2 or strip[2] - strip[0] < 2:
        return False
    rel = rel_fn(strip)
    return rel is not None and min_rel <= rel <= MAX_REL


def stage_title(log: dict[str, Any]) -> str:
    if log.get("skip"):
        return f"6 sym off ({log['skip']})"
    return (
        f"6 sym win={log['window_sym']:.2f} "
        f"+{log['added']} add, {log['aligned']} align, -{log['dropped']} drop"
    )
