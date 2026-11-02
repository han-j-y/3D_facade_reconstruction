"""Open-balcony post-rules after the overlap filter (user, 2026-11-01).

Applied to the filtered SAM3 balcony boxes, in this order:

1. ``off_facade`` - drop a box that lies mostly outside the window grid
   (less than half of its width over the grid's x-extent), or a small box
   touching the image border (area under half the photo's median box area,
   within 1% of the border). Cut-off balconies and balconies on another wall.
2. ``oriel`` - drop an open box on a predicted oriel cell (same floor, shared
   bay; ``enclosed.drop_open_under_enclosed``).
3. ``no_window_near`` - drop a box with no window just above it in its bay
   span: a window whose center is above the box center and whose bottom is
   at most half a window height above the box top. Boxes centred in the top
   floor band are kept. Removes roofs, cornices and ornament between floors.
4. ``solid_juliet`` - drop a box the railing classifier calls ``solid`` +
   ``juliet`` (a window sill with no railing).
5. ``juliet_split`` - a ``juliet`` box over two or more windows is replaced by
   one box per window: the window's x-range, the box's y-range.

6. ``edge_depth`` (user's wording, 2026-11-01) - drop a small box (area under
   half the photo's median box area) that is near the image border (within 3%
   of width / height) or whose median DA3 depth differs from the main wall by
   more than 20%. Main wall = median depth over the window-grid extent.

7. ``side_wall`` (rule 1 v2, 2026-11-01) - drop a box on a side wall or a
   neighbouring building: (a) its median DA3 depth is more than 8% farther than
   the main wall, or (b) it touches the left / right image border (within 1% of
   the width) and is taller than wide (width / height < 1.0, a cut-off or
   foreshortened balcony). No size condition. Thresholds were chosen on test100
   after the user marked 23 such boxes in six photos (see agent.md).

8. ``below_floor`` (rule 3 v2, user 2026-11-01) - drop a box whose top edge is
   below the bottom of its floor's window row. Window row = top of the highest
   to bottom of the lowest window of that floor (``snap._bands_from_window_instances``;
   the structural ``floors_y`` bands have no gaps, so they cannot be used). The
   box's floor is the row it overlaps vertically, so the rule drops a box that
   overlaps no window row (it lies in a gap between floors or below the lowest
   row). A box entirely above every row (top floor) is kept. No window has to be
   found over the box. (First version took the nearest row starting above the
   box top; tall balconies whose top reaches above their own windows were then
   compared with the floor above and removed.)

9. ``window_below`` (hood rule, user 2026-11-01) - drop a box that sits on top
   of a window: a detected window centred in the box x-range has its top below
   the box centre and within ``BELOW_GAP_FRAC`` (0.25) window heights of the box
   bottom (above or below), and no window is just above the box
   (``windows_just_above``, the balcony's own door). Removes window hoods and
   pediments. A missed window keeps the box. Thresholds fixed before scoring.

10. ``wide_split`` (user 2026-11-01) - ``juliet_split`` without the classifier
    condition: any box with two or more windows just above it is replaced by one
    box per window (the window's x-range, the box's y-range). Targets continuous
    balustrades boxed once over several windows.

11. ``masonry_juliet_split`` (user 2026-11-01) - ``juliet_split`` only for a box
    the classifier calls ``juliet`` + ``open_work`` + ``masonry`` (a baluster
    balustrade); same split, one box per window just above (two or more).

Rules 1-6 were fixed when written; ``side_wall`` was chosen by looking at test100. Rules 4-5 need the classifier labels of each box (``labels``),
rule 2 the predicted oriel cells (floor, bay).
"""

from __future__ import annotations

import statistics
from typing import Any

import numpy as np

from enclosed import drop_open_under_enclosed
from snap import _bands_from_window_instances, _bands_from_windows_dsl, snap_units

RULES = ("off_facade", "oriel", "no_window_near", "solid_juliet", "juliet_split")
USER_RULES = ("side_wall", "solid_juliet", "no_window_near")
SIDE_DEPTH_FRAC = 0.08
SIDE_EDGE_FRAC = 0.01
SIDE_MAX_ASPECT = 1.0
NEAR_EDGE_FRAC = 0.03
DEPTH_OFF_FRAC = 0.20
GRID_COVER_MIN = 0.5
EDGE_FRAC = 0.01
SMALL_AREA_FRAC = 0.5
NEAR_WINDOW_FRAC = 0.5
BELOW_GAP_FRAC = 0.25


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def window_boxes(windows_dsl: dict[str, Any]) -> list[dict[str, Any]]:
    """Window instances (doors excluded) with a box and a bay."""
    return [u for u in windows_dsl.get("instances") or []
            if u.get("box_xyxy") and str(u.get("kind", "window")) == "window"]


def off_facade(box, size: tuple[int, int], grid_x: tuple[float, float] | None, median_area: float) -> bool:
    iw, ih = size
    x0, y0, x1, y1 = box
    if grid_x is not None:
        if _overlap_1d(x0, x1, *grid_x) < GRID_COVER_MIN * max(1.0, x1 - x0):
            return True
    touches = (x0 <= EDGE_FRAC * iw or y0 <= EDGE_FRAC * ih
               or x1 >= (1 - EDGE_FRAC) * iw or y1 >= (1 - EDGE_FRAC) * ih)
    return touches and _area(box) < SMALL_AREA_FRAC * median_area


def edge_depth(box, size: tuple[int, int], median_area: float, depth: np.ndarray | None,
               wall_depth: float | None) -> str | None:
    """``"edge"`` or ``"depth"`` when the small box should go, else None."""
    if _area(box) >= SMALL_AREA_FRAC * median_area:
        return None
    iw, ih = size
    x0, y0, x1, y1 = box
    if (x0 <= NEAR_EDGE_FRAC * iw or x1 >= (1 - NEAR_EDGE_FRAC) * iw
            or y0 <= NEAR_EDGE_FRAC * ih or y1 >= (1 - NEAR_EDGE_FRAC) * ih):
        return "edge"
    if depth is None or not wall_depth:
        return None
    patch = depth[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)]
    if patch.size == 0:
        return None
    return "depth" if abs(float(np.median(patch)) - wall_depth) > DEPTH_OFF_FRAC * wall_depth else None


def main_wall_depth(depth: np.ndarray, floor_y, bay_x) -> float | None:
    """Median depth over the window-grid extent."""
    if depth is None or not floor_y or not bay_x:
        return None
    gx0 = min(min(b) for b in bay_x.values())
    gx1 = max(max(b) for b in bay_x.values())
    gy0 = min(min(b) for b in floor_y.values())
    gy1 = max(max(b) for b in floor_y.values())
    region = depth[int(gy0):int(gy1), int(gx0):int(gx1)]
    return float(np.median(region)) if region.size else None


def side_wall(box, size: tuple[int, int], depth: np.ndarray | None, wall_depth: float | None) -> str | None:
    """``"depth"`` or ``"edge"`` when the box is on a side wall / neighbour, else None."""
    iw, _ih = size
    x0, y0, x1, y1 = box
    if depth is not None and wall_depth:
        patch = depth[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)]
        if patch.size and float(np.median(patch)) > (1 + SIDE_DEPTH_FRAC) * wall_depth:
            return "depth"
    if (x0 <= SIDE_EDGE_FRAC * iw or x1 >= (1 - SIDE_EDGE_FRAC) * iw) and (x1 - x0) < SIDE_MAX_ASPECT * (y1 - y0):
        return "edge"
    return None


def below_floor(box, rows: list[tuple[int, int]]) -> bool:
    """True when the box overlaps no window row and is not above all of them."""
    y0, y1 = box[1], box[3]
    if y1 <= min(min(r) for r in rows):
        return False
    return not any(_overlap_1d(y0, y1, min(r), max(r)) > 0 for r in rows)


def window_right_below(box, windows: list[dict[str, Any]]) -> bool:
    """A window centred in the box x-range whose top is at the box bottom (box sits on it)."""
    x0, y0, x1, y1 = box
    cy = 0.5 * (y0 + y1)
    for w in windows:
        wx0, wy0, wx1, wy1 = w["box_xyxy"]
        if not x0 <= 0.5 * (wx0 + wx1) <= x1 or wy0 <= cy:
            continue
        if abs(wy0 - y1) <= BELOW_GAP_FRAC * (wy1 - wy0):
            return True
    return False


def windows_just_above(box, windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Windows over the box x-range whose bottom sits at most half a window height above the box top."""
    x0, y0, x1, y1 = box
    cy = 0.5 * (y0 + y1)
    out = []
    for w in windows:
        wx0, wy0, wx1, wy1 = w["box_xyxy"]
        wcx = 0.5 * (wx0 + wx1)
        if not x0 <= wcx <= x1:
            continue
        if 0.5 * (wy0 + wy1) >= cy:
            continue
        if y0 - wy1 > NEAR_WINDOW_FRAC * (wy1 - wy0):
            continue
        out.append(w)
    return out


def in_top_floor(box, floor_y: dict[int, tuple[int, int]]) -> bool:
    if not floor_y:
        return False
    top = min(floor_y.values(), key=lambda b: min(b))
    lo, hi = min(top), max(top)
    return lo <= 0.5 * (box[1] + box[3]) <= hi


def apply_rules(
    boxes: list[list[int]],
    size: tuple[int, int],
    windows_dsl: dict[str, Any],
    *,
    labels: list[dict[str, Any] | None] | None = None,
    oriel_cells: list[tuple[int, int]] | None = None,
    depth: np.ndarray | None = None,
    rules: tuple[str, ...] = RULES,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Return (kept boxes, log). ``labels[i]`` is the classifier output for ``boxes[i]``."""
    labels = labels if labels is not None else [None] * len(boxes)
    windows = window_boxes(windows_dsl)
    floor_y, bay_x = _bands_from_windows_dsl(windows_dsl)
    grid_x = None
    if bay_x:
        grid_x = (min(min(b) for b in bay_x.values()), max(max(b) for b in bay_x.values()))
    median_area = statistics.median([_area(b) for b in boxes]) if boxes else 0.0

    log: list[dict[str, Any]] = []
    alive = list(range(len(boxes)))

    def drop(rule: str, idxs: list[int]) -> None:
        for i in idxs:
            log.append({"rule": rule, "box": boxes[i]})
        for i in idxs:
            alive.remove(i)

    if "edge_depth" in rules:
        wall = None
        if depth is not None and floor_y and bay_x:
            gx0, gx1 = grid_x
            gy0 = min(min(b) for b in floor_y.values())
            gy1 = max(max(b) for b in floor_y.values())
            region = depth[int(gy0):int(gy1), int(gx0):int(gx1)]
            wall = float(np.median(region)) if region.size else None
        why = {i: edge_depth(boxes[i], size, median_area, depth, wall) for i in alive}
        hit = [i for i in alive if why[i]]
        drop("edge_depth", hit)
        for e, i in zip(log[-len(hit):] if hit else [], hit):
            e["reason"] = why[i]

    if "side_wall" in rules:
        wall = main_wall_depth(depth, floor_y, bay_x)
        why = {i: side_wall(boxes[i], size, depth, wall) for i in alive}
        hit = [i for i in alive if why[i]]
        drop("side_wall", hit)
        for e, i in zip(log[-len(hit):] if hit else [], hit):
            e["reason"] = why[i]

    if "off_facade" in rules:
        drop("off_facade", [i for i in alive if off_facade(boxes[i], size, grid_x, median_area)])

    if "oriel" in rules and oriel_cells:
        units = snap_units([boxes[i] for i in alive], windows_dsl)
        enc = [{"floor": int(f), "bay": int(b)} for f, b in oriel_cells]
        _kept, dropped = drop_open_under_enclosed(units, enc)
        drop("oriel", [alive[u["unit_id"]] for u in dropped])

    if "below_floor" in rules:
        rows = list(_bands_from_window_instances(windows_dsl.get("instances") or [])[0].values())
        if rows:
            drop("below_floor", [i for i in alive if below_floor(boxes[i], rows)])

    if "window_below" in rules and windows:
        drop("window_below", [i for i in alive if window_right_below(boxes[i], windows)
                              and not windows_just_above(boxes[i], windows)])

    if "no_window_near" in rules and windows:
        drop("no_window_near", [i for i in alive
                                if not in_top_floor(boxes[i], floor_y) and not windows_just_above(boxes[i], windows)])

    if "solid_juliet" in rules:
        drop("solid_juliet", [i for i in alive if labels[i]
                              and labels[i].get("kind") == "solid" and labels[i].get("structure") == "juliet"])

    out: list[list[int]] = []
    for i in alive:
        b = boxes[i]
        lab = labels[i]
        masonry_juliet = bool(lab) and lab.get("structure") == "juliet" and lab.get("kind") == "open_work"             and lab.get("material") == "masonry"
        if ("wide_split" in rules or ("juliet_split" in rules and lab and lab.get("structure") == "juliet")
                or ("masonry_juliet_split" in rules and masonry_juliet)):
            above = windows_just_above(b, windows)
            if len(above) >= 2:
                pieces = [[int(w["box_xyxy"][0]), b[1], int(w["box_xyxy"][2]), b[3]]
                          for w in sorted(above, key=lambda w: w["box_xyxy"][0])]
                log.append({"rule": next(r for r in ("wide_split", "masonry_juliet_split", "juliet_split")
                                         if r in rules), "box": b, "into": pieces})
                out.extend(pieces)
                continue
        out.append(list(b))
    return out, log
