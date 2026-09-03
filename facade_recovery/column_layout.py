"""Structural column inference: anchor side stacks + middle bays + colspan."""

from __future__ import annotations

from typing import Any

import numpy as np


def assign_floors(cy: np.ndarray, row_tol: float) -> np.ndarray:
    """Greedy 1D floor labels from normalized y centers."""
    order = np.argsort(cy)
    floor = np.full(len(cy), -1, dtype=np.int32)
    cur = -1
    mean_y = None
    n = 0
    for i in order:
        yi = float(cy[i])
        if mean_y is None or abs(yi - mean_y) > row_tol:
            cur += 1
            mean_y = yi
            n = 1
        else:
            n += 1
            mean_y += (yi - mean_y) / n
        floor[i] = cur
    return floor


def infer_structural_columns(
    boxes: list[list[int]],
    floors: np.ndarray,
    *,
    iw: int,
    left_frac: float = 0.19,
    right_frac: float = 0.80,
) -> list[tuple[float, float]]:
    """Infer global columns: left stack | middle bays | right stack."""
    med_w = float(np.median([max(1.0, b[2] - b[0]) for b in boxes]))
    left_edge = left_frac * iw
    right_edge = right_frac * iw

    left_idxs: list[int] = []
    right_idxs: list[int] = []
    mid_idxs: list[int] = []
    for i, b in enumerate(boxes):
        if b[2] <= left_edge + 0.35 * med_w:
            left_idxs.append(i)
        elif b[0] >= right_edge - 0.35 * med_w:
            right_idxs.append(i)
        else:
            mid_idxs.append(i)

    if left_idxs:
        left_x0 = min(float(boxes[i][0]) for i in left_idxs)
        left_x1 = max(float(boxes[i][2]) for i in left_idxs)
    else:
        left_x0, left_x1 = 0.05 * iw, 0.17 * iw

    if right_idxs:
        right_x0 = min(float(boxes[i][0]) for i in right_idxs)
        right_x1 = max(float(boxes[i][2]) for i in right_idxs)
    else:
        right_x0, right_x1 = 0.83 * iw, 0.95 * iw

    split_pairs: list[tuple[float, float]] = []
    for f in sorted(set(int(v) for v in floors.tolist())):
        f_mid = [i for i in mid_idxs if int(floors[i]) == f]
        if len(f_mid) < 3:
            continue
        order = sorted(f_mid, key=lambda i: boxes[i][0])
        xs0 = [float(boxes[i][0]) for i in order]
        xs1 = [float(boxes[i][2]) for i in order]
        split_pairs.append((0.5 * (xs1[0] + xs0[1]), 0.5 * (xs1[1] + xs0[2])))

    if split_pairs:
        split1 = float(np.median([p[0] for p in split_pairs]))
        split2 = float(np.median([p[1] for p in split_pairs]))
    else:
        mid_cx = [0.5 * (boxes[i][0] + boxes[i][2]) for i in mid_idxs]
        if len(mid_cx) >= 3:
            qs = np.quantile(mid_cx, [0.33, 0.66])
            split1, split2 = float(qs[0]), float(qs[1])
        else:
            gap = (right_x0 - left_x1) / 3.0
            split1 = left_x1 + gap
            split2 = left_x1 + 2 * gap

    split1 = max(left_x1 + 4.0, min(split1, split2 - 8.0))
    split2 = max(split1 + 8.0, min(split2, right_x0 - 4.0))

    return [
        (left_x0, left_x1),
        (left_x1, split1),
        (split1, split2),
        (split2, right_x0),
        (right_x0, right_x1),
    ]


def _overlap(box: list[int], col: tuple[float, float]) -> float:
    x0, x1 = float(box[0]), float(box[2])
    cl, cr = col
    return max(0.0, min(x1, cr) - max(x0, cl))


def assign_box_to_columns(
    box: list[int],
    columns: list[tuple[float, float]],
    *,
    span_overlap_frac: float = 0.28,
) -> tuple[int, int]:
    """Return (start_col, colspan) for a unit box."""
    overlaps = [_overlap(box, col) for col in columns]
    bw = max(1.0, float(box[2] - box[0]))
    if not overlaps or max(overlaps) <= 0:
        return 0, 1

    active = [i for i, o in enumerate(overlaps) if o >= span_overlap_frac * bw]
    if not active:
        j = int(np.argmax(overlaps))
        return j, 1

    start, end = active[0], active[-1]
    if end > start:
        covered = sum(overlaps[start : end + 1])
        if covered < 0.55 * bw:
            j = int(np.argmax(overlaps))
            return j, 1
    return start, end - start + 1


def assign_layout(
    boxes: list[list[int]],
    floors: np.ndarray,
    columns: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, box in enumerate(boxes):
        bay, span = assign_box_to_columns(box, columns)
        out.append({"floor": int(floors[i]), "bay": bay, "colspan": span})
    return out


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(values))


def bay_box_envelopes(
    units: list[dict[str, Any]],
    n_bays: int,
) -> tuple[list[float | None], list[float | None]]:
    """Per-bay left/right x from window boxes (image coordinates)."""
    lo_acc: list[list[float]] = [[] for _ in range(n_bays)]
    hi_acc: list[list[float]] = [[] for _ in range(n_bays)]

    for u in units:
        if u.get("kind") == "door":
            continue
        b = int(u["bay"])
        if b < 0 or b >= n_bays:
            continue
        span = max(1, int(u.get("colspan", 1)))
        box = u["box_xyxy"]
        x0, x1 = float(box[0]), float(box[2])
        if span == 1:
            lo_acc[b].append(x0)
            hi_acc[b].append(x1)
        else:
            end = min(n_bays, b + span)
            lo_acc[b].append(x0)
            hi_acc[end - 1].append(x1)

    lo = [_median_or_none(a) for a in lo_acc]
    hi = [_median_or_none(a) for a in hi_acc]
    return lo, hi


def infer_inter_bay_gaps(
    units: list[dict[str, Any]],
    n_bays: int,
) -> list[float | None]:
    """Median image-space gap (px) between adjacent bay window groups on each floor.

    Gap between bay ``b`` and ``b+1`` is ``x0(next) - x1(prev)`` for single-column
    units on the same floor. Returns ``None`` when no same-floor pair is observed.
    """
    if n_bays < 2:
        return []
    floor_units: dict[int, list[dict[str, Any]]] = {}
    for u in units:
        if u.get("kind") == "door":
            continue
        if int(u.get("colspan", 1)) != 1:
            continue
        floor_units.setdefault(int(u["floor"]), []).append(u)

    gap_acc: list[list[float]] = [[] for _ in range(n_bays - 1)]
    for fus in floor_units.values():
        by_bay: dict[int, tuple[float, float]] = {}
        for u in fus:
            b = int(u["bay"])
            if b < 0 or b >= n_bays:
                continue
            x0, x1 = float(u["box_xyxy"][0]), float(u["box_xyxy"][2])
            by_bay[b] = (x0, x1)
        bays_present = sorted(by_bay)
        for i, b in enumerate(bays_present[:-1]):
            b_next = bays_present[i + 1]
            if b_next != b + 1:
                continue
            gap = by_bay[b_next][0] - by_bay[b][1]
            if gap >= 0:
                gap_acc[b].append(gap)
    return [_median_or_none(acc) for acc in gap_acc]


def infer_inter_bay_gaps_robust(
    units: list[dict[str, Any]],
    n_bays: int,
    *,
    outer_bay_count: int = 1,
) -> list[float]:
    """Median gap (px) between bay ``b`` and ``b+1`` from segmented box edges.

    Inner gaps (between middle bays) use only floors where every middle bay has a
    window, so a missing center bay on one floor does not shrink inner gap estimates.
    Outer gaps (side stack ↔ middle) use all floors with both bays present.
    """
    if n_bays < 2:
        return []
    mid_start = outer_bay_count
    mid_end = n_bays - outer_bay_count - 1
    middle_bays = list(range(mid_start, mid_end + 1))

    floor_units: dict[int, list[dict[str, Any]]] = {}
    for u in units:
        if u.get("kind") == "door":
            continue
        if int(u.get("colspan", 1)) != 1:
            continue
        floor_units.setdefault(int(u["floor"]), []).append(u)

    gap_acc: list[list[float]] = [[] for _ in range(n_bays - 1)]

    for fus in floor_units.values():
        by_bay: dict[int, tuple[float, float]] = {}
        for u in fus:
            b = int(u["bay"])
            if b < 0 or b >= n_bays:
                continue
            x0, x1 = float(u["box_xyxy"][0]), float(u["box_xyxy"][2])
            by_bay[b] = (x0, x1)

        has_full_middle = all(b in by_bay for b in middle_bays)

        for b in range(n_bays - 1):
            if b + 1 not in by_bay or b not in by_bay:
                continue
            is_inner = mid_start <= b < mid_end
            if is_inner and not has_full_middle:
                continue
            gap = by_bay[b + 1][0] - by_bay[b][1]
            if gap >= 0:
                gap_acc[b].append(gap)

    out: list[float] = []
    for b in range(n_bays - 1):
        med = _median_or_none(gap_acc[b])
        if med is None:
            lo, hi = bay_box_envelopes(units, n_bays)
            if lo[b] is not None and hi[b] is not None and lo[b + 1] is not None:
                med = max(0.0, float(lo[b + 1]) - float(hi[b]))
            else:
                med = 0.0
        out.append(float(med))
    return out


def infer_gap_norms(
    units: list[dict[str, Any]],
    n_bays: int,
    iw: int,
) -> list[float]:
    """Gap between bay ``b`` and ``b+1`` as a fraction of image width (like ``w_norm``)."""
    gaps_px = infer_inter_bay_gaps_robust(units, n_bays)
    fw = max(1.0, float(iw))
    return [max(0.0, float(g) / fw) for g in gaps_px]


def infer_bay_window_width_norms(
    units: list[dict[str, Any]],
    n_bays: int,
    iw: int,
    lo: list[float | None],
    hi: list[float | None],
) -> list[float]:
    """Median unit box width per bay, normalized by image width."""
    acc: list[list[float]] = [[] for _ in range(n_bays)]
    fw = max(1.0, float(iw))
    for u in units:
        if u.get("kind") == "door":
            continue
        b = int(u["bay"])
        if b < 0 or b >= n_bays or int(u.get("colspan", 1)) != 1:
            continue
        box = u["box_xyxy"]
        acc[b].append((float(box[2]) - float(box[0])) / fw)
    out: list[float] = []
    for b in range(n_bays):
        if acc[b]:
            out.append(max(0.03, float(np.median(acc[b]))))
        elif lo[b] is not None and hi[b] is not None:
            out.append(max(0.03, (float(hi[b]) - float(lo[b])) / fw))
        else:
            out.append(0.08)
    return out


def _half_gap_outer_margin(gaps: list[float], *, end: str, min_px: float = 8.0) -> float:
    """Half of the adjacent inter-bay / inter-floor gap for façade outer edges."""
    if not gaps:
        return max(min_px * 0.5, min_px)
    g = float(gaps[0]) if end == "start" else float(gaps[-1])
    return max(min_px * 0.5, g / 2.0)


def infer_proportional_bay_widths(
    units: list[dict[str, Any]],
    n_bays: int,
    iw: int,
) -> tuple[list[float], list[float]]:
    """Bay ``w_norm`` = window width fraction + half of adjacent gap fractions."""
    lo, hi = bay_box_envelopes(units, n_bays)
    gap_norms = infer_gap_norms(units, n_bays, iw)
    window_w = infer_bay_window_width_norms(units, n_bays, iw, lo, hi)
    col_w: list[float] = []
    for b in range(n_bays):
        gl = (
            gap_norms[b - 1] / 2.0
            if b > 0
            else (gap_norms[0] / 2.0 if gap_norms else 0.0)
        )
        gr = (
            gap_norms[b] / 2.0
            if b < n_bays - 1
            else (gap_norms[-1] / 2.0 if gap_norms else 0.0)
        )
        col_w.append(max(0.03, window_w[b] + gl + gr))
    return col_w, gap_norms


def infer_n_window_bays(
    units: list[dict[str, Any]],
    iw: int,
    *,
    min_gap_mult: float = 0.35,
) -> int:
    """Bay count = max detected windows on any single floor (one bay per window column)."""
    from collections import defaultdict

    window_units = [u for u in units if u.get("kind") != "door"]
    if not window_units:
        return 1

    by_floor: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for u in window_units:
        by_floor[int(u["floor"])].append(u)
    return max(len(fus) for fus in by_floor.values())


def _expand_unit_colspan(
    floor_units: list[dict[str, Any]],
    columns: list[tuple[float, float]],
) -> None:
    """Grow a unit into empty bays its box overlaps (left and right of its primary bay)."""
    occupied = {int(u["bay"]) for u in floor_units}
    n_bays = len(columns)
    for u in floor_units:
        primary = int(u["bay"])
        others = occupied - {primary}
        start, span = assign_box_to_columns(u["box_xyxy"], columns)
        lo = min(start, primary)
        hi = max(start + max(1, span), primary + 1)
        lo = max(0, lo)
        hi = min(n_bays, hi)
        while lo < primary and lo in others:
            lo += 1
        while hi > primary + 1 and (hi - 1) in others:
            hi -= 1
        u["bay"] = lo
        u["colspan"] = max(1, hi - lo)


def infer_window_bay_columns(
    units: list[dict[str, Any]],
    iw: int,
    *,
    min_gap_mult: float = 0.35,
    min_col_px: float = 8.0,
) -> list[tuple[float, float]]:
    """Bay columns from the densest floor (one column per window); others span those columns."""
    from collections import defaultdict

    window_indices: list[int] = [
        i for i, u in enumerate(units) if u.get("kind") != "door"
    ]
    if not window_indices:
        return [(0.0, float(iw))]

    by_floor: dict[int, list[int]] = defaultdict(list)
    for i in window_indices:
        by_floor[int(units[i]["floor"])].append(i)

    ref_floor = max(by_floor, key=lambda f: len(by_floor[f]))
    ref_indices = sorted(
        by_floor[ref_floor],
        key=lambda i: float(units[i]["box_xyxy"][0]),
    )
    n_bays = len(ref_indices)
    ref_lo = [float(units[i]["box_xyxy"][0]) for i in ref_indices]
    ref_hi = [float(units[i]["box_xyxy"][2]) for i in ref_indices]

    # One bay per window on the densest floor — never let a wide lower-floor
    # box (e.g. T0 spanning two T3s) merge those columns.
    for bay_i, idx in enumerate(ref_indices):
        units[idx]["bay"] = bay_i
        units[idx]["colspan"] = 1

    for floor, indices in by_floor.items():
        if floor == ref_floor:
            continue
        for i in indices:
            u = units[i]
            x0, _, x1, _ = [float(v) for v in u["box_xyxy"]]
            overlaps = [
                max(0.0, min(x1, ref_hi[j]) - max(x0, ref_lo[j]))
                for j in range(n_bays)
            ]
            u["bay"] = int(np.argmax(overlaps))
            u["colspan"] = 1

    columns = infer_bay_column_bounds_from_units(
        [units[i] for i in ref_indices],
        n_bays,
        iw=iw,
        min_col_px=min_col_px,
    )

    for indices in by_floor.values():
        _expand_unit_colspan([units[i] for i in indices], columns)

    return columns


def infer_bay_column_bounds_from_units(
    units: list[dict[str, Any]],
    n_bays: int,
    *,
    iw: int,
    structural_columns: list[tuple[float, float]] | None = None,
    min_col_px: float = 8.0,
) -> list[tuple[float, float]]:
    """Column bounds at gap midpoints from segmented box edges (gaps ÷ image width)."""
    lo, hi = bay_box_envelopes(units, n_bays)
    gaps_px = infer_inter_bay_gaps_robust(units, n_bays)

    if structural_columns is not None:
        for i in range(n_bays):
            sx0, sx1 = structural_columns[i]
            if lo[i] is None:
                lo[i] = float(sx0)
            if hi[i] is None:
                hi[i] = float(sx1)

    for i in range(n_bays):
        if lo[i] is None or hi[i] is None:
            raise ValueError(f"bay {i}: missing box envelope and no structural fallback")

    splits: list[float] = []
    for i in range(n_bays - 1):
        g = max(0.0, float(gaps_px[i]))
        mid = float(hi[i]) + g / 2.0
        mid = max(float(hi[i]) + min_col_px * 0.5, min(mid, float(lo[i + 1]) - min_col_px * 0.5))
        splits.append(mid)

    outer_left = _half_gap_outer_margin(gaps_px, end="start", min_px=min_col_px)
    outer_right = _half_gap_outer_margin(gaps_px, end="end", min_px=min_col_px)
    facade_left = max(0.0, float(lo[0]) - outer_left)
    facade_right = min(float(iw), float(hi[n_bays - 1]) + outer_right)

    out: list[tuple[float, float]] = []
    for i in range(n_bays):
        xl = facade_left if i == 0 else splits[i - 1]
        xr = facade_right if i == n_bays - 1 else splits[i]
        if xr - xl < min_col_px:
            cx = 0.5 * (float(lo[i]) + float(hi[i]))
            half = max(min_col_px / 2.0, (float(hi[i]) - float(lo[i])) / 2.0)
            xl, xr = cx - half, cx + half
        out.append((xl, xr))

    return out


def _floor_index_map(floor_ids: list[int]) -> dict[int, int]:
    ordered = sorted({int(f) for f in floor_ids})
    return {f: i for i, f in enumerate(ordered)}


def floor_box_envelopes(
    units: list[dict[str, Any]],
    floor_ids: list[int],
) -> tuple[list[float | None], list[float | None]]:
    """Per-floor top/bottom y from window boxes (image y grows downward)."""
    ordered = sorted({int(f) for f in floor_ids})
    n = len(ordered)
    top_acc: list[list[float]] = [[] for _ in range(n)]
    bot_acc: list[list[float]] = [[] for _ in range(n)]
    fmap = _floor_index_map(floor_ids)

    for u in units:
        fi = fmap.get(int(u["floor"]))
        if fi is None:
            continue
        y0, y1 = float(u["box_xyxy"][1]), float(u["box_xyxy"][3])
        top_acc[fi].append(y0)
        bot_acc[fi].append(y1)

    tops = [_median_or_none(a) for a in top_acc]
    bots = [_median_or_none(a) for a in bot_acc]
    return tops, bots


def infer_inter_floor_gaps_robust(
    units: list[dict[str, Any]],
    floor_ids: list[int],
) -> list[float]:
    """Median vertical gap (px) between consecutive floors from segmented boxes.

    Uses same-bay floor pairs when available; falls back to per-floor envelopes.
    """
    ordered = sorted({int(f) for f in floor_ids})
    if len(ordered) < 2:
        return []

    tops, bots = floor_box_envelopes(units, floor_ids)
    gap_acc: list[list[float]] = [[] for _ in range(len(ordered) - 1)]

    by_bay: dict[int, list[dict[str, Any]]] = {}
    for u in units:
        by_bay.setdefault(int(u["bay"]), []).append(u)

    for bay_units in by_bay.values():
        by_floor = {int(u["floor"]): u for u in bay_units}
        for i in range(len(ordered) - 1):
            f0, f1 = ordered[i], ordered[i + 1]
            if f0 not in by_floor or f1 not in by_floor:
                continue
            y1_prev = float(by_floor[f0]["box_xyxy"][3])
            y0_next = float(by_floor[f1]["box_xyxy"][1])
            gap = y0_next - y1_prev
            if gap >= 0:
                gap_acc[i].append(gap)

    out: list[float] = []
    for i in range(len(ordered) - 1):
        med = _median_or_none(gap_acc[i])
        if med is None and tops[i + 1] is not None and bots[i] is not None:
            med = max(0.0, float(tops[i + 1]) - float(bots[i]))
        out.append(float(med or 0.0))
    return out


def infer_floor_gap_norms(
    units: list[dict[str, Any]],
    floor_ids: list[int],
    ih: int,
) -> list[float]:
    gaps_px = infer_inter_floor_gaps_robust(units, floor_ids)
    fh = max(1.0, float(ih))
    return [max(0.0, float(g) / fh) for g in gaps_px]


def infer_floor_window_height_norms(
    units: list[dict[str, Any]],
    floor_ids: list[int],
    ih: int,
    tops: list[float | None],
    bots: list[float | None],
) -> list[float]:
    ordered = sorted({int(f) for f in floor_ids})
    fmap = _floor_index_map(floor_ids)
    acc: list[list[float]] = [[] for _ in range(len(ordered))]
    fh = max(1.0, float(ih))

    for u in units:
        fi = fmap.get(int(u["floor"]))
        if fi is None:
            continue
        box = u["box_xyxy"]
        acc[fi].append((float(box[3]) - float(box[1])) / fh)

    out: list[float] = []
    for i in range(len(ordered)):
        if acc[i]:
            out.append(max(0.03, float(np.median(acc[i]))))
        elif tops[i] is not None and bots[i] is not None:
            out.append(max(0.03, (float(bots[i]) - float(tops[i])) / fh))
        else:
            out.append(0.08)
    return out


def infer_proportional_floor_heights(
    units: list[dict[str, Any]],
    floor_ids: list[int],
    ih: int,
) -> tuple[list[float], list[float]]:
    """Floor ``h_norm`` = window height fraction + half of adjacent floor-gap fractions."""
    tops, bots = floor_box_envelopes(units, floor_ids)
    gap_norms = infer_floor_gap_norms(units, floor_ids, ih)
    window_h = infer_floor_window_height_norms(units, floor_ids, ih, tops, bots)
    row_h: list[float] = []
    for i in range(len(window_h)):
        gl = (
            gap_norms[i - 1] / 2.0
            if i > 0
            else (gap_norms[0] / 2.0 if gap_norms else 0.0)
        )
        gr = (
            gap_norms[i] / 2.0
            if i < len(gap_norms)
            else (gap_norms[-1] / 2.0 if gap_norms else 0.0)
        )
        row_h.append(max(0.03, window_h[i] + gl + gr))
    return row_h, gap_norms


def infer_floor_row_bounds_from_units(
    units: list[dict[str, Any]],
    floor_ids: list[int],
    *,
    ih: int,
    min_row_px: float = 8.0,
) -> list[tuple[float, float]]:
    """Floor row bounds at vertical gap midpoints (image y, top → bottom)."""
    tops, bots = floor_box_envelopes(units, floor_ids)
    gaps_px = infer_inter_floor_gaps_robust(units, floor_ids)
    n = len(tops)

    for i in range(n):
        if tops[i] is None or bots[i] is None:
            raise ValueError(f"floor index {i}: missing box envelope")

    splits: list[float] = []
    for i in range(n - 1):
        g = max(0.0, float(gaps_px[i]))
        mid = float(bots[i]) + g / 2.0
        mid = max(
            float(bots[i]) + min_row_px * 0.5,
            min(mid, float(tops[i + 1]) - min_row_px * 0.5),
        )
        splits.append(mid)

    outer_top = _half_gap_outer_margin(gaps_px, end="start", min_px=min_row_px)
    outer_bottom = _half_gap_outer_margin(gaps_px, end="end", min_px=min_row_px)
    facade_top = max(0.0, float(tops[0]) - outer_top)
    facade_bottom = min(float(ih), float(bots[n - 1]) + outer_bottom)

    out: list[tuple[float, float]] = []
    for i in range(n):
        yt = facade_top if i == 0 else splits[i - 1]
        yb = facade_bottom if i == n - 1 else splits[i]
        if yb - yt < min_row_px:
            cy = 0.5 * (float(tops[i]) + float(bots[i]))
            half = max(min_row_px / 2.0, (float(bots[i]) - float(tops[i])) / 2.0)
            yt, yb = cy - half, cy + half
        out.append((yt, yb))

    return out

