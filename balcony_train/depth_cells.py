"""Depth-first enclosed detection on window-grid cells.

Each floor×bay cell is judged on its own; enclosed and half-enclosed are not
told apart. Steps, each a set of cells:

    1_depth   L or R >= DEPTH_REL. L = cell vs the wall strips beside it,
              R = cell vs the median of its storey row.
    2_upper   the upper two thirds of the cell is mostly in front of the
              wall (an open balcony only projects at the slab and railing).
    3_sam3    a SAM3 enclosure-prompt box overlaps the cell.
    4_column  the cell is part of a run of >= MIN_RUN floors in one bay.
    5_mirror  windows mirror-symmetric (>= SYM_GATE): a missing mirror cell
              that passed step 2 is added, a cell whose mirror failed step 2
              is dropped (cells on the axis are kept).
    6_edge    outermost bays touching the photo border are dropped.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from balcony_train.enclosed_symmetry import SYM_GATE, grid_axis_x, window_symmetry
from balcony_train.try_sam3_da3_volumes import (
    SAM3_PROMPTS,
    box_rel_delta,
    front_fill_frac,
    robust_median,
)

Cell = tuple[int, int]
Bands = dict[int, tuple[int, int]]

DEPTH_REL = 0.015
UPPER_FRAC = 2.0 / 3.0
UPPER_MIN = 0.40
SAM3_COVER = 0.25
SAM3_MIN_SCORE = 0.15
MIN_RUN = 2
EDGE_FRAC = 0.02

STEP_TITLES: tuple[tuple[str, str], ...] = (
    ("1_depth", "1 depth L or R >= 1.5%"),
    ("2_upper", "2 upper 2/3 in front (not open)"),
    ("3_sam3", "3 SAM3 box on cell"),
    ("4_column", "4 >= 2 floors in one bay"),
    ("5_mirror", "5 mirror add / drop"),
    ("6_edge", "6 drop photo-edge bay"),
)


def cell_box(cell: Cell, floor_y: Bands, bay_x: Bands) -> list[int]:
    fid, bid = cell
    y0, y1 = floor_y[fid]
    x0, x1 = bay_x[bid]
    return [int(x0), int(y0), int(x1), int(y1)]


def _cover(cell: list[int], box: list[float]) -> float:
    ix = max(0.0, min(cell[2], box[2]) - max(cell[0], box[0]))
    iy = max(0.0, min(cell[3], box[3]) - max(cell[1], box[1]))
    area = max(1.0, (cell[2] - cell[0]) * (cell[3] - cell[1]))
    return ix * iy / area


def cell_features(
    depth: np.ndarray,
    veg: np.ndarray | None,
    floor_y: Bands,
    bay_x: Bands,
    sam3_records: list[dict[str, Any]],
) -> dict[Cell, dict[str, float | None]]:
    """Per cell: L, R, upper (front fraction of the upper part), sam3 (best score)."""
    seeds = [
        r
        for r in sam3_records
        if r.get("prompt") in SAM3_PROMPTS and float(r.get("score") or 0.0) >= SAM3_MIN_SCORE
    ]
    out: dict[Cell, dict[str, float | None]] = {}
    for fid in floor_y:
        meds = {bid: robust_median(depth, cell_box((fid, bid), floor_y, bay_x)) for bid in bay_x}
        vals = [m for m in meds.values() if m is not None]
        row_med = float(np.median(vals)) if vals else None
        for bid in bay_x:
            box = cell_box((fid, bid), floor_y, bay_x)
            m = meds[bid]
            upper_box = [box[0], box[1], box[2], box[1] + max(2, int(round(UPPER_FRAC * (box[3] - box[1]))))]
            sam3 = 0.0
            for r in seeds:
                if _cover(box, [float(v) for v in r["box_xyxy"]]) >= SAM3_COVER:
                    sam3 = max(sam3, float(r.get("score") or 0.0))
            out[(int(fid), int(bid))] = {
                "L": box_rel_delta(depth, box, veg=veg),
                "R": (row_med - m) / row_med if (m is not None and row_med) else None,
                "upper": front_fill_frac(depth, upper_box, veg=veg, min_rel=DEPTH_REL),
                "sam3": sam3,
            }
    return out


def _ge(v: float | None, thr: float) -> bool:
    return v is not None and v >= thr


def column_runs(cells: set[Cell], floor_y: Bands, *, min_run: int = MIN_RUN) -> set[Cell]:
    """Keep cells in a run of >= ``min_run`` vertically adjacent floors (same bay)."""
    order = sorted(floor_y, key=lambda f: floor_y[f][0])
    rank = {f: i for i, f in enumerate(order)}
    keep: set[Cell] = set()
    for bid in {b for _, b in cells}:
        ranks = sorted(rank[f] for f, b in cells if b == bid and f in rank)
        run: list[int] = []
        for r in ranks + [None]:  # type: ignore[list-item]
            if r is not None and run and r == run[-1] + 1:
                run.append(r)
                continue
            if len(run) >= min_run:
                keep.update((order[k], bid) for k in run)
            run = [r] if r is not None else []
    return keep


def mirror_bay(bid: int, bay_x: Bands, cx: float) -> int | None:
    x0, x1 = bay_x[bid]
    mx = 2.0 * cx - 0.5 * (x0 + x1)
    best, best_d = None, None
    for b, (a0, a1) in bay_x.items():
        d = abs(0.5 * (a0 + a1) - mx)
        if best_d is None or d < best_d:
            best, best_d = b, d
    if best is None:
        return None
    a0, a1 = bay_x[best]
    if best_d > 0.5 * (a1 - a0) + 0.25 * (x1 - x0):
        return None
    return int(best)


def mirror_cells(
    cells: set[Cell],
    support: set[Cell],
    bay_x: Bands,
    *,
    cx: float | None,
    window_sym: float,
    gate: float = SYM_GATE,
) -> set[Cell]:
    if cx is None or window_sym < gate:
        return set(cells)
    out = set(cells)
    for fid, bid in sorted(cells):
        mb = mirror_bay(bid, bay_x, cx)
        if mb is None or mb == bid or (fid, mb) in cells:
            continue
        if (fid, mb) in support:
            out.add((fid, mb))
        else:
            out.discard((fid, bid))
    return out


def drop_edge_bays(cells: set[Cell], bay_x: Bands, iw: int, *, frac: float = EDGE_FRAC) -> set[Cell]:
    if not bay_x:
        return set(cells)
    left = min(bay_x, key=lambda b: bay_x[b][0])
    right = max(bay_x, key=lambda b: bay_x[b][1])
    edge = set()
    if bay_x[left][0] <= frac * iw:
        edge.add(left)
    if bay_x[right][1] >= (1.0 - frac) * iw:
        edge.add(right)
    return {c for c in cells if c[1] not in edge}


def depth_cell_steps(
    feats: dict[Cell, dict[str, float | None]],
    floor_y: Bands,
    bay_x: Bands,
    *,
    iw: int,
    windows: list[list[int]] | None = None,
    depth_rel: float = DEPTH_REL,
    upper_min: float = UPPER_MIN,
) -> tuple[list[tuple[str, str, set[Cell]]], dict[str, Any]]:
    """Every step as (id, title, cells), plus a small log."""
    s1 = {c for c, f in feats.items() if _ge(f["L"], depth_rel) or _ge(f["R"], depth_rel)}
    s2 = {c for c in s1 if _ge(feats[c]["upper"], upper_min)}
    s3 = {c for c in s2 if float(feats[c]["sam3"] or 0.0) > 0.0}
    s4 = column_runs(s3, floor_y)
    cx = grid_axis_x(bay_x)
    wsym = window_symmetry(windows or [], cx) if cx is not None else 0.0
    s5 = column_runs(mirror_cells(s4, s2, bay_x, cx=cx, window_sym=wsym), floor_y)
    s6 = drop_edge_bays(s5, bay_x, iw)
    sets = (s1, s2, s3, s4, s5, s6)
    steps = [(sid, title, cells) for (sid, title), cells in zip(STEP_TITLES, sets)]
    return steps, {"window_sym": wsym, "axis_x": cx}
