"""Where each balcony sits on the façade grid (no Blender dependency).

``balcony_frames`` turns ``balcony_placement`` into world-space frames
(slab span, sill height, plan outline). It also links stacked balconies:
a half-enclosed or enclosed balcony with a balcony directly above it
takes that balcony's slab as its roof, and the upper slab is aligned to
the lower footprint. ``front_window_targets`` moves windows that share a
cell with an enclosed balcony onto that balcony's front face.
"""

from __future__ import annotations

from typing import Any, Sequence

from balcony_plan import (
    DEFAULT_SLAB_DEPTH_M,
    ENCLOSED_CLEAR_H_M,
    centroid,
    even_along_polyline,
    inset_point,
    slab_outline,
    support_polyline,
)
from facade_spec import get_cell, total_grid_size

STACKABLE = frozenset({"half_enclosed", "enclosed"})
OUTLINE_SHAPES = frozenset({"triangle", "circle", "trapezoid", "hexagon"})

STACK_MIN_OVERLAP = 0.5
MIN_CLEAR_H_M = 1.2
FRONT_WINDOW_MARGIN_M = 0.08
# A window keeps at least this share of its width on the front, else it stays on the wall.
FRONT_WINDOW_MIN_FIT = 0.6
MIN_FRONT_WIDTH_M = 0.4
COLUMN_WIDTH_M = 0.25


def span_xz(
    spec: dict[str, Any],
    row: int,
    col0: int,
    col1: int,
    *,
    mirror_x: bool,
) -> tuple[float, float, float, float]:
    a = get_cell(spec, int(row), int(col0), mirror_x=mirror_x)
    b = get_cell(spec, int(row), int(col1), mirror_x=mirror_x)
    return (
        min(float(a["x0"]), float(b["x0"])),
        max(float(a["x1"]), float(b["x1"])),
        float(a["z0"]),
        float(a["z1"]),
    )


def mean_column_cx(
    spec: dict[str, Any],
    row: int,
    col0: int,
    col1: int,
    *,
    mirror_x: bool,
) -> float:
    """Mean X center of grid columns col0..col1 (inclusive)."""
    c0, c1 = min(int(col0), int(col1)), max(int(col0), int(col1))
    centers: list[float] = []
    for col in range(c0, c1 + 1):
        cell = get_cell(spec, int(row), col, mirror_x=mirror_x)
        centers.append(0.5 * (float(cell["x0"]) + float(cell["x1"])))
    return sum(centers) / len(centers)


def photo_x_span(
    spec: dict[str, Any],
    rec: dict[str, Any],
    *,
    mirror_x: bool,
    bay_x0: float,
    bay_x1: float,
    row: int,
    col0: int,
    col1: int,
) -> tuple[float, float, float]:
    """Map photo box width/center to world X. Returns (x0, x1, cx).

    Photo u=0 is left; with mirror_x the camera shows world +X on the left,
    so ``world_x = (0.5 - cx_norm) * total_w``.
    When cx_norm is missing, uses window_cx_norm (paired window boxes), then
    bay_cx_norm (mean of photo bay bands), then the mean of associated grid
    column centers (multi-bay balconies).
    """
    total_w, _ = total_grid_size(spec["grid"])
    total_w = max(0.3, float(total_w))
    bay_cx = mean_column_cx(spec, row, col0, col1, mirror_x=mirror_x)

    w_norm = rec.get("width_norm")
    cx_norm = rec.get("cx_norm")
    window_cx_norm = rec.get("window_cx_norm")
    bay_cx_norm = rec.get("bay_cx_norm")
    try:
        w_norm_f = float(w_norm) if w_norm is not None else None
        cx_norm_f = float(cx_norm) if cx_norm is not None else None
        window_cx_norm_f = (
            float(window_cx_norm) if window_cx_norm is not None else None
        )
        bay_cx_norm_f = float(bay_cx_norm) if bay_cx_norm is not None else None
    except (TypeError, ValueError):
        w_norm_f, cx_norm_f, window_cx_norm_f, bay_cx_norm_f = None, None, None, None

    if w_norm_f is None or w_norm_f <= 0:
        return bay_x0, bay_x1, bay_cx

    span_w = max(0.25, min(total_w * 0.98, w_norm_f * total_w))
    if cx_norm_f is not None:
        cx_n = min(1.0, max(0.0, cx_norm_f))
    elif window_cx_norm_f is not None:
        cx_n = min(1.0, max(0.0, window_cx_norm_f))
    elif bay_cx_norm_f is not None:
        cx_n = min(1.0, max(0.0, bay_cx_norm_f))
    else:
        cx = bay_cx
        x0 = cx - span_w / 2.0
        x1 = cx + span_w / 2.0
        half = total_w / 2.0
        if x0 < -half:
            x1 += -half - x0
            x0 = -half
        if x1 > half:
            x0 -= x1 - half
            x1 = half
        return x0, x1, 0.5 * (x0 + x1)

    if mirror_x:
        cx = (0.5 - cx_n) * total_w
    else:
        cx = (cx_n - 0.5) * total_w
    x0 = cx - span_w / 2.0
    x1 = cx + span_w / 2.0
    half = total_w / 2.0
    if x0 < -half:
        x1 += -half - x0
        x0 = -half
    if x1 > half:
        x0 -= x1 - half
        x1 = half
    return x0, x1, 0.5 * (x0 + x1)


def _frame(
    spec: dict[str, Any],
    index: int,
    rec: dict[str, Any],
    ir: dict[str, Any],
    *,
    mirror_x: bool,
    bottom_m: float,
) -> dict[str, Any] | None:
    grid = spec.get("grid") or {}
    n_rows = len(grid.get("rows") or [])
    n_cols = len(grid.get("cols") or [])
    row = int(rec.get("row", 0))
    c0 = int(rec.get("col0", rec.get("col", 0)))
    c1 = int(rec.get("col1", c0))
    if not (0 <= row < n_rows and 0 <= c0 < n_cols and 0 <= c1 < n_cols):
        print(f"warn: balcony placement out of grid {rec}")
        return None

    wall = spec.get("wall") or {}
    front_y = float(wall.get("base_front_y", 0.0))
    x0, x1, z0, z1 = span_xz(spec, row, c0, c1, mirror_x=mirror_x)
    x0, x1, cx = photo_x_span(
        spec, rec, mirror_x=mirror_x, bay_x0=x0, bay_x1=x1, row=row, col0=c0, col1=c1
    )
    span_w = max(0.3, x1 - x0)
    cell_h = max(0.3, z1 - z0)
    floor = ir.get("floor") or {}
    params = floor.get("params") or {}
    shape = str(floor.get("shape") or "rectangle").lower()
    structure = str(ir.get("structure") or "projecting")
    enclosure = str(ir.get("enclosure") or "open")
    thick = float((ir.get("output") or {}).get("slab_thickness") or 0.12)
    depth = float(DEFAULT_SLAB_DEPTH_M)

    sill_z = z0 + cell_h * bottom_m
    if structure == "free_standing":
        deck_h = float(params.get("height") or 0.0)
        if deck_h > 0.05:
            sill_z = deck_h

    pad = span_w * 0.01
    bx0, bx1 = x0 + pad, x1 - pad
    if shape in OUTLINE_SHAPES or enclosure in STACKABLE:
        tw = float(params.get("width") or params.get("diameter") or 0.0)
        if tw > 0.3:
            tw = min(tw, span_w * 0.98)
            bx0 = cx - tw / 2.0
            bx1 = cx + tw / 2.0
    if shape not in OUTLINE_SHAPES and enclosure in STACKABLE:
        # Rectangle railings size from span_w * 0.98; keep them on this slab.
        span_w = (bx1 - bx0) / 0.98

    stacks = structure not in ("inset", "composite")
    outline = (
        slab_outline(
            shape if shape in OUTLINE_SHAPES else "rectangle",
            x0=bx0,
            x1=bx1,
            y_wall=front_y,
            depth=depth,
            structure=structure,
        )
        if stacks
        else None
    )
    return {
        "index": index,
        "name": str(rec.get("type") or ""),
        "row": row,
        "col0": min(c0, c1),
        "col1": max(c0, c1),
        "x0": x0,
        "x1": x1,
        "cx": cx,
        "z0": z0,
        "z1": z1,
        "span_w": span_w,
        "cell_h": cell_h,
        "shape": shape,
        "structure": structure,
        "enclosure": enclosure,
        "depth": depth,
        "thick": thick,
        "sill_z": sill_z,
        "bx0": bx0,
        "bx1": bx1,
        "y_wall": front_y,
        "outline": outline,
        "aligned": False,
        "roof": None,
        "above": None,
        "top_z": None,
    }


def _own_top_z(frame: dict[str, Any], ir: dict[str, Any]) -> float:
    """Top of the enclosure when nothing sits above it."""
    params = (ir.get("floor") or {}).get("params") or {}
    floor_top = frame["sill_z"] + frame["thick"]
    clear = float(params.get("height") or 0.0)
    if clear <= 0.05:
        # A roof slab whose top lands where the next storey's sill would be.
        next_sill = frame["z1"] + (frame["sill_z"] - frame["z0"])
        clear = next_sill - frame["thick"] - floor_top
        if clear < MIN_CLEAR_H_M:
            clear = ENCLOSED_CLEAR_H_M
    return floor_top + max(MIN_CLEAR_H_M, clear)


def _x_overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    lo = max(a["bx0"], b["bx0"])
    hi = min(a["bx1"], b["bx1"])
    narrow = min(a["bx1"] - a["bx0"], b["bx1"] - b["bx0"])
    if narrow <= 1e-6:
        return 0.0
    return max(0.0, hi - lo) / narrow


def _align_to(upper: dict[str, Any], lower: dict[str, Any]) -> None:
    """Upper slab takes the lower footprint so the stack reads as one bay."""
    for key in ("bx0", "bx1", "depth", "shape", "outline"):
        upper[key] = lower[key]
    upper["cx"] = 0.5 * (lower["bx0"] + lower["bx1"])
    upper["span_w"] = (lower["bx1"] - lower["bx0"]) / 0.98
    upper["aligned"] = True


def link_stacks(
    frames: Sequence[dict[str, Any] | None],
    library: dict[str, Any],
) -> None:
    """Share floors up the stack (in place).

    Bottom rows first, so a chain of enclosures aligns from the lowest one.
    The highest member of a chain keeps its own roof slab.
    """
    live = [f for f in frames if f is not None]
    for f in sorted(live, key=lambda fr: -fr["row"]):
        if f["enclosure"] not in STACKABLE or f["outline"] is None:
            continue
        best: dict[str, Any] | None = None
        best_ov = STACK_MIN_OVERLAP
        for g in live:
            if g["row"] != f["row"] - 1 or g["outline"] is None:
                continue
            ov = _x_overlap(f, g)
            if ov >= best_ov:
                best, best_ov = g, ov
        if best is not None and best["sill_z"] > f["sill_z"] + f["thick"] + MIN_CLEAR_H_M:
            f["roof"] = "shared"
            f["above"] = best["index"]
            f["top_z"] = best["sill_z"]
            _align_to(best, f)
        else:
            f["roof"] = "own"
            f["top_z"] = _own_top_z(f, library.get(f["name"]) or {})


def balcony_frames(spec: dict[str, Any]) -> list[dict[str, Any] | None]:
    """One frame per ``balcony_placement`` entry (None when it cannot be placed)."""
    library = spec.get("balconies") or {}
    pp = spec.get("placement_params") or {}
    mirror_x = bool(pp.get("mirror_x", True))
    bottom_m = float(pp.get("bottom_margin_ratio", 0.14))
    frames: list[dict[str, Any] | None] = []
    for i, rec in enumerate(spec.get("balcony_placement") or []):
        name = str(rec.get("type") or "")
        ir = library.get(name)
        if not isinstance(ir, dict):
            print(f"warn: unknown balcony type {name!r}")
            frames.append(None)
            continue
        frames.append(_frame(spec, i, rec, ir, mirror_x=mirror_x, bottom_m=bottom_m))
    link_stacks(frames, library)
    return frames


def front_edge_x(frame: dict[str, Any]) -> tuple[float, float] | None:
    """X range of the flat front edge parallel to the wall, if the plan has one."""
    outline = frame.get("outline")
    if not outline:
        return None
    y_far = max(p[1] for p in outline)
    if abs(y_far - frame["y_wall"]) < 1e-4:
        return None
    xs = [p[0] for p in outline if abs(p[1] - y_far) < 1e-4]
    if len(xs) < 2:
        return None
    lo, hi = min(xs), max(xs)
    if hi - lo < MIN_FRONT_WIDTH_M:
        return None
    return lo, hi


def front_y(frame: dict[str, Any]) -> float:
    return max(p[1] for p in frame["outline"])


def front_window_targets(
    frames: Sequence[dict[str, Any] | None],
    windows: Sequence[dict[str, Any]],
    *,
    margin: float = FRONT_WINDOW_MARGIN_M,
) -> dict[int, tuple[int, tuple[float, float, float, float]]]:
    """Windows that belong on an enclosed balcony's front face.

    ``windows`` carry ``r``, ``c``, ``span`` and world ``ox``, ``oz``, ``ww``,
    ``hh``. Returns ``{window_index: (frame_index, (ox, oz, ww, hh))}`` with the
    box clamped inside the front face. A window matches when it is on the same
    floor, its bays overlap the balcony's bays, and its center lies on the
    front face between the floor and the top of the enclosure.
    """
    out: dict[int, tuple[int, tuple[float, float, float, float]]] = {}
    for wi, w in enumerate(windows):
        r = int(w["r"])
        c_lo = int(w["c"])
        c_hi = c_lo + max(1, int(w.get("span", 1))) - 1
        wcx = float(w["ox"]) + 0.5 * float(w["ww"])
        wcz = float(w["oz"]) + 0.5 * float(w["hh"])
        for f in frames:
            if f is None or f["enclosure"] != "enclosed" or f["outline"] is None:
                continue
            if f["row"] != r or c_hi < f["col0"] or c_lo > f["col1"]:
                continue
            fx = front_edge_x(f)
            if fx is None or not (fx[0] < wcx < fx[1]):
                continue
            z_lo = f["sill_z"] + f["thick"]
            z_hi = float(f["top_z"])
            if not (z_lo < wcz < z_hi):
                continue
            x0 = max(float(w["ox"]), fx[0] + margin)
            x1 = min(float(w["ox"]) + float(w["ww"]), fx[1] - margin)
            z0 = max(float(w["oz"]), z_lo + margin)
            z1 = min(float(w["oz"]) + float(w["hh"]), z_hi - margin)
            if x1 - x0 < max(0.3, FRONT_WINDOW_MIN_FIT * float(w["ww"])) or z1 - z0 < 0.3:
                continue
            out[wi] = (f["index"], (x0, z0, x1 - x0, z1 - z0))
            break
    return out


def column_points(
    frame: dict[str, Any],
    *,
    count: int = 0,
    width: float = COLUMN_WIDTH_M,
) -> list[tuple[float, float]]:
    """Column XY on the slab. count=0 puts one at every bend of the outer edge."""
    outline = frame["outline"]
    y_wall = frame["y_wall"]
    shape = frame["shape"]
    mid = centroid(outline)
    chain = support_polyline(shape, outline, y_wall)
    if count <= 0:
        if shape == "circle":
            pts = even_along_polyline(chain, 3)
        else:
            pts = [p for p in outline if abs(p[1] - y_wall) > 1e-4]
    else:
        pts = even_along_polyline(chain, int(count))
    return [inset_point(p, mid, width * 0.75) for p in pts]
