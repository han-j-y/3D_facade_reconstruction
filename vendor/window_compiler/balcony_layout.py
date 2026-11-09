"""Where each balcony sits on the façade grid (no Blender dependency).

``balcony_frames`` turns ``balcony_placement`` into world-space frames
(slab span, sill height, plan outline). It also links stacked balconies:
a half-enclosed or enclosed balcony with a balcony directly above it
takes that balcony's slab as its roof, and the upper slab is aligned to
the lower footprint. An enclosed circle is different: touching cells
become one semicircular drum (depth = half the diameter) from the lowest
sill to the top of the stack. ``front_window_targets`` moves windows that
share a cell with an enclosed balcony onto that balcony's front face.
"""

from __future__ import annotations

import statistics
from typing import Any, Sequence

from balcony_plan import (
    DEFAULT_SLAB_DEPTH_M,
    ENCLOSED_CLEAR_H_M,
    JULIET_MASONRY_DIMS,
    centroid,
    curtain_circle_panes,
    even_along_polyline,
    inset_point,
    slab_outline,
    support_polyline,
)
from facade_spec import JULIET_WINDOW_OVERLAP, get_cell, image_facade_extent_x, image_facade_extent_y, total_grid_size, wall_world_extent

STACKABLE = frozenset({"half_enclosed", "enclosed"})
PHOTO_RAIL_MIN_M = 0.3  # railing height from the photo box, clamped
PHOTO_RAIL_MAX_M = 1.3
PROJECTING_RAIL_M = 1.1  # a projecting balcony's railing is always this high (user, 2026-11-08)
STACK_OVERHANG_M = 0.5  # an open slab reaching this far past the enclosed block under it keeps its own span
OUTLINE_SHAPES = frozenset({"triangle", "circle"})

STACK_MIN_OVERLAP = 0.5
MIN_CLEAR_H_M = 1.2
FRONT_WINDOW_MARGIN_M = 0.08
# A window keeps at least this share of its width on the front, else it stays on the wall.
FRONT_WINDOW_MIN_FIT = 0.6
MIN_FRONT_WIDTH_M = 0.4
COLUMN_WIDTH_M = 0.25
# Wall kept on each side of an enclosed block's outer windows, as a share of their median width.
ENCLOSED_WINDOW_MARGIN = 0.25
SIDE_WINDOW_MARGIN_M = 0.12
SIDE_WALL_THICK_M = 0.12  # front panel the side window keeps clear of


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

    # The norms are fractions of image width, the wall spans only the grid's extent of the image.
    iw = float(((spec.get("meta") or {}).get("image_size") or [0])[0] or 0.0)
    xl, xr = image_facade_extent_x(spec) if iw > 0 else (0.0, 1.0)
    iw = iw if iw > 0 else 1.0
    grid_w = max(1e-6, xr - xl)

    # The wall can reach past the grid, so a balcony beyond the outermost windows keeps its place on it.
    wmin, wmax, _, _ = wall_world_extent(spec, mirror_x=mirror_x)
    n_lo = (-wmax / total_w + 0.5) if mirror_x else (wmin / total_w + 0.5)
    n_hi = (0.5 - wmin / total_w) if mirror_x else (wmax / total_w + 0.5)

    def to_grid(norm: float) -> float:
        return min(n_hi, max(n_lo, (norm * iw - xl) / grid_w))

    span_w = max(0.25, min(total_w * 0.98, w_norm_f * iw / grid_w * total_w))
    if cx_norm_f is not None:
        cx_n = to_grid(cx_norm_f)
    elif window_cx_norm_f is not None:
        cx_n = to_grid(window_cx_norm_f)
    elif bay_cx_norm_f is not None:
        cx_n = to_grid(bay_cx_norm_f)
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
    if x0 < wmin:
        x1 += wmin - x0
        x0 = wmin
    if x1 > wmax:
        x0 -= x1 - wmax
        x1 = wmax
    return x0, x1, 0.5 * (x0 + x1)


def _frame(
    spec: dict[str, Any],
    index: int,
    rec: dict[str, Any],
    ir: dict[str, Any],
    *,
    mirror_x: bool,
    bottom_m: float,
    x_span: tuple[float, float] | None = None,
    type_width: float | None = None,
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
    if x_span is not None:
        x0, x1 = x_span
        cx = 0.5 * (x0 + x1)
    else:
        x0, x1, cx = photo_x_span(
            spec, rec, mirror_x=mirror_x, bay_x0=x0, bay_x1=x1, row=row, col0=c0, col1=c1
        )
    span_w = max(0.3, x1 - x0)
    cell_h = max(0.3, z1 - z0)
    floor = ir.get("floor") or {}
    params = floor.get("params") or {}
    shape = str(floor.get("shape") or "rectangle").lower()
    # A placement may override its type's structure (rooftop is per balcony).
    structure = str(rec.get("structure") or ir.get("structure") or "projecting")
    enclosure = str(ir.get("enclosure") or "open")
    thick = float((ir.get("output") or {}).get("slab_thickness") or 0.12)
    depth = float(DEFAULT_SLAB_DEPTH_M)

    sill_z = z0 + cell_h * bottom_m
    photo_rail_h = None
    photo_top_z = None
    rail_measured = False  # railing height read off the photo box (juliets vote it per type)
    if rec.get("photo_bottom_y") is not None:
        yt, yb = image_facade_extent_y(spec)
        to_z = total_grid_size(grid)[1] / max(1e-6, yb - yt)
        sill_z = (yb - float(rec["photo_bottom_y"])) * to_z
        if rec.get("photo_top_y") is not None:
            slab = 0.0 if structure in ("juliet", "rooftop") else thick  # no slab under these railings
            top_z = (yb - float(rec["photo_top_y"])) * to_z
            photo_rail_h = min(PHOTO_RAIL_MAX_M, max(PHOTO_RAIL_MIN_M, top_z - sill_z - slab))
            rail_measured = top_z - sill_z - slab <= PHOTO_RAIL_MAX_M
            if top_z - sill_z - slab > PHOTO_RAIL_MAX_M:
                rail_ir = min(PHOTO_RAIL_MAX_M, float((ir.get("railing") or {}).get("height") or 1.1))
                if structure == "juliet" and float(rec.get("window_overlap") or 0.0) >= JULIET_WINDOW_OVERLAP:
                    # A juliet whose window reaches into its box: the window stands behind the railing, so the
                    # railing stands on the box bottom (above it the box is the opening).
                    photo_rail_h = rail_ir
                else:
                    # Box taller than any railing (it holds the bracket below): hang from the box top.
                    photo_rail_h = rail_ir
                    sill_z = top_z - photo_rail_h - slab
                    photo_top_z = top_z
            if structure == "projecting" and enclosure == "open":
                # Fixed railing height; the slab stays where the photo puts it (or under the box top when the box
                # holds the bracket below).
                if photo_top_z is not None:
                    sill_z = photo_top_z - PROJECTING_RAIL_M - slab
                photo_rail_h = PROJECTING_RAIL_M
    if structure == "free_standing":
        deck_h = float(params.get("height") or 0.0)
        if deck_h > 0.05:
            sill_z = deck_h

    pad = span_w * 0.01
    bx0, bx1 = x0 + pad, x1 - pad
    if shape in OUTLINE_SHAPES or enclosure in STACKABLE:
        tw = float(params.get("width") or params.get("diameter") or 0.0)
        if type_width is not None and enclosure == "open":
            tw = type_width  # measured on the photo at the facade's scale, not the IR's 12 m-image guess
        if tw > 0.3:
            tw = min(tw, span_w * 0.98)
            bx0 = cx - tw / 2.0
            bx1 = cx + tw / 2.0
    if shape not in OUTLINE_SHAPES and enclosure in STACKABLE:
        # Rectangle railings size from span_w * 0.98; keep them on this slab.
        span_w = (bx1 - bx0) / 0.98

    # A juliet railing has no slab, so nothing can stack on or under it.
    stacks = structure not in ("inset", "composite", "juliet")
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
        "from_bays": rec.get("width_norm") is None,
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
        "curtain_wall": bool((ir.get("walls") or {}).get("curtain_wall")),
        "depth": depth,
        "thick": thick,
        "sill_z": sill_z,
        "photo_z": rec.get("photo_bottom_y") is not None,
        "photo_rail_h": photo_rail_h,
        "rail_measured": rail_measured,
        "photo_top_z": photo_top_z,
        "bx0": bx0,
        "bx1": bx1,
        "y_wall": front_y,
        "outline": outline,
        "aligned": False,
        "roof": None,
        "above": None,
        "top_z": None,
        "carries_stack": False,
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


def _block_footprint(lowers: list[dict[str, Any]], own: dict[str, float] | None = None) -> dict[str, Any]:
    """Footprint of the slab over enclosed cells: every cell under it (an oriel cut into bays is one block), and the
    slab's own photo span when that reaches further (a long balcony over two oriels keeps its length)."""
    first = lowers[0]
    lo, hi = min(f["bx0"] for f in lowers), max(f["bx1"] for f in lowers)
    if own is not None and (own["bx0"] < lo - STACK_OVERHANG_M or own["bx1"] > hi + STACK_OVERHANG_M):
        lo, hi = min(lo, own["bx0"]), max(hi, own["bx1"])
    elif len(lowers) == 1:
        return first
    depth = max(f["depth"] for f in lowers)
    shape = first["shape"] if first["shape"] in OUTLINE_SHAPES else "rectangle"
    return {
        "bx0": lo,
        "bx1": hi,
        "depth": depth,
        "shape": first["shape"],
        "outline": slab_outline(shape, x0=lo, x1=hi, y_wall=first["y_wall"], depth=depth, structure=first["structure"]),
    }


def link_stacks(
    frames: Sequence[dict[str, Any] | None],
    library: dict[str, Any],
) -> None:
    """Share floors up the stack (in place).

    Bottom rows first, so a chain of enclosures aligns from the lowest one.
    The highest member of a chain keeps its own roof slab.
    """
    live = [f for f in frames if f is not None]
    carried: dict[int, list[dict[str, Any]]] = {}
    own = {g["index"]: {"bx0": g["bx0"], "bx1": g["bx1"]} for g in live}  # spans before any alignment
    for f in sorted(live, key=lambda fr: -fr["row"]):
        if f["enclosure"] not in STACKABLE or f["outline"] is None:
            continue
        best: dict[str, Any] | None = None
        best_ov = STACK_MIN_OVERLAP
        for g in live:
            if g["row"] != f["row"] - 1 or g["outline"] is None:
                continue
            ov = _x_overlap(own[f["index"]], own[g["index"]])
            if ov >= best_ov:
                best, best_ov = g, ov
        if best is not None and best["sill_z"] > f["sill_z"] + f["thick"] + MIN_CLEAR_H_M:
            f["roof"] = "shared"
            f["above"] = best["index"]
            f["top_z"] = best["sill_z"]
            best["carries_stack"] = True
            carried.setdefault(best["index"], []).append(f)
            _align_to(best, _block_footprint(carried[best["index"]], own[best["index"]]))
        else:
            f["roof"] = "own"
            f["top_z"] = _own_top_z(f, library.get(f["name"]) or {})
    _open_on_oriel_takes_its_width(live, carried, own)


ORIEL_BLOCK_GAP_M = 0.3  # enclosed cells side by side closer than this are one oriel
ORIEL_BLOCK_SILL_TOL_M = 0.3


def _open_on_oriel_takes_its_width(
    live: list[dict[str, Any]], carried: dict[int, list[dict[str, Any]]], own: dict[int, dict[str, float]]
) -> None:
    """An open balcony on an oriel is exactly as wide as the oriel (user, 2026-11-08), in place.

    The oriel is every enclosed cell on the row under the balcony that is joined side by side (gap below
    ``ORIEL_BLOCK_GAP_M``, same sill) to a cell the balcony already carries. Cells the balcony did not pick up (too little
    overlap) now carry it too, and the balcony takes the block's footprint, not its own photo span. A gallery whose photo
    span reaches past the block by more than half a cell width on a side (it runs on over bays with no oriel) keeps its
    length."""
    by_index = {f["index"]: f for f in live}
    for idx, lowers in carried.items():
        top = by_index.get(idx)
        if top is None or top["enclosure"] != "open":
            continue
        block = [f for f in lowers if f["enclosure"] in STACKABLE and f["enclosure"] != "open"]
        if not block:
            continue
        row, sill = block[0]["row"], float(block[0]["sill_z"])
        grown = True
        while grown:
            grown = False
            for g in live:
                if any(g is b for b in block) or g["row"] != row or g["enclosure"] == "open" or g["outline"] is None:
                    continue
                if g.get("above") is not None or abs(float(g["sill_z"]) - sill) > ORIEL_BLOCK_SILL_TOL_M:
                    continue
                if any(g["bx0"] <= b["bx1"] + ORIEL_BLOCK_GAP_M and b["bx0"] <= g["bx1"] + ORIEL_BLOCK_GAP_M for b in block):
                    if top["sill_z"] <= g["sill_z"] + g["thick"] + MIN_CLEAR_H_M:
                        continue
                    g["roof"] = "shared"
                    g["above"] = idx
                    g["top_z"] = top["sill_z"]
                    lowers.append(g)
                    block.append(g)
                    grown = True
        # Gallery test on photo spans: the oriel's cells may have been narrowed to their windows already
        # (else a balcony as wide as its oriel on the photo reads as a gallery).
        lo = min(b.get("photo_bx0", b["bx0"]) for b in block)
        hi = max(b.get("photo_bx1", b["bx1"]) for b in block)
        half_cell = 0.5 * statistics.median(b.get("photo_bx1", b["bx1"]) - b.get("photo_bx0", b["bx0"]) for b in block)
        span = own.get(idx)
        gallery = span is not None and (span["bx0"] < lo - half_cell or span["bx1"] > hi + half_cell)
        _align_to(top, _block_footprint(block, span) if gallery else _block_footprint(block))


def _is_enclosed_circle(frame: dict[str, Any] | None) -> bool:
    return bool(
        frame is not None
        and frame.get("enclosure") == "enclosed"
        and frame.get("shape") == "circle"
        and frame.get("outline") is not None
    )


def _component_neighbors(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """4-connected: same floor and touching bays, or the floor above with overlapping bays."""
    if a["row"] == b["row"]:
        return a["col0"] <= b["col1"] + 1 and b["col0"] <= a["col1"] + 1
    if abs(int(a["row"]) - int(b["row"])) == 1:
        return a["col0"] <= b["col1"] and b["col0"] <= a["col1"]
    return False


def fuse_enclosed_circles(
    frames: Sequence[dict[str, Any] | None],
    library: dict[str, Any],
) -> None:
    """Turn each connected enclosed circle into one semicircular drum, curtain wall or punched.

    The host is the lowest cell. Its plan spans every bay in the component,
    its depth is half that width, and its skin runs up to the top of the
    highest cell. The other cells stay in the list but draw nothing. Cells only
    join cells with the same skin.
    """
    live = [f for f in frames if _is_enclosed_circle(f)]
    seen = [False] * len(live)
    for i in range(len(live)):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp: list[dict[str, Any]] = []
        while stack:
            k = stack.pop()
            comp.append(live[k])
            for j, other in enumerate(live):
                if seen[j] or not _component_neighbors(live[k], other):
                    continue
                if bool(other.get("curtain_wall")) != bool(live[k].get("curtain_wall")):
                    continue
                seen[j] = True
                stack.append(j)
        _fuse_circle_component(comp, frames, library)


def _fuse_circle_component(
    comp: list[dict[str, Any]],
    frames: Sequence[dict[str, Any] | None],
    library: dict[str, Any],
) -> None:
    bottom_row = max(int(fr["row"]) for fr in comp)
    top_row = min(int(fr["row"]) for fr in comp)
    bottoms = [fr for fr in comp if int(fr["row"]) == bottom_row]
    tops = [fr for fr in comp if int(fr["row"]) == top_row]
    host = min(bottoms, key=lambda fr: int(fr["col0"]))
    x0 = min(float(fr["x0"]) for fr in comp)
    x1 = max(float(fr["x1"]) for fr in comp)
    pad = max(0.0, (x1 - x0) * 0.01)
    bx0, bx1 = x0 + pad, x1 - pad
    width = bx1 - bx0
    if width < 0.3:
        return
    depth = width / 2.0
    host["x0"] = x0
    host["x1"] = x1
    host["bx0"] = bx0
    host["bx1"] = bx1
    host["cx"] = 0.5 * (bx0 + bx1)
    host["depth"] = depth
    host["shape"] = "circle"
    host["span_w"] = width / 0.98
    curtain = bool(host.get("curtain_wall"))
    y_wall = float(host["y_wall"])
    host["outline"] = (
        curtain_circle_panes(bx0, bx1, y_wall, depth)
        if curtain
        else slab_outline("circle", x0=bx0, x1=bx1, y_wall=y_wall, depth=depth)
    )
    host["col0"] = min(int(fr["col0"]) for fr in comp)
    host["col1"] = max(int(fr["col1"]) for fr in comp)
    host["row_lo"] = top_row
    host["row_hi"] = bottom_row
    host["sill_z"] = min(float(fr["sill_z"]) for fr in bottoms)
    top_z = max((float(fr["top_z"]) for fr in tops if fr.get("top_z") is not None), default=None)
    if top_z is None:
        top_z = _own_top_z(host, library.get(host["name"]) or {})
    host["top_z"] = top_z
    host["roof"] = "own" if any(fr.get("roof") == "own" for fr in tops) else (tops[0].get("roof") or "own")
    host["curtain_drum"] = curtain
    host["drum"] = True
    host["skip_body"] = False
    by_row: dict[int, list[dict[str, Any]]] = {}
    for fr in comp:
        by_row.setdefault(int(fr["row"]), []).append(fr)
    host["floor_sills"] = sorted(
        min(float(fr["sill_z"]) for fr in group) for group in by_row.values()
    )
    for fr in comp:
        if fr is host:
            continue
        fr["skip_body"] = True
    lids = {fr["above"] for fr in tops if fr.get("above") is not None}
    for f in frames:  # a balcony on top of the drum keeps its footprint
        if f is not None and f["index"] in lids and not any(f is fr for fr in comp):
            _align_to(f, host)


def _fit_spans(
    frames: Sequence[dict[str, Any] | None],
    windows: Sequence[dict[str, Any]],
) -> dict[int, tuple[float, float] | None]:
    """X span per enclosed cell, from the windows of its connected block; None removes the cell.

    Cells placed from grid bays take the bay's whole width, wall included. Every row of a block gets the same outer
    edges, so a stack stays straight. A block with no windows keeps its bays, and curtain walls are left alone.

    A block with side-wall windows on one side has its true position between the front and side windows: the photo
    shows the front face pushed away from the camera. It is centred on the span of both, as wide as the front face
    (outer edge to the corner between the windows), on the front windows' column, and the front windows move with it
    (``ox``, in place).
    """
    live = [
        f for f in frames
        if f is not None and f["enclosure"] == "enclosed" and f["from_bays"] and not f["curtain_wall"]
    ]
    out: dict[int, tuple[float, float] | None] = {}
    seen: set[int] = set()
    for start in live:
        if start["index"] in seen:
            continue
        seen.add(start["index"])
        comp, stack = [], [start]
        while stack:
            f = stack.pop()
            comp.append(f)
            for g in live:
                if g["index"] not in seen and _component_neighbors(f, g):
                    seen.add(g["index"])
                    stack.append(g)
        lo = min(f["x0"] for f in comp)
        hi = max(f["x1"] for f in comp)
        hits, sides = [], []
        rectangular = all(f["shape"] == "rectangle" for f in comp)
        for w in windows:
            c_lo = int(w["c"])
            c_hi = c_lo + max(1, int(w.get("span", 1))) - 1
            cx = float(w["ox"]) + 0.5 * float(w["ww"])
            if lo < cx < hi and any(f["row"] == w["r"] and c_lo <= f["col1"] and c_hi >= f["col0"] for f in comp):
                (sides if w.get("side_face") and rectangular else hits).append(w)
        if not hits:
            continue
        widths = sorted(float(w["ww"]) for w in hits)
        margin = ENCLOSED_WINDOW_MARGIN * widths[len(widths) // 2]
        left = max(lo, min(float(w["ox"]) for w in hits) - margin)
        right = min(hi, max(float(w["ox"]) + float(w["ww"]) for w in hits) + margin)
        front_cols = {int(w["c"]) for w in hits}
        front_lo = min(float(w["ox"]) for w in hits)
        front_hi = max(float(w["ox"]) + float(w["ww"]) for w in hits)
        right_side = [w for w in sides if float(w["ox"]) + 0.5 * float(w["ww"]) > 0.5 * (front_lo + front_hi)]
        if sides and len(front_cols) == 1 and len(right_side) in (0, len(sides)):
            # Mirror the left case onto the right one: u runs from the block's outer edge towards its side wall.
            sign = 1.0 if right_side else -1.0
            def u_lo(w: dict[str, Any]) -> float:
                return float(w["ox"]) if sign > 0 else -(float(w["ox"]) + float(w["ww"]))
            def u_hi(w: dict[str, Any]) -> float:
                return float(w["ox"]) + float(w["ww"]) if sign > 0 else -float(w["ox"])
            outer = min(u_lo(w) for w in hits) - margin
            corner = 0.5 * (max(u_hi(w) for w in hits) + min(u_lo(w) for w in sides))
            centre = sign * 0.5 * (outer + max(u_hi(w) for w in sides) + margin)
            half = 0.5 * (corner - outer)
            shift = centre - 0.5 * (front_lo + front_hi)
            for w in hits:
                w["ox"] = float(w["ox"]) + shift
            col = next(iter(front_cols))
            for f in comp:
                out[f["index"]] = (centre - half, centre + half) if f["col0"] <= col <= f["col1"] else None
            continue
        for f in comp:
            x0, x1 = max(f["x0"], left), min(f["x1"], right)
            # a round block fuses into one drum, so a narrow cell must stay to keep it connected
            keep = x1 - x0 >= MIN_FRONT_WIDTH_M or (not rectangular and x1 > x0)
            out[f["index"]] = (x0, x1) if keep else None
    return out


def _plain_frames(
    spec: dict[str, Any],
    fits: dict[int, tuple[float, float] | None] | None = None,
) -> list[dict[str, Any] | None]:
    library = spec.get("balconies") or {}
    pp = spec.get("placement_params") or {}
    mirror_x = bool(pp.get("mirror_x", True))
    bottom_m = float(pp.get("bottom_margin_ratio", 0.14))
    fits = fits or {}
    frames: list[dict[str, Any] | None] = []
    for i, rec in enumerate(spec.get("balcony_placement") or []):
        name = str(rec.get("type") or "")
        ir = library.get(name)
        if not isinstance(ir, dict):
            print(f"warn: unknown balcony type {name!r}")
            frames.append(None)
            continue
        if i in fits and fits[i] is None:
            frames.append(None)
            continue
        frames.append(_frame(spec, i, rec, ir, mirror_x=mirror_x, bottom_m=bottom_m, x_span=fits.get(i)))
    widths = _outline_type_widths(frames)
    if widths:
        recs = spec.get("balcony_placement") or []
        for k, f in enumerate(frames):
            if f is not None and f["name"] in widths:
                frames[k] = _frame(spec, k, recs[k], library[f["name"]], mirror_x=mirror_x, bottom_m=bottom_m,
                                   x_span=fits.get(k), type_width=widths[f["name"]])
    recs = spec.get("balcony_placement") or []
    for k, f in enumerate(frames):
        if f is None:
            continue
        if fits.get(k) is not None:
            # Span before the window fit narrowed it: what the photo shows (compared with other photo spans).
            raw = _frame(spec, k, recs[k], library[f["name"]], mirror_x=mirror_x, bottom_m=bottom_m)
            f["photo_bx0"], f["photo_bx1"] = raw["bx0"], raw["bx1"]
        else:
            f["photo_bx0"], f["photo_bx1"] = f["bx0"], f["bx1"]
    return frames


def _outline_type_widths(frames: Sequence[dict[str, Any] | None]) -> dict[str, float]:
    """One width per open round/triangle type: the median photo width of its members (metres on this facade)."""
    by_type: dict[str, list[float]] = {}
    for f in frames:
        if f is None or f["shape"] not in OUTLINE_SHAPES or f["enclosure"] != "open" or f["from_bays"] or f["structure"] == "rooftop":
            continue
        by_type.setdefault(f["name"], []).append(f["x1"] - f["x0"])
    return {name: float(statistics.median(ws)) for name, ws in by_type.items()}


def enclosed_window_fits(
    spec: dict[str, Any],
    windows: Sequence[dict[str, Any]],
) -> dict[int, tuple[float, float] | None]:
    """Spans for ``balcony_frames``. Take them once, before any window moves onto a balcony front."""
    return _fit_spans(_plain_frames(spec), windows)


def unify_juliet_rail_heights(frames: Sequence[dict[str, Any] | None], library: dict[str, Any]) -> None:
    """Open juliets of one type share one railing height: the median of the heights read off their photo boxes
    (user, 2026-11-08), in place. A type with no measured member uses its IR height (1.1 m). A juliet hung from its box
    top keeps its top; the others keep their bottom."""
    by_type: dict[str, list[dict[str, Any]]] = {}
    for f in frames:
        if f is not None and f["enclosure"] == "open" and f["structure"] == "juliet":
            by_type.setdefault(f["name"], []).append(f)
    for name, members in by_type.items():
        measured = [float(f["photo_rail_h"]) for f in members if f.get("rail_measured") and f.get("photo_rail_h")]
        h = statistics.median(measured) if measured else float(((library.get(name) or {}).get("railing") or {}).get("height") or 1.1)
        for f in members:
            if f.get("photo_top_z") is not None:
                f["sill_z"] = float(f["photo_top_z"]) - h
            f["photo_rail_h"] = h


def balcony_frames(
    spec: dict[str, Any],
    fits: dict[int, tuple[float, float] | None] | None = None,
) -> list[dict[str, Any] | None]:
    """One frame per ``balcony_placement`` entry (None when it cannot be placed)."""
    library = spec.get("balconies") or {}
    frames = _plain_frames(spec, fits)
    unify_juliet_rail_heights(frames, library)
    link_stacks(frames, library)
    fuse_enclosed_circles(frames, library)
    return frames


def center_on_windows(
    frames: Sequence[dict[str, Any] | None],
    placements: Sequence[dict[str, Any]],
    windows: Sequence[dict[str, Any]],
) -> None:
    """Windows are the reference: an open balcony moves sideways (in place) onto the centre of its own window(s)
    as compiled, after type sizing and column alignment. Height and width stay. Balconies without a window,
    rooftop railings, slabs that carry an enclosure stack and galleries with other windows over them keep their
    place."""
    by_unit: dict[str, list[dict[str, Any]]] = {}
    for w in windows:
        if w.get("unit_id") is not None and not w.get("side_face"):
            by_unit.setdefault(str(w["unit_id"]), []).append(w)
    for f in frames:
        if f is None or f["enclosure"] != "open" or f["structure"] == "rooftop" or f.get("aligned"):
            continue
        place = placements[f["index"]] if f["index"] < len(placements) else {}
        if place.get("column_aligned"):
            continue  # its type's column shares one centre (merge_dsl.align_type_columns)
        ids = place.get("window_unit_ids") or []
        wins = [w for u in ids for w in by_unit.get(str(u), [])]
        if not wins:
            continue
        rows = {w.get("r") for w in wins}
        others = [
            w for w in windows
            if w.get("r") in rows and not w.get("side_face") and all(w is not v for v in wins)
            and f["bx0"] < float(w["ox"]) + 0.5 * float(w["ww"]) < f["bx1"]
        ]
        if others:
            continue  # a gallery under several windows is not any one window's balcony
        cx = 0.5 * (min(float(w["ox"]) for w in wins) + max(float(w["ox"]) + float(w["ww"]) for w in wins))
        dx = cx - 0.5 * (f["bx0"] + f["bx1"])
        for key in ("x0", "x1", "cx", "bx0", "bx1"):
            f[key] += dx
        if f.get("outline"):
            f["outline"] = [(x + dx, y) for x, y in f["outline"]]
        f["window_dx"] = dx


def seat_on_window_bottoms(
    frames: Sequence[dict[str, Any] | None],
    windows: Sequence[dict[str, Any]],
) -> None:
    """An open balcony above its window's bottom drops down to it (in place); one at or below it stays.

    Slab top on the lowest window bottom the balcony overlaps; a juliet railing, which has no slab, starts there.
    Balconies with no window on their row keep their height, and so do rooftop railings and enclosures.
    """
    live = [f for f in frames if f is not None]
    for f in live:
        if f["enclosure"] != "open" or f["structure"] not in ("projecting", "juliet"):
            continue
        if f.get("photo_z") and f.get("photo_top_z") is None:
            continue
        bottoms = [
            float(w["oz"])
            for w in windows
            if w["r"] == f["row"] and "on_balcony" not in w
            and min(f["bx1"], w["ox"] + w["ww"]) - max(f["bx0"], w["ox"]) > 0.0
        ]
        if not bottoms:
            continue
        drop = f["thick"] if f["structure"] != "juliet" else 0.0
        target = max(0.0, min(bottoms) - drop)
        if target >= f["sill_z"]:
            continue
        f["sill_z"] = target  # the whole balcony moves down; its railing height stays (1.1 m / the type's juliet height)
        for g in live:
            if g.get("above") == f["index"]:
                g["top_z"] = f["sill_z"]


def _juliet_rail_material(frame: dict[str, Any], library: dict[str, Any]) -> str | None:
    """'masonry' or 'metal' for an open juliet with an open_work railing, else None."""
    if frame["structure"] != "juliet" or frame["enclosure"] != "open":
        return None
    rail = (library.get(frame["name"]) or {}).get("railing") or {}
    if str(rail.get("kind") or "open_work").strip().lower() != "open_work":
        return None
    material = str(rail.get("material") or "metal").strip().lower()
    return "masonry" if material in ("masonry", "stone", "concrete_post") else "metal"


def _balcony_windows(
    f: dict[str, Any], placements: Sequence[dict[str, Any]], windows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The wall windows behind an open balcony: its partner windows; without links, the windows on its row that
    overlap it in x. Not windows on a balcony front or a side wall."""
    ids = {str(u) for u in (placements[f["index"]] if f["index"] < len(placements) else {}).get("window_unit_ids") or []}
    wins = [
        w for w in windows
        if not w.get("side_face") and w.get("on_balcony") is None and w.get("unit_id") is not None
        and str(w["unit_id"]) in ids
    ]
    if not wins:
        wins = [
            w for w in windows
            if w.get("r") == f["row"] and not w.get("side_face") and w.get("on_balcony") is None
            and min(f["bx1"], w["ox"] + w["ww"]) - max(f["bx0"], w["ox"]) > 0.0
        ]
    return wins


def fit_juliet_windows(
    frames: Sequence[dict[str, Any] | None],
    placements: Sequence[dict[str, Any]],
    windows: Sequence[dict[str, Any]],
    library: dict[str, Any],
    *,
    min_window_h: float = 0.35,
) -> None:
    """Open balconies (projecting, juliet) and their windows: window bottom = balcony bottom (user, 2026-11-08).

    Balcony bottom = slab top of a projecting balcony, railing bottom of a juliet. A window reaching below a
    projecting slab moves the whole balcony down onto it (railing stays 1.1 m); a juliet keeps its type's railing
    height. Then every window is set to the balcony bottom (window top kept; up or down), a masonry open_work juliet's
    to the top of its bottom rail. Juliets keep their
    windows as ``juliet_windows`` so the compiler can make the railing as wide as the windows after the window IR fit.
    Windows are the balcony's partner windows; without links, the windows on its row that overlap it."""
    bot_h = float(JULIET_MASONRY_DIMS.get("bot_h", 0.0))
    live = [f for f in frames if f is not None]
    for f in live:
        if f["enclosure"] != "open" or f["structure"] not in ("projecting", "juliet"):
            continue
        wins = [w for w in _balcony_windows(f, placements, windows) if not w.get("bottom_fitted")]
        if not wins:
            continue
        juliet = f["structure"] == "juliet"
        material = _juliet_rail_material(f, library)
        if juliet:
            f["juliet_windows"] = wins
            f["juliet_material"] = material
        drop = 0.0 if juliet else float(f["thick"])
        floor = float(f["sill_z"]) + drop
        low = min(float(w["oz"]) for w in wins)
        if low < floor - 1e-6 and not juliet:
            # A window reaching below a projecting slab: the whole balcony moves down onto it (railing stays 1.1 m).
            f["sill_z"] = low - drop
            floor = low
            for g in live:
                if g.get("above") == f["index"]:
                    g["top_z"] = f["sill_z"]
        # A juliet keeps its type's railing height: a window reaching lower is raised to the railing bottom (below).
        rail_kind = str(((library.get(f["name"]) or {}).get("railing") or {}).get("kind") or "open_work").strip().lower()
        if juliet and (material == "metal" or rail_kind in ("solid", "surface_panel", "panel", "parapet", "concrete")):
            # Metal, solid and surface-panel juliets go into the opening (window bottom = railing bottom now); solid
            # and panel ones with their outer face on the wall face (user, 2026-11-08).
            f["juliet_in_opening"] = True
        target = floor + (bot_h if juliet and material == "masonry" else 0.0)
        for w in wins:
            top = float(w["oz"]) + float(w["hh"])
            if top - target >= min_window_h:
                w["oz"], w["hh"] = target, top - target
            w["bottom_fitted"] = True


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
            if f.get("curtain_wall") or (w.get("side_face") and f["shape"] == "rectangle"):
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


def side_wall_targets(
    frames: Sequence[dict[str, Any] | None],
    windows: Sequence[dict[str, Any]],
    *,
    margin: float = SIDE_WINDOW_MARGIN_M,
) -> dict[int, dict[str, Any]]:
    """Windows flagged ``side_face``, placed on the nearest side wall of a rectangular punched block on their floor.

    Returns ``{window_index: {frame, sign, edge_x, y0, ww, oz, hh, mirror}}``. ``sign`` is +1 on the block's max-x side.
    The photo shows a side wall foreshortened, so the width is capped by the wall's length; height and bottom follow
    the block's front window on that floor, as they would on a real bay. The photo cannot see the far side wall:
    ``mirror`` (frame, sign, edge_x, y0) is the same window there, assuming a symmetric bay.
    """
    out: dict[int, dict[str, Any]] = {}
    for wi, w in enumerate(windows):
        if not w.get("side_face"):
            continue
        c = int(w["c"])
        wcx = float(w["ox"]) + 0.5 * float(w["ww"])
        near = [
            f for f in frames
            if f is not None and f["row"] == w["r"] and f["enclosure"] == "enclosed" and not f.get("curtain_wall")
            and f["shape"] == "rectangle" and f["outline"] is not None and f["col0"] - 1 <= c <= f["col1"] + 1
        ]
        if not near:
            continue
        lo, hi = min(f["bx0"] for f in near), max(f["bx1"] for f in near)
        sign = 1 if wcx > 0.5 * (lo + hi) else -1
        f = max(near, key=lambda fr: sign * (fr["bx1"] if sign > 0 else fr["bx0"]))
        # Walk across touching cells on this floor to the block's far wall.
        row_cells = [
            fr for fr in frames
            if fr is not None and fr["row"] == w["r"] and fr["enclosure"] == "enclosed" and not fr.get("curtain_wall")
            and fr["shape"] == "rectangle" and fr["outline"] is not None
        ]
        far = f
        while True:
            edge = far["bx0"] if sign > 0 else far["bx1"]
            next_cell = next(
                (fr for fr in row_cells if fr is not far and abs((fr["bx1"] if sign > 0 else fr["bx0"]) - edge) < 0.2),
                None,
            )
            if next_cell is None:
                break
            far = next_cell
        y_wall = float(f["y_wall"])
        length = front_y(f) - y_wall
        usable = length - 2.0 * margin - SIDE_WALL_THICK_M
        ww = min(float(w["ww"]), usable)
        if ww < 0.3:
            continue
        z_lo, z_hi = f["sill_z"] + f["thick"] + margin, float(f["top_z"]) - margin
        front = [o for o in windows if o.get("on_balcony") == f["index"] and o["r"] == w["r"] and not o.get("side_face")]
        oz, hh = (float(front[0]["oz"]), float(front[0]["hh"])) if front else (float(w["oz"]), float(w["hh"]))
        oz = max(oz, z_lo)
        hh = min(hh, z_hi - oz)
        if hh < 0.3:
            continue
        out[wi] = {
            "frame": int(f["index"]),
            "sign": sign,
            "edge_x": float(f["bx1"] if sign > 0 else f["bx0"]),
            "y0": y_wall + margin + 0.5 * (usable - ww),
            "ww": ww,
            "oz": oz,
            "hh": hh,
            "mirror": {
                "frame": int(far["index"]),
                "sign": -sign,
                "edge_x": float(far["bx0"] if sign > 0 else far["bx1"]),
                "y0": y_wall + margin + 0.5 * (usable - ww),
            },
        }
    return out


def windows_behind_glass(
    frames: Sequence[dict[str, Any] | None],
    windows: Sequence[dict[str, Any]],
) -> set[int]:
    """Windows on the same row whose bays overlap an enclosed curtain-wall frame.

    Those windows are dropped: not moved to the front and not cut into the main wall.
    A fused curtain-circle drum covers every floor between its lowest and highest row.
    """
    drop: set[int] = set()
    for wi, w in enumerate(windows):
        row = int(w["r"])
        c_lo = int(w["c"])
        c_hi = c_lo + max(1, int(w.get("span", 1))) - 1
        for frame in frames:
            if frame is None or frame.get("enclosure") != "enclosed" or not frame.get("curtain_wall"):
                continue
            row_lo = int(frame.get("row_lo", frame["row"]))
            row_hi = int(frame.get("row_hi", frame["row"]))
            if row_hi < row_lo:
                row_lo, row_hi = row_hi, row_lo
            if row < row_lo or row > row_hi or c_hi < frame["col0"] or c_lo > frame["col1"]:
                continue
            drop.add(wi)
            break
    return drop


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
