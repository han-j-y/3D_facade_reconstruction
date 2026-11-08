"""Merge balcony types into a copy of the window recovery DSL (do not edit build_facade_dsl)."""

from __future__ import annotations

import copy
import statistics
from typing import Any

from enclosed import drop_open_under_enclosed
from heuristic_ir import balcony_type_token

from partners import mean_window_center_norm, partner_windows_for_unit
from snap import _bands_from_windows_dsl


def _photo_norms(
    box: list[int] | list[float],
    image_size: tuple[int, int] | list[int] | None,
) -> tuple[float, float]:
    """Return (width_norm, cx_norm) in [0,1] relative to image width."""
    iw = 1.0
    if image_size and len(image_size) >= 1:
        iw = max(1.0, float(image_size[0]))
    x0, x1 = float(box[0]), float(box[2])
    width_norm = max(1e-4, (x1 - x0) / iw)
    cx_norm = (0.5 * (x0 + x1)) / iw
    return float(width_norm), float(cx_norm)


def _bay_ids(unit: dict[str, Any]) -> list[int]:
    """Placement span bays (overlap with balcony width)."""
    raw = unit.get("bays")
    if isinstance(raw, list) and raw:
        return sorted(int(b) for b in raw)
    bs = int(unit.get("bay_start", 0))
    be = int(unit.get("bay_end", bs))
    lo, hi = min(bs, be), max(bs, be)
    return list(range(lo, hi + 1))


def _center_bay_ids(unit: dict[str, Any]) -> list[int]:
    """Bays that define horizontal center (full-cover set or max-overlap bay)."""
    raw = unit.get("bays_center")
    if isinstance(raw, list) and raw:
        return sorted(int(b) for b in raw)
    return _bay_ids(unit)


def _mean_bay_center_norm(
    bay_ids: list[int],
    bay_x: dict[int, tuple[int, int]],
    image_width: float,
) -> float | None:
    """Mean horizontal center of window bay bands (photo pixels), normalized."""
    iw = max(1.0, float(image_width))
    centers: list[float] = []
    for bid in bay_ids:
        band = bay_x.get(int(bid))
        if band is None:
            continue
        bx0, bx1 = band
        centers.append(0.5 * (float(bx0) + float(bx1)))
    if not centers:
        return None
    return sum(centers) / len(centers) / iw


def _normalize_center_mode(center_mode: str) -> str:
    mode = str(center_mode).lower()
    if mode in ("photo", "bay", "window"):
        return mode
    return "window"


STACK_MIN_WIDTH_RATIO = 0.6
OUTER_BAY_MIN_WINDOW_WIDTHS = 0.5


def floor_of_top_edge(box: list[float], floor_y: dict[int, tuple[int, int]], default: int) -> int:
    """The floor whose band holds ``box``'s top edge, else ``default``."""
    top = float(box[1])
    for fid, (y0, y1) in floor_y.items():
        if float(y0) <= top < float(y1):
            return int(fid)
    return default


WINDOW_ABOVE_MAX_FH = 0.4  # window bottom at most this many floor heights above the balcony top
WINDOW_BELOW_MAX_FH = 0.1  # or reaching down behind the railing, at most this far past the balcony bottom


def attach_to_window(
    box: list[float],
    partners: list[dict[str, Any]],
    floor_i: int,
    floor_y: dict[int, tuple[int, int]],
) -> tuple[list[dict[str, Any]], int, list[Any]]:
    """A balcony belongs to the window just above it: take that window's floor, if the window is attached.

    Floors are bands between window rows, so a balcony hanging below a window whose row is offset (a corner bay)
    can fall in the next band. The window is attached when its bottom is at most ``WINDOW_ABOVE_MAX_FH`` floor
    heights above the balcony top, or reaches down behind the railing to at most ``WINDOW_BELOW_MAX_FH`` past
    the balcony bottom. A window further away is not this balcony's: the link is dropped (ids returned) and the
    balcony keeps its band floor.
    """
    wins = [w for w in partners if w.get("box_xyxy") and isinstance(w.get("floor"), int)]
    if not wins or not floor_y:
        return partners, floor_i, []
    fh = statistics.median(float(y1 - y0) for y0, y1 in floor_y.values())
    top, bottom = float(box[1]), float(box[3])
    near = min(wins, key=lambda w: abs(top - float(w["box_xyxy"][3])))
    wb = float(near["box_xyxy"][3])
    if wb < top - WINDOW_ABOVE_MAX_FH * fh or wb > bottom + WINDOW_BELOW_MAX_FH * fh:
        return [], floor_i, [w.get("unit_id") for w in partners]
    return partners, int(near["floor"]), []


def outer_bay_center(box: list[float], meta: dict[str, Any], instances: list[dict[str, Any]]) -> tuple[str, float] | None:
    """``(side, center_px)`` of the bay between the window grid and the facade region's edge that holds ``box``.

    None when the box's center is on the grid, there is no region, or the margin is too thin to be a bay.
    """
    region, cols = meta.get("facade_region_xyxy"), meta.get("columns_xy")
    if not region or not cols:
        return None
    cx = 0.5 * (float(box[0]) + float(box[2]))
    if cx < float(cols[0][0]):
        side, lo, hi = "left", float(region[0]), float(cols[0][0])
    elif cx > float(cols[-1][1]):
        side, lo, hi = "right", float(cols[-1][1]), float(region[2])
    else:
        return None
    widths = [u["box_xyxy"][2] - u["box_xyxy"][0] for u in instances if u.get("kind", "window") == "window"]
    typical = statistics.median(widths) if widths else 0.0
    if hi - lo < OUTER_BAY_MIN_WINDOW_WIDTHS * typical:
        return None
    return side, 0.5 * (lo + hi)


def _open_center(rec: dict[str, Any]) -> float | None:
    for key in ("window_cx_norm", "bay_cx_norm"):
        if key in rec:
            return float(rec[key])
    return None


def _move_stacks_together(records: list[dict[str, Any]], image_width: float) -> None:
    """Open balconies of one type stacked over each other share one center (in place).

    The center is the median of the stack's window centers, so a floor whose window was not detected follows its
    stack; a stack with no window at all stays at its bay. A balcony left at its bay that would then overlap another
    one on its floor uses its own box. Balconies placed from their box (beside an enclosed volume) are left alone.
    """
    def x_span(rec: dict[str, Any]) -> tuple[float, float]:
        return float(rec["box_xyxy"][0]), float(rec["box_xyxy"][2])

    movable = [r for r in records if "width_norm" in r and "cx_norm" not in r and _open_center(r) is not None]
    seen: set[int] = set()
    for i, start in enumerate(movable):
        if i in seen:
            continue
        seen.add(i)
        stack, todo = [], [i]
        while todo:
            k = todo.pop()
            stack.append(movable[k])
            a0, a1 = x_span(movable[k])
            for j, other in enumerate(movable):
                if j in seen or other["type"] != movable[k]["type"]:
                    continue
                b0, b1 = x_span(other)
                # A long gallery of the same type is not stacked with the small balconies it spans.
                similar = min(a1 - a0, b1 - b0) >= STACK_MIN_WIDTH_RATIO * max(a1 - a0, b1 - b0)
                free_floor = all(r["floor"] != other["floor"] for r in stack) and all(
                    movable[q]["floor"] != other["floor"] for q in todo
                )
                if similar and free_floor and min(a1, b1) - max(a0, b0) >= 0.5 * min(a1 - a0, b1 - b0):
                    seen.add(j)
                    todo.append(j)
        windows = [float(r["window_cx_norm"]) for r in stack if "window_cx_norm" in r]
        key = "window_cx_norm" if windows else "bay_cx_norm"
        center = statistics.median(windows or [float(r["bay_cx_norm"]) for r in stack])
        for r in stack:
            r.pop("window_cx_norm", None)
            r.pop("bay_cx_norm", None)
            r[key] = round(center, 6)

    def span(rec: dict[str, Any]) -> tuple[float, float]:
        c, w = _open_center(rec) if "cx_norm" not in rec else float(rec["cx_norm"]), float(rec["width_norm"])
        return c - w / 2.0, c + w / 2.0

    for rec in movable:
        if "bay_cx_norm" not in rec or "outer_bay" in rec:
            continue
        lo, hi = span(rec)
        if any(
            o is not rec and o["floor"] == rec["floor"] and "width_norm" in o and min(hi, span(o)[1]) > max(lo, span(o)[0])
            for o in records
        ):
            del rec["bay_cx_norm"]
            rec["cx_norm"] = round(0.5 * (rec["box_xyxy"][0] + rec["box_xyxy"][2]) / max(1.0, image_width), 6)


def merge_balcony_into_windows_dsl(
    windows_dsl: dict[str, Any],
    *,
    balcony_types: list[dict[str, Any]],
    units: list[dict[str, Any]],
    image_size: tuple[int, int] | list[int] | None = None,
    per_unit_railing: bool = False,
    per_unit_floor: bool = False,
    center_mode: str = "window",
) -> dict[str, Any]:
    dsl = copy.deepcopy(windows_dsl)
    layout = dsl.setdefault("layout", {})
    meta = dsl.setdefault("meta", {})
    if image_size is None:
        image_size = meta.get("image_size")

    mode = _normalize_center_mode(center_mode)
    enclosed_units = [u for u in units if u.get("source") == "enclosed_volume"]
    # A balcony's window is the one behind it, on every facade. Side-wall windows are not behind anything; a balcony
    # sharing a bay with an enclosed volume uses its own box below, so it cannot drift into the volume.
    open_mode = mode
    instances = [u for u in windows_dsl.get("instances") or [] if not u.get("side_face")]
    floor_y, bay_x = _bands_from_windows_dsl(windows_dsl)
    iw = 1.0
    ih = 800.0
    if image_size and len(image_size) >= 1:
        iw = max(1.0, float(image_size[0]))
    if image_size and len(image_size) >= 2:
        ih = max(1.0, float(image_size[1]))

    balc_place = []
    seen: set[tuple] = set()
    for u in units:
        ir = u.get("structure_ir") or {}
        volume = u.get("source") == "enclosed_volume"
        if (not volume) and enclosed_units:
            kept, dropped = drop_open_under_enclosed([u], enclosed_units)
            if dropped:
                continue
        if volume:
            name = "balc_enclosed"
        elif per_unit_railing or per_unit_floor:
            name = f"balc_{balcony_type_token(ir, include_floor=per_unit_floor)}"
        else:
            name = f"balc_type_{int(u['type_id']):02d}"
        box = u.get("box_xyxy") or [0, 0, 1, 1]
        w_norm, cx_norm = _photo_norms(box, image_size)
        floor_i = int(u["floor"])
        bs = int(u["bay_start"])
        be = int(u["bay_end"])
        span_bays = _bay_ids(u)
        center_bays = _center_bay_ids(u)
        place_mode = mode if volume else open_mode
        outer = None if volume else outer_bay_center(box, meta, instances)
        if outer is not None:
            # Floors come from windows and a balcony hangs below its window, so its top edge names the floor.
            floor_i = floor_of_top_edge(box, floor_y, floor_i)
        # An open balcony that shares a bay with a block sits beside it: the bay centre would put it inside.
        if not volume and any(
            int(e["floor"]) == floor_i and set(_bay_ids(e)) & set(span_bays) for e in enclosed_units
        ):
            place_mode = "photo"

        partners: list[dict[str, Any]] = []
        pids: tuple = ()
        if volume:
            key = (floor_i, tuple(span_bays), name)
        elif outer is not None:
            key = (floor_i, outer[0], int(box[1]), name)
        elif place_mode == "photo":
            key = (floor_i, int(round(cx_norm * 1000)), name)
        elif place_mode == "bay":
            key = (floor_i, tuple(center_bays), name)
        else:
            partners = partner_windows_for_unit(
                u,
                instances,
                center_bays=center_bays,
                span_bays=span_bays,
                image_size=(iw, ih),
            )
            partners, floor_i, far_window_ids = attach_to_window(box, partners, floor_i, floor_y)
            pids = tuple(
                # Door partners carry ids like "door_0": window ids first, then doors.
                sorted(
                    (w["unit_id"] for w in partners if w.get("unit_id") is not None),
                    key=lambda v: (isinstance(v, str), v if isinstance(v, int) else 0, str(v)),
                )
            )
            # An unlinked balcony keys on the window it was paired with, as before, so it is not taken for a duplicate.
            # Window ids and bay indices are different things: tag them, or window 1 and bay 1 look like one balcony.
            if pids:
                anchor = ("window", pids)
            elif far_window_ids:
                anchor = ("window", tuple(far_window_ids))
            else:
                anchor = ("bay", tuple(center_bays))
            key = (floor_i, anchor, name)

        if key in seen:
            continue
        seen.add(key)

        rec: dict[str, Any] = {
            "floor": floor_i,
            "bay_start": bs,
            "bay_end": be,
            "bays": span_bays,
            "bays_center": center_bays,
            "bay": int(u.get("bay", u["bay_start"])),
            "type": name,
            "box_xyxy": [int(v) for v in box],
        }
        if not volume:
            rec["width_norm"] = round(w_norm, 6)
            if u.get("rooftop"):
                # Per placement, not per type: the balcony keeps its type.
                rec["structure"] = "rooftop"

        if volume:
            bay_cx = _mean_bay_center_norm(span_bays, bay_x, iw)
            if bay_cx is not None:
                rec["bay_cx_norm"] = round(bay_cx, 6)
        elif outer is not None:
            rec["outer_bay"] = outer[0]
            rec["bay_cx_norm"] = round(outer[1] / iw, 6)
        elif place_mode == "photo":
            rec["cx_norm"] = round(cx_norm, 6)
        elif place_mode == "bay":
            bay_cx = _mean_bay_center_norm(center_bays, bay_x, iw)
            if bay_cx is not None:
                rec["bay_cx_norm"] = round(bay_cx, 6)
        else:
            win_cx = mean_window_center_norm(partners, iw)
            if win_cx is not None:
                rec["window_cx_norm"] = round(win_cx, 6)
            else:  # no window detected behind it: it stays at its bay
                bay_cx = _mean_bay_center_norm(center_bays, bay_x, iw)
                if bay_cx is not None:
                    rec["bay_cx_norm"] = round(bay_cx, 6)
            if pids:
                rec["partner_window_unit_ids"] = list(pids)
            if far_window_ids:
                rec["far_window_unit_ids"] = far_window_ids

        balc_place.append(rec)

    if mode == "window":
        _move_stacks_together(balc_place, iw)
    layout["balconies"] = balc_place
    dsl["balcony_types"] = balcony_types
    meta["n_balcony_units"] = len(units)
    meta["n_balcony_types"] = len(balcony_types)
    meta["balcony_center_mode"] = mode
    notes = str(meta.get("notes") or "")
    if per_unit_railing and per_unit_floor:
        mode_label = "per-unit railing and floor IR"
    elif per_unit_floor:
        mode_label = "voted railing, per-unit floor IR"
    elif per_unit_railing:
        mode_label = "per-unit railing IR"
    else:
        mode_label = "heuristic IR vote"
    note_mode = open_mode if enclosed_units else mode
    if note_mode == "photo":
        placement_note = "slab width/center from photo boxes (width_norm/cx_norm)."
    elif note_mode == "bay":
        placement_note = (
            "slab width from photo (width_norm); horizontal center = mean X of "
            "bays_center (bay_cx_norm)."
        )
        if enclosed_units and mode == "window":
            placement_note += " Open slabs use bay center when enclosed volumes are present."
    else:
        placement_note = (
            "slab width from photo (width_norm); horizontal center = mean X of "
            "paired window boxes on same floor+bay (window_cx_norm)."
        )
    extra = f" Balcony types appended by balcony_pipeline ({mode_label}); {placement_note}"
    if "Balcony types appended by balcony_pipeline" not in notes:
        meta["notes"] = (notes + extra).strip()
    # floor / bay order lists for editing (vendor/window_compiler/grid_order.py); window DSLs from before them get them here
    layout.setdefault("floor_order", [int(f["id"]) for f in layout.get("floors") or []])
    layout.setdefault("bay_order", [int(b["id"]) for b in layout.get("bays") or []])
    return dsl
