"""Merge balcony types into a copy of the window recovery DSL (do not edit build_facade_dsl)."""

from __future__ import annotations

import copy
from typing import Any

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


def merge_balcony_into_windows_dsl(
    windows_dsl: dict[str, Any],
    *,
    balcony_types: list[dict[str, Any]],
    units: list[dict[str, Any]],
    image_size: tuple[int, int] | list[int] | None = None,
    per_unit_railing: bool = False,
    center_mode: str = "window",
) -> dict[str, Any]:
    dsl = copy.deepcopy(windows_dsl)
    layout = dsl.setdefault("layout", {})
    meta = dsl.setdefault("meta", {})
    if image_size is None:
        image_size = meta.get("image_size")

    mode = _normalize_center_mode(center_mode)
    instances = windows_dsl.get("instances") or []
    _, bay_x = _bands_from_windows_dsl(windows_dsl)
    iw = 1.0
    ih = 800.0
    if image_size and len(image_size) >= 1:
        iw = max(1.0, float(image_size[0]))
    if image_size and len(image_size) >= 2:
        ih = max(1.0, float(image_size[1]))

    balc_place = []
    seen: set[tuple] = set()
    for u in units:
        if per_unit_railing:
            ir = u.get("structure_ir") or {}
            name = f"balc_{balcony_type_token(ir)}"
        else:
            name = f"balc_type_{int(u['type_id']):02d}"
        box = u.get("box_xyxy") or [0, 0, 1, 1]
        w_norm, cx_norm = _photo_norms(box, image_size)
        floor_i = int(u["floor"])
        bs = int(u["bay_start"])
        be = int(u["bay_end"])
        span_bays = _bay_ids(u)
        center_bays = _center_bay_ids(u)

        if mode == "photo":
            key = (floor_i, int(round(cx_norm * 1000)), name)
        elif mode == "bay":
            key = (floor_i, tuple(center_bays), name)
        else:
            partners = partner_windows_for_unit(
                u,
                instances,
                center_bays=center_bays,
                span_bays=span_bays,
                image_size=(iw, ih),
            )
            pids = tuple(
                sorted(int(w["unit_id"]) for w in partners if w.get("unit_id") is not None)
            )
            key = (floor_i, pids or tuple(center_bays), name)

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
            "width_norm": round(w_norm, 6),
            "box_xyxy": [int(v) for v in box],
        }

        if mode == "photo":
            rec["cx_norm"] = round(cx_norm, 6)
        elif mode == "bay":
            bay_cx = _mean_bay_center_norm(center_bays, bay_x, iw)
            if bay_cx is not None:
                rec["bay_cx_norm"] = round(bay_cx, 6)
        else:
            win_cx = mean_window_center_norm(partners, iw)
            if win_cx is None:
                win_cx = _mean_bay_center_norm(center_bays, bay_x, iw)
            if win_cx is not None:
                rec["window_cx_norm"] = round(win_cx, 6)
            if pids:
                rec["partner_window_unit_ids"] = list(pids)

        balc_place.append(rec)

    layout["balconies"] = balc_place
    dsl["balcony_types"] = balcony_types
    meta["n_balcony_units"] = len(units)
    meta["n_balcony_types"] = len(balcony_types)
    meta["balcony_center_mode"] = mode
    notes = str(meta.get("notes") or "")
    mode_label = "per-unit railing IR" if per_unit_railing else "heuristic IR vote"
    if mode == "photo":
        placement_note = "slab width/center from photo boxes (width_norm/cx_norm)."
    elif mode == "bay":
        placement_note = (
            "slab width from photo (width_norm); horizontal center = mean X of "
            "bays_center (bay_cx_norm)."
        )
    else:
        placement_note = (
            "slab width from photo (width_norm); horizontal center = mean X of "
            "paired window boxes on same floor+bay (window_cx_norm)."
        )
    extra = f" Balcony types appended by balcony_pipeline ({mode_label}); {placement_note}"
    if "Balcony types appended by balcony_pipeline" not in notes:
        meta["notes"] = (notes + extra).strip()
    return dsl
