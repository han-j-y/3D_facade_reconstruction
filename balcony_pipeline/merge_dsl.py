"""Merge balcony types into a copy of the window recovery DSL (do not edit build_facade_dsl)."""

from __future__ import annotations

import copy
from typing import Any


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


def merge_balcony_into_windows_dsl(
    windows_dsl: dict[str, Any],
    *,
    balcony_types: list[dict[str, Any]],
    units: list[dict[str, Any]],
    image_size: tuple[int, int] | list[int] | None = None,
) -> dict[str, Any]:
    dsl = copy.deepcopy(windows_dsl)
    layout = dsl.setdefault("layout", {})
    meta = dsl.setdefault("meta", {})
    if image_size is None:
        image_size = meta.get("image_size")

    balc_place = []
    # One placement per unit; photo box width/center drive mesh size.
    seen: set[tuple[int, int, str]] = set()
    for u in units:
        name = f"balc_type_{int(u['type_id']):02d}"
        box = u.get("box_xyxy") or [0, 0, 1, 1]
        w_norm, cx_norm = _photo_norms(box, image_size)
        # Dedup near-identical centers on the same floor+type (px quantization).
        cx_q = int(round(cx_norm * 1000))
        key = (int(u["floor"]), cx_q, name)
        if key in seen:
            continue
        seen.add(key)
        balc_place.append(
            {
                "floor": int(u["floor"]),
                "bay_start": int(u["bay_start"]),
                "bay_end": int(u["bay_end"]),
                "bay": int(u.get("bay", u["bay_start"])),
                "type": name,
                "width_norm": round(w_norm, 6),
                "cx_norm": round(cx_norm, 6),
                "box_xyxy": [int(v) for v in box],
            }
        )
    layout["balconies"] = balc_place
    dsl["balcony_types"] = balcony_types
    meta["n_balcony_units"] = len(units)
    meta["n_balcony_types"] = len(balcony_types)
    notes = str(meta.get("notes") or "")
    extra = (
        " Balcony types appended by balcony_pipeline (heuristic IR vote); "
        "slab width/center from photo boxes (width_norm/cx_norm)."
    )
    if "width_norm/cx_norm" not in notes:
        meta["notes"] = (notes + extra).strip()
    return dsl
