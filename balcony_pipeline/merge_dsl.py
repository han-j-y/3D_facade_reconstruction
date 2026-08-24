"""Merge balcony types into a copy of the window recovery DSL (do not edit build_facade_dsl)."""

from __future__ import annotations

import copy
from typing import Any


def merge_balcony_into_windows_dsl(
    windows_dsl: dict[str, Any],
    *,
    balcony_types: list[dict[str, Any]],
    units: list[dict[str, Any]],
) -> dict[str, Any]:
    dsl = copy.deepcopy(windows_dsl)
    layout = dsl.setdefault("layout", {})
    balc_place = []
    seen: set[tuple[int, int, int, str]] = set()
    for u in units:
        name = f"balc_type_{int(u['type_id']):02d}"
        key = (int(u["floor"]), int(u["bay_start"]), int(u["bay_end"]), name)
        if key in seen:
            continue
        seen.add(key)
        balc_place.append(
            {
                "floor": int(u["floor"]),
                "bay_start": int(u["bay_start"]),
                "bay_end": int(u["bay_end"]),
                "type": name,
            }
        )
    layout["balconies"] = balc_place
    dsl["balcony_types"] = balcony_types
    meta = dsl.setdefault("meta", {})
    meta["n_balcony_units"] = len(units)
    meta["n_balcony_types"] = len(balcony_types)
    notes = str(meta.get("notes") or "")
    extra = " Balcony types appended by balcony_pipeline (heuristic IR vote)."
    if extra.strip() not in notes:
        meta["notes"] = (notes + extra).strip()
    return dsl
