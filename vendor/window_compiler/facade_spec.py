"""Façade document schema helpers (no Blender dependency).

See ``facade_compile.py`` for scene compilation.
"""

from __future__ import annotations

import copy
from typing import Any

EMPTY_TOKENS = frozenset({None, "", "empty", "spacer", "s", "null"})


def is_facade_spec(spec: dict[str, Any]) -> bool:
    if spec.get("type") == "facade":
        return True
    if spec.get("schema") == "facade_recovery_dsl_v1":
        return True
    if "layout" in spec and "window_types" in spec:
        return True
    return False


def is_window_spec(spec: dict[str, Any]) -> bool:
    return spec.get("type") == "window" or (
        "boundary" in spec and "layout" not in spec and not is_facade_spec(spec)
    )


def unwrap_window_ir(obj: Any) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    if obj.get("type") == "window" or "boundary" in obj:
        return obj
    inner = obj.get("ir")
    if isinstance(inner, dict) and (inner.get("type") == "window" or "boundary" in inner):
        return inner
    return None


def normalize_facade_spec(
    spec: dict[str, Any],
    *,
    storey_height: float = 3.0,
    facade_width: float | None = None,
) -> dict[str, Any]:
    """Map recovery DSL (or already-normalized façade) → window_compiler_facade_v1."""
    if spec.get("type") == "facade" and "grid" in spec and "windows" in spec:
        out = copy.deepcopy(spec)
        out.setdefault("schema", "window_compiler_facade_v1")
        return out

    layout = spec.get("layout") or {}
    floors = list(layout.get("floors") or [])
    bays = list(layout.get("bays") or [])
    placement_src = layout.get("placement") or []
    if not floors or not bays:
        raise ValueError("facade recovery layout missing floors/bays")

    h_norms = [max(0.05, float(f.get("h_norm", 1.0))) for f in floors]
    w_norms = [max(0.05, float(b.get("w_norm", 1.0))) for b in bays]
    h_sum, w_sum = sum(h_norms), sum(w_norms)
    n_floors, n_bays = len(floors), len(bays)
    total_h = storey_height * n_floors
    total_w = facade_width if facade_width is not None else max(6.0, 2.4 * n_bays)

    rows = [
        {
            "name": str(floors[i].get("name", f"F{i}")),
            "h": round(total_h * (h_norms[i] / h_sum), 4),
        }
        for i in range(n_floors)
    ]
    cols = [
        {
            "name": str(bays[j].get("name", f"B{j}")),
            "w": round(total_w * (w_norms[j] / w_sum), 4),
        }
        for j in range(n_bays)
    ]

    windows: dict[str, Any] = {}
    for wt in spec.get("window_types") or []:
        name = str(wt.get("name") or f"win_type_{int(wt['type_id']):02d}")
        ir = unwrap_window_ir(wt.get("structure_ir"))
        if ir is None:
            continue
        ir = copy.deepcopy(ir)
        ir["type"] = "window"
        ir["debug"] = False
        windows[name] = ir

    placement: list[list[str | None]] = []
    for r in range(n_floors):
        src = placement_src[r] if r < len(placement_src) else []
        row: list[str | None] = []
        for c in range(n_bays):
            tok = src[c] if c < len(src) else None
            if tok in EMPTY_TOKENS:
                row.append(None)
            else:
                row.append(str(tok) if tok in windows else None)
        placement.append(row)

    type_ratios: dict[str, dict[str, float]] = {}
    iw, ih = (spec.get("meta") or {}).get("image_size") or [1, 1]
    floor_h_px = {
        int(f["id"]): max(1.0, float(f["h_norm"]) * float(ih) / 0.55) for f in floors
    }
    bay_w_px = {
        int(b["id"]): max(1.0, float(b["w_norm"]) * float(iw) / 0.55) for b in bays
    }
    acc: dict[str, list[tuple[float, float]]] = {}
    for inst in spec.get("instances") or []:
        name = f"win_type_{int(inst['type_id']):02d}"
        x0, y0, x1, y1 = inst["box_xyxy"]
        bw, bh = max(1.0, float(x1 - x0)), max(1.0, float(y1 - y0))
        cw = bay_w_px.get(int(inst["bay"]), bw / 0.5)
        ch = floor_h_px.get(int(inst["floor"]), bh / 0.55)
        wr = min(0.85, max(0.28, bw / cw))
        hr = min(0.85, max(0.35, bh / ch))
        acc.setdefault(name, []).append((wr, hr))
    for name, pairs in acc.items():
        type_ratios[name] = {
            "width_ratio": float(sum(p[0] for p in pairs) / len(pairs)),
            "height_ratio": float(sum(p[1] for p in pairs) / len(pairs)),
        }

    return {
        "type": "facade",
        "schema": "window_compiler_facade_v1",
        "meta": {
            "source_schema": spec.get("schema"),
            "facade_id": (spec.get("meta") or {}).get("facade_id"),
            "source_image": (spec.get("meta") or {}).get("image"),
            "type_ratios": type_ratios,
        },
        "wall": {
            "depth": 0.42,
            "base_front_y": 0.0,
            "material": "wall",
        },
        "grid": {"rows": rows, "cols": cols},
        "placement": placement,
        "windows": windows,
        "placement_params": {
            "width_ratio": 0.55,
            "height_ratio": 0.60,
            "bottom_margin_ratio": 0.14,
            "recess": 0.01,
            "mirror_x": True,
        },
    }


def total_grid_size(grid: dict[str, Any]) -> tuple[float, float]:
    return (
        float(sum(c["w"] for c in grid["cols"])),
        float(sum(r["h"] for r in grid["rows"])),
    )


def get_cell(
    spec: dict[str, Any],
    row_idx: int,
    col_idx: int,
    *,
    mirror_x: bool = True,
) -> dict[str, Any]:
    """Cell bounds: X centered, Z from top band downward (dsl7-compatible).

    ``mirror_x`` (default True): negate X so photo-left columns appear on the
    *left* of the compiler camera view. The façade camera sits on +Y looking −Y,
    which mirrors world +X onto the image left (see window_ast.structure_lines).
    """
    grid = spec["grid"]
    rows, cols = grid["rows"], grid["cols"]
    total_w, total_h = total_grid_size(grid)
    wall = spec.get("wall") or {}
    front_y = float(wall.get("base_front_y", 0.0))

    x0 = -total_w / 2.0
    for c in range(col_idx):
        x0 += float(cols[c]["w"])
    x1 = x0 + float(cols[col_idx]["w"])

    if mirror_x:
        x0, x1 = -x1, -x0

    z1 = total_h
    for r in range(row_idx):
        z1 -= float(rows[r]["h"])
    z0 = z1 - float(rows[row_idx]["h"])

    return {
        "row": row_idx,
        "col": col_idx,
        "row_name": rows[row_idx]["name"],
        "col_name": cols[col_idx]["name"],
        "x0": x0,
        "x1": x1,
        "z0": z0,
        "z1": z1,
        "cx": 0.5 * (x0 + x1),
        "cz": 0.5 * (z0 + z1),
        "w": x1 - x0,
        "h": z1 - z0,
        "front_y": front_y,
        "mirror_x": mirror_x,
    }


def fit_window_ir_to_cell(
    ir: dict[str, Any],
    cell: dict[str, Any],
    *,
    width_ratio: float,
    height_ratio: float,
) -> dict[str, Any]:
    out = copy.deepcopy(ir)
    out["type"] = "window"
    out["debug"] = False
    boundary = out.setdefault("boundary", {"id": "root", "shape": "rectangle"})
    shape = boundary.get("shape", "rectangle")
    params = boundary.setdefault("params", {})
    tw = max(0.2, float(cell["w"]) * float(width_ratio))
    th = max(0.2, float(cell["h"]) * float(height_ratio))
    if shape == "rectangle":
        params["width"] = round(tw, 4)
        params["height"] = round(th, 4)
    elif shape == "circle":
        params["diameter"] = round(min(tw, th), 4)
    else:
        params["width"] = round(tw, 4)
        params["height"] = round(th, 4)
    return out
