"""Façade document schema helpers (no Blender dependency).

See ``facade_compile.py`` for scene compilation.
"""

from __future__ import annotations

import copy
from typing import Any

EMPTY_TOKENS = frozenset({None, "", "empty", "spacer", "s", "null"})


def door_type_token(type_id: int) -> str:
    return f"d{int(type_id):02d}"


def is_door_type_token(token: str) -> bool:
    s = str(token).strip().lower()
    return len(s) >= 2 and s.startswith("d") and s[1:].isdigit()


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
    meta_early = spec.get("meta") or {}
    iw_meta, ih_meta = meta_early.get("image_size") or [1, 1]
    image_aspect = max(0.25, float(iw_meta) / max(1.0, float(ih_meta)))
    content_aspect = image_aspect
    columns_xy_meta = meta_early.get("columns_xy")
    floors_y_meta = meta_early.get("floors_y")
    if columns_xy_meta and floors_y_meta and len(columns_xy_meta) >= 1 and len(floors_y_meta) >= 1:
        cw = float(columns_xy_meta[-1][1]) - float(columns_xy_meta[0][0])
        ch = float(floors_y_meta[-1][1]) - float(floors_y_meta[0][0])
        if cw > 1.0 and ch > 1.0:
            content_aspect = max(0.25, cw / ch)
    aspect = content_aspect
    total_w = facade_width if facade_width is not None else total_h * aspect

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

    # Known type names come from the recovery library, independent of IR unwrap.
    # Previously placement was gated on successful unwrap (``tok in windows``), which
    # silently wiped the whole floor×bay map whenever structure_ir was missing.
    type_names: dict[str, str] = {}
    windows: dict[str, Any] = {}
    for wt in spec.get("window_types") or []:
        name = str(wt.get("name") or f"win_type_{int(wt['type_id']):02d}")
        type_names[name] = name
        if wt.get("type_id") is not None:
            type_names[str(int(wt["type_id"]))] = name
            type_names[f"win_type_{int(wt['type_id']):02d}"] = name
        ir = unwrap_window_ir(wt.get("structure_ir"))
        if ir is None:
            continue
        ir = copy.deepcopy(ir)
        ir["type"] = "window"
        ir["debug"] = False
        windows[name] = ir

    doors: dict[str, dict[str, Any]] = {}
    for dt in spec.get("door_types") or []:
        tid = int(dt.get("type_id", 0))
        name = str(dt.get("name") or door_type_token(tid))
        type_names[name] = name
        type_names[door_type_token(tid)] = name
        doors[name] = {
            "type_id": tid,
            "kind": "door",
            "shape": str(dt.get("shape") or dt.get("primitive") or "rectangle"),
            "primitive": str(dt.get("primitive") or dt.get("shape") or "rectangle"),
            "arch_rise_ratio": float(dt.get("arch_rise_ratio") or 0.0),
            "material": str(dt.get("material") or "door_wood"),
            "placeholder": True,
        }

    placement: list[list[str | None]] = []
    dropped = 0
    missing_ir = 0
    placement_src = layout.get("placement") or []
    placement_spans_src = layout.get("placement_spans") or []
    for r in range(n_floors):
        src = placement_src[r] if r < len(placement_src) else []
        row: list[str | None] = []
        for c in range(n_bays):
            tok = src[c] if c < len(src) else None
            if tok in EMPTY_TOKENS:
                row.append(None)
                continue
            key = str(tok)
            name = type_names.get(key)
            if name is None:
                dropped += 1
                row.append(None)
                continue
            if name not in windows and name not in doors:
                missing_ir += 1
            row.append(name)
        placement.append(row)
    placement_spans: list[list[int]] = []
    for r in range(n_floors):
        span_src = placement_spans_src[r] if r < len(placement_spans_src) else []
        span_row: list[int] = []
        for c in range(n_bays):
            if c < len(span_src) and span_src[c] is not None:
                span_row.append(max(0, int(span_src[c])))
            else:
                span_row.append(0 if (placement[r][c] is None) else 1)
        placement_spans.append(span_row)
    if dropped:
        print(
            f"warn: normalize_facade_spec dropped {dropped} placement token(s) "
            f"with unknown type names"
        )
    if missing_ir:
        print(
            f"warn: normalize_facade_spec kept {missing_ir} placement cell(s) "
            f"whose window type has no structure_ir (Blender will skip them)"
        )

    meta = spec.get("meta") or {}
    iw, ih = meta.get("image_size") or [1, 1]
    columns_xy = meta.get("columns_xy")
    floor_id_to_row = {int(f["id"]): i for i, f in enumerate(floors)}
    bay_id_to_col = {int(b["id"]): j for j, b in enumerate(bays)}

    if columns_xy and len(columns_xy) == n_bays:
        bay_w_px = {
            int(bays[j]["id"]): max(1.0, float(columns_xy[j][1]) - float(columns_xy[j][0]))
            for j in range(n_bays)
        }
    else:
        bay_w_px = {
            int(b["id"]): max(1.0, float(b["w_norm"]) * float(iw)) for b in bays
        }
    floor_h_px = {
        int(f["id"]): max(1.0, float(f["h_norm"]) * float(ih)) for f in floors
    }

    type_ratios: dict[str, dict[str, float]] = {}
    placement_fit: list[list[dict[str, float] | None]] = [
        [None for _ in range(n_bays)] for _ in range(n_floors)
    ]
    acc: dict[str, list[tuple[float, float]]] = {}

    for inst in spec.get("instances") or []:
        if inst.get("kind") == "door":
            name = door_type_token(int(inst["type_id"]))
        else:
            name = f"win_type_{int(inst['type_id']):02d}"
        x0, y0, x1, y1 = inst["box_xyxy"]
        bw, bh = max(1.0, float(x1 - x0)), max(1.0, float(y1 - y0))
        span = max(1, int(inst.get("colspan", 1)))
        bay0 = int(inst["bay"])
        floor_id = int(inst["floor"])
        r = floor_id_to_row.get(floor_id)
        c = bay_id_to_col.get(bay0)
        if r is None or c is None or c < 0 or c >= n_bays:
            continue

        cw = 0.0
        col_left: float | None = None
        col_right: float | None = None
        if columns_xy and len(columns_xy) == n_bays:
            end = min(n_bays, bay0 + span)
            col_left = float(columns_xy[bay0][0])
            col_right = float(columns_xy[end - 1][1])
            cw = max(1.0, col_right - col_left)
        else:
            for j in range(span):
                cw += bay_w_px.get(bay0 + j, bw)
            cw = max(1.0, cw)

        ch = floor_h_px.get(floor_id, bh)
        row_top: float | None = None
        row_bottom: float | None = None
        floors_y = meta.get("floors_y")
        if floors_y and r < len(floors_y):
            row_top = float(floors_y[r][0])
            row_bottom = float(floors_y[r][1])
            ch = max(1.0, row_bottom - row_top)

        wr = float(bw / max(1.0, cw))
        hr = float(bh / max(1.0, ch))
        wr = min(0.98, max(0.05, wr))
        hr = min(0.98, max(0.05, hr))
        box_cx = 0.5 * (float(x0) + float(x1))
        box_cy = 0.5 * (float(y0) + float(y1))
        if col_left is not None and col_right is not None:
            cx_ratio = (box_cx - col_left) / max(1.0, col_right - col_left)
        else:
            cx_ratio = 0.5
        cx_ratio = min(0.98, max(0.02, float(cx_ratio)))
        if row_top is not None and row_bottom is not None:
            cy_ratio = (box_cy - row_top) / max(1.0, row_bottom - row_top)
        else:
            cy_ratio = (box_cy - float(y0)) / max(1.0, bh)
        cy_ratio = min(0.98, max(0.02, float(cy_ratio)))

        acc.setdefault(name, []).append((wr, hr))
        placement_fit[r][c] = {
            "width_ratio": wr,
            "height_ratio": hr,
            "cx_ratio": cx_ratio,
            "cy_ratio": cy_ratio,
            "box_aspect": round(bw / max(1.0, bh), 4),
        }

    for name, pairs in acc.items():
        type_ratios[name] = {
            "width_ratio": float(sum(p[0] for p in pairs) / len(pairs)),
            "height_ratio": float(sum(p[1] for p in pairs) / len(pairs)),
        }

    image_extent: dict[str, float] | None = None
    if columns_xy and len(columns_xy) >= 1:
        image_extent = {
            "x0": float(columns_xy[0][0]),
            "x1": float(columns_xy[-1][1]),
        }
    floors_y = meta.get("floors_y")
    if floors_y and len(floors_y) >= 1:
        if image_extent is None:
            image_extent = {}
        image_extent["y0"] = float(floors_y[0][0])
        image_extent["y1"] = float(floors_y[-1][1])

    floor_id_to_row = {int(floors[i]["id"]): i for i in range(n_floors)}
    bay_id_to_col = {int(bays[j]["id"]): j for j in range(n_bays)}
    balcony_lib: dict[str, Any] = {}
    for bt in spec.get("balcony_types") or []:
        bname = str(bt.get("name") or f"balc_type_{int(bt.get('type_id', 0)):02d}")
        bir = bt.get("structure_ir")
        if isinstance(bir, dict):
            balcony_lib[bname] = copy.deepcopy(bir)
    balcony_placement: list[dict[str, Any]] = []
    for rec in (spec.get("layout") or {}).get("balconies") or []:
        bname = str(rec.get("type") or "")
        if bname not in balcony_lib:
            continue
        try:
            rr = floor_id_to_row[int(rec["floor"])]
            cc0 = bay_id_to_col[int(rec["bay_start"])]
            cc1 = bay_id_to_col[int(rec["bay_end"])]
        except (KeyError, TypeError, ValueError):
            continue
        if cc1 < cc0:
            cc0, cc1 = cc1, cc0
        balcony_placement.append(
            {
                "row": rr,
                "col0": cc0,
                "col1": cc1,
                "type": bname,
                **(
                    {"width_norm": float(rec["width_norm"])}
                    if rec.get("width_norm") is not None
                    else {}
                ),
                **(
                    {"cx_norm": float(rec["cx_norm"])}
                    if rec.get("cx_norm") is not None
                    else {}
                ),
                **(
                    {"window_cx_norm": float(rec["window_cx_norm"])}
                    if rec.get("window_cx_norm") is not None
                    else {}
                ),
                **(
                    {"bay_cx_norm": float(rec["bay_cx_norm"])}
                    if rec.get("bay_cx_norm") is not None
                    else {}
                ),
                **(
                    {"bays": [int(b) for b in rec["bays"]]}
                    if rec.get("bays")
                    else {}
                ),
                **(
                    {"bays_center": [int(b) for b in rec["bays_center"]]}
                    if rec.get("bays_center")
                    else {}
                ),
            }
        )

    return {
        "type": "facade",
        "schema": "window_compiler_facade_v1",
        "meta": {
            "source_schema": spec.get("schema"),
            "facade_id": meta.get("facade_id"),
            "source_image": meta.get("image"),
            "image_size": [iw, ih],
            "type_ratios": type_ratios,
            "inter_bay_gap_norm": meta.get("inter_bay_gap_norm"),
            "inter_floor_gap_norm": meta.get("inter_floor_gap_norm"),
            "columns_xy": meta.get("columns_xy"),
            "floors_y": meta.get("floors_y"),
        },
        "wall": {
            "depth": 0.42,
            "base_front_y": 0.0,
            "material": "wall",
        },
        "grid": {"rows": rows, "cols": cols},
        "placement": placement,
        "placement_spans": placement_spans,
        "placement_fit": placement_fit,
        "windows": windows,
        "doors": doors,
        "balconies": balcony_lib,
        "balcony_placement": balcony_placement,
        "instances": list(spec.get("instances") or []),
        "floor_ids": [int(f["id"]) for f in floors],
        "image_extent": image_extent,
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


def image_facade_extent_x(spec: dict[str, Any]) -> tuple[float, float]:
    """Horizontal extent of the façade in image pixels (for proportional layout)."""
    ext = spec.get("image_extent")
    if isinstance(ext, dict) and ext.get("x0") is not None and ext.get("x1") is not None:
        return float(ext["x0"]), float(ext["x1"])
    meta = spec.get("meta") or {}
    cols = meta.get("columns_xy")
    if cols and len(cols) >= 1:
        return float(cols[0][0]), float(cols[-1][1])
    xs: list[float] = []
    for inst in spec.get("instances") or []:
        box = inst.get("box_xyxy") or []
        if len(box) >= 4:
            xs.extend([float(box[0]), float(box[2])])
    if xs:
        return min(xs), max(xs)
    return 0.0, float((meta.get("image_size") or [1, 1])[0])


def box_xyxy_to_world_x(
    box_xyxy: list[float],
    spec: dict[str, Any],
    total_w: float,
    *,
    mirror_x: bool = True,
) -> tuple[float, float]:
    """Map image box horizontal edges to world X (proportional to façade width)."""
    xl, xr = image_facade_extent_x(spec)
    span = max(1.0, xr - xl)
    u0 = (float(box_xyxy[0]) - xl) / span
    u1 = (float(box_xyxy[2]) - xl) / span
    u0 = min(1.0, max(0.0, u0))
    u1 = min(1.0, max(0.0, u1))
    if u1 < u0:
        u0, u1 = u1, u0
    if mirror_x:
        return total_w * (0.5 - u1), total_w * (0.5 - u0)
    return -total_w / 2.0 + u0 * total_w, -total_w / 2.0 + u1 * total_w


def image_facade_extent_y(spec: dict[str, Any]) -> tuple[float, float]:
    """Vertical extent of the façade in image pixels (top y, bottom y)."""
    ext = spec.get("image_extent")
    if isinstance(ext, dict) and ext.get("y0") is not None and ext.get("y1") is not None:
        return float(ext["y0"]), float(ext["y1"])
    meta = spec.get("meta") or {}
    rows_y = meta.get("floors_y")
    if rows_y and len(rows_y) >= 1:
        return float(rows_y[0][0]), float(rows_y[-1][1])
    ys: list[float] = []
    for inst in spec.get("instances") or []:
        box = inst.get("box_xyxy") or []
        if len(box) >= 4:
            ys.extend([float(box[1]), float(box[3])])
    if ys:
        return min(ys), max(ys)
    return 0.0, float((meta.get("image_size") or [1, 1])[1])


def box_xyxy_to_world_z(
    box_xyxy: list[float],
    spec: dict[str, Any],
    total_h: float,
) -> tuple[float, float]:
    """Map image box vertical edges to world Z (image top = high Z)."""
    yt, yb = image_facade_extent_y(spec)
    span = max(1.0, yb - yt)
    y0, y1 = float(box_xyxy[1]), float(box_xyxy[3])
    y0 = min(y0, y1)
    y1 = max(y0, y1)
    u_top = (y0 - yt) / span
    u_bot = (y1 - yt) / span
    u_top = min(1.0, max(0.0, u_top))
    u_bot = min(1.0, max(0.0, u_bot))
    z_top = total_h * (1.0 - u_top)
    z_bot = total_h * (1.0 - u_bot)
    return z_bot, z_top


def build_instance_map(spec: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    """Map (floor_id, start_bay) → instance for image-anchored placement."""
    floor_ids = spec.get("floor_ids") or []
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for inst in spec.get("instances") or []:
        floor_id = int(inst["floor"])
        bay = int(inst["bay"])
        out[(floor_id, bay)] = inst
    if not floor_ids and out:
        floor_ids = sorted({k[0] for k in out})
    return out


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


def get_cell_span(
    spec: dict[str, Any],
    row_idx: int,
    col_start: int,
    col_end: int,
    *,
    mirror_x: bool = True,
) -> dict[str, Any]:
    """Merged cell covering columns [col_start, col_end) on one floor row."""
    grid = spec["grid"]
    rows, cols = grid["rows"], grid["cols"]
    total_w, total_h = total_grid_size(grid)
    wall = spec.get("wall") or {}
    front_y = float(wall.get("base_front_y", 0.0))
    col_end = min(col_end, len(cols))

    x0 = -total_w / 2.0
    for c in range(col_start):
        x0 += float(cols[c]["w"])
    x1 = x0
    for c in range(col_start, col_end):
        x1 += float(cols[c]["w"])

    if mirror_x:
        x0, x1 = -x1, -x0

    z1 = total_h
    for r in range(row_idx):
        z1 -= float(rows[r]["h"])
    z0 = z1 - float(rows[row_idx]["h"])

    return {
        "row": row_idx,
        "col": col_start,
        "col_end": col_end,
        "row_name": rows[row_idx]["name"],
        "col_name": cols[col_start]["name"],
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


def segmentation_box_cell_fit(
    box_xyxy: list[float],
    *,
    col_left: float,
    col_right: float,
    row_top: float,
    row_bottom: float,
) -> dict[str, float]:
    """Relative window geometry inside a floor×bay cell (from segmentation box)."""
    x0, y0, x1, y1 = (
        float(box_xyxy[0]),
        float(box_xyxy[1]),
        float(box_xyxy[2]),
        float(box_xyxy[3]),
    )
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    cw = max(1.0, col_right - col_left)
    ch = max(1.0, row_bottom - row_top)
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    wr = min(0.98, max(0.05, bw / cw))
    hr = min(0.98, max(0.05, bh / ch))
    cx_ratio = min(0.98, max(0.02, (cx - col_left) / cw))
    cy_ratio = min(0.98, max(0.02, (cy - row_top) / ch))
    return {
        "width_ratio": wr,
        "height_ratio": hr,
        "cx_ratio": cx_ratio,
        "cy_ratio": cy_ratio,
        "box_aspect": bw / bh,
    }


def image_cell_bounds_px(
    spec: dict[str, Any],
    floor_id: int,
    bay: int,
    *,
    colspan: int = 1,
) -> tuple[float, float, float, float] | None:
    """Image pixel bounds (x_left, y_top, x_right, y_bottom) for floor×bay cell."""
    meta = spec.get("meta") or {}
    columns_xy = meta.get("columns_xy")
    floors_y = meta.get("floors_y")
    if not columns_xy or not floors_y:
        return None
    floor_ids = spec.get("floor_ids") or []
    try:
        row_idx = floor_ids.index(int(floor_id))
    except ValueError:
        return None
    if row_idx >= len(floors_y):
        return None
    bay = int(bay)
    end = min(len(columns_xy), bay + max(1, int(colspan)))
    if bay < 0 or bay >= len(columns_xy):
        return None
    col_left = float(columns_xy[bay][0])
    col_right = float(columns_xy[end - 1][1])
    row_top = float(floors_y[row_idx][0])
    row_bottom = float(floors_y[row_idx][1])
    return col_left, row_top, col_right, row_bottom


def placement_fit_for_instance(
    spec: dict[str, Any],
    inst: dict[str, Any],
    row_idx: int,
    col_idx: int,
) -> dict[str, float] | None:
    """Cell-relative fit from placement_fit table or segmentation box."""
    placement_fit = spec.get("placement_fit") or []
    if row_idx < len(placement_fit):
        fit_row = placement_fit[row_idx]
        if col_idx < len(fit_row) and fit_row[col_idx]:
            return dict(fit_row[col_idx])
    box = inst.get("box_xyxy")
    if not box or len(box) < 4:
        return None
    bounds = image_cell_bounds_px(
        spec,
        int(inst["floor"]),
        int(inst["bay"]),
        colspan=max(1, int(inst.get("colspan", 1))),
    )
    if bounds is None:
        return None
    col_left, row_top, col_right, row_bottom = bounds
    return segmentation_box_cell_fit(
        box,
        col_left=col_left,
        col_right=col_right,
        row_top=row_top,
        row_bottom=row_bottom,
    )


def cell_cx_ratio_world(cell: dict[str, Any], cx_ratio: float) -> float:
    """Map image-space cx_ratio into world cell coordinates.

    ``cx_ratio`` is measured from image column left (low x) → right (high x).
    ``get_cell(mirror_x=True)`` negates cell X so photo-left stays image-left;
    the horizontal center must be flipped within the cell.
    """
    cx = float(cx_ratio)
    if cell.get("mirror_x", False):
        cx = 1.0 - cx
    return cx


def world_placement_in_cell(
    cell: dict[str, Any],
    fit: dict[str, float],
) -> tuple[float, float, float, float]:
    """World (ox, oz, ww, hh) from grid cell + relative segmentation fit.

    Parameters (all relative to the floor×bay cell from columns_xy / floors_y):
      width_ratio  — box width ÷ cell width
      height_ratio — box height ÷ cell height
      cx_ratio     — box center x within cell (0=left, 1=right)
      cy_ratio     — box center y within cell (0=top, 1=bottom; image y down)
    """
    wr = float(fit.get("width_ratio", 0.55))
    hr = float(fit.get("height_ratio", 0.6))
    cx_ratio = cell_cx_ratio_world(cell, float(fit.get("cx_ratio", 0.5)))
    cy_ratio = float(fit.get("cy_ratio", 0.5))
    ww = max(0.15, float(cell["w"]) * wr)
    hh = max(0.15, float(cell["h"]) * hr)
    ox = float(cell["x0"]) + cx_ratio * float(cell["w"]) - ww / 2.0
    cz = float(cell["z1"]) - cy_ratio * float(cell["h"])
    oz = cz - hh / 2.0
    return ox, oz, ww, hh


def unified_pixel_scale(spec: dict[str, Any], total_w: float) -> float:
    """Single px→world scale (m/px) from horizontal façade extent."""
    xl, xr = image_facade_extent_x(spec)
    return float(total_w) / max(1.0, xr - xl)


def native_boundary_size(ir: dict[str, Any]) -> tuple[float, float]:
    boundary = ir.get("boundary") or {}
    params = boundary.get("params") or {}
    shape = boundary.get("shape", "rectangle")
    if shape == "circle":
        d = float(params.get("diameter", 1.0))
        return d, d
    w = float(params.get("width", 1.0))
    h = float(params.get("height", params.get("width", 1.0)))
    return w, h


def image_y_to_world_z(y_px: float, spec: dict[str, Any], total_h: float) -> float:
    """World Z at image row y (image top = high Z)."""
    yt, yb = image_facade_extent_y(spec)
    span = max(1.0, yb - yt)
    u = (float(y_px) - yt) / span
    u = min(1.0, max(0.0, u))
    return float(total_h) * (1.0 - u)


def image_x_to_world_x(
    x_px: float,
    spec: dict[str, Any],
    total_w: float,
    *,
    mirror_x: bool = True,
) -> float:
    """World X at image column x."""
    xl, xr = image_facade_extent_x(spec)
    span = max(1.0, xr - xl)
    u = (float(x_px) - xl) / span
    u = min(1.0, max(0.0, u))
    if mirror_x:
        return float(total_w) * (0.5 - u)
    return -float(total_w) / 2.0 + u * float(total_w)


def box_xyxy_to_world_placement(
    box_xyxy: list[float],
    spec: dict[str, Any],
    total_w: float,
    total_h: float,
    *,
    mirror_x: bool = True,
) -> tuple[float, float, float, float]:
    """Map image box → (ox, oz, ww, hh) with isotropic px scale and preserved aspect."""
    xl, xr = image_facade_extent_x(spec)
    yt, yb = image_facade_extent_y(spec)
    scale = unified_pixel_scale(spec, total_w)
    x0, y0, x1, y1 = (
        float(box_xyxy[0]),
        float(box_xyxy[1]),
        float(box_xyxy[2]),
        float(box_xyxy[3]),
    )
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    ww = max(0.2, (x1 - x0) * scale)
    hh = max(0.2, (y1 - y0) * scale)
    cx_px = 0.5 * (x0 + x1)
    cy_px = 0.5 * (y0 + y1)
    ucx = (cx_px - xl) / max(1.0, xr - xl)
    ucy = (cy_px - yt) / max(1.0, yb - yt)
    ucx = min(1.0, max(0.0, ucx))
    ucy = min(1.0, max(0.0, ucy))
    if mirror_x:
        wcx = float(total_w) * (0.5 - ucx)
    else:
        wcx = -float(total_w) / 2.0 + ucx * float(total_w)
    wcz = float(total_h) * (1.0 - ucy)
    ox = wcx - ww / 2.0
    oz = wcz - hh / 2.0
    return ox, oz, ww, hh


def fit_window_ir_to_bounds(
    ir: dict[str, Any],
    target_w: float,
    target_h: float,
    *,
    preserve_aspect: bool = True,
) -> dict[str, Any]:
    """Resize window IR to target bounds; optionally keep native width/height ratio."""
    out = copy.deepcopy(ir)
    out["type"] = "window"
    out["debug"] = False
    boundary = out.setdefault("boundary", {"id": "root", "shape": "rectangle"})
    shape = boundary.get("shape", "rectangle")
    params = boundary.setdefault("params", {})
    tw = max(0.2, float(target_w))
    th = max(0.2, float(target_h))
    if preserve_aspect:
        nw, nh = native_boundary_size(out)
        if nw > 0 and nh > 0:
            s = min(tw / nw, th / nh)
            tw, th = nw * s, nh * s
    if shape == "rectangle":
        params["width"] = round(tw, 4)
        params["height"] = round(th, 4)
    elif shape == "circle":
        params["diameter"] = round(min(tw, th), 4)
    else:
        params["width"] = round(tw, 4)
        params["height"] = round(th, 4)
    return out


def fit_window_ir_to_cell(
    ir: dict[str, Any],
    cell: dict[str, Any],
    *,
    width_ratio: float,
    height_ratio: float,
    preserve_aspect: bool = True,
) -> dict[str, Any]:
    target_w = max(0.2, float(cell["w"]) * float(width_ratio))
    target_h = max(0.2, float(cell["h"]) * float(height_ratio))
    return fit_window_ir_to_bounds(
        ir, target_w, target_h, preserve_aspect=preserve_aspect
    )
