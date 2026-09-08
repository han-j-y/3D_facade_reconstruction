"""Compile a façade of windows onto a wall grid (Blender)."""

from __future__ import annotations

from typing import Any

import bpy

from balcony_compile import add_balcony_meshes
from blender_scene import build_blender_scene
from compiler import compile_spec
from door_scene import build_default_door, door_opening_bounds
from facade_spec import (
    EMPTY_TOKENS,
    build_instance_map,
    cell_cx_ratio_world,
    fit_window_ir_to_bounds,
    fit_window_ir_to_cell,
    get_cell,
    get_cell_span,
    is_door_type_token,
    is_facade_spec,
    is_window_spec,
    normalize_facade_spec,
    placement_fit_for_instance,
    total_grid_size,
    unwrap_window_ir,
    world_placement_in_cell,
)
from geometry import assign_mat
from materials import MATS

__all__ = [
    "EMPTY_TOKENS",
    "compile_facade_scene",
    "fit_window_ir_to_cell",
    "get_cell",
    "is_facade_spec",
    "is_window_spec",
    "normalize_facade_spec",
    "total_grid_size",
    "unwrap_window_ir",
]


def _build_wall(spec: dict[str, Any], parent: bpy.types.Collection) -> bpy.types.Object:
    grid = spec["grid"]
    wall = spec.get("wall") or {}
    total_w, total_h = total_grid_size(grid)
    depth = float(wall.get("depth", 0.42))
    front_y = float(wall.get("base_front_y", 0.0))
    cx = 0.0
    cz = total_h / 2.0
    cy = front_y - depth / 2.0

    bpy.ops.mesh.primitive_cube_add(location=(cx, cy, cz))
    obj = bpy.context.active_object
    obj.name = "FacadeWall"
    obj.scale = (total_w / 2.0, depth / 2.0, total_h / 2.0)
    bpy.ops.object.transform_apply(scale=True)
    mat = MATS.get(wall.get("material", "wall"), MATS.get("wall"))
    if mat is not None:
        assign_mat(obj, mat)
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    parent.objects.link(obj)
    return obj


def _cut_opening(
    wall_obj: bpy.types.Object,
    *,
    x0: float,
    z0: float,
    x1: float,
    z1: float,
    front_y: float,
    depth: float,
    pad: float = 0.005,
) -> None:
    """Boolean-cut a rectangular opening through the wall slab (XZ aperture)."""
    cx = 0.5 * (x0 + x1)
    cz = 0.5 * (z0 + z1)
    cy = front_y - depth / 2.0
    bpy.ops.mesh.primitive_cube_add(location=(cx, cy, cz))
    cutter = bpy.context.active_object
    cutter.name = "OpeningCutter"
    cutter.scale = (
        max(0.02, (x1 - x0) / 2.0 + pad),
        depth / 2.0 + 0.02,
        max(0.02, (z1 - z0) / 2.0 + pad),
    )
    bpy.ops.object.transform_apply(scale=True)

    mod = wall_obj.modifiers.new(name="CutOpening", type="BOOLEAN")
    mod.operation = "DIFFERENCE"
    try:
        mod.solver = "EXACT"
    except (AttributeError, TypeError):
        pass
    mod.object = cutter

    bpy.ops.object.select_all(action="DESELECT")
    wall_obj.select_set(True)
    bpy.context.view_layer.objects.active = wall_obj
    bpy.ops.object.modifier_apply(modifier=mod.name)

    mesh_to_drop = cutter.data
    bpy.data.objects.remove(cutter, do_unlink=True)
    if mesh_to_drop is not None and mesh_to_drop.users == 0:
        bpy.data.meshes.remove(mesh_to_drop)


def compile_facade_scene(spec: dict[str, Any]) -> dict[str, Any]:
    """Compile wall + all placed windows into the current Blender scene."""
    if spec.get("type") != "facade":
        raise ValueError('spec.type must be "facade" (call normalize_facade_spec first)')

    grid = spec["grid"]
    placement = spec.get("placement") or []
    placement_spans = spec.get("placement_spans") or []
    windows = spec.get("windows") or {}
    doors = spec.get("doors") or {}
    pp = spec.get("placement_params") or {}
    type_ratios = (spec.get("meta") or {}).get("type_ratios") or {}
    placement_fit = spec.get("placement_fit") or []
    default_wr = float(pp.get("width_ratio", 0.55))
    default_hr = float(pp.get("height_ratio", 0.60))
    recess = float(pp.get("recess", 0.01))
    wall_depth = float((spec.get("wall") or {}).get("depth", 0.42))
    front_y = float((spec.get("wall") or {}).get("base_front_y", 0.0))

    facade_coll = bpy.data.collections.new("Generated_Facade")
    bpy.context.scene.collection.children.link(facade_coll)
    windows_coll = bpy.data.collections.new("Windows")
    facade_coll.children.link(windows_coll)
    doors_coll = bpy.data.collections.new("Doors")
    facade_coll.children.link(doors_coll)

    wall_obj = _build_wall(spec, facade_coll)

    total_w, total_h = total_grid_size(grid)
    n_placed = 0
    n_segments = 0
    n_rows = len(grid["rows"])
    n_cols = len(grid["cols"])

    planned: list[dict[str, Any]] = []
    mirror_x = bool((spec.get("placement_params") or {}).get("mirror_x", True))
    floor_ids = spec.get("floor_ids") or []
    inst_map = build_instance_map(spec)

    n_doors = 0
    ground_floor_id = max(floor_ids) if floor_ids else None

    for r in range(n_rows):
        row = placement[r] if r < len(placement) else []
        span_row = placement_spans[r] if r < len(placement_spans) else []
        c = 0
        floor_id = int(floor_ids[r]) if r < len(floor_ids) else r
        while c < n_cols:
            tok = row[c] if c < len(row) else None
            span = 1
            if c < len(span_row) and span_row[c]:
                span = max(1, int(span_row[c]))
            if tok in EMPTY_TOKENS:
                c += max(1, span)
                continue
            tok = str(tok)
            is_door = tok in doors or is_door_type_token(tok)
            if is_door and tok not in doors:
                print(f"warn: unknown door type {tok!r} at ({r},{c})")
                c += span
                continue
            if not is_door and tok not in windows:
                print(f"warn: unknown window type {tok!r} at ({r},{c})")
                c += span
                continue
            if span > 1:
                cell = get_cell_span(spec, r, c, c + span, mirror_x=mirror_x)
            else:
                cell = get_cell(spec, r, c, mirror_x=mirror_x)
            fit_row = placement_fit[r] if r < len(placement_fit) else []
            fit = fit_row[c] if c < len(fit_row) and fit_row[c] else {}
            inst = inst_map.get((floor_id, c))

            if is_door:
                door_def = doors[tok]
                seg_fit = (
                    placement_fit_for_instance(spec, inst, r, c)
                    if inst is not None
                    else fit
                )
                if seg_fit:
                    ox, oz, ww, hh = world_placement_in_cell(cell, seg_fit)
                else:
                    ratios = type_ratios.get(tok) or {}
                    wr = float(fit.get("width_ratio", ratios.get("width_ratio", default_wr)))
                    hr = float(fit.get("height_ratio", ratios.get("height_ratio", default_hr)))
                    ww = max(0.25, float(cell["w"]) * wr)
                    hh = max(0.35, float(cell["h"]) * hr)
                    cx_ratio = cell_cx_ratio_world(cell, float(fit.get("cx_ratio", 0.5)))
                    cy_ratio = float(fit.get("cy_ratio", 0.5))
                    ox = float(cell["x0"]) + cx_ratio * float(cell["w"]) - ww / 2.0
                    cz = float(cell["z1"]) - cy_ratio * float(cell["h"])
                    oz = cz - hh / 2.0
                oy = front_y - max(0.0, recess)
                open_to_floor = 0.0 if floor_id == ground_floor_id else oz
                open_x0, open_z0, open_x1, open_z1 = door_opening_bounds(
                    ox=ox, oz=oz, ww=ww, hh=hh, open_to_floor_z=open_to_floor
                )
                planned.append(
                    {
                        "r": r,
                        "c": c,
                        "tok": tok,
                        "kind": "door",
                        "door_def": door_def,
                        "ox": ox,
                        "oy": oy,
                        "oz": oz,
                        "ww": ww,
                        "hh": hh,
                        "open_x0": open_x0,
                        "open_x1": open_x1,
                        "open_z0": open_z0,
                        "open_z1": open_z1,
                    }
                )
                c += span
                continue

            ratios = type_ratios.get(tok) or {}
            seg_fit = (
                placement_fit_for_instance(spec, inst, r, c)
                if inst is not None
                else fit
            )
            if not seg_fit:
                seg_fit = {
                    "width_ratio": float(
                        fit.get("width_ratio", ratios.get("width_ratio", default_wr))
                    ),
                    "height_ratio": float(
                        fit.get("height_ratio", ratios.get("height_ratio", default_hr))
                    ),
                    "cx_ratio": float(fit.get("cx_ratio", 0.5)),
                    "cy_ratio": float(fit.get("cy_ratio", 0.5)),
                }
            ox, oz, ww, hh = world_placement_in_cell(cell, seg_fit)
            ir = fit_window_ir_to_bounds(
                windows[tok], ww, hh, preserve_aspect=False
            )
            ctx = compile_spec(ir)
            hh = float(ctx.region("root").height)
            ww = float(ctx.region("root").width)
            open_x0, open_x1 = ox, ox + ww
            open_z0, open_z1 = oz, oz + hh
            oy = front_y - max(0.0, recess)
            planned.append(
                {
                    "r": r,
                    "c": c,
                    "tok": tok,
                    "kind": "window",
                    "ctx": ctx,
                    "ox": ox,
                    "oy": oy,
                    "oz": oz,
                    "ww": ww,
                    "hh": hh,
                    "open_x0": open_x0,
                    "open_x1": open_x1,
                    "open_z0": open_z0,
                    "open_z1": open_z1,
                }
            )
            c += span

    for p in planned:
        _cut_opening(
            wall_obj,
            x0=p["open_x0"],
            z0=p["open_z0"],
            x1=p["open_x1"],
            z1=p["open_z1"],
            front_y=front_y,
            depth=wall_depth,
        )

    for p in planned:
        if p.get("kind") == "door":
            dd = p.get("door_def") or {}
            build_default_door(
                name_prefix=f"Door_r{p['r']}_c{p['c']}_{p['tok']}",
                parent_collection=doors_coll,
                origin=(p["ox"], p["oy"], p["oz"]),
                width=p["ww"],
                height=p["hh"],
                shape=str(dd.get("shape") or "rectangle"),
                arch_rise_ratio=float(dd.get("arch_rise_ratio") or 0.0),
                recess=recess,
                open_to_floor_z=p["open_z0"],
            )
            n_doors += 1
            continue
        prefix = f"Win_r{p['r']}_c{p['c']}_{p['tok']}"
        build_blender_scene(
            p["ctx"],
            name_prefix=prefix,
            parent_collection=windows_coll,
            origin=(p["ox"], p["oy"], p["oz"]),
        )
        n_placed += 1
        n_segments += len(p["ctx"].segments)

    n_balc = add_balcony_meshes(
        spec, wall_obj, facade_coll, cut_opening=_cut_opening
    )

    bounds = (-total_w / 2.0, total_w / 2.0, 0.0, total_h)
    print(
        f"compiled façade: {n_placed} windows, {n_doors} doors, {n_balc} balconies, "
        f"{n_segments} muntin segments, "
        f"size={total_w:.2f}x{total_h:.2f}m"
    )
    return {
        "n_windows": n_placed,
        "n_doors": n_doors,
        "n_balconies": n_balc,
        "n_segments": n_segments,
        "total_w": total_w,
        "total_h": total_h,
        "bounds": bounds,
    }
