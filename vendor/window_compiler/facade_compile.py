"""Compile a façade of windows onto a wall grid (Blender)."""

from __future__ import annotations

from typing import Any

import bpy

from blender_scene import build_blender_scene
from compiler import compile_spec
from facade_spec import (
    EMPTY_TOKENS,
    fit_window_ir_to_cell,
    get_cell,
    is_facade_spec,
    is_window_spec,
    normalize_facade_spec,
    total_grid_size,
    unwrap_window_ir,
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
    # slightly thicker than the wall so the boolean cleanly pierces both faces
    bpy.ops.mesh.primitive_cube_add(location=(cx, cy, cz))
    cutter = bpy.context.active_object
    cutter.name = "OpeningCutter"
    cutter.scale = (
        max(0.05, (x1 - x0) / 2.0 + pad),
        depth / 2.0 + 0.08,
        max(0.05, (z1 - z0) / 2.0 + pad),
    )
    bpy.ops.object.transform_apply(scale=True)

    mod = wall_obj.modifiers.new(name="WinOpen", type="BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.solver = "EXACT"
    try:
        mod.operand_type = "OBJECT"
    except (AttributeError, TypeError):
        pass
    mod.object = cutter

    # Apply on the wall (must be active / object mode)
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
    windows = spec.get("windows") or {}
    pp = spec.get("placement_params") or {}
    type_ratios = (spec.get("meta") or {}).get("type_ratios") or {}
    default_wr = float(pp.get("width_ratio", 0.55))
    default_hr = float(pp.get("height_ratio", 0.60))
    bottom_m = float(pp.get("bottom_margin_ratio", 0.14))
    # Recess into the opening (toward -Y). Must stay within a cut hole —
    # without openings, any recess buries windows inside the solid wall.
    recess = float(pp.get("recess", 0.01))
    wall_depth = float((spec.get("wall") or {}).get("depth", 0.42))
    front_y = float((spec.get("wall") or {}).get("base_front_y", 0.0))

    facade_coll = bpy.data.collections.new("Generated_Facade")
    bpy.context.scene.collection.children.link(facade_coll)
    windows_coll = bpy.data.collections.new("Windows")
    facade_coll.children.link(windows_coll)

    wall_obj = _build_wall(spec, facade_coll)

    total_w, total_h = total_grid_size(grid)
    n_placed = 0
    n_segments = 0
    n_rows = len(grid["rows"])
    n_cols = len(grid["cols"])

    # First pass: gather placements + cut openings (before window meshes).
    planned: list[dict[str, Any]] = []
    # Photo-left ↔ image-left: compiler +Y camera mirrors world X (see get_cell).
    mirror_x = bool((spec.get("placement_params") or {}).get("mirror_x", True))

    for r in range(n_rows):
        row = placement[r] if r < len(placement) else []
        for c in range(n_cols):
            tok = row[c] if c < len(row) else None
            if tok in EMPTY_TOKENS:
                continue
            tok = str(tok)
            if tok not in windows:
                print(f"warn: unknown window type {tok!r} at ({r},{c})")
                continue
            cell = get_cell(spec, r, c, mirror_x=mirror_x)
            ratios = type_ratios.get(tok) or {}
            wr = float(ratios.get("width_ratio", default_wr))
            hr = float(ratios.get("height_ratio", default_hr))
            ir = fit_window_ir_to_cell(windows[tok], cell, width_ratio=wr, height_ratio=hr)
            ctx = compile_spec(ir)
            ww = float(ctx.region("root").width)
            hh = float(ctx.region("root").height)
            ox = float(cell["cx"]) - ww / 2.0
            oz = float(cell["z0"]) + float(cell["h"]) * bottom_m
            if oz + hh > float(cell["z1"]) - 0.02:
                oz = max(float(cell["z0"]) + 0.02, float(cell["z1"]) - hh - 0.02)
            # Exterior face flush with / slightly into the wall front after opening cut
            oy = front_y - max(0.0, recess)
            planned.append(
                {
                    "r": r,
                    "c": c,
                    "tok": tok,
                    "ctx": ctx,
                    "ox": ox,
                    "oy": oy,
                    "oz": oz,
                    "ww": ww,
                    "hh": hh,
                }
            )

    for p in planned:
        _cut_opening(
            wall_obj,
            x0=p["ox"],
            z0=p["oz"],
            x1=p["ox"] + p["ww"],
            z1=p["oz"] + p["hh"],
            front_y=front_y,
            depth=wall_depth,
        )

    for p in planned:
        prefix = f"Win_r{p['r']}_c{p['c']}_{p['tok']}"
        build_blender_scene(
            p["ctx"],
            name_prefix=prefix,
            parent_collection=windows_coll,
            origin=(p["ox"], p["oy"], p["oz"]),
        )
        n_placed += 1
        n_segments += len(p["ctx"].segments)

    bounds = (-total_w / 2.0, total_w / 2.0, 0.0, total_h)
    print(
        f"compiled façade: {n_placed} windows, {n_segments} muntin segments, "
        f"size={total_w:.2f}×{total_h:.2f}m"
    )
    return {
        "n_windows": n_placed,
        "n_segments": n_segments,
        "total_w": total_w,
        "total_h": total_h,
        "bounds": bounds,
    }
