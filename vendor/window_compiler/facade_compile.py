"""Compile a façade of windows onto a wall grid (Blender)."""

from __future__ import annotations

from typing import Any

import bpy

from balcony_compile import add_balcony_meshes
from balcony_layout import balcony_frames, front_window_targets
from balcony_layout import front_y as balcony_front_y
from blender_scene import build_blender_scene
from compiler import compile_spec
from door_scene import (
    build_default_door,
    door_opening_bounds,
    door_outline_world,
    effective_arch_rise,
    make_prism,
)
from facade_spec import (
    EMPTY_TOKENS,
    align_type_bottoms,
    build_instance_map,
    fit_window_ir_to_bounds,
    fit_window_ir_to_cell,
    cell_cx_ratio_world,
    get_cell,
    get_cell_span,
    is_door_type_token,
    is_facade_spec,
    is_window_spec,
    normalize_facade_spec,
    placement_fit_for_instance,
    total_grid_size,
    unify_type_sizes,
    unwrap_window_ir,
    world_placement_in_cell,
)
from geometry import assign_mat
from shapes import contour_for_region
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
    _apply_cutter(wall_obj, cutter)


def _cut_opening_poly(
    wall_obj: bpy.types.Object,
    poly_xz: list[tuple[float, float]],
    *,
    front_y: float,
    depth: float,
) -> None:
    """Boolean-cut an opening with an arbitrary convex XZ outline (arched doors)."""
    cutter = make_prism(
        "OpeningCutter",
        poly_xz,
        front_y - depth - 0.08,
        front_y + 0.08,
        wall_obj.users_collection[0],
    )
    _apply_cutter(wall_obj, cutter)


def _padded_outline(
    pts: list[tuple[float, float]], pad: float = 0.005
) -> list[tuple[float, float]]:
    """Grow an XZ outline by ~``pad`` about its bbox center so the cut clears the frame."""
    dedup: list[tuple[float, float]] = []
    for q in pts:
        if not dedup or abs(q[0] - dedup[-1][0]) + abs(q[1] - dedup[-1][1]) > 1e-6:
            dedup.append(q)
    if len(dedup) > 1 and abs(dedup[0][0] - dedup[-1][0]) + abs(dedup[0][1] - dedup[-1][1]) <= 1e-6:
        dedup.pop()
    pts = dedup
    xs, zs = [x for x, _ in pts], [z for _, z in pts]
    cx, cz = 0.5 * (min(xs) + max(xs)), 0.5 * (min(zs) + max(zs))
    sx = 1.0 + 2.0 * pad / max(max(xs) - min(xs), 1e-6)
    sz = 1.0 + 2.0 * pad / max(max(zs) - min(zs), 1e-6)
    return [(cx + (x - cx) * sx, cz + (z - cz) * sz) for x, z in pts]


def _apply_cutter(wall_obj: bpy.types.Object, cutter: bpy.types.Object) -> None:
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
    placement_spans = spec.get("placement_spans") or []
    windows = spec.get("windows") or {}
    doors = spec.get("doors") or {}
    pp = spec.get("placement_params") or {}
    type_ratios = (spec.get("meta") or {}).get("type_ratios") or {}
    placement_fit = spec.get("placement_fit") or []
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
    doors_coll = bpy.data.collections.new("Doors")
    facade_coll.children.link(doors_coll)

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
                door_def = dict(doors[tok])
                # door types can mix rectangular and arched members
                if inst is not None and inst.get("kind") == "door" and inst.get("shape"):
                    door_def["shape"] = inst["shape"]
                    door_def["arch_rise_ratio"] = float(inst.get("arch_rise_ratio") or 0.0)
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
            planned.append(
                {
                    "r": r,
                    "c": c,
                    "tok": tok,
                    "kind": "window",
                    "span": span,
                    "cell": cell,
                    "ox": ox,
                    "oy": front_y - max(0.0, recess),
                    "oz": oz,
                    "ww": ww,
                    "hh": hh,
                    "seg_fit": seg_fit,
                }
            )
            c += span

    planned_wins = [p for p in planned if p["kind"] == "window"]
    if bool(pp.get("unify_type_size", True)):
        unify_type_sizes(planned_wins)
    if bool(pp.get("align_type_bottom", True)):
        align_type_bottoms(planned_wins)

    frames = balcony_frames(spec) if spec.get("balcony_placement") else []
    for wi, (fi, (ox, oz, ww, hh)) in front_window_targets(frames, planned_wins).items():
        p = planned_wins[wi]
        p.update(ox=ox, oz=oz, ww=ww, hh=hh, on_balcony=fi)
        p["oy"] = balcony_front_y(frames[fi]) - max(0.0, recess)

    for p in planned:
        if p["kind"] != "window":
            continue
        ir = fit_window_ir_to_bounds(
            windows[p["tok"]], p["ww"], p["hh"], preserve_aspect=False
        )
        ctx = compile_spec(ir)
        p["ctx"] = ctx
        p["ww"] = float(ctx.region("root").width)
        p["hh"] = float(ctx.region("root").height)
        p["open_x0"], p["open_x1"] = p["ox"], p["ox"] + p["ww"]
        p["open_z0"], p["open_z1"] = p["oz"], p["oz"] + p["hh"]
        root = ctx.region("root")
        if root.shape != "rectangle":
            p["outline"] = _padded_outline(
                [(p["ox"] + q.x, p["oz"] + q.y) for q in contour_for_region(root)]
            )

    front_openings: dict[int, list[dict[str, Any]]] = {}
    for p in planned_wins:
        if p.get("on_balcony") is None:
            continue
        front_openings.setdefault(int(p["on_balcony"]), []).append(
            {
                "x0": p["open_x0"],
                "x1": p["open_x1"],
                "z0": p["open_z0"],
                "z1": p["open_z1"],
                "outline": p.get("outline"),
            }
        )

    for p in planned:
        if p.get("on_balcony") is not None:
            continue
        if p.get("kind") == "door":
            dd = p.get("door_def") or {}
            top_z = p["oz"] + max(0.35, p["hh"])
            rise = effective_arch_rise(
                max(0.25, p["ww"]),
                top_z - p["open_z0"],
                str(dd.get("shape") or "rectangle"),
                float(dd.get("arch_rise_ratio") or 0.0),
            )
            if rise > 0.0:
                _cut_opening_poly(
                    wall_obj,
                    door_outline_world(
                        ox=p["ox"],
                        z_base=p["open_z0"],
                        width=max(0.25, p["ww"]),
                        top_z=top_z,
                        rise=rise,
                        pad=0.005,
                    ),
                    front_y=front_y,
                    depth=wall_depth,
                )
                continue
        if p.get("outline"):
            _cut_opening_poly(wall_obj, p["outline"], front_y=front_y, depth=wall_depth)
            continue
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
        spec,
        wall_obj,
        facade_coll,
        cut_opening=_cut_opening,
        cut_opening_poly=_cut_opening_poly,
        frames=frames or None,
        front_openings=front_openings,
    )

    bounds = (-total_w / 2.0, total_w / 2.0, 0.0, total_h)
    print(
        f"compiled façade: {n_placed} windows, {n_doors} doors, {n_balc} balconies, "
        f"{n_segments} muntin segments, "
        f"size={total_w:.2f}×{total_h:.2f}m"
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
