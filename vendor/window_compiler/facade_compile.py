"""Compile a façade of windows onto a wall grid (Blender)."""

from __future__ import annotations

import math
import os

from typing import Any

import bmesh
import bpy
from mathutils import Matrix, Vector

from balcony_compile import ENCLOSED_WALL_M, add_balcony_meshes
from balcony_layout import balcony_frames, enclosed_window_fits, front_edge_x, front_window_targets, seat_on_window_bottoms, side_wall_targets, windows_behind_glass
from balcony_plan import outer_edges
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
    wall_world_extent,
    EMPTY_TOKENS,
    align_type_bottoms,
    build_instance_map,
    cell_fit,
    cell_tokens,
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
    align_column_centers,
    align_columns_enabled,
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
    xmin, xmax, zmin, zmax = wall_world_extent(spec, mirror_x=bool((spec.get("placement_params") or {}).get("mirror_x", True)))
    depth = float(wall.get("depth", 0.42))
    front_y = float(wall.get("base_front_y", 0.0))
    cx = 0.5 * (xmin + xmax)
    cz = 0.5 * (zmin + zmax)
    cy = front_y - depth / 2.0

    bpy.ops.mesh.primitive_cube_add(location=(cx, cy, cz))
    obj = bpy.context.active_object
    obj.name = "FacadeWall"
    obj.scale = ((xmax - xmin) / 2.0, depth / 2.0, (zmax - zmin) / 2.0)
    bpy.ops.object.transform_apply(scale=True)
    mat = MATS.get(wall.get("material", "wall"), MATS.get("wall"))
    if mat is not None:
        assign_mat(obj, mat)
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    parent.objects.link(obj)
    return obj


def _slot_suffix(p: dict[str, Any]) -> str:
    """Object-name suffix for items stacked in one cell (slot > 0), so their names stay unique."""
    return f"_s{p['slot']}" if p.get("slot") else ""


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


def _cut_side_opening(
    panel: bpy.types.Object,
    *,
    x_face: float,
    sign: int,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
    depth: float,
    pad: float = 0.005,
) -> None:
    """Boolean-cut an opening through a panel square to the wall (YZ aperture, outer face at ``x_face``)."""
    bpy.ops.mesh.primitive_cube_add(location=(x_face - sign * depth / 2.0, 0.5 * (y0 + y1), 0.5 * (z0 + z1)))
    cutter = bpy.context.active_object
    cutter.name = "OpeningCutter"
    cutter.scale = (depth / 2.0 + 0.08, max(0.05, (y1 - y0) / 2.0 + pad), max(0.05, (z1 - z0) / 2.0 + pad))
    bpy.ops.object.transform_apply(scale=True)
    _apply_cutter(panel, cutter)


def _turn_onto_side_wall(objs: list[bpy.types.Object], p: dict[str, Any], recess: float) -> None:
    """A window is built facing +y from its origin; turn it about z to face out of the block's side wall."""
    s = p["side"]
    r_in = min(max(0.02, recess), 0.06)
    pivot = Vector((p["ox"], p["oy"], p["oz"]))
    if s["sign"] > 0:  # faces +x: local width runs towards the wall
        to = Vector((s["edge_x"] - r_in, s["y0"] + p["ww"], p["oz"]))
    else:
        to = Vector((s["edge_x"] + r_in, s["y0"], p["oz"]))
    turn = Matrix.Translation(to) @ Matrix.Rotation(-s["sign"] * math.pi / 2.0, 4, "Z") @ Matrix.Translation(-pivot)
    bpy.context.view_layer.update()
    inside = set(objs)
    for o in objs:
        if o.parent not in inside:
            o.matrix_world = turn @ o.matrix_world


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


def _curve_y(outline: list[tuple[float, float]], y_wall: float, x: float) -> float | None:
    """Depth (y) of a balcony plan's outer boundary at ``x`` (max over its outer edges), None outside it."""
    best = None
    for (ax, ay), (bx, by) in outer_edges(outline, y_wall):
        lo, hi = min(ax, bx), max(ax, bx)
        if not (lo - 1e-6 <= x <= hi + 1e-6):
            continue
        y = ay if abs(bx - ax) < 1e-9 else ay + (by - ay) * (x - ax) / (bx - ax)
        best = y if best is None else max(best, y)
    return best


def _curved_front_targets(
    frames: list[dict[str, Any] | None], wins: list[dict[str, Any]], taken: set[int], recess: float, margin: float = 0.05
) -> list[dict[str, Any]]:
    """Windows of punched enclosed balconies whose plan has no flat front (circle). A window belongs to a frame on
    its floor whose bays overlap its own and whose plan spans its centre x, between floor top and enclosure top. It
    moves to just inside the shell (the shallowest boundary depth over its width, minus the recess) and is marked
    ``on_balcony``; returns the openings to cut through the shell once it is built."""
    cuts = []
    for wi, w in enumerate(wins):
        if wi in taken:
            continue
        r, c_lo = int(w["r"]), int(w["c"])
        c_hi = c_lo + max(1, int(w.get("span", 1))) - 1
        wcx, wcz = float(w["ox"]) + 0.5 * float(w["ww"]), float(w["oz"]) + 0.5 * float(w["hh"])
        for f in frames:
            if f is None or f.get("enclosure") != "enclosed" or f.get("outline") is None or f.get("curtain_wall"):
                continue
            if front_edge_x(f) is not None or f["row"] != r or c_hi < f["col0"] or c_lo > f["col1"]:
                continue
            xs = [q[0] for q in f["outline"]]
            if not (min(xs) + margin < wcx < max(xs) - margin):
                continue
            z_lo, z_hi = f["sill_z"] + f["thick"], float(f["top_z"])
            if not (z_lo < wcz < z_hi):
                continue
            x0 = max(float(w["ox"]), min(xs) + margin)
            x1 = min(float(w["ox"]) + float(w["ww"]), max(xs) - margin)
            ys = [_curve_y(f["outline"], float(f["y_wall"]), x) for x in (x0, 0.5 * (x0 + x1), x1)]
            if x1 - x0 < 0.3 or any(y is None for y in ys):
                continue
            w.update(ox=x0, ww=x1 - x0, on_balcony=int(f["index"]))
            r_in = min(max(0.02, recess), 0.06)
            w["oy"] = min(ys) - r_in
            cuts.append({"frame": f, "win": w, "recess": r_in})
            break
    return cuts


def _bend_to_curve(objs: list[bpy.types.Object], f: dict[str, Any], recess: float, step: float = 0.04) -> None:
    """Wrap a flat window (built facing +y) onto its balcony's curved shell: sliced every ``step`` along x, then each
    vertex at (x, y) goes to the shell point at x, offset along the shell's outward normal by its depth below the
    window's front face (minus the recess), so the frame follows the curve and sits in the shell thickness."""
    outline, y_wall = f["outline"], float(f["y_wall"])
    xs_o = [q[0] for q in outline]
    lo, hi = min(xs_o) + 1e-3, max(xs_o) - 1e-3

    def curve(x: float) -> float:
        return _curve_y(outline, y_wall, min(hi, max(lo, x))) or y_wall

    def normal(x: float, h: float = 0.12) -> tuple[float, float]:
        a, b = max(lo, x - h), min(hi, x + h)
        dy = (curve(b) - curve(a)) / max(b - a, 1e-6)
        n = (dy * dy + 1.0) ** 0.5
        return -dy / n, 1.0 / n

    dg = bpy.context.evaluated_depsgraph_get()
    meshes = []
    for o in objs:  # world-space, single-user mesh data; curve objects (muntins) become meshes
        if o.type not in ("MESH", "CURVE") or o.data is None:
            continue
        mw = o.matrix_world.copy()
        if o.type == "CURVE":
            me = bpy.data.meshes.new_from_object(o.evaluated_get(dg))
            m = bpy.data.objects.new(o.name, me)
            for c in o.users_collection:
                c.objects.link(m)
            bpy.data.objects.remove(o, do_unlink=True)
        else:
            m = o
            m.data = m.data.copy()
            m.parent = None
        m.data.transform(mw)
        m.matrix_world = Matrix.Identity(4)
        meshes.append(m)
    if not meshes:
        return
    y_front = max(v.co.y for o in meshes for v in o.data.vertices)
    for o in meshes:
        bm = bmesh.new()
        bm.from_mesh(o.data)
        vx = [v.co.x for v in bm.verts]
        if vx:
            x = min(vx) + step
            while x < max(vx) - 1e-4:
                geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
                bmesh.ops.bisect_plane(bm, geom=geom, plane_co=(x, 0.0, 0.0), plane_no=(1.0, 0.0, 0.0))
                x += step
        for v in bm.verts:
            d = v.co.y - y_front - recess
            x = v.co.x
            nx, ny = normal(x)
            v.co.x, v.co.y = x + nx * d, curve(x) + ny * d
        bm.to_mesh(o.data)
        bm.free()
        o.data.update()


def _cut_curved_openings(cuts: list[dict[str, Any]]) -> int:
    """Boolean-cut each curved-front window's opening through the shell panels of its balcony."""
    coll = bpy.data.collections.get("Balconies")
    if coll is None or not cuts:
        return 0
    n = 0
    for cut in cuts:
        w = cut["win"]
        x0, x1 = w.get("open_x0", w["ox"]), w.get("open_x1", w["ox"] + w["ww"])
        z0, z1 = w.get("open_z0", w["oz"]), w.get("open_z1", w["oz"] + w["hh"])
        prefix = f"Balc_{int(cut['frame']['index'])}_"
        for obj in list(coll.all_objects):
            if not obj.name.startswith(prefix) or "_wall_" not in obj.name or obj.type != "MESH":
                continue
            bx = [(obj.matrix_world @ v.co).x for v in obj.data.vertices]
            if not bx or max(bx) < x0 or min(bx) > x1:
                continue
            # box cutter from 0.08 m outside the shell to past its back face, bent like the window so the
            # opening's reveals are normal to the curve
            pad, depth = 0.005, ENCLOSED_WALL_M + 0.16
            if w.get("outline"):  # arched etc.: the window's padded outline
                cutter = make_prism("OpeningCutter", w["outline"], -depth, 0.0, coll)
            else:
                bpy.ops.mesh.primitive_cube_add(location=(0.5 * (x0 + x1), -depth / 2.0, 0.5 * (z0 + z1)))
                cutter = bpy.context.active_object
                cutter.name = "OpeningCutter"
                cutter.scale = ((x1 - x0) / 2.0 + pad, depth / 2.0, (z1 - z0) / 2.0 + pad)
                bpy.ops.object.transform_apply(scale=True)
            _bend_to_curve([cutter], cut["frame"], -0.08)
            _apply_cutter(obj, cutter)
            n += 1
    return n


def _glaze_railing_panels() -> int:
    """See-through material on balcony railing infill panels (``surface_panel``, which ``glass`` aliases to).

    ``balcony_compile`` builds them with the opaque ``railing`` material; mullions and rails keep it.
    """
    coll = bpy.data.collections.get("Balconies")
    mat = MATS.get("railing_glass")
    if coll is None or mat is None:
        return 0
    n = 0
    for obj in coll.all_objects:
        if "_panel_" in obj.name:
            assign_mat(obj, mat)
            n += 1
    return n


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
    recess = float(os.environ.get("FACADE_DSL_RECESS") or pp.get("recess", 0.01))
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
            span = 1
            if c < len(span_row) and span_row[c]:
                span = max(1, int(span_row[c]))
            toks = cell_tokens(row[c] if c < len(row) else None)  # several when items are stacked in the cell
            if not toks:
                c += max(1, span)
                continue
            for slot, tok in enumerate(toks):
                is_door = tok in doors or is_door_type_token(tok)
                if is_door and tok not in doors:
                    print(f"warn: unknown door type {tok!r} at ({r},{c})")
                    continue
                if not is_door and tok not in windows:
                    print(f"warn: unknown window type {tok!r} at ({r},{c})")
                    continue
                if span > 1:
                    cell = get_cell_span(spec, r, c, c + span, mirror_x=mirror_x)
                else:
                    cell = get_cell(spec, r, c, mirror_x=mirror_x)
                fit_row = placement_fit[r] if r < len(placement_fit) else []
                fit = cell_fit(fit_row[c], slot) if c < len(fit_row) else {}
                inst = inst_map.get((floor_id, c, slot))

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
                        "slot": slot,
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
                        "slot": slot,
                        "kind": "window",
                        "span": span,
                        "cell": cell,
                        "ox": ox,
                        "oy": front_y - max(0.0, recess),
                        "oz": oz,
                        "ww": ww,
                        "hh": hh,
                        "seg_fit": seg_fit,
                        "side_face": bool(inst and inst.get("side_face")),
                    }
                )
            c += span

    planned_wins = [p for p in planned if p["kind"] == "window"]
    # Side-wall windows look narrow only through foreshortening: they must not pull their type's sizes or columns.
    front_wins = [p for p in planned_wins if not p.get("side_face")]
    if bool(pp.get("unify_type_size", True)):
        unify_type_sizes(
            front_wins,
            separate_spans=not bool(pp.get("unify_type_size_across_spans", True)),
        )
    if bool(pp.get("align_type_bottom", True)):
        align_type_bottoms(front_wins)
    if align_columns_enabled(pp):
        align_column_centers(front_wins, tol=float(pp.get("align_column_tol", 0.25)))

    # balconies (hiroo/balcony-pipeline hooks), after type sizing and bay alignment: windows behind an enclosed
    # balcony's glass are dropped; windows on a balcony front move onto it (the balcony mesh cuts their opening)
    fits = enclosed_window_fits(spec, planned_wins) if spec.get("balcony_placement") else {}
    frames = balcony_frames(spec, fits) if spec.get("balcony_placement") else []
    behind_glass = windows_behind_glass(frames, planned_wins)
    if behind_glass:
        drop_ids = {id(planned_wins[i]) for i in behind_glass}
        planned = [p for p in planned if id(p) not in drop_ids]
        planned_wins = [p for i, p in enumerate(planned_wins) if i not in behind_glass]
    flat_targets = front_window_targets(frames, planned_wins)
    for wi, (fi, (ox, oz, ww, hh)) in flat_targets.items():
        p = planned_wins[wi]
        p.update(ox=ox, oz=oz, ww=ww, hh=hh, on_balcony=fi)
        p["oy"] = balcony_front_y(frames[fi]) - max(0.0, recess)
    # punched (wall-skin) enclosed balconies with a curved front (circle plan): front_window_targets only handles a
    # flat front, so place those windows here, just inside the curved shell, and cut their openings after the
    # balcony meshes exist (see _cut_curved_openings)
    curved_cuts = _curved_front_targets(frames, planned_wins, set(flat_targets), recess)

    frames = balcony_frames(spec, fits) if spec.get("balcony_placement") else []
    behind_glass = windows_behind_glass(frames, planned_wins)
    if behind_glass:
        drop_ids = {id(planned_wins[i]) for i in behind_glass}
        planned = [p for p in planned if id(p) not in drop_ids]
        planned_wins = [p for i, p in enumerate(planned_wins) if i not in behind_glass]
    for wi, (fi, (ox, oz, ww, hh)) in front_window_targets(frames, planned_wins).items():
        p = planned_wins[wi]
        p.update(ox=ox, oz=oz, ww=ww, hh=hh, on_balcony=fi)
        p["oy"] = balcony_front_y(frames[fi]) - max(0.0, recess)
    mirrored: list[dict[str, Any]] = []
    for wi, t in side_wall_targets(frames, planned_wins).items():
        p = planned_wins[wi]
        p.update(ww=t["ww"], oz=t["oz"], hh=t["hh"], on_balcony=t["frame"], side=t)
        q = dict(p)
        q.update(on_balcony=t["mirror"]["frame"], side=t["mirror"], slot=f"{p.get('slot', 0)}m")
        planned.append(q)
        mirrored.append(q)
    planned_wins.extend(mirrored)
    seat_on_window_bottoms(frames, planned_wins)

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
        if p.get("side"):
            s = p["side"]
            front_openings.setdefault(int(p["on_balcony"]), []).append(
                {"side": s["sign"], "y0": s["y0"], "y1": s["y0"] + p["ww"], "z0": p["open_z0"], "z1": p["open_z1"]}
            )
            continue
        front_openings.setdefault(int(p["on_balcony"]), []).append(
            {"x0": p["open_x0"], "x1": p["open_x1"], "z0": p["open_z0"], "z1": p["open_z1"],
             "outline": p.get("outline")}
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
                name_prefix=f"Door_r{p['r']}_c{p['c']}{_slot_suffix(p)}_{p['tok']}",
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
        prefix = f"Win_r{p['r']}_c{p['c']}{_slot_suffix(p)}_{p['tok']}"
        before = set(bpy.data.objects)
        build_blender_scene(
            p["ctx"],
            name_prefix=prefix,
            parent_collection=windows_coll,
            origin=(p["ox"], p["oy"], p["oz"]),
        )
        if p.get("side"):
            _turn_onto_side_wall([o for o in bpy.data.objects if o not in before], p, recess)
        bend = next((c for c in curved_cuts if c["win"] is p), None)
        if bend is not None:
            _bend_to_curve([o for o in bpy.data.objects if o not in before], bend["frame"], bend["recess"])
        n_placed += 1
        n_segments += len(p["ctx"].segments)

    n_balc = add_balcony_meshes(
        spec,
        wall_obj,
        facade_coll,
        cut_opening=_cut_opening,
        cut_opening_poly=_cut_opening_poly,
        cut_side_opening=_cut_side_opening,
        frames=frames or None,
        front_openings=front_openings,
    )
    _glaze_railing_panels()
    _cut_curved_openings(curved_cuts)

    bounds = wall_world_extent(spec, mirror_x=bool(pp.get("mirror_x", True)))
    print(
        f"compiled façade: {n_placed} windows, {n_doors} doors, {n_balc} balconies, {n_segments} muntin segments, "
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
