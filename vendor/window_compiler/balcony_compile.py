"""Place balcony slabs / railings on a compiled façade (Blender)."""

from __future__ import annotations

import math
from typing import Any, Callable

import bmesh
import bpy
from mathutils import Vector

from balcony_plan import (  # noqa: E402
    BALUSTER_DIAMETER_M,
    BALUSTER_SPACING_M,
    DEFAULT_SLAB_DEPTH_M,
    ENCLOSED_CLEAR_H_M,
    MASONRY_BALUSTER_DIAMETER_M,
    MASONRY_BALUSTER_GAP_M,
    MASONRY_BALUSTER_SPACING_M,
    MASONRY_POST_M,
    MASONRY_RAIL_INSET_M,
    MASONRY_TOP_RAIL_HEIGHT_M,
    MASONRY_TOP_RAIL_WIDTH_M,
    MULLION_SPACING_M,
    MULLION_WIDTH_M,
    SURFACE_PANEL_M,
    TOP_RAIL_DIAMETER_M,
    centroid,
    even_along_polyline,
    extend_segment_ends,
    glass_bays,
    grid_stations,
    inset_edge,
    inset_point,
    is_front_edge,
    masonry_baluster_centers,
    masonry_inset_rail_frame,
    outer_edges,
    polyline_length,
    rect_corner_posts,
    shorten_segment_end,
    slab_outline,
    spaced_count,
    support_polyline,
    surface_panel_parts,
)
from facade_spec import get_cell, total_grid_size
from geometry import assign_mat
from materials import MATS
from parse_bdsl import RAIL_ALIASES, RAIL_KINDS


def _span_xz(
    spec: dict[str, Any],
    row: int,
    col0: int,
    col1: int,
    *,
    mirror_x: bool,
) -> tuple[float, float, float, float]:
    a = get_cell(spec, int(row), int(col0), mirror_x=mirror_x)
    b = get_cell(spec, int(row), int(col1), mirror_x=mirror_x)
    return (
        min(float(a["x0"]), float(b["x0"])),
        max(float(a["x1"]), float(b["x1"])),
        float(a["z0"]),
        float(a["z1"]),
    )


def _mean_column_cx(
    spec: dict[str, Any],
    row: int,
    col0: int,
    col1: int,
    *,
    mirror_x: bool,
) -> float:
    """Mean X center of grid columns col0..col1 (inclusive)."""
    c0, c1 = min(int(col0), int(col1)), max(int(col0), int(col1))
    centers: list[float] = []
    for col in range(c0, c1 + 1):
        cell = get_cell(spec, int(row), col, mirror_x=mirror_x)
        centers.append(0.5 * (float(cell["x0"]) + float(cell["x1"])))
    return sum(centers) / len(centers)


def _photo_x_span(
    spec: dict[str, Any],
    rec: dict[str, Any],
    *,
    mirror_x: bool,
    bay_x0: float,
    bay_x1: float,
    row: int,
    col0: int,
    col1: int,
) -> tuple[float, float, float]:
    """Map photo box width/center to world X. Returns (x0, x1, cx).

    Photo u=0 is left; with mirror_x the camera shows world +X on the left,
    so ``world_x = (0.5 - cx_norm) * total_w``.
    When cx_norm is missing, uses window_cx_norm (paired window boxes), then
    bay_cx_norm (mean of photo bay bands), then the mean of associated grid
    column centers (multi-bay balconies).
    """
    total_w, _ = total_grid_size(spec["grid"])
    total_w = max(0.3, float(total_w))
    bay_cx = _mean_column_cx(spec, row, col0, col1, mirror_x=mirror_x)

    w_norm = rec.get("width_norm")
    cx_norm = rec.get("cx_norm")
    window_cx_norm = rec.get("window_cx_norm")
    bay_cx_norm = rec.get("bay_cx_norm")
    try:
        w_norm_f = float(w_norm) if w_norm is not None else None
        cx_norm_f = float(cx_norm) if cx_norm is not None else None
        window_cx_norm_f = (
            float(window_cx_norm) if window_cx_norm is not None else None
        )
        bay_cx_norm_f = float(bay_cx_norm) if bay_cx_norm is not None else None
    except (TypeError, ValueError):
        w_norm_f, cx_norm_f, window_cx_norm_f, bay_cx_norm_f = None, None, None, None

    if w_norm_f is None or w_norm_f <= 0:
        return bay_x0, bay_x1, bay_cx

    span_w = max(0.25, min(total_w * 0.98, w_norm_f * total_w))
    if cx_norm_f is not None:
        cx_n = min(1.0, max(0.0, cx_norm_f))
    elif window_cx_norm_f is not None:
        cx_n = min(1.0, max(0.0, window_cx_norm_f))
    elif bay_cx_norm_f is not None:
        cx_n = min(1.0, max(0.0, bay_cx_norm_f))
    else:
        cx = bay_cx
        x0 = cx - span_w / 2.0
        x1 = cx + span_w / 2.0
        half = total_w / 2.0
        if x0 < -half:
            x1 += -half - x0
            x0 = -half
        if x1 > half:
            x0 -= x1 - half
            x1 = half
        return x0, x1, 0.5 * (x0 + x1)

    if mirror_x:
        cx = (0.5 - cx_n) * total_w
    else:
        cx = (cx_n - 0.5) * total_w
    x0 = cx - span_w / 2.0
    x1 = cx + span_w / 2.0
    # Keep on façade
    half = total_w / 2.0
    if x0 < -half:
        x1 += -half - x0
        x0 = -half
    if x1 > half:
        x0 -= x1 - half
        x1 = half
    return x0, x1, 0.5 * (x0 + x1)


def _link(obj: bpy.types.Object, coll: bpy.types.Collection) -> None:
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)


def _cube(
    *,
    name: str,
    cx: float,
    cy: float,
    cz: float,
    sx: float,
    sy: float,
    sz: float,
    mat_key: str,
    parent: bpy.types.Collection,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=(cx, cy, cz))
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (max(0.004, sx / 2.0), max(0.004, sy / 2.0), max(0.004, sz / 2.0))
    bpy.ops.object.transform_apply(scale=True)
    assign_mat(obj, MATS.get(mat_key) or MATS.get("wall"))
    _link(obj, parent)
    return obj


SOLID_PARAPET_M = 0.20
GLASS_PANEL_M = 0.01


def _rail_thickness(ir: dict[str, Any], kind: str) -> float:
    authored = float((ir.get("output") or {}).get("railing_thickness") or 0.0)
    if kind == "solid":
        return authored if authored > 0.05 else SOLID_PARAPET_M
    if kind == "surface_panel":
        if authored > 0.001 and abs(authored - 0.04) > 1e-9:
            return authored
        return SURFACE_PANEL_M
    if kind == "glass":
        if authored > 0.001 and abs(authored - 0.04) > 1e-9:
            return authored
        return GLASS_PANEL_M
    return authored if authored > 0.005 else 0.04


def _add_solid_rect_parapet(
    *,
    prefix: str,
    cx: float,
    x0: float,
    x1: float,
    y_wall: float,
    depth: float,
    z_lo: float,
    rail_h: float,
    thick: float,
    mat_key: str,
    parent: bpy.types.Collection,
) -> None:
    """U-shaped parapet: front + left/right, open at the wall. Outer faces flush with the slab."""
    slab_w = max(0.2, x1 - x0)
    t = min(thick, slab_w / 3.0, max(0.05, depth) * 0.45)
    cz = z_lo + rail_h / 2.0
    y_front = y_wall + depth
    _cube(
        name=f"{prefix}_rail_f",
        cx=cx,
        cy=y_front - t / 2.0,
        cz=cz,
        sx=slab_w,
        sy=t,
        sz=rail_h,
        mat_key=mat_key,
        parent=parent,
    )
    side_d = max(0.08, depth - t)
    side_cy = y_wall + side_d / 2.0
    for side, sx_c in (("L", x0 + t / 2.0), ("R", x1 - t / 2.0)):
        _cube(
            name=f"{prefix}_rail_{side}",
            cx=sx_c,
            cy=side_cy,
            cz=cz,
            sx=t,
            sy=side_d,
            sz=rail_h,
            mat_key=mat_key,
            parent=parent,
        )


def _add_enclosed_rect_box(
    *,
    prefix: str,
    cx: float,
    x0: float,
    x1: float,
    y_wall: float,
    depth: float,
    z_floor_top: float,
    clear_h: float,
    slab_t: float,
    glass_t: float,
    parent: bpy.types.Collection,
) -> None:
    """Floor already placed. Ceiling + 3-sided glass + 1 m vertical mullions."""
    if x1 < x0:
        x0, x1 = x1, x0
    width = max(0.4, x1 - x0)
    depth = max(0.4, float(depth))
    clear_h = max(1.2, float(clear_h))
    glass_t = max(0.006, float(glass_t))
    mull_w = MULLION_WIDTH_M
    y_front = y_wall + depth
    z_mid = z_floor_top + clear_h / 2.0
    z_ceil = z_floor_top + clear_h + slab_t / 2.0

    _cube(
        name=f"{prefix}_ceil",
        cx=cx,
        cy=y_wall + depth / 2.0,
        cz=z_ceil,
        sx=width,
        sy=depth,
        sz=slab_t,
        mat_key="slab",
        parent=parent,
    )

    for i, (a, b) in enumerate(
        glass_bays(width, MULLION_SPACING_M, mull_w, start_post=True, end_post=True)
    ):
        _cube(
            name=f"{prefix}_glass_f_{i}",
            cx=x0 + 0.5 * (a + b),
            cy=y_front - glass_t / 2.0,
            cz=z_mid,
            sx=b - a,
            sy=glass_t,
            sz=clear_h,
            mat_key="glass",
            parent=parent,
        )
    for side, x_face, inward in (
        ("L", x0, 1.0),
        ("R", x1, -1.0),
    ):
        for i, (a, b) in enumerate(
            glass_bays(
                depth, MULLION_SPACING_M, mull_w, start_post=False, end_post=True
            )
        ):
            _cube(
                name=f"{prefix}_glass_{side}_{i}",
                cx=x_face + inward * glass_t / 2.0,
                cy=y_wall + 0.5 * (a + b),
                cz=z_mid,
                sx=glass_t,
                sy=b - a,
                sz=clear_h,
                mat_key="glass",
                parent=parent,
            )

    for i, s in enumerate(
        grid_stations(width, MULLION_SPACING_M, include_start=True, include_end=True)
    ):
        if s <= 1e-9:
            mx = x0 + mull_w / 2.0
        elif s >= width - 1e-9:
            mx = x1 - mull_w / 2.0
        else:
            mx = x0 + s
        _cube(
            name=f"{prefix}_mull_f_{i}",
            cx=mx,
            cy=y_front - mull_w / 2.0,
            cz=z_mid,
            sx=mull_w,
            sy=mull_w,
            sz=clear_h,
            mat_key="slab",
            parent=parent,
        )
    for side, x_face, inward in (
        ("L", x0, 1.0),
        ("R", x1, -1.0),
    ):
        for i, s in enumerate(
            grid_stations(
                depth, MULLION_SPACING_M, include_start=False, include_end=False
            )
        ):
            _cube(
                name=f"{prefix}_mull_{side}_{i}",
                cx=x_face + inward * mull_w / 2.0,
                cy=y_wall + s,
                cz=z_mid,
                sx=mull_w,
                sy=mull_w,
                sz=clear_h,
                mat_key="slab",
                parent=parent,
            )


def _prism_xy(
    *,
    name: str,
    verts_xy: list[tuple[float, float]],
    z0: float,
    z1: float,
    mat_key: str,
    parent: bpy.types.Collection,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bm = bmesh.new()
    bottom = [bm.verts.new((float(x), float(y), float(z0))) for x, y in verts_xy]
    top = [bm.verts.new((float(x), float(y), float(z1))) for x, y in verts_xy]
    bm.verts.ensure_lookup_table()
    bm.faces.new(bottom)
    bm.faces.new(list(reversed(top)))
    n = len(verts_xy)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((bottom[i], bottom[j], top[j], top[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    assign_mat(obj, MATS.get(mat_key) or MATS.get("wall"))
    parent.objects.link(obj)
    return obj


def _box_along_xy(
    *,
    name: str,
    p0: tuple[float, float],
    p1: tuple[float, float],
    z_lo: float,
    z_hi: float,
    thickness: float,
    mat_key: str,
    parent: bpy.types.Collection,
) -> bpy.types.Object | None:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1e-4:
        return None
    cx = 0.5 * (p0[0] + p1[0])
    cy = 0.5 * (p0[1] + p1[1])
    cz = 0.5 * (z_lo + z_hi)
    bpy.ops.mesh.primitive_cube_add(
        location=(cx, cy, cz),
        rotation=(0.0, 0.0, math.atan2(dy, dx)),
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (
        max(0.004, length / 2.0),
        max(0.004, thickness / 2.0),
        max(0.004, abs(z_hi - z_lo) / 2.0),
    )
    bpy.ops.object.transform_apply(scale=True, rotation=True)
    assign_mat(obj, MATS.get(mat_key) or MATS.get("wall"))
    _link(obj, parent)
    return obj


def _cylinder_between(
    *,
    name: str,
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    radius: float,
    mat_key: str,
    parent: bpy.types.Collection,
    vertices: int = 12,
) -> bpy.types.Object | None:
    a = Vector(p0)
    b = Vector(p1)
    vec = b - a
    length = float(vec.length)
    if length < 1e-4:
        return None
    mid = (a + b) * 0.5
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=int(vertices),
        radius=max(0.004, radius),
        depth=length,
        location=(mid.x, mid.y, mid.z),
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_euler = vec.to_track_quat("Z", "Y").to_euler()
    bpy.ops.object.transform_apply(rotation=True, scale=True)
    assign_mat(obj, MATS.get(mat_key) or MATS.get("railing"))
    _link(obj, parent)
    return obj


def _rail_kind(raw: str) -> str:
    k = (raw or "open_work").strip().lower()
    k = RAIL_ALIASES.get(k, k)
    if k in RAIL_KINDS:
        return k
    if k in ("panel", "parapet", "concrete"):
        return "solid"
    return "open_work"


def _rail_material(raw: Any) -> str:
    m = str(raw or "metal").strip().lower()
    if m in ("masonry", "stone", "concrete_post"):
        return "masonry"
    return "metal"


def _add_baluster_guard(
    *,
    prefix: str,
    outline: list[tuple[float, float]],
    y_wall: float,
    z_slab_top: float,
    rail_h: float,
    parent: bpy.types.Collection,
    mat_key: str = "railing",
    front_only: bool = False,
) -> None:
    """Vertical φ20 mm rods at ~100 mm, φ50 mm top pipe on the outer edges."""
    if len(outline) < 2:
        return
    inset = TOP_RAIL_DIAMETER_M / 2.0
    pipe_r = TOP_RAIL_DIAMETER_M / 2.0
    bal_r = BALUSTER_DIAMETER_M / 2.0
    z_pipe = z_slab_top + max(0.15, rail_h) - pipe_r
    z_bal_top = max(z_slab_top + 0.05, z_pipe - pipe_r)
    if front_only:
        p0, p1 = outline[0], outline[-1]
        toward = (0.5 * (p0[0] + p1[0]), min(p0[1], p1[1]) - 1.0)
        a, b = inset_edge(p0, p1, toward, inset)
        chain = [a, b]
        edges = [(p0, p1)]
        mid = toward
    else:
        mid = centroid(outline)
        chain = [inset_point(p, mid, inset) for p in outline]
        edges = outer_edges(outline, y_wall)
    n = spaced_count(polyline_length(chain), BALUSTER_SPACING_M)
    for i, (x, y) in enumerate(even_along_polyline(chain, n)):
        _cylinder_between(
            name=f"{prefix}_bal_{i}",
            p0=(x, y, z_slab_top),
            p1=(x, y, z_bal_top),
            radius=bal_r,
            mat_key=mat_key,
            parent=parent,
            vertices=10,
        )
    for e, (p0, p1) in enumerate(edges):
        a, b = inset_edge(p0, p1, mid, inset)
        _cylinder_between(
            name=f"{prefix}_toprail_{e}",
            p0=(a[0], a[1], z_pipe),
            p1=(b[0], b[1], z_pipe),
            radius=pipe_r,
            mat_key=mat_key,
            parent=parent,
            vertices=16,
        )


def _add_masonry_openwork_guard(
    *,
    prefix: str,
    outline: list[tuple[float, float]],
    y_wall: float,
    z_slab_top: float,
    rail_h: float,
    parent: bpy.types.Collection,
    mat_key: str = "slab",
    front_only: bool = False,
) -> None:
    """Masonry open_work: 250 mm posts, φ150 balusters (100 mm gaps), 300×150 top rail.

    Builds one inset open-U frame (shared corners) so each corner has a single
    post. Front top rail spans to the slab left/right edges; side rails butt
    the front rail (no corner notch).
    """
    if len(outline) < 2:
        return
    post = float(MASONRY_POST_M)
    half_post = post / 2.0
    bal_r = float(MASONRY_BALUSTER_DIAMETER_M) / 2.0
    pitch = float(MASONRY_BALUSTER_SPACING_M)
    top_w = float(MASONRY_TOP_RAIL_WIDTH_M)
    top_h = float(MASONRY_TOP_RAIL_HEIGHT_M)
    rail_inset = float(MASONRY_RAIL_INSET_M)
    clear_h = max(0.2, float(rail_h) - top_h)
    z_post_top = z_slab_top + clear_h
    z_rail0 = z_post_top
    z_rail1 = z_rail0 + top_h
    half_rail = top_w / 2.0

    corners, edges = masonry_inset_rail_frame(
        outline, y_wall, rail_inset, front_only=front_only
    )
    if not edges:
        return

    for i, (x, y) in enumerate(corners):
        _cube(
            name=f"{prefix}_m_post_{i}",
            cx=x,
            cy=y,
            cz=z_slab_top + clear_h / 2.0,
            sx=post,
            sy=post,
            sz=clear_h,
            mat_key=mat_key,
            parent=parent,
        )

    bal_i = 0
    for e, (p0, p1) in enumerate(edges):
        length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if length < post + 0.05:
            continue
        front = front_only or is_front_edge(p0, p1, y_wall)
        if front:
            # Span inset post centers, then continue to the slab left/right edges.
            rail_seg = extend_segment_ends(
                p0, p1, ext0=rail_inset, ext1=rail_inset
            )
        else:
            # Stop at the inner face of the front rail (plus a hairline gap).
            trim = half_rail + 1e-3
            d0 = abs(p0[1] - y_wall)
            d1 = abs(p1[1] - y_wall)
            if d1 >= d0:
                rail_seg = shorten_segment_end(p0, p1, trim0=0.0, trim1=trim)
            else:
                rail_seg = shorten_segment_end(p0, p1, trim0=trim, trim1=0.0)
        if rail_seg is not None:
            _box_along_xy(
                name=f"{prefix}_m_top_{e}",
                p0=rail_seg[0],
                p1=rail_seg[1],
                z_lo=z_rail0,
                z_hi=z_rail1,
                thickness=top_w,
                mat_key=mat_key,
                parent=parent,
            )

        end_clear = half_post + bal_r + float(MASONRY_BALUSTER_GAP_M) * 0.5
        for x, y in masonry_baluster_centers(
            p0, p1, end_clear=end_clear, pitch=pitch
        ):
            _cylinder_between(
                name=f"{prefix}_m_bal_{bal_i}",
                p0=(x, y, z_slab_top),
                p1=(x, y, z_post_top),
                radius=bal_r,
                mat_key=mat_key,
                parent=parent,
                vertices=16,
            )
            bal_i += 1


def _add_open_work_guard(
    *,
    prefix: str,
    outline: list[tuple[float, float]],
    y_wall: float,
    z_slab_top: float,
    rail_h: float,
    parent: bpy.types.Collection,
    material: str = "metal",
    mat_key: str = "railing",
    front_only: bool = False,
) -> None:
    if _rail_material(material) == "masonry":
        _add_masonry_openwork_guard(
            prefix=prefix,
            outline=outline,
            y_wall=y_wall,
            z_slab_top=z_slab_top,
            rail_h=rail_h,
            parent=parent,
            mat_key="slab",
            front_only=front_only,
        )
        return
    _add_baluster_guard(
        prefix=prefix,
        outline=outline,
        y_wall=y_wall,
        z_slab_top=z_slab_top,
        rail_h=rail_h,
        parent=parent,
        mat_key=mat_key,
        front_only=front_only,
    )


def _add_surface_panel_guard(
    *,
    prefix: str,
    cx: float,
    x0: float,
    x1: float,
    y_wall: float,
    depth: float,
    z_lo: float,
    rail_h: float,
    parent: bpy.types.Collection,
    mat_key: str = "railing",
    panel_t: float = SURFACE_PANEL_M,
    front_only: bool = False,
) -> None:
    """U-shaped 60×60 mm mullion frames with 25 mm infill panels (open at the wall)."""
    del cx
    for part in surface_panel_parts(
        x0=x0,
        x1=x1,
        y_wall=y_wall,
        depth=depth,
        z_lo=z_lo,
        rail_h=rail_h,
        panel_t=panel_t,
        front_only=front_only,
    ):
        _cube(
            name=f"{prefix}_{part['name']}",
            cx=float(part["cx"]),
            cy=float(part["cy"]),
            cz=float(part["cz"]),
            sx=float(part["sx"]),
            sy=float(part["sy"]),
            sz=float(part["sz"]),
            mat_key=mat_key,
            parent=parent,
        )


INSET_LINING_M = 0.16
INSET_VOID_H_DEFAULT = 2.5


def _add_inset_loggia(
    *,
    prefix: str,
    cx: float,
    front_y: float,
    wall_depth: float,
    void_w: float,
    void_d: float,
    void_h: float,
    z_floor: float,
    floor_thick: float,
    parent: bpy.types.Collection,
) -> None:
    """Floor + inner side walls + back wall + ceiling for an inset void.

    The façade slab is only ``wall_depth`` thick, so linings sit *behind* it.
    Total void depth from the front face is ``void_d``.
    """
    lining = INSET_LINING_M
    room_d = max(0.08, void_d - max(0.05, wall_depth))
    y_room_front = front_y - max(0.05, wall_depth)
    cy = y_room_front - room_d / 2.0
    z_mid = z_floor + void_h / 2.0
    _cube(
        name=f"{prefix}_slab",
        cx=cx,
        cy=front_y - void_d / 2.0,
        cz=z_floor - floor_thick / 2.0,
        sx=void_w,
        sy=void_d,
        sz=floor_thick,
        mat_key="slab",
        parent=parent,
    )
    _cube(
        name=f"{prefix}_ceil",
        cx=cx,
        cy=cy,
        cz=z_floor + void_h + lining / 2.0,
        sx=void_w,
        sy=room_d,
        sz=lining,
        mat_key="wall",
        parent=parent,
    )
    _cube(
        name=f"{prefix}_side_L",
        cx=cx - void_w / 2.0 - lining / 2.0,
        cy=cy,
        cz=z_mid,
        sx=lining,
        sy=room_d,
        sz=void_h,
        mat_key="wall",
        parent=parent,
    )
    _cube(
        name=f"{prefix}_side_R",
        cx=cx + void_w / 2.0 + lining / 2.0,
        cy=cy,
        cz=z_mid,
        sx=lining,
        sy=room_d,
        sz=void_h,
        mat_key="wall",
        parent=parent,
    )
    _cube(
        name=f"{prefix}_back",
        cx=cx,
        cy=front_y - void_d - lining / 2.0,
        cz=z_mid,
        sx=void_w,
        sy=lining,
        sz=void_h,
        mat_key="wall",
        parent=parent,
    )


def add_balcony_meshes(
    spec: dict[str, Any],
    wall_obj: bpy.types.Object,
    facade_coll: bpy.types.Collection,
    *,
    cut_opening: Callable[..., None],
) -> int:
    """Slab + railing. Triangle plans use a prism; others stay boxes."""
    library = spec.get("balconies") or {}
    placements = spec.get("balcony_placement") or []
    if not library or not placements:
        return 0

    wall = spec.get("wall") or {}
    front_y = float(wall.get("base_front_y", 0.0))
    wall_depth = float(wall.get("depth", 0.42))
    pp = spec.get("placement_params") or {}
    mirror_x = bool(pp.get("mirror_x", True))
    bottom_m = float(pp.get("bottom_margin_ratio", 0.14))

    balc_coll = bpy.data.collections.new("Balconies")
    facade_coll.children.link(balc_coll)

    n = 0
    for i, rec in enumerate(placements):
        name = str(rec.get("type") or "")
        ir = library.get(name)
        if not isinstance(ir, dict):
            print(f"warn: unknown balcony type {name!r}")
            continue
        row = int(rec.get("row", 0))
        c0 = int(rec.get("col0", rec.get("col", 0)))
        c1 = int(rec.get("col1", c0))
        n_cols = len((spec.get("grid") or {}).get("cols") or [])
        n_rows = len((spec.get("grid") or {}).get("rows") or [])
        if not (0 <= row < n_rows and 0 <= c0 < n_cols and 0 <= c1 < n_cols):
            print(f"warn: balcony placement out of grid {rec}")
            continue

        x0, x1, z0, z1 = _span_xz(spec, row, c0, c1, mirror_x=mirror_x)
        x0, x1, cx = _photo_x_span(
            spec,
            rec,
            mirror_x=mirror_x,
            bay_x0=x0,
            bay_x1=x1,
            row=row,
            col0=c0,
            col1=c1,
        )
        span_w = max(0.3, x1 - x0)
        cell_h = max(0.3, z1 - z0)
        floor = ir.get("floor") or {}
        params = floor.get("params") or {}
        shape = str(floor.get("shape") or "rectangle").lower()
        depth = float(DEFAULT_SLAB_DEPTH_M)
        thick = float((ir.get("output") or {}).get("slab_thickness") or 0.12)
        rail_h = float((ir.get("railing") or {}).get("height") or 1.1)
        kind = _rail_kind(str((ir.get("railing") or {}).get("kind") or "open_work"))
        rail_material = _rail_material((ir.get("railing") or {}).get("material"))
        structure = str(ir.get("structure") or "projecting")
        enclosure = str(ir.get("enclosure") or "open")

        sill_z = z0 + cell_h * bottom_m
        if structure == "free_standing":
            deck_h = float(params.get("height") or 0.0)
            if deck_h > 0.05:
                sill_z = deck_h
        slab_z = sill_z + thick / 2.0
        pad = span_w * 0.01
        bx0, bx1 = x0 + pad, x1 - pad

        prefix = f"Balc_{i}_{name}"
        if kind == "solid" or (kind == "open_work" and rail_material == "masonry"):
            rail_mat = "slab"
        else:
            rail_mat = "railing"
        rail_t = _rail_thickness(ir, kind)
        z_rail0 = sill_z + thick
        z_rail1 = z_rail0 + rail_h

        if structure in ("inset", "composite"):
            void_w = max(0.5, float(params.get("width") or span_w * 0.7))
            void_w = min(void_w, span_w * 0.92)
            void_h = max(1.2, float(params.get("height") or INSET_VOID_H_DEFAULT))
            void_h = min(void_h, max(1.2, cell_h - 0.45))
            vx0 = cx - void_w / 2.0
            vx1 = cx + void_w / 2.0
            z_floor = sill_z
            cut_opening(
                wall_obj,
                x0=vx0,
                z0=z_floor,
                x1=vx1,
                z1=z_floor + void_h,
                front_y=front_y,
                depth=wall_depth,
            )
            _add_inset_loggia(
                prefix=prefix,
                cx=cx,
                front_y=front_y,
                wall_depth=wall_depth,
                void_w=void_w,
                void_d=depth,
                void_h=void_h,
                z_floor=z_floor,
                floor_thick=thick,
                parent=balc_coll,
            )
            if structure == "composite":
                _cube(
                    name=f"{prefix}_slab_out",
                    cx=cx,
                    cy=front_y + depth / 2.0,
                    cz=z_floor - thick / 2.0,
                    sx=void_w,
                    sy=depth,
                    sz=thick,
                    mat_key="slab",
                    parent=balc_coll,
                )
                proj_outline = slab_outline(
                    "rectangle",
                    x0=vx0,
                    x1=vx1,
                    y_wall=front_y,
                    depth=depth,
                    structure="projecting",
                )
                if kind == "open_work":
                    _add_open_work_guard(
                        prefix=prefix,
                        outline=proj_outline,
                        y_wall=front_y,
                        z_slab_top=z_floor,
                        rail_h=rail_h,
                        parent=balc_coll,
                        material=rail_material,
                        mat_key=rail_mat,
                    )
                elif kind == "surface_panel":
                    _add_surface_panel_guard(
                        prefix=prefix,
                        cx=cx,
                        x0=vx0,
                        x1=vx1,
                        y_wall=front_y,
                        depth=depth,
                        z_lo=z_floor,
                        rail_h=rail_h,
                        parent=balc_coll,
                        mat_key=rail_mat,
                        panel_t=rail_t,
                    )
                else:
                    _add_solid_rect_parapet(
                        prefix=prefix,
                        cx=cx,
                        x0=vx0,
                        x1=vx1,
                        y_wall=front_y,
                        depth=depth,
                        z_lo=z_floor,
                        rail_h=rail_h,
                        thick=rail_t,
                        mat_key=rail_mat,
                        parent=balc_coll,
                    )
            elif kind == "open_work":
                _add_open_work_guard(
                    prefix=prefix,
                    outline=[(vx0, front_y), (vx1, front_y)],
                    y_wall=front_y,
                    z_slab_top=z_floor,
                    rail_h=rail_h,
                    parent=balc_coll,
                    material=rail_material,
                    mat_key=rail_mat,
                    front_only=True,
                )
            elif kind == "surface_panel":
                _add_surface_panel_guard(
                    prefix=prefix,
                    cx=cx,
                    x0=vx0,
                    x1=vx1,
                    y_wall=front_y - MULLION_WIDTH_M,
                    depth=MULLION_WIDTH_M,
                    z_lo=z_floor,
                    rail_h=rail_h,
                    parent=balc_coll,
                    mat_key=rail_mat,
                    panel_t=rail_t,
                    front_only=True,
                )
            else:
                _cube(
                    name=f"{prefix}_rail_f",
                    cx=cx,
                    cy=front_y - rail_t / 2.0,
                    cz=z_floor + rail_h / 2.0,
                    sx=void_w,
                    sy=rail_t,
                    sz=rail_h,
                    mat_key=rail_mat,
                    parent=balc_coll,
                )
            n += 1
            continue

        slab_y = front_y + depth / 2.0
        y_wall = front_y

        if shape in {"triangle", "circle"} or (
            enclosure == "enclosed" and shape == "rectangle"
        ):
            tw = float(params.get("width") or params.get("diameter") or 0.0)
            if tw > 0.3:
                tw = min(tw, span_w * 0.98)
                bx0 = cx - tw / 2.0
                bx1 = cx + tw / 2.0

        outline = slab_outline(
            shape if shape in {"triangle", "circle"} else "rectangle",
            x0=bx0,
            x1=bx1,
            y_wall=y_wall,
            depth=depth,
            structure=structure,
        )

        if shape in {"triangle", "circle"}:
            _prism_xy(
                name=f"{prefix}_slab",
                verts_xy=outline,
                z0=sill_z,
                z1=sill_z + thick,
                mat_key="slab",
                parent=balc_coll,
            )
        else:
            sx_slab = (bx1 - bx0) if enclosure == "enclosed" else span_w * 0.98
            _cube(
                name=f"{prefix}_slab",
                cx=cx,
                cy=slab_y,
                cz=slab_z,
                sx=sx_slab,
                sy=depth,
                sz=thick,
                mat_key="slab",
                parent=balc_coll,
            )

        if enclosure == "enclosed":
            if shape == "rectangle":
                _add_enclosed_rect_box(
                    prefix=prefix,
                    cx=cx,
                    x0=bx0,
                    x1=bx1,
                    y_wall=y_wall,
                    depth=depth,
                    z_floor_top=sill_z + thick,
                    clear_h=max(1.8, float(params.get("height") or ENCLOSED_CLEAR_H_M)),
                    slab_t=thick,
                    glass_t=GLASS_PANEL_M,
                    parent=balc_coll,
                )
        elif kind == "open_work":
            _add_open_work_guard(
                prefix=prefix,
                outline=outline,
                y_wall=y_wall,
                z_slab_top=sill_z + thick,
                rail_h=rail_h,
                parent=balc_coll,
                material=rail_material,
                mat_key=rail_mat,
            )
        elif kind == "surface_panel" and shape not in {"triangle", "circle"}:
            sx_slab = span_w * 0.98
            _add_surface_panel_guard(
                prefix=prefix,
                cx=cx,
                x0=cx - sx_slab / 2.0,
                x1=cx + sx_slab / 2.0,
                y_wall=y_wall,
                depth=depth,
                z_lo=sill_z + thick,
                rail_h=rail_h,
                parent=balc_coll,
                mat_key=rail_mat,
                panel_t=rail_t,
            )
        elif shape in {"triangle", "circle"}:
            mid = centroid(outline)
            for e, (p0, p1) in enumerate(outer_edges(outline, y_wall)):
                a, b = inset_edge(p0, p1, mid, rail_t / 2.0)
                _box_along_xy(
                    name=f"{prefix}_rail_{e}",
                    p0=a,
                    p1=b,
                    z_lo=z_rail0,
                    z_hi=z_rail1,
                    thickness=rail_t,
                    mat_key=rail_mat,
                    parent=balc_coll,
                )
        else:
            sx_slab = span_w * 0.98
            _add_solid_rect_parapet(
                prefix=prefix,
                cx=cx,
                x0=cx - sx_slab / 2.0,
                x1=cx + sx_slab / 2.0,
                y_wall=y_wall,
                depth=depth,
                z_lo=sill_z + thick,
                rail_h=rail_h,
                thick=rail_t,
                mat_key=rail_mat,
                parent=balc_coll,
            )

        if structure == "free_standing":
            n_post = max(2, int((ir.get("supports") or {}).get("count") or 2))
            if shape == "triangle":
                mid = centroid(outline)
                chain = support_polyline("triangle", outline, y_wall)
                for p, (px, py) in enumerate(even_along_polyline(chain, n_post)):
                    ipx, ipy = inset_point((px, py), mid, 0.06)
                    _cube(
                        name=f"{prefix}_post_{p}",
                        cx=ipx,
                        cy=ipy,
                        cz=sill_z / 2.0,
                        sx=0.12,
                        sy=0.12,
                        sz=max(0.3, sill_z),
                        mat_key="railing",
                        parent=balc_coll,
                    )
            else:
                n_post = int((ir.get("supports") or {}).get("count") or 0)
                if n_post < 2:
                    n_post = 4
                post_w = 0.20
                sx_slab = span_w * 0.98
                for p, (px, py) in enumerate(
                    rect_corner_posts(
                        cx - sx_slab / 2.0,
                        cx + sx_slab / 2.0,
                        front_y,
                        front_y + depth,
                        n_post,
                        inset=0.18,
                    )
                ):
                    _cube(
                        name=f"{prefix}_post_{p}",
                        cx=px,
                        cy=py,
                        cz=sill_z / 2.0,
                        sx=post_w,
                        sy=post_w,
                        sz=max(0.3, sill_z),
                        mat_key="slab",
                        parent=balc_coll,
                    )
        n += 1

    print(f"compiled balconies: {n} instances")
    return n
