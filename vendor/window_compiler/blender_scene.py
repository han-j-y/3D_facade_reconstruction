"""Export CompileContext to Blender collections and objects."""

from __future__ import annotations

import math

import bpy
from mathutils import Vector

from compiler import output_params
from context import CompileContext
from geom2d import Region, Segment, Vec2
from geometry import assign_mat
from materials import MATS
from shapes import contour_for_region

Origin3 = tuple[float, float, float]


def build_blender_scene(
    ctx: CompileContext,
    *,
    name_prefix: str = "Generated_Window",
    parent_collection: bpy.types.Collection | None = None,
    origin: Origin3 = (0.0, 0.0, 0.0),
) -> bpy.types.Collection:
    """Build glass/frame/muntins for one window.

    ``origin`` translates the window's local (0,0) corner into world space
    (grammar xy → Blender XZ, Y = depth). Distinct ``name_prefix`` values are
    required when placing multiple windows in one scene.
    """
    root_region = ctx.region("root")
    params = output_params(ctx.spec, root_region)
    frame_spec = ctx.spec.get("frame") or {}
    frame_mat_name = frame_spec.get("material", "painted_wood")

    mats = MATS
    parent = parent_collection or bpy.context.scene.collection
    root_coll = _ensure_collection(name_prefix, parent)
    glass_coll = _ensure_collection(f"{name_prefix}_Glass", root_coll)
    frame_coll = _ensure_collection(f"{name_prefix}_Frame", root_coll)
    muntin_coll = _ensure_collection(f"{name_prefix}_Muntins", root_coll)
    debug_coll = None
    if ctx.debug:
        debug_coll = _ensure_collection(f"{name_prefix}_Debug", root_coll)

    glass_region_id = _glass_region_id(ctx)
    glass_region = ctx.regions.get(glass_region_id, root_region)

    _build_glass(
        glass_region,
        root_region,
        params,
        glass_coll,
        mats["glass"],
        name_prefix=name_prefix,
        origin=origin,
    )
    _build_frame(
        root_region,
        glass_region,
        params,
        frame_coll,
        mats.get(frame_mat_name, mats["frame"]),
        name_prefix=name_prefix,
        origin=origin,
    )
    _build_muntins(
        ctx.segments,
        params,
        muntin_coll,
        mats["muntin"],
        name_prefix=name_prefix,
        origin=origin,
    )

    if debug_coll is not None:
        _build_debug_anchors(ctx, debug_coll, name_prefix=name_prefix, origin=origin)

    return root_coll


def _glass_region_id(ctx: CompileContext) -> str:
    if "glass" in ctx.regions:
        return "glass"
    for rid in ctx.regions:
        if rid != "root":
            return rid
    return "root"


def _ensure_collection(name: str, parent: bpy.types.Collection) -> bpy.types.Collection:
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        parent.children.link(coll)
    elif coll.name not in [c.name for c in parent.children]:
        parent.children.link(coll)
    return coll


def _link_mesh(obj, coll, mat):
    assign_mat(obj, mat)
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)


def _link_object(obj, coll) -> None:
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)


def _offset(p: Vec2, origin: Origin3) -> tuple[float, float, float]:
    ox, oy, oz = origin
    return (p.x + ox, oy, p.y + oz)


def _build_glass(
    inner: Region,
    outer: Region,
    params: dict[str, float],
    coll,
    mat,
    *,
    name_prefix: str,
    origin: Origin3,
) -> None:
    contour = contour_for_region(inner, outer)
    depth = _glass_extrusion_depth(params)
    _link_mesh(
        _mesh_from_contour(f"{name_prefix}_Glass", contour, depth, origin=origin),
        coll,
        mat,
    )


def _glass_extrusion_depth(params: dict[str, float]) -> float:
    return params["frame_depth"]


def _muntin_front_y(params: dict[str, float], origin: Origin3) -> float:
    return origin[1] + params["muntin_width"] / 2.0


def _extrude_faces_into_wall(bm, depth: float) -> None:
    """Exterior façade at local Y=0 (toward camera at +Y); depth into wall (-Y)."""
    import bmesh

    _ensure_faces_normal_plus_y(bm, list(bm.faces))
    geom = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
    verts = [v for v in geom["geom"] if isinstance(v, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, verts=verts, vec=(0.0, -depth, 0.0))


def _build_frame(
    outer: Region,
    inner: Region,
    params: dict[str, float],
    coll: bpy.types.Collection,
    mat,
    *,
    name_prefix: str,
    origin: Origin3,
) -> None:
    depth = params["frame_depth"]
    oc = contour_for_region(outer)
    ic = contour_for_region(inner, outer)
    _link_mesh(
        _ring_from_contours(f"{name_prefix}_Frame", oc, ic, depth, origin=origin),
        coll,
        mat,
    )


def _ensure_faces_normal_plus_y(bm, faces) -> None:
    import bmesh

    bm.normal_update()
    if faces and faces[0].normal.y < 0:
        bmesh.ops.reverse_faces(bm, faces=faces)


def _mesh_from_contour(
    name: str,
    contour: list[Vec2],
    depth: float,
    *,
    origin: Origin3 = (0.0, 0.0, 0.0),
):
    import bmesh

    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bm = bmesh.new()
    if len(contour) < 3:
        bm.free()
        return obj
    rim = [bm.verts.new(_offset(p, origin)) for p in contour]
    face = bm.faces.new(rim)
    bmesh.ops.triangulate(bm, faces=[face])
    _extrude_faces_into_wall(bm, depth)
    bm.to_mesh(mesh)
    bm.free()
    return obj


def _ring_from_contours(
    name: str,
    outer: list[Vec2],
    inner: list[Vec2],
    depth: float,
    *,
    origin: Origin3 = (0.0, 0.0, 0.0),
):
    import bmesh

    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bm = bmesh.new()
    n = len(outer)
    inner_r = _resample_contour(inner, n)
    for i in range(n):
        j = (i + 1) % n
        o0, o1 = outer[i], outer[j]
        i0, i1 = inner_r[i], inner_r[j]
        verts = [
            bm.verts.new(_offset(o0, origin)),
            bm.verts.new(_offset(o1, origin)),
            bm.verts.new(_offset(i1, origin)),
            bm.verts.new(_offset(i0, origin)),
        ]
        bm.faces.new(verts)
    _extrude_faces_into_wall(bm, depth)
    bm.to_mesh(mesh)
    bm.free()
    return obj


def _resample_contour(contour: list[Vec2], n: int) -> list[Vec2]:
    if len(contour) == n:
        return list(contour)
    if len(contour) < 2:
        return list(contour)
    lengths = []
    total = 0.0
    for i in range(len(contour)):
        a = contour[i]
        b = contour[(i + 1) % len(contour)]
        seg_len = math.hypot(b.x - a.x, b.y - a.y)
        lengths.append(seg_len)
        total += seg_len
    if total < 1e-9:
        return [contour[0]] * n
    out: list[Vec2] = []
    step = total / n
    dist = 0.0
    edge = 0
    edge_pos = 0.0
    for _ in range(n):
        target = dist
        while edge < len(lengths) and edge_pos + lengths[edge] < target + 1e-9:
            edge_pos += lengths[edge]
            edge += 1
        if edge >= len(lengths):
            edge = len(lengths) - 1
        a = contour[edge]
        b = contour[(edge + 1) % len(contour)]
        t = (target - edge_pos) / max(lengths[edge], 1e-9)
        out.append(Vec2(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)))
        dist += step
    return out


def _build_muntins(
    segments: list[Segment],
    params: dict[str, float],
    coll,
    mat,
    *,
    name_prefix: str,
    origin: Origin3,
) -> None:
    width = params["muntin_width"]
    depth = params["muntin_depth"]
    y = _muntin_front_y(params, origin)
    half = width / 2.0
    ox, _, oz = origin
    for seg in _merge_collinear_segments(segments):
        seg = _extend_segment_caps(seg, half)
        obj = _segment_curve_object(
            seg,
            width,
            depth,
            y,
            name=f"{name_prefix}_Muntin_{seg.id}",
            offset_x=ox,
            offset_z=oz,
        )
        _link_mesh(obj, coll, mat)


def _merge_collinear_segments(segments: list[Segment], eps: float = 1e-4) -> list[Segment]:
    buckets: dict[tuple[str, float], list[Segment]] = {}
    rest: list[Segment] = []
    for seg in segments:
        dx = seg.b.x - seg.a.x
        dy = seg.b.y - seg.a.y
        if abs(dy) < eps:
            buckets.setdefault(("h", round(seg.a.y, 6)), []).append(seg)
        elif abs(dx) < eps:
            buckets.setdefault(("v", round(seg.a.x, 6)), []).append(seg)
        else:
            rest.append(seg)

    out = list(rest)
    for (axis, coord), segs in buckets.items():
        out.extend(_merge_intervals(axis, coord, segs, eps))
    return out


def _merge_intervals(axis: str, coord: float, segs: list[Segment], eps: float) -> list[Segment]:
    intervals: list[tuple[float, float, str]] = []
    for s in segs:
        if axis == "h":
            intervals.append((min(s.a.x, s.b.x), max(s.a.x, s.b.x), s.id))
        else:
            intervals.append((min(s.a.y, s.b.y), max(s.a.y, s.b.y), s.id))
    intervals.sort()
    merged: list[tuple[float, float, str]] = []
    for lo, hi, sid in intervals:
        if merged and lo <= merged[-1][1] + eps:
            prev = merged[-1]
            merged[-1] = (prev[0], max(prev[1], hi), prev[2])
        else:
            merged.append((lo, hi, sid))
    out: list[Segment] = []
    for lo, hi, sid in merged:
        if axis == "h":
            out.append(Segment(sid, Vec2(lo, coord), Vec2(hi, coord)))
        else:
            out.append(Segment(sid, Vec2(coord, lo), Vec2(coord, hi)))
    return out


def _extend_segment_caps(seg: Segment, half_width: float) -> Segment:
    dx = seg.b.x - seg.a.x
    dy = seg.b.y - seg.a.y
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return seg
    ux, uy = dx / length, dy / length
    return Segment(
        seg.id,
        Vec2(seg.a.x - ux * half_width, seg.a.y - uy * half_width),
        Vec2(seg.b.x + ux * half_width, seg.b.y + uy * half_width),
        layer=seg.layer,
        clip_region_id=seg.clip_region_id,
    )


def _segment_curve_object(
    seg: Segment,
    width: float,
    depth: float,
    y: float,
    *,
    name: str | None = None,
    offset_x: float = 0.0,
    offset_z: float = 0.0,
):
    curve = bpy.data.curves.new(name or seg.id, type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = width / 2.0
    curve.bevel_resolution = 4
    spline = curve.splines.new("POLY")
    spline.points.add(1)
    a = Vector((seg.a.x + offset_x, y, seg.a.y + offset_z))
    b = Vector((seg.b.x + offset_x, y, seg.b.y + offset_z))
    spline.points[0].co = (*a, 1.0)
    spline.points[1].co = (*b, 1.0)
    obj = bpy.data.objects.new(name or f"Muntin_{seg.id}", curve)
    return obj


def _build_debug_anchors(
    ctx: CompileContext,
    coll,
    *,
    name_prefix: str,
    origin: Origin3,
) -> None:
    ox, oy, oz = origin
    y = oy + 0.04
    for aid, anchor in ctx.anchors.items():
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.015,
            location=(anchor.position.x + ox, y, anchor.position.y + oz),
        )
        obj = bpy.context.active_object
        obj.name = f"{name_prefix}_Anchor_{aid}"
        _link_object(obj, coll)
