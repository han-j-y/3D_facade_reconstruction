"""Default door placeholder meshes (solid panel + frame, not window glass)."""

from __future__ import annotations

import math

import bpy
import bmesh

from geometry import assign_mat
from materials import MATS

ARCH_SEGMENTS = 32


def effective_arch_rise(width: float, height: float, shape: str, arch_rise_ratio: float) -> float:
    """Arch rise (m) for a door of ``width``; 0 for rectangular doors.

    ``arch_rise_ratio`` is rise/width measured on the photo crop, which reads
    roughly half the true rise (the crop edge clips the top of the arch), so it
    is doubled and snapped to a semicircle when close.
    """
    if str(shape) != "arched" or float(arch_rise_ratio) <= 0.02:
        return 0.0
    r = min(0.5, 2.0 * float(arch_rise_ratio))
    if r >= 0.42:
        r = 0.5
    return min(r * float(width), 0.6 * float(height))


def _arch_frame(width: float, spring_z: float, rise: float) -> tuple[float, float, float]:
    """Circle (cx, cz, R) of a segmental arch over chord [0, width] at ``spring_z``."""
    R = (width * width / 4.0 + rise * rise) / (2.0 * rise)
    return width / 2.0, spring_z + rise - R, R


def door_profile(
    width: float,
    height: float,
    rise: float,
    *,
    inset: float = 0.0,
    segments: int = ARCH_SEGMENTS,
) -> list[tuple[float, float]]:
    """Open polyline (x, z) up the left jamb, over the head, down the right jamb.

    Local coords: x in [0, width], z in [0, height]; bottom edge is open. With
    ``inset`` > 0 the profile is offset inward by that amount (concentric arch),
    keeping the same vertex count as ``inset=0`` so rings can be bridged.
    """
    w, h, t = float(width), float(height), float(inset)
    if rise <= 1e-4:
        return [(t, 0.0), (t, h - t), (w - t, h - t), (w - t, 0.0)]
    cx, cz, R = _arch_frame(w, h - rise, rise)
    Ri = max(R - t, 1e-4)
    half = max(w / 2.0 - t, 1e-4)
    a = math.acos(max(-1.0, min(1.0, half / Ri)))
    pts = [(t, 0.0)]
    for i in range(segments + 1):
        th = (math.pi - a) - (math.pi - 2.0 * a) * i / segments
        pts.append((cx + Ri * math.cos(th), cz + Ri * math.sin(th)))
    pts.append((w - t, 0.0))
    return pts


def _link_obj(obj: bpy.types.Object, coll: bpy.types.Collection, mat) -> bpy.types.Object:
    if mat is not None:
        assign_mat(obj, mat)
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)
    return obj


def _mesh_obj(name: str, verts, faces, coll, mat) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return _link_obj(obj, coll, mat)


def make_prism(
    name: str,
    poly_xz: list[tuple[float, float]],
    y0: float,
    y1: float,
    coll: bpy.types.Collection,
    mat=None,
) -> bpy.types.Object:
    """Extrude a closed convex XZ polygon between depths ``y0`` and ``y1``."""
    n = len(poly_xz)
    verts = [(x, y1, z) for x, z in poly_xz] + [(x, y0, z) for x, z in poly_xz]
    faces = [list(range(n)), list(range(2 * n - 1, n - 1, -1))]
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j, n + i])
    return _mesh_obj(name, verts, faces, coll, mat)


def _make_ring(
    name: str,
    outer: list[tuple[float, float]],
    inner: list[tuple[float, float]],
    y0: float,
    y1: float,
    coll: bpy.types.Collection,
    mat=None,
) -> bpy.types.Object:
    """Bridge two open XZ polylines of equal length into a solid frame band."""
    n = len(outer)
    verts = (
        [(x, y1, z) for x, z in outer]
        + [(x, y1, z) for x, z in inner]
        + [(x, y0, z) for x, z in outer]
        + [(x, y0, z) for x, z in inner]
    )
    OF, IF, OB, IB = 0, n, 2 * n, 3 * n
    faces = []
    for i in range(n - 1):
        j = i + 1
        faces.append([OF + i, OF + j, IF + j, IF + i])
        faces.append([OB + i, IB + i, IB + j, OB + j])
        faces.append([OF + i, OB + i, OB + j, OF + j])
        faces.append([IF + i, IF + j, IB + j, IB + i])
    for i in (0, n - 1):
        faces.append([OF + i, IF + i, IB + i, OB + i])
    return _mesh_obj(name, verts, faces, coll, mat)


def door_outline_world(
    *,
    ox: float,
    z_base: float,
    width: float,
    top_z: float,
    rise: float,
    pad: float = 0.0,
) -> list[tuple[float, float]]:
    """Closed world XZ outline of a door opening (for wall boolean cutters)."""
    w = float(width) + 2.0 * pad
    h = float(top_z) - float(z_base) + pad
    prof = door_profile(w, h, rise + pad if rise > 0 else 0.0)
    return [(ox - pad + x, z_base + z) for x, z in prof]


def _ensure_collection(name: str, parent: bpy.types.Collection) -> bpy.types.Collection:
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        parent.children.link(coll)
    return coll


def build_default_door(
    *,
    name_prefix: str,
    parent_collection: bpy.types.Collection,
    origin: tuple[float, float, float],
    width: float,
    height: float,
    shape: str = "rectangle",
    arch_rise_ratio: float = 0.0,
    recess: float = 0.02,
    open_to_floor_z: float = 0.0,
) -> bpy.types.Collection:
    """Place a door filling its opening: frame + wood panel + threshold.

    The door spans from the opening floor (``open_to_floor_z``) to the top of
    the detected box; arched doors keep the arch inside that box.
    """
    ox, oy, oz = origin
    ww = max(0.25, float(width))
    top_z = float(oz) + max(0.35, float(height))
    z_base = max(0.0, min(float(oz), float(open_to_floor_z)))
    hh = top_z - z_base
    left = float(ox)
    center_x = left + ww / 2.0
    frame_t = min(0.09, ww * 0.1, hh * 0.06)
    panel_depth = 0.045
    y_panel = oy - recess - panel_depth * 0.5
    rise = effective_arch_rise(ww, hh, shape, arch_rise_ratio)

    root = _ensure_collection(name_prefix, parent_collection)
    frame_coll = _ensure_collection(f"{name_prefix}_Frame", root)
    panel_coll = _ensure_collection(f"{name_prefix}_Panel", root)

    frame_mat = MATS.get("door_frame", MATS.get("painted_wood", MATS.get("frame")))
    panel_mat = MATS.get("door_wood", MATS.get("painted_wood", MATS.get("frame")))
    thresh_mat = MATS.get("threshold", frame_mat)

    def _box(name: str, coll: bpy.types.Collection, loc, scale, mat):
        bpy.ops.mesh.primitive_cube_add(location=loc)
        obj = bpy.context.active_object
        obj.name = name
        obj.scale = scale
        bpy.ops.object.transform_apply(scale=True)
        return _link_obj(obj, coll, mat)

    def _world(pts):
        return [(left + x, z_base + z) for x, z in pts]

    outer = door_profile(ww, hh, rise)
    inner = door_profile(ww, hh, rise, inset=frame_t)
    _make_ring(
        f"{name_prefix}_Frame",
        _world(outer),
        _world(inner),
        oy - panel_depth / 2.0,
        oy + panel_depth / 2.0,
        frame_coll,
        frame_mat,
    )

    y0p, y1p = y_panel - panel_depth / 2.0, y_panel + panel_depth / 2.0
    make_prism(f"{name_prefix}_Panel", _world(inner), y0p, y1p, panel_coll, panel_mat)

    floor_z = z_base
    _box(
        f"{name_prefix}_Threshold",
        panel_coll,
        (center_x, oy + 0.02, floor_z + 0.02),
        (ww * 0.55 / 2.0, 0.05, 0.02),
        thresh_mat,
    )

    return root


def door_opening_bounds(
    *,
    ox: float,
    oz: float,
    ww: float,
    hh: float,
    open_to_floor_z: float = 0.0,
) -> tuple[float, float, float, float]:
    """Opening aperture for a door (extends down to floor, never below the wall)."""
    z0 = max(0.0, min(float(oz), float(open_to_floor_z)))
    return float(ox), z0, float(ox) + float(ww), float(oz) + float(hh)
