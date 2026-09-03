"""Default door placeholder meshes (solid panel + frame, not window glass)."""

from __future__ import annotations

import bpy

from geometry import assign_mat
from materials import MATS


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
    """Place a simple door: frame ring + solid wood panel + threshold."""
    ox, oy, oz = origin
    ww = max(0.25, float(width))
    hh = max(0.35, float(height))
    left = float(ox)
    center_x = left + ww / 2.0
    frame_t = min(0.09, ww * 0.1, hh * 0.06)
    panel_depth = 0.045
    y_panel = oy - recess - panel_depth * 0.5

    root = _ensure_collection(name_prefix, parent_collection)
    frame_coll = _ensure_collection(f"{name_prefix}_Frame", root)
    panel_coll = _ensure_collection(f"{name_prefix}_Panel", root)

    frame_mat = MATS.get("door_frame", MATS.get("painted_wood", MATS.get("frame")))
    panel_mat = MATS.get("door_wood", MATS.get("painted_wood", MATS.get("frame")))
    thresh_mat = MATS.get("threshold", frame_mat)

    z_mid = oz + hh / 2.0
    z_top = oz + hh
    side_h = max(0.1, hh - frame_t)
    side_z = oz + side_h / 2.0

    def _box(name: str, coll: bpy.types.Collection, loc, scale, mat):
        bpy.ops.mesh.primitive_cube_add(location=loc)
        obj = bpy.context.active_object
        obj.name = name
        obj.scale = scale
        bpy.ops.object.transform_apply(scale=True)
        if mat is not None:
            assign_mat(obj, mat)
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        coll.objects.link(obj)
        return obj

    _box(
        f"{name_prefix}_FrameL",
        frame_coll,
        (left + frame_t / 2.0, oy, side_z),
        (frame_t / 2.0, panel_depth / 2.0, side_h / 2.0),
        frame_mat,
    )
    _box(
        f"{name_prefix}_FrameR",
        frame_coll,
        (left + ww - frame_t / 2.0, oy, side_z),
        (frame_t / 2.0, panel_depth / 2.0, side_h / 2.0),
        frame_mat,
    )
    _box(
        f"{name_prefix}_FrameT",
        frame_coll,
        (center_x, oy, z_top - frame_t / 2.0),
        (ww / 2.0, panel_depth / 2.0, frame_t / 2.0),
        frame_mat,
    )

    # Arch hint: small segment above frame top (placeholder, not boolean arch)
    if str(shape) == "arched" and float(arch_rise_ratio) > 0.02:
        rise = min(hh * 0.25, ww * 0.5, hh * float(arch_rise_ratio) * 4.0)
        bpy.ops.mesh.primitive_uv_sphere_add(
            location=(center_x, oy, z_top + rise * 0.35),
            scale=(ww / 2.0, panel_depth / 2.0, rise),
        )
        arch = bpy.context.active_object
        arch.name = f"{name_prefix}_ArchHint"
        if panel_mat is not None:
            assign_mat(arch, frame_mat)
        for c in list(arch.users_collection):
            c.objects.unlink(arch)
        frame_coll.objects.link(arch)

    inner_w = max(0.1, ww - 2.0 * frame_t)
    inner_h = max(0.1, hh - frame_t)
    _box(
        f"{name_prefix}_Panel",
        panel_coll,
        (center_x, y_panel, oz + inner_h / 2.0),
        (inner_w / 2.0, panel_depth / 2.0, inner_h / 2.0),
        panel_mat,
    )

    floor_z = max(0.0, float(open_to_floor_z))
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
    """Opening aperture for a door (extends down to floor)."""
    z0 = min(float(oz), max(0.0, float(open_to_floor_z)))
    return float(ox), z0, float(ox) + float(ww), float(oz) + float(hh)
