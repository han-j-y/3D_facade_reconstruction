import os

import bpy


MATS = {}


def make_principled_mat(name, base_color=(0.8, 0.8, 0.8), roughness=0.5, transmission_weight=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*base_color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = transmission_weight
    return mat


def make_see_through_mat(name, base_color, roughness=0.05, alpha=0.3):
    """Thin see-through glass: alpha cut-out (Cycles and EEVEE), so what is behind stays visible.

    Separate from ``MATS["glass"]`` (window glass, which the ``real`` render style makes dark and opaque).
    """
    mat = make_principled_mat(name, base_color=base_color, roughness=roughness)
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = alpha
    for attr, val in (("blend_method", "HASHED"), ("shadow_method", "HASHED"), ("surface_render_method", "DITHERED")):
        try:
            setattr(mat, attr, val)
        except (AttributeError, TypeError):
            pass
    return mat


def _env_rgb(name, default):
    """``FACADE_DSL_<name>_RGB="r,g,b"`` (linear 0–1) overrides a material colour."""
    raw = os.environ.get(f"FACADE_DSL_{name}_RGB", "").strip()
    if not raw:
        return default
    try:
        rgb = tuple(float(v) for v in raw.split(","))
        return rgb if len(rgb) == 3 else default
    except ValueError:
        return default


def init_materials(frame_name="painted_wood"):
    global MATS
    MATS.clear()
    MATS["painted_wood"] = make_principled_mat(
        "FrameWood", base_color=_env_rgb("FRAME", (0.92, 0.90, 0.86)), roughness=0.55
    )
    MATS["frame"] = MATS["painted_wood"]
    MATS["muntin"] = make_principled_mat(
        "Muntin", base_color=_env_rgb("MUNTIN", _env_rgb("FRAME", (0.88, 0.86, 0.82))), roughness=0.5
    )
    MATS["glass"] = make_principled_mat(
        "Glass", base_color=_env_rgb("GLASS", (0.18, 0.28, 0.38)), roughness=0.03,
        transmission_weight=float(os.environ.get("FACADE_DSL_GLASS_TRANSMISSION", 0.85))
    )
    MATS["wall"] = make_principled_mat(
        "Wall", base_color=_env_rgb("WALL", (0.78, 0.78, 0.76)), roughness=0.85
    )
    MATS["door_wood"] = make_principled_mat(
        "DoorWood", base_color=_env_rgb("DOOR", (0.42, 0.30, 0.22)), roughness=0.62
    )
    MATS["door_frame"] = MATS["painted_wood"]
    MATS["threshold"] = make_principled_mat(
        "Threshold", base_color=(0.52, 0.50, 0.46), roughness=0.78
    )
    # balcony meshes (balcony_compile, hiroo/balcony-pipeline)
    MATS["slab"] = make_principled_mat(
        "Slab", base_color=(0.62, 0.62, 0.60), roughness=0.75
    )
    MATS["railing"] = make_principled_mat(
        "Railing", base_color=(0.12, 0.12, 0.13), roughness=0.35
    )
    # infill of surface_panel / glass railings (assigned by facade_compile after add_balcony_meshes)
    MATS["railing_glass"] = make_see_through_mat(
        "RailingGlass", base_color=_env_rgb("RAILING_GLASS", (0.55, 0.62, 0.66)),
        alpha=float(os.environ.get("FACADE_DSL_RAILING_GLASS_ALPHA", 0.3)),
    )
    if frame_name not in MATS:
        MATS[frame_name] = MATS["frame"]
    return MATS
