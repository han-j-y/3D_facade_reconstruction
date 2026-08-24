import bpy


MATS = {}


def make_principled_mat(
    name,
    base_color=(0.8, 0.8, 0.8),
    roughness=0.5,
    transmission_weight=0.0,
    alpha=1.0,
    ior=1.45,
):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*base_color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = transmission_weight
    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = alpha
    if "IOR" in bsdf.inputs:
        bsdf.inputs["IOR"].default_value = ior
    translucent = transmission_weight > 0.0 or alpha < 0.999
    if translucent:
        try:
            mat.blend_method = "HASHED"
        except (AttributeError, TypeError):
            pass
        try:
            mat.shadow_method = "HASHED"
        except (AttributeError, TypeError):
            pass
        try:
            mat.use_screen_refraction = True
            mat.refraction_depth = 0.01
        except (AttributeError, TypeError):
            pass
        try:
            mat.surface_render_method = "DITHERED"
        except (AttributeError, TypeError):
            pass
    return mat


def init_materials(frame_name="painted_wood"):
    global MATS
    MATS.clear()
    MATS["painted_wood"] = make_principled_mat(
        "FrameWood", base_color=(0.92, 0.90, 0.86), roughness=0.55
    )
    MATS["frame"] = MATS["painted_wood"]
    MATS["muntin"] = make_principled_mat(
        "Muntin", base_color=(0.88, 0.86, 0.82), roughness=0.5
    )
    MATS["glass"] = make_principled_mat(
        "Glass",
        base_color=(0.62, 0.70, 0.76),
        roughness=0.06,
        transmission_weight=0.92,
        alpha=0.38,
    )
    MATS["wall"] = make_principled_mat(
        "Wall", base_color=(0.78, 0.78, 0.76), roughness=0.85
    )
    MATS["slab"] = make_principled_mat(
        "Slab", base_color=(0.62, 0.62, 0.60), roughness=0.75
    )
    MATS["railing"] = make_principled_mat(
        "Railing", base_color=(0.12, 0.12, 0.13), roughness=0.35
    )
    if frame_name not in MATS:
        MATS[frame_name] = MATS["frame"]
    return MATS
