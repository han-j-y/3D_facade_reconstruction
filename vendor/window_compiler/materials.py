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
        "Glass", base_color=(0.18, 0.28, 0.38), roughness=0.03, transmission_weight=0.85
    )
    MATS["wall"] = make_principled_mat(
        "Wall", base_color=(0.78, 0.78, 0.76), roughness=0.85
    )
    if frame_name not in MATS:
        MATS[frame_name] = MATS["frame"]
    return MATS
