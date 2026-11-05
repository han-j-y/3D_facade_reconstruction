"""Blender: build our façade from a recovery DSL (as window_compiler/main.py) and render an oblique orthographic view
like Astra's 'oblique' camera (up and to the right of the façade, ortho scale 1.27 x max(W, H)), transparent film.

blender -b -P oblique_ours.py -- <facade_dsl.json> <out.png> [res]
"""
import math
import os
import sys
from pathlib import Path

ROOT = Path("/local-scratch/localhome/jiyeonh/Research/arch/experiments/facade_e2e/vendor/window_compiler")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402
from load_spec import load_spec  # noqa: E402
from facade_compile import compile_facade_scene  # noqa: E402
from facade_spec import normalize_facade_spec  # noqa: E402
from geometry import clear_scene  # noqa: E402
from materials import init_materials  # noqa: E402
from render_setup import configure_render, setup_camera_for_facade  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
dsl, out = argv[0], argv[1]
res = int(argv[2]) if len(argv) > 2 else 1200

spec = load_spec(Path(dsl))
clear_scene()
init_materials((spec.get("frame") or {}).get("material", "painted_wood"))
configure_render()
facade = normalize_facade_spec(spec, storey_height=3.0, facade_width=None)
summary = compile_facade_scene(facade)
x0, x1, z0, z1 = summary["bounds"]
setup_camera_for_facade(tuple(summary["bounds"]))          # keeps the pipeline's lights / sun
W, H = x1 - x0, z1 - z0
cx, cz = (x0 + x1) / 2, (z0 + z1) / 2

# our wall faces +Y and the front camera looks along -Y, so the viewer's right is -X (photo right, mirror_x)
scene = bpy.context.scene
cam_data = bpy.data.cameras.new("oblique")
cam_data.type = "ORTHO"
cam_data.ortho_scale = max(W, H) * 1.27
cam_data.clip_end = 1000.0
cam = bpy.data.objects.new("oblique", cam_data)
scene.collection.objects.link(cam)
cam.location = (cx - 0.9 * W, 38.0, z0 + 0.83 * H)
target = Vector((cx, 0.0, z0 + 0.48 * H))
cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
scene.camera = cam
scene.render.resolution_x = scene.render.resolution_y = res
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.filepath = str(Path(out).resolve())
bpy.ops.render.render(write_still=True)
print("oblique ->", out)
