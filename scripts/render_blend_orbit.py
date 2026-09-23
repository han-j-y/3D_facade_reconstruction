#!/usr/bin/env python3
"""360° turntable video from a saved façade ``.blend``.

Looks like a clay architectural turntable: gray ground, gray-blue sky,
perspective camera, 4s loop.

Run from the host:

    python scripts/render_blend_orbit.py \\
        --blend runs/batch/_Summary/facade_scene_cmp_x0006.blend

Or inside Blender:

    blender -b facade_scene.blend -P scripts/render_blend_orbit.py -- --out orbit.mp4
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

_SKIP = frozenset(
    {
        "Camera",
        "Sun",
        "Ground",
        "OrbitPivot",
        "LookAt",
        "OrbitFill",
        "OrbitCamera",
        "WindowCam",
        "Key",
    }
)


def _in_blender() -> bool:
    try:
        import bpy  # noqa: F401
    except ImportError:
        return False
    return True


def _argv_tail() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--blend",
        type=Path,
        default=None,
        help="façade .blend (host launcher only; Blender already has the file open)",
    )
    ap.add_argument("--out", type=Path, default=None, help="MP4 output path")
    ap.add_argument("--frames", type=int, default=96, help="frames in the 360° loop")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--res", default="720x960", help="WxH, portrait like the example")
    ap.add_argument("--lens", type=float, default=35.0, help="perspective focal length mm")
    ap.add_argument("--start-angle", type=float, default=22.0, help="degrees; 0 = front +Y")
    ap.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help=">1 moves the camera closer (1.3 ≈ small side margins)",
    )
    ap.add_argument("--preview", action="store_true", help="render frame 1 PNG only")
    ap.add_argument(
        "--blender",
        default=None,
        help="Blender binary (host launcher). Default: $BLENDER or blender on PATH",
    )
    return ap.parse_args(argv if argv is not None else _argv_tail())


def _parse_res(res: str) -> tuple[int, int]:
    a, b = res.lower().split("x", 1)
    return max(16, int(a.strip())), max(16, int(b.strip()))


def _mesh_aabb():
    import bpy
    from mathutils import Vector

    xmin = ymin = zmin = float("inf")
    xmax = ymax = zmax = float("-inf")
    for obj in bpy.context.scene.objects:
        if obj.type not in {"MESH", "CURVE"} or obj.name in _SKIP:
            continue
        mat = obj.matrix_world
        for c in obj.bound_box:
            p = mat @ Vector(c)
            xmin, xmax = min(xmin, float(p.x)), max(xmax, float(p.x))
            ymin, ymax = min(ymin, float(p.y)), max(ymax, float(p.y))
            zmin, zmax = min(zmin, float(p.z)), max(zmax, float(p.z))
    if xmin == float("inf"):
        raise RuntimeError("no mesh objects in blend to orbit")
    return xmin, xmax, ymin, ymax, zmin, zmax


def _diffuse_mat(name: str, color: tuple[float, float, float], roughness: float = 0.85):
    import bpy

    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    principled = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if principled is not None:
        principled.inputs["Base Color"].default_value = (*color, 1.0)
        if "Roughness" in principled.inputs:
            principled.inputs["Roughness"].default_value = roughness
        if "Specular IOR Level" in principled.inputs:
            principled.inputs["Specular IOR Level"].default_value = 0.05
        elif "Specular" in principled.inputs:
            principled.inputs["Specular"].default_value = 0.05
    return mat


def _ensure_ground(zmin: float, span: float):
    import bpy

    if "Ground" in bpy.data.objects:
        ground = bpy.data.objects["Ground"]
    else:
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, min(0.0, zmin) - 0.005))
        ground = bpy.context.active_object
        ground.name = "Ground"
    size = max(80.0, span * 8.0)
    ground.scale = (size, size, 1.0)
    ground.location.z = min(0.0, zmin) - 0.005
    mat = _diffuse_mat("GroundMat", (0.46, 0.47, 0.48), roughness=0.95)
    if ground.data.materials:
        ground.data.materials[0] = mat
    else:
        ground.data.materials.append(mat)
    return ground


def _set_world_sky():
    import bpy

    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    world = scene.world
    color = (0.38, 0.44, 0.54, 1.0)
    strength = 0.85
    try:
        world.use_nodes = True
    except Exception:
        pass
    bg = None
    if getattr(world, "node_tree", None) is not None:
        bg = world.node_tree.nodes.get("Background")
        if bg is None:
            bg = next((n for n in world.node_tree.nodes if n.type == "BACKGROUND"), None)
    if bg is not None:
        bg.inputs["Color"].default_value = color
        bg.inputs["Strength"].default_value = strength
    if hasattr(world, "color"):
        world.color = color[:3]


def _setup_sun(cx: float, cz: float, radius: float, h: float):
    import bpy

    scene = bpy.context.scene
    sun = scene.objects.get("Sun")
    if sun is None or sun.type != "LIGHT":
        data = bpy.data.lights.new("Sun", type="SUN")
        sun = bpy.data.objects.new("Sun", data)
        scene.collection.objects.link(sun)
    sun.data.type = "SUN"
    sun.data.energy = 3.2
    if hasattr(sun.data, "angle"):
        sun.data.angle = 0.02
    sun.location = (cx + h * 0.55, radius * 0.55, cz + h * 0.55)
    sun.rotation_euler = (0.85, 0.15, 0.85)
    if scene.objects.get("OrbitFill") is None:
        fill_data = bpy.data.lights.new("OrbitFill", type="SUN")
        fill = bpy.data.objects.new("OrbitFill", fill_data)
        scene.collection.objects.link(fill)
        fill.location = (cx - h * 0.4, -radius * 0.45, cz + h * 0.2)
        fill.rotation_euler = (1.05, -0.25, -0.7)
        fill.data.energy = 1.1
        if hasattr(fill.data, "angle"):
            fill.data.angle = 0.4


def _setup_engine(scene) -> None:
    for cand in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH"):
        try:
            scene.render.engine = cand
            break
        except Exception:
            continue
    eng = scene.render.engine
    print(f"[orbit] render.engine={eng}", flush=True)
    if eng.startswith("BLENDER_EEVEE"):
        eevee = getattr(scene, "eevee", None)
        if eevee is not None:
            for attr, val in (
                ("taa_render_samples", 8),
                ("use_shadows", True),
                ("use_raytracing", False),
                ("use_volumetric_shadows", False),
            ):
                if hasattr(eevee, attr):
                    try:
                        setattr(eevee, attr, val)
                    except Exception:
                        pass


def _setup_camera_orbit(
    *,
    cx: float,
    cz: float,
    radius: float,
    cam_z: float,
    lens_mm: float,
    n_frames: int,
    fps: int,
    start_angle_deg: float,
):
    import bpy

    scene = bpy.context.scene
    cam = scene.camera
    if cam is None or cam.type != "CAMERA":
        cam_data = bpy.data.cameras.new("OrbitCamera")
        cam = bpy.data.objects.new("OrbitCamera", cam_data)
        scene.collection.objects.link(cam)
        scene.camera = cam

    cam.parent = None
    cam.constraints.clear()
    cam.data.type = "PERSP"
    cam.data.lens = float(lens_mm)
    cam.data.clip_start = 0.05
    cam.data.clip_end = max(400.0, radius * 12.0)
    cam.data.shift_x = 0.0
    cam.data.shift_y = 0.0

    pivot = bpy.data.objects.get("OrbitPivot")
    if pivot is None:
        pivot = bpy.data.objects.new("OrbitPivot", None)
        scene.collection.objects.link(pivot)
    pivot.empty_display_type = "PLAIN_AXES"
    pivot.location = (cx, 0.0, 0.0)
    pivot.rotation_euler = (0.0, 0.0, 0.0)
    pivot.rotation_mode = "XYZ"

    look = bpy.data.objects.get("LookAt")
    if look is None:
        look = bpy.data.objects.new("LookAt", None)
        scene.collection.objects.link(look)
    look.empty_display_type = "SPHERE"
    look.location = (cx, 0.0, cz * 0.48)

    start = math.radians(float(start_angle_deg))
    cam.parent = pivot
    cam.location = (
        radius * math.sin(start),
        radius * math.cos(start),
        cam_z,
    )
    track = cam.constraints.new(type="TRACK_TO")
    track.target = look
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    n_frames = max(8, int(n_frames))
    fps = max(1, int(fps))
    scene.frame_start = 1
    scene.frame_end = n_frames
    scene.render.fps = fps
    scene.frame_set(1)

    try:
        bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    except Exception:
        pass

    if pivot.animation_data and pivot.animation_data.action:
        pivot.animation_data_clear()
    pivot.rotation_euler = (0.0, 0.0, 0.0)
    pivot.keyframe_insert("rotation_euler", frame=1)
    pivot.rotation_euler = (0.0, 0.0, 2.0 * math.pi)
    pivot.keyframe_insert("rotation_euler", frame=n_frames + 1)
    _linearize_fcurves(pivot)

    bpy.context.view_layer.update()
    return cam, pivot


def _linearize_fcurves(obj) -> None:
    """Force linear key interpolation (Blender 4.x fcurves + 5.x slotted actions)."""
    ad = obj.animation_data
    if ad is None or ad.action is None:
        return
    action = ad.action
    curves = []
    if hasattr(action, "fcurves"):
        try:
            curves.extend(list(action.fcurves))
        except Exception:
            pass
    try:
        slot = getattr(ad, "action_slot", None)
        for layer in getattr(action, "layers", []):
            for strip in getattr(layer, "strips", []):
                bag = None
                if hasattr(strip, "channelbag") and slot is not None:
                    try:
                        bag = strip.channelbag(slot)
                    except Exception:
                        bag = None
                if bag is None:
                    bags = getattr(strip, "channelbags", None)
                    if bags:
                        bag = bags[0]
                if bag is not None and hasattr(bag, "fcurves"):
                    curves.extend(list(bag.fcurves))
    except Exception:
        pass
    for fc in curves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def blender_main(args: argparse.Namespace) -> None:
    import bpy

    xmin, xmax, ymin, ymax, zmin, zmax = _mesh_aabb()
    w = max(1e-3, xmax - xmin)
    d = max(1e-3, ymax - ymin)
    h = max(1e-3, zmax - zmin)
    cx = 0.5 * (xmin + xmax)
    cz = 0.5 * (zmin + zmax)
    span = max(w, h, d)

    scene = bpy.context.scene
    rx, ry = _parse_res(args.res)
    scene.render.resolution_x = rx
    scene.render.resolution_y = ry
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.render.image_settings.color_mode = "RGB"

    _set_world_sky()
    _ensure_ground(zmin, span)

    # Distance so the façade width + height fit the portrait frustum with margin.
    aspect = rx / max(ry, 1)
    lens = float(args.lens)
    sensor = 24.0 if aspect < 1.0 else 36.0
    fov = 2.0 * math.atan((sensor * 0.5) / lens)
    if aspect < 1.0:
        hfov = 2.0 * math.atan(math.tan(fov * 0.5) * aspect)
        needed = (w * 0.5) / max(1e-4, math.tan(hfov * 0.5))
        needed_v = (h * 0.72) / max(1e-4, math.tan(fov * 0.5))
        radius = max(needed, needed_v) * 1.08
    else:
        radius = (span * 0.5) / max(1e-4, math.tan(fov * 0.5)) * 1.2
    radius = max(12.0, radius / max(0.1, float(args.zoom)))
    cam_z = max(3.0, h * 0.52)

    _setup_sun(cx, cz, radius, h)
    _setup_engine(scene)
    _setup_camera_orbit(
        cx=cx,
        cz=cz,
        radius=radius,
        cam_z=cam_z,
        lens_mm=lens,
        n_frames=args.frames,
        fps=args.fps,
        start_angle_deg=args.start_angle,
    )

    blend_path = Path(bpy.data.filepath) if bpy.data.filepath else Path.cwd() / "facade.blend"
    out = args.out
    if out is None:
        out = blend_path.with_name(blend_path.stem.replace("facade_scene", "facade_orbit") + ".mp4")
        if out == blend_path.with_suffix(".mp4"):
            out = blend_path.with_name(blend_path.stem + "_orbit.mp4")
    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[orbit] bounds w={w:.2f} h={h:.2f} d={d:.2f} radius={radius:.2f} "
        f"cam_z={cam_z:.2f} frames={args.frames} fps={args.fps} res={rx}x{ry}",
        flush=True,
    )

    if args.preview:
        preview = out.with_suffix(".png")
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(preview)
        scene.frame_set(1)
        bpy.ops.render.render(write_still=True)
        print(f"[orbit] preview → {preview}", flush=True)
        return

    frames_dir = out.parent / f"{out.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(frames_dir / "frame_")
    print(f"[orbit] rendering {args.frames} frames → {frames_dir}", flush=True)
    bpy.ops.render.render(animation=True)
    encoded = encode_png_sequence(frames_dir, out, fps=args.fps)
    if encoded is not None:
        print(f"[orbit] video → {encoded}", flush=True)
    else:
        print(
            f"[orbit] frames ready at {frames_dir} (no ffmpeg in Blender; "
            "encode from the host launcher)",
            flush=True,
        )


def encode_png_sequence(frames_dir: Path, video_out: Path, *, fps: int) -> Path | None:
    """Encode ``frame_*.png`` to H.264 MP4. Returns the video path, or None."""
    import shutil

    frames_dir = Path(frames_dir)
    video_out = Path(video_out)
    pngs = sorted(frames_dir.glob("frame_*.png"))
    if not pngs:
        print(f"[orbit] no frames in {frames_dir}", flush=True)
        return None
    video_out.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = os.environ.get("FACADE_FFMPEG", "").strip() or shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if not ffmpeg:
        return None

    pattern = str(frames_dir / "frame_%04d.png")
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(max(1, int(fps))),
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(video_out),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return video_out


def _resolve_blender(explicit: str | None) -> str:
    import shutil
    from facade_recovery.paths import resolve_blender

    path = resolve_blender(explicit)
    if path:
        return path
    which = shutil.which("blender")
    if which:
        return which
    raise SystemExit("Blender not found. Pass --blender or set $BLENDER.")


def host_main(args: argparse.Namespace) -> None:
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    blend = args.blend
    if blend is None:
        raise SystemExit("--blend is required when launching from Python")
    blend = blend.expanduser().resolve()
    if not blend.is_file():
        raise SystemExit(f"blend not found: {blend}")
    blender = _resolve_blender(args.blender)
    env = os.environ.copy()
    try:
        import imageio_ffmpeg

        env["FACADE_FFMPEG"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    cmd = [
        blender,
        "-b",
        str(blend),
        "-P",
        str(Path(__file__).resolve()),
        "--",
    ]
    passthrough = []
    if args.out:
        passthrough += ["--out", str(Path(args.out).expanduser().resolve())]
    passthrough += ["--frames", str(args.frames), "--fps", str(args.fps), "--res", args.res]
    passthrough += ["--lens", str(args.lens), "--start-angle", str(args.start_angle)]
    passthrough += ["--zoom", str(args.zoom)]
    if args.preview:
        passthrough.append("--preview")
    cmd.extend(passthrough)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)
    out = args.out
    if out is None:
        out = blend.with_name(blend.stem.replace("facade_scene", "facade_orbit") + ".mp4")
    out = Path(out).expanduser().resolve()
    if not out.is_file():
        frames_dir = out.parent / f"{out.stem}_frames"
        encoded = encode_png_sequence(frames_dir, out, fps=args.fps)
        if encoded is None:
            raise SystemExit(f"render finished but could not encode MP4 from {frames_dir}")


if __name__ == "__main__":
    args = parse_args()
    if _in_blender():
        blender_main(args)
    else:
        host_main(args)
