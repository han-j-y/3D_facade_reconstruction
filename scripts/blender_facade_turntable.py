#!/usr/bin/env python3
"""Blender-side turntable renderer for a compiled façade spec.

Builds the façade scene with the vendored ``window_compiler``, then flies a
perspective camera around it and writes a PNG frame sequence. The ortho render
used elsewhere reads as a flat drawing; the moving parallax here shows the wall
slab, the cut openings and the recessed window frames as actual 3D.

Run inside Blender::

  blender -b -P scripts/blender_facade_turntable.py -- \\
    runs/.../blender/facade_compile.json --out /tmp/frames --frames 96
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
COMPILER_ROOT = Path(
    os.environ.get("FACADE_COMPILER_ROOT")
    or _ROOT / "vendor" / "window_compiler"
).resolve()
if str(COMPILER_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPILER_ROOT))

from facade_compile import (  # noqa: E402
    compile_facade_scene,
    is_facade_spec,
    normalize_facade_spec,
)
from facade_spec import total_grid_size  # noqa: E402
from geometry import clear_scene  # noqa: E402
from load_spec import load_spec  # noqa: E402
from materials import init_materials, make_principled_mat  # noqa: E402

SENSOR_MM = 36.0
# Frame aspect is matched to the façade so wide walls don't drown in sky.
ASPECT_MIN, ASPECT_MAX = 0.60, 2.40
LONG_SIDE = 1152


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("spec", type=Path, help="facade_compile.json (or recovery DSL)")
    ap.add_argument("--out", type=Path, required=True, help="frame output directory")
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument(
        "--res", default="auto", help="WxH, or 'auto' to match the façade aspect"
    )
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument(
        "--motion",
        choices=("sweep", "orbit"),
        default="sweep",
        help="sweep = ping-pong arc in front of the façade; orbit = full 360°",
    )
    ap.add_argument("--yaw-deg", type=float, default=52.0, help="sweep half-angle")
    ap.add_argument("--elev-deg", type=float, default=9.0)
    ap.add_argument("--elev-bob-deg", type=float, default=4.0)
    ap.add_argument("--lens", type=float, default=42.0)
    ap.add_argument("--margin", type=float, default=1.06, help="framing headroom")
    ap.add_argument(
        "--wall-depth",
        type=float,
        default=None,
        help="override wall slab depth in metres (thicker = more readable 3D)",
    )
    ap.add_argument("--storey-height", type=float, default=3.0)
    ap.add_argument("--facade-width", type=float, default=None)
    ap.add_argument("--no-ground", action="store_true")
    return ap.parse_args(argv)


def resolve_res(res: str, *, width: float, height: float) -> tuple[int, int]:
    """Frame size: explicit ``WxH``, or an aspect matched to the façade."""
    if res.strip().lower() != "auto":
        rx, ry = (int(v) for v in res.lower().split("x", 1))
    else:
        aspect = min(ASPECT_MAX, max(ASPECT_MIN, width / max(height, 1e-3)))
        if aspect >= 1.0:
            rx, ry = LONG_SIDE, round(LONG_SIDE / aspect)
        else:
            rx, ry = round(LONG_SIDE * aspect), LONG_SIDE
    return max(320, rx - rx % 2), max(320, ry - ry % 2)


def setup_cycles(samples: int, res: tuple[int, int]) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == kind]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == kind
            scene.cycles.device = "GPU"
            print(f"[turntable] cycles {kind}: {[d.name for d in gpus]}", flush=True)
            break
    else:
        scene.cycles.device = "CPU"

    scene.cycles.samples = max(8, int(samples))
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.max_bounces = 6
    scene.cycles.transmission_bounces = 8

    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"


def setup_environment(bounds: tuple[float, float, float, float], *, ground: bool) -> None:
    """Sky-ish world, a key/fill sun pair, and an optional shadow-catching ground."""
    xmin, xmax, zmin, zmax = bounds
    w = max(1e-3, xmax - xmin)
    h = max(1e-3, zmax - zmin)
    cx = 0.5 * (xmin + xmax)
    scene = bpy.context.scene

    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg is not None:
        # Dim, cool surround so the pale façade reads with contrast.
        bg.inputs["Color"].default_value = (0.26, 0.33, 0.45, 1.0)
        bg.inputs["Strength"].default_value = 0.55

    key_data = bpy.data.lights.new("TurntableKey", type="SUN")
    key_data.energy = 4.0
    key_data.color = (1.0, 0.96, 0.9)
    key_data.angle = math.radians(2.5)
    key = bpy.data.objects.new("TurntableKey", key_data)
    scene.collection.objects.link(key)
    key.rotation_euler = (math.radians(58.0), 0.0, math.radians(38.0))

    fill_data = bpy.data.lights.new("TurntableFill", type="SUN")
    fill_data.energy = 1.1
    fill_data.angle = math.radians(20.0)
    fill = bpy.data.objects.new("TurntableFill", fill_data)
    scene.collection.objects.link(fill)
    fill.rotation_euler = (math.radians(72.0), 0.0, math.radians(-125.0))

    if not ground:
        return
    span = max(w, h) * 6.0
    bpy.ops.mesh.primitive_plane_add(size=span, location=(cx, 0.0, zmin - 0.01))
    plane = bpy.context.active_object
    plane.name = "TurntableGround"
    plane.data.materials.append(
        make_principled_mat("TurntableGround", base_color=(0.16, 0.17, 0.19), roughness=0.95)
    )


def place_camera(
    cam: bpy.types.Object,
    *,
    center: Vector,
    radius: float,
    azimuth: float,
    elevation: float,
) -> None:
    """Camera on a sphere around ``center``; ``azimuth=0`` faces the façade front (+Y)."""
    horiz = radius * math.cos(elevation)
    cam.location = (
        center.x + horiz * math.sin(azimuth),
        center.y + horiz * math.cos(azimuth),
        center.z + radius * math.sin(elevation),
    )
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()


def fit_radius(
    cam: bpy.types.Object,
    *,
    center: Vector,
    corners: list[Vector],
    poses: list[tuple[float, float]],
    lens: float,
    margin: float,
    width: float,
    height: float,
) -> float:
    """Smallest orbit radius keeping the façade in frame at every camera pose.

    Projected size scales ~1/distance, so rescaling by the worst overshoot
    converges in a few passes — and it stays correct whatever ``sensor_fit``
    and frame aspect Blender ends up using.
    """
    scene = bpy.context.scene
    half_fov = math.atan(SENSOR_MM / (2.0 * lens))
    radius = max(width, height) / (2.0 * math.tan(half_fov))
    for _ in range(16):
        worst = 0.0
        for azimuth, elevation in poses:
            place_camera(
                cam,
                center=center,
                radius=radius,
                azimuth=azimuth,
                elevation=elevation,
            )
            bpy.context.view_layer.update()
            for p in corners:
                v = world_to_camera_view(scene, cam, p)
                worst = max(worst, abs(v.x - 0.5) * 2.0, abs(v.y - 0.5) * 2.0)
        scale = worst * margin
        radius *= scale
        if 0.995 <= scale <= 1.005:
            break
    return radius


def main() -> None:
    args = parse_args()
    spec_path = args.spec.expanduser().resolve()
    spec = load_spec(spec_path)
    if not is_facade_spec(spec):
        raise SystemExit(f"not a façade spec: {spec_path}")

    clear_scene()
    init_materials((spec.get("frame") or {}).get("material", "painted_wood"))

    facade = normalize_facade_spec(
        spec, storey_height=args.storey_height, facade_width=args.facade_width
    )
    wall = facade.setdefault("wall", {})
    if args.wall_depth is not None:
        wall["depth"] = float(args.wall_depth)
    wall_depth = float(wall.get("depth", 0.42))
    front_y = float(wall.get("base_front_y", 0.0))
    grid_w, grid_h = total_grid_size(facade["grid"])
    setup_cycles(args.samples, resolve_res(args.res, width=grid_w, height=grid_h))

    summary = compile_facade_scene(facade)
    xmin, xmax, zmin, zmax = (float(v) for v in summary["bounds"])
    width = max(1e-3, xmax - xmin)
    height = max(1e-3, zmax - zmin)
    # Orbit the slab's mid-depth so a full turntable doesn't wobble around the front face.
    center = Vector(
        (0.5 * (xmin + xmax), front_y - 0.5 * wall_depth, 0.5 * (zmin + zmax))
    )

    setup_environment((xmin, xmax, zmin, zmax), ground=not args.no_ground)

    cam_data = bpy.data.cameras.new("TurntableCam")
    cam_data.type = "PERSP"
    cam_data.lens = float(args.lens)
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = SENSOR_MM
    cam_data.clip_start = 0.05
    cam_data.clip_end = 10_000.0
    cam = bpy.data.objects.new("TurntableCam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    elev = math.radians(args.elev_deg)
    bob = math.radians(args.elev_bob_deg)
    yaw = math.radians(args.yaw_deg)
    azimuths = (
        [i * math.pi / 4.0 for i in range(8)]
        if args.motion == "orbit"
        else [-yaw, 0.0, yaw]
    )
    poses = [(a, e) for a in azimuths for e in (elev - bob, elev, elev + bob)]
    corners = [
        Vector((x, y, z))
        for x in (xmin, xmax)
        for y in (front_y - wall_depth - 0.2, front_y + 0.2)
        for z in (zmin, zmax)
    ]
    radius = fit_radius(
        cam,
        center=center,
        corners=corners,
        poses=poses,
        lens=args.lens,
        margin=args.margin,
        width=width,
        height=height,
    )

    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    total = max(2, int(args.frames))
    scene = bpy.context.scene
    print(
        f"[turntable] {spec_path.name} {summary['n_windows']} windows "
        f"{width:.1f}×{height:.1f}m radius={radius:.1f} motion={args.motion} "
        f"frames={total} res={scene.render.resolution_x}x{scene.render.resolution_y}",
        flush=True,
    )

    for fi in range(total):
        t = fi / total
        if args.motion == "orbit":
            azimuth = 2.0 * math.pi * t
            elevation = elev
        else:
            azimuth = math.radians(args.yaw_deg) * math.sin(2.0 * math.pi * t)
            elevation = elev + math.radians(args.elev_bob_deg) * math.sin(
                4.0 * math.pi * t
            )
        place_camera(
            cam, center=center, radius=radius, azimuth=azimuth, elevation=elevation
        )
        bpy.context.view_layer.update()
        scene.render.filepath = str(out_dir / f"frame_{fi + 1:04d}.png")
        bpy.ops.render.render(write_still=True)
        if fi == 0 or fi + 1 == total or (fi + 1) % 12 == 0:
            print(f"  frame {fi + 1}/{total}", flush=True)

    print(f"[turntable] frames → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
