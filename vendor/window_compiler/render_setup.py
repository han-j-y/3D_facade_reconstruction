"""
Rendering + orthographic camera for window_compiler.

Grammar coords: x horizontal, y vertical. Blender wall: XZ façade, Y depth.
Camera sits on +Y and looks straight at the wall (orthographic).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

if TYPE_CHECKING:
    from context import CompileContext

DEFAULT_ORTHO_ZOOM = 1.0
# Multiplier on fitted span; >1 adds margin around the window in ortho renders.
DEFAULT_FRAMING_PADDING = 1.55
_SKIP_OBJECTS = frozenset(
    {"Camera", "Sun", "Ground", "Wall", "FacadeWall", "WindowCam", "Key"}
)

def configure_render() -> None:
    scene = bpy.context.scene
    eng = os.environ.get("FACADE_DSL_RENDER_ENGINE", "CYCLES").strip().upper()
    if eng in ("EEVEE", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        eng = "BLENDER_EEVEE"
    try:
        scene.render.engine = eng
    except Exception:
        scene.render.engine = "CYCLES"

    res_s = (
        os.environ.get("FACADE_DSL_RENDER_RES", "").strip()
        or os.environ.get("FACADE_LAB_RENDER_RES", "").strip()
        or "1024x1024"
    )
    try:
        a, b = res_s.lower().split("x", 1)
        rx, ry = max(1, int(a.strip())), max(1, int(b.strip()))
    except (ValueError, IndexError):
        rx, ry = 1024, 1024
    scene.render.resolution_x = rx
    scene.render.resolution_y = ry
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = os.environ.get("FACADE_DSL_FILM_TRANSPARENT", "").lower() in (
        "1",
        "true",
        "yes",
    )
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if scene.render.film_transparent else "RGB"
    scene.render.image_settings.color_depth = "8"

    if scene.render.engine == "CYCLES":
        samples_s = os.environ.get("FACADE_DSL_SAMPLES", "64").strip()
        try:
            scene.cycles.samples = int(samples_s)
        except ValueError:
            pass


def render_still_image(output_path: str | Path) -> Path:
    p = Path(os.path.expanduser(str(output_path))).resolve()
    if not p.suffix:
        p = p.with_suffix(".png")
    p.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.filepath = str(p)
    bpy.ops.render.render(write_still=True)
    written = Path(bpy.path.abspath(scene.render.filepath))
    print(f"[window_compiler] Render saved → {written}", flush=True)
    return written


def setup_orbit_animation_for_facade(
    bounds: tuple[float, float, float, float],
    *,
    n_frames: int = 72,
    fps: int = 24,
    radius_scale: float = 1.85,
    elev_scale: float = 0.18,
    lens_mm: float = 50.0,
) -> dict[str, float]:
    """Perspective turntable: orbit 360° around façade center (vertical Z).

    Starts at front (+Y), rotates CCW around Z. Returns orbit params used.
    Camera pose is applied per-frame in ``render_orbit_animation`` (no Action API).
    """
    xmin, xmax, zmin, zmax = bounds
    objs = _window_objects()
    mesh_bounds = _world_xz_bounds(objs)
    if mesh_bounds is not None:
        mx0, mx1, mz0, mz1 = mesh_bounds
        xmin, xmax = min(xmin, mx0), max(xmax, mx1)
        zmin, zmax = min(zmin, mz0), max(zmax, mz1)

    w = max(1e-3, xmax - xmin)
    h = max(1e-3, zmax - zmin)
    cx = 0.5 * (xmin + xmax)
    cz = 0.5 * (zmin + zmax)
    span = max(w, h)
    radius = max(6.0, span * radius_scale)
    elev = max(0.5, h * elev_scale)

    n_frames = max(8, int(n_frames))
    fps = max(1, int(fps))

    scene = bpy.context.scene
    cam = scene.camera
    if cam is None or cam.type != "CAMERA":
        cam_data = bpy.data.cameras.new("OrbitCamera")
        cam = bpy.data.objects.new("OrbitCamera", cam_data)
        scene.collection.objects.link(cam)
        scene.camera = cam
    cam.data.type = "PERSP"
    cam.data.lens = float(lens_mm)
    cam.data.clip_start = 0.05
    cam.data.clip_end = max(200.0, radius * 8.0)
    if hasattr(cam.data, "shift_x"):
        cam.data.shift_x = 0.0
    if hasattr(cam.data, "shift_y"):
        cam.data.shift_y = 0.0
    # Clear any parent from prior ortho setup; we place in world space each frame.
    cam.parent = None

    scene.frame_start = 1
    scene.frame_end = n_frames
    scene.render.fps = fps

    # Soft fill so the back of the wall isn't black.
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        try:
            bg.inputs["Color"].default_value = (0.88, 0.91, 0.96, 1.0)
            bg.inputs["Strength"].default_value = 0.55
        except (KeyError, IndexError):
            pass

    if "OrbitFill" not in bpy.data.objects:
        fill_data = bpy.data.lights.new("OrbitFill", type="SUN")
        fill = bpy.data.objects.new("OrbitFill", fill_data)
        scene.collection.objects.link(fill)
        fill.location = (cx - w * 0.4, -radius * 0.5, cz + h * 0.3)
        fill.rotation_euler = (1.0, -0.3, -0.6)
        fill.data.energy = 1.6

    # Stash orbit params on scene for the renderer.
    scene["facade_orbit_cx"] = float(cx)
    scene["facade_orbit_cz"] = float(cz)
    scene["facade_orbit_radius"] = float(radius)
    scene["facade_orbit_elev"] = float(elev)
    scene["facade_orbit_n_frames"] = int(n_frames)

    # Pose frame 1 (front).
    _place_orbit_camera(cam, cx=cx, cz=cz, radius=radius, elev=elev, angle=0.0)
    bpy.context.view_layer.update()

    params = {
        "cx": cx,
        "cz": cz,
        "radius": radius,
        "elev": elev,
        "n_frames": float(n_frames),
        "fps": float(fps),
    }
    print(
        f"[window_compiler] orbit setup frames={n_frames} fps={fps} "
        f"radius={radius:.2f} elev={elev:.2f}",
        flush=True,
    )
    return params


def _place_orbit_camera(
    cam: bpy.types.Object,
    *,
    cx: float,
    cz: float,
    radius: float,
    elev: float,
    angle: float,
) -> None:
    """Place camera on a horizontal circle around (cx,0,cz); ``angle=0`` is +Y."""
    x = cx + radius * math.sin(angle)
    y = radius * math.cos(angle)
    z = cz + elev
    cam.location = (x, y, z)
    target = Vector((cx, 0.0, cz))
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()


def render_orbit_animation(output_path: str | Path, *, n_frames: int | None = None) -> Path:
    """Render the orbit turntable to an MP4 (PNG sequence + ffmpeg) or frame dir."""
    import shutil
    import subprocess
    import tempfile

    out = Path(os.path.expanduser(str(output_path))).resolve()
    scene = bpy.context.scene
    cam = scene.camera
    if cam is None:
        raise RuntimeError("no scene camera for orbit render")

    cx = float(scene.get("facade_orbit_cx", 0.0))
    cz = float(scene.get("facade_orbit_cz", 0.0))
    radius = float(scene.get("facade_orbit_radius", 10.0))
    elev = float(scene.get("facade_orbit_elev", 1.0))
    total = int(n_frames or scene.get("facade_orbit_n_frames", scene.frame_end))
    total = max(8, total)
    scene.frame_start = 1
    scene.frame_end = total

    # Faster engine for multi-frame turntables unless user overrode.
    eng_env = os.environ.get("FACADE_DSL_RENDER_ENGINE", "").strip()
    if not eng_env:
        for cand in ("BLENDER_EEVEE", "BLENDER_WORKBENCH"):
            try:
                scene.render.engine = cand
                break
            except Exception:
                continue
    print(f"[window_compiler] orbit render.engine={scene.render.engine}", flush=True)

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False

    if out.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"}:
        frames_dir = Path(tempfile.mkdtemp(prefix="facade_orbit_"))
        cleanup_frames = True
        video_out = out
    else:
        frames_dir = out
        frames_dir.mkdir(parents=True, exist_ok=True)
        cleanup_frames = False
        video_out = out / "orbit.mp4" if out.suffix == "" else out.with_suffix(".mp4")

    print(
        f"[window_compiler] rendering orbit frames 1–{total} → {frames_dir}",
        flush=True,
    )
    for fi in range(1, total + 1):
        angle = 2.0 * math.pi * (fi - 1) / total
        _place_orbit_camera(cam, cx=cx, cz=cz, radius=radius, elev=elev, angle=angle)
        scene.frame_set(fi)
        bpy.context.view_layer.update()
        frame_path = frames_dir / f"frame_{fi:04d}.png"
        scene.render.filepath = str(frame_path)
        bpy.ops.render.render(write_still=True)
        if fi == 1 or fi == total or fi % 12 == 0:
            print(f"  frame {fi}/{total}", flush=True)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print(f"[window_compiler] ffmpeg not found; frames left at {frames_dir}", flush=True)
        return frames_dir

    video_out.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1, int(scene.render.fps))
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
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
    print(f"[window_compiler] orbit video → {video_out}", flush=True)

    if cleanup_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
    return video_out


def setup_opaque_world_background() -> None:
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        try:
            bg.inputs["Color"].default_value = (0.92, 0.95, 1.0, 1.0)
            bg.inputs["Strength"].default_value = 0.85
        except (KeyError, IndexError):
            pass


def apply_ortho_zoom(factor: float = DEFAULT_ORTHO_ZOOM) -> None:
    if factor <= 0:
        return
    scene = bpy.context.scene
    cam = scene.camera
    if cam is None or cam.type != "CAMERA":
        return
    try:
        scale = float(cam.data.ortho_scale)
    except (TypeError, ValueError):
        return
    shift_y = float(getattr(cam.data, "shift_y", 0.0) or 0.0)
    shift_x = float(getattr(cam.data, "shift_x", 0.0) or 0.0)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = max(1e-6, scale * factor)
    if abs(factor - 1.0) > 1e-6:
        if hasattr(cam.data, "shift_y") and abs(shift_y) > 1e-12:
            cam.data.shift_y = shift_y / factor
        if hasattr(cam.data, "shift_x") and abs(shift_x) > 1e-12:
            cam.data.shift_x = shift_x / factor


def _window_objects() -> list[bpy.types.Object]:
    return [
        o
        for o in bpy.context.scene.objects
        if o.type in {"MESH", "CURVE"} and o.name not in _SKIP_OBJECTS
    ]


def _world_xz_bounds(objects: list[bpy.types.Object]) -> tuple[float, float, float, float] | None:
    """Return xmin, xmax, zmin, zmax on the wall plane (grammar y → Blender Z)."""
    xmin = zmin = float("inf")
    xmax = zmax = float("-inf")
    for obj in objects:
        if getattr(obj, "data", None) is None:
            continue
        mat = obj.matrix_world
        for c in obj.bound_box:
            p = mat @ Vector(c)
            xmin = min(xmin, float(p.x))
            xmax = max(xmax, float(p.x))
            zmin = min(zmin, float(p.z))
            zmax = max(zmax, float(p.z))
    if xmin == float("inf"):
        return None
    return xmin, xmax, zmin, zmax


def _refit_ortho_to_bounds(
    cam: bpy.types.Object,
    bounds: tuple[float, float, float, float],
    *,
    padding: float = 1.08,
) -> None:
    xmin, xmax, zmin, zmax = bounds
    w = xmax - xmin
    h = zmax - zmin
    scene = bpy.context.scene
    aspect = scene.render.resolution_x / max(scene.render.resolution_y, 1)
    needed = max(h, w / aspect) * padding
    cam.data.ortho_scale = max(needed, 1e-4)
    cam.location.x = (xmin + xmax) * 0.5
    cam.location.z = (zmin + zmax) * 0.5


def _projected_bounds(
    objects: list[bpy.types.Object], cam: bpy.types.Object
) -> tuple[float, float, float, float] | None:
    scene = bpy.context.scene
    xs: list[float] = []
    ys: list[float] = []
    for obj in objects:
        if getattr(obj, "data", None) is None:
            continue
        mat = obj.matrix_world
        for c in obj.bound_box:
            p = world_to_camera_view(scene, cam, mat @ Vector(c))
            xs.append(float(p.x))
            ys.append(float(p.y))
    if not xs:
        return None
    return min(xs), max(xs), min(ys), max(ys)


def _auto_center_camera_shift(
    objects: list[bpy.types.Object],
    cam: bpy.types.Object,
    *,
    allow_x: bool,
    allow_y: bool,
) -> None:
    if not hasattr(cam.data, "shift_x") and not hasattr(cam.data, "shift_y"):
        return

    def _center_xy() -> tuple[float, float] | None:
        b = _projected_bounds(objects, cam)
        if b is None:
            return None
        x0, x1, y0, y1 = b
        return (x0 + x1) * 0.5, (y0 + y1) * 0.5

    bpy.context.view_layer.update()
    if _center_xy() is None:
        return

    eps = 1e-3
    for axis, enabled in (("shift_x", allow_x), ("shift_y", allow_y)):
        if not enabled or not hasattr(cam.data, axis):
            continue
        s0 = float(getattr(cam.data, axis) or 0.0)
        c_base = _center_xy()
        if c_base is None:
            continue
        idx = 0 if axis == "shift_x" else 1
        v_base = c_base[idx]

        setattr(cam.data, axis, s0 + eps)
        bpy.context.view_layer.update()
        c_eps = _center_xy()
        if c_eps is None:
            setattr(cam.data, axis, s0)
            continue
        slope = (c_eps[idx] - v_base) / eps
        if abs(slope) < 1e-8:
            setattr(cam.data, axis, s0)
            continue
        solved = s0 + (0.5 - v_base) / slope
        setattr(cam.data, axis, max(-2.0, min(2.0, solved)))
        bpy.context.view_layer.update()


def _ensure_ortho_camera_and_sun(
    *,
    cx: float,
    cz: float,
    span_w: float,
    span_h: float,
) -> bpy.types.Object:
    cam_dist = max(8.0, max(span_w, span_h) * 1.2)

    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.location = (cx, cam_dist, cz)
    target = Vector((cx, 0.0, cz))
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam_data.type = "ORTHO"
    cam_data.clip_start = 0.01
    cam_data.clip_end = max(100.0, cam_dist * 4.0)
    if hasattr(cam.data, "shift_y"):
        cam.data.shift_y = 0.0
    if hasattr(cam.data, "shift_x"):
        cam.data.shift_x = 0.0

    raw_cx = os.environ.get("FACADE_DSL_CAM_X", "").strip()
    if raw_cx:
        try:
            cam.location.x = float(raw_cx)
        except ValueError:
            pass

    light_data = bpy.data.lights.new("Sun", type="SUN")
    light = bpy.data.objects.new("Sun", light_data)
    bpy.context.scene.collection.objects.link(light)
    light.location = (cx + span_w * 0.5, cam_dist * 0.6, cz + span_h * 0.5)
    light.rotation_euler = (0.9, 0.2, 0.8)
    light.data.energy = 3.0
    return cam


def setup_camera_for_window(
    ctx: CompileContext,
    *,
    ortho_zoom: float = DEFAULT_ORTHO_ZOOM,
) -> None:
    """Orthographic camera on +Y, aimed at the window center on the XZ wall."""
    region = ctx.region("root")
    w = float(region.width)
    h = float(region.height)
    cx = float(region.center.x)
    cz = float(region.center.y)

    cam = _ensure_ortho_camera_and_sun(cx=cx, cz=cz, span_w=w, span_h=h)
    bpy.context.view_layer.update()

    # Grammar y → Blender Z on the wall.
    region_bounds = (region.min_x, region.max_x, region.min_y, region.max_y)
    objs = _window_objects()
    mesh_bounds = _world_xz_bounds(objs)
    bounds = mesh_bounds
    if bounds is None:
        bounds = region_bounds
    else:
        rx0, rx1, rz0, rz1 = region_bounds
        mx0, mx1, mz0, mz1 = bounds
        bounds = (min(rx0, mx0), max(rx1, mx1), min(rz0, mz0), max(rz1, mz1))

    zoom_s = os.environ.get("FACADE_DSL_ORTHO_ZOOM", "").strip()
    zoom = float(zoom_s) if zoom_s else ortho_zoom

    _refit_ortho_to_bounds(
        cam,
        bounds,
        padding=DEFAULT_FRAMING_PADDING / max(zoom, 1e-3),
    )

    raw_sx = os.environ.get("FACADE_DSL_CAM_SHIFT_X", "").strip()
    raw_sy = os.environ.get("FACADE_DSL_CAM_SHIFT_Y", "").strip()
    if raw_sy and hasattr(cam.data, "shift_y"):
        try:
            cam.data.shift_y = float(raw_sy)
        except ValueError:
            pass
    if raw_sx and hasattr(cam.data, "shift_x"):
        try:
            cam.data.shift_x = float(raw_sx)
        except ValueError:
            pass

    auto_center = os.environ.get("FACADE_DSL_AUTO_CENTER", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if auto_center and objs:
        _auto_center_camera_shift(
            objs,
            cam,
            allow_x=(raw_sx == ""),
            allow_y=(raw_sy == ""),
        )

    if not bpy.context.scene.render.film_transparent:
        setup_opaque_world_background()

    if os.environ.get("FACADE_DSL_DEBUG_CAMERA", "").strip() not in ("", "0", "false", "no"):
        loc = tuple(round(x, 4) for x in cam.location)
        rot = tuple(round(math.degrees(x), 2) for x in cam.rotation_euler)
        sx = float(getattr(cam.data, "shift_x", 0.0) or 0.0)
        sy = float(getattr(cam.data, "shift_y", 0.0) or 0.0)
        proj = _projected_bounds(objs, cam) if objs else None
        print(
            "[window_compiler camera]",
            f"location_xyz={loc}",
            f"rotation_euler_deg={rot}",
            f"ortho_scale={cam.data.ortho_scale:.4f}",
            f"shift=({sx:.4f},{sy:.4f})",
            f"projected_center={proj}",
            flush=True,
        )

    _align_viewport_to_camera()


def setup_camera_for_facade(
    bounds: tuple[float, float, float, float],
    *,
    ortho_zoom: float = DEFAULT_ORTHO_ZOOM,
    padding: float | None = None,
) -> None:
    """Orthographic camera framing a façade XZ AABB ``(xmin, xmax, zmin, zmax)``."""
    xmin, xmax, zmin, zmax = bounds
    w = max(1e-3, xmax - xmin)
    h = max(1e-3, zmax - zmin)
    cx = 0.5 * (xmin + xmax)
    cz = 0.5 * (zmin + zmax)

    cam = _ensure_ortho_camera_and_sun(cx=cx, cz=cz, span_w=w, span_h=h)
    bpy.context.view_layer.update()

    objs = _window_objects()
    mesh_bounds = _world_xz_bounds(objs)
    if mesh_bounds is not None:
        mx0, mx1, mz0, mz1 = mesh_bounds
        bounds = (min(xmin, mx0), max(xmax, mx1), min(zmin, mz0), max(zmax, mz1))

    zoom_s = os.environ.get("FACADE_DSL_ORTHO_ZOOM", "").strip()
    zoom = float(zoom_s) if zoom_s else ortho_zoom
    # Slightly looser than single-window framing so edge windows aren't clipped.
    pad = padding if padding is not None else (1.45 / max(zoom, 1e-3))

    _refit_ortho_to_bounds(cam, bounds, padding=pad)

    if not bpy.context.scene.render.film_transparent:
        setup_opaque_world_background()

    _align_viewport_to_camera()


def _align_viewport_to_camera() -> None:
    """Switch the first 3D view to camera view (interactive sessions)."""
    if bpy.app.background:
        return
    cam = bpy.context.scene.camera
    if cam is None:
        return
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for region in area.regions:
                if region.type != "WINDOW":
                    continue
                override = {
                    "window": window,
                    "screen": screen,
                    "area": area,
                    "region": region,
                }
                with bpy.context.temp_override(**override):
                    bpy.ops.view3d.view_camera()
                return
