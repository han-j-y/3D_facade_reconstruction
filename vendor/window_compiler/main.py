#!/usr/bin/env python3
"""
Compile a window or façade spec into a Blender scene.

  # single window (existing)
  blender -b -P main.py -- examples/example_grid.json

  # full façade (grid of window IRs)
  blender -b -P main.py -- path/to/facade.json --render-image out.png

Façade docs use ``type: "facade"`` (see facade_compile.py). Recovery DSL
(``facade_recovery_dsl_v1``) is accepted and normalized automatically.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_script_dir = str(Path(__file__).resolve().parent)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from load_spec import load_spec

import bpy

from blender_scene import build_blender_scene
from compiler import compile_spec
from facade_compile import (
    compile_facade_scene,
    is_facade_spec,
    is_window_spec,
    normalize_facade_spec,
)
from geometry import clear_scene
from materials import init_materials
from render_setup import (
    configure_render,
    render_orbit_animation,
    render_still_image,
    setup_camera_for_facade,
    setup_orbit_animation_for_facade,
    setup_camera_for_window,
)


def _argv_tail() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def _resolve_spec(path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_file():
        return p.resolve()
    alt = Path(__file__).resolve().parent / path
    if alt.is_file():
        return alt.resolve()
    raise FileNotFoundError(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="window/façade Blender compiler")
    ap.add_argument("spec_file", nargs="?", default="examples/example_grid.json")
    ap.add_argument(
        "--render-image",
        dest="render_image",
        default="",
        help="PNG output path (or set FACADE_DSL_RENDER_IMAGE)",
    )
    ap.add_argument(
        "--render-orbit",
        dest="render_orbit",
        default="",
        help="360° turntable MP4 (or frame dir). Env: FACADE_DSL_RENDER_ORBIT",
    )
    ap.add_argument("--orbit-frames", type=int, default=72, help="frames for --render-orbit")
    ap.add_argument("--orbit-fps", type=int, default=24, help="fps for --render-orbit")
    ap.add_argument(
        "--storey-height",
        type=float,
        default=3.0,
        help="metres per storey when normalizing recovery DSL",
    )
    ap.add_argument(
        "--facade-width",
        type=float,
        default=None,
        help="total façade width (m) when normalizing recovery DSL",
    )
    args = ap.parse_args(_argv_tail())

    spec_path = _resolve_spec(args.spec_file)
    spec = load_spec(spec_path)

    clear_scene()
    init_materials((spec.get("frame") or {}).get("material", "painted_wood"))

    render_out = os.environ.get("FACADE_DSL_RENDER_IMAGE", "").strip() or args.render_image
    configure_render()

    facade_bounds = None
    if is_facade_spec(spec):
        facade = normalize_facade_spec(
            spec,
            storey_height=args.storey_height,
            facade_width=args.facade_width,
        )
        # persist normalized form next to source when useful for debugging
        summary = compile_facade_scene(facade)
        facade_bounds = tuple(summary["bounds"])
        setup_camera_for_facade(facade_bounds)
        print(
            f"compiled façade {spec_path.name}: "
            f"{summary['n_windows']} windows, "
            f"{summary['total_w']:.2f}×{summary['total_h']:.2f}m"
        )
    elif is_window_spec(spec):
        if spec.get("type") != "window":
            spec = {**spec, "type": "window"}
        ctx = compile_spec(spec)
        build_blender_scene(ctx)
        print(f"compiled {spec_path.name}: {len(ctx.segments)} muntin segments")
        setup_camera_for_window(ctx)
    else:
        raise SystemExit(
            f"Unrecognized spec (need type=window or type=facade / recovery DSL): {spec_path}"
        )

    if render_out:
        render_still_image(render_out)

    orbit_out = os.environ.get("FACADE_DSL_RENDER_ORBIT", "").strip() or args.render_orbit
    if orbit_out:
        if facade_bounds is None:
            raise SystemExit("--render-orbit requires a façade spec (not a single window)")
        n_frames = int(os.environ.get("FACADE_DSL_ORBIT_FRAMES", args.orbit_frames))
        fps = int(os.environ.get("FACADE_DSL_ORBIT_FPS", args.orbit_fps))
        setup_orbit_animation_for_facade(facade_bounds, n_frames=n_frames, fps=fps)
        render_orbit_animation(orbit_out, n_frames=n_frames)


if __name__ == "__main__":
    main()
