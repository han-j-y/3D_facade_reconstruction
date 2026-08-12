#!/usr/bin/env python3
"""Compile recovered façade end-to-end with facade_dsl8 (window + façade compiler).

Writes a normalized ``facade_dsl8.json`` (wall + grid + window IRs) and renders
the full façade in Blender via ``facade_dsl8/main.py``.

Also overlays cluster/type boxes on photo + render for side-by-side compares.

Example:
  python scripts/recovery_to_blender.py \\
    --recovery runs/facade_e2e_221/facade_dsl.json \\
    --out-dir runs/facade_e2e_221/blender \\
    --render
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Standalone package: scripts live under facade_e2e/vendor/scripts
# or facade_e2e/scripts → vendor/scripts
_PKG = ROOT if (ROOT / "checkpoints").is_dir() or (ROOT / "vendor").is_dir() else ROOT.parent
if (_PKG / "vendor" / "facade_dsl8" / "main.py").is_file():
    DSL8 = _PKG / "vendor" / "facade_dsl8"
elif (_PKG.parent / "facade_dsl8" / "main.py").is_file():
    DSL8 = _PKG.parent / "facade_dsl8"
else:
    DSL8 = _PKG.parent / "facade_dsl8"  # may be missing; render will fail clearly

if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
if str(DSL8) not in sys.path:
    sys.path.insert(0, str(DSL8))

from window_ast.paths import resolve_blender  # noqa: E402
from facade_spec import (  # noqa: E402
    EMPTY_TOKENS,
    fit_window_ir_to_cell,
    get_cell,
    normalize_facade_spec,
    total_grid_size,
)


TYPE_PALETTE = [
    (230, 70, 70),
    (70, 160, 230),
    (70, 200, 120),
    (230, 180, 50),
    (180, 90, 220),
    (50, 200, 200),
    (230, 120, 50),
    (120, 120, 230),
    (200, 80, 140),
    (100, 180, 80),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--recovery",
        type=Path,
        required=True,
        help="facade_recovery_dsl_v1 JSON (or e2e out-dir with facade_dsl.json)",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--blender",
        default=None,
        help="Blender binary (default: $BLENDER or blender on PATH)",
    )
    ap.add_argument(
        "--render",
        action="store_true",
        help="compile + render full façade with facade_dsl8",
    )
    ap.add_argument("--res", default="1024x1280", help="FACADE_DSL_RENDER_RES")
    ap.add_argument("--samples", type=int, default=48, help="FACADE_DSL_SAMPLES")
    ap.add_argument(
        "--ortho-zoom",
        type=float,
        default=0.95,
        help="ortho zoom factor (<1 zooms out, >1 zooms in). Sets FACADE_DSL_ORTHO_ZOOM",
    )
    ap.add_argument("--storey-height", type=float, default=3.0)
    ap.add_argument("--facade-width", type=float, default=None)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def resolve_recovery(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        cand = path / "facade_dsl.json"
        if not cand.is_file():
            raise SystemExit(f"no facade_dsl.json in {path}")
        return cand
    if not path.is_file():
        raise SystemExit(f"recovery JSON not found: {path}")
    return path


def type_color(type_id: int) -> tuple[int, int, int]:
    return TYPE_PALETTE[int(type_id) % len(TYPE_PALETTE)]


def type_id_from_name(name: str) -> int:
    try:
        return int(str(name).rsplit("_", 1)[-1])
    except ValueError:
        return 0


def planned_windows(facade: dict) -> list[dict]:
    """Mirror facade_compile placement (world XZ boxes per placed window)."""
    windows = facade.get("windows") or {}
    placement = facade.get("placement") or []
    pp = facade.get("placement_params") or {}
    type_ratios = (facade.get("meta") or {}).get("type_ratios") or {}
    default_wr = float(pp.get("width_ratio", 0.55))
    default_hr = float(pp.get("height_ratio", 0.60))
    bottom_m = float(pp.get("bottom_margin_ratio", 0.14))
    mirror_x = bool(pp.get("mirror_x", True))
    n_rows = len(facade["grid"]["rows"])
    n_cols = len(facade["grid"]["cols"])
    out: list[dict] = []
    for r in range(n_rows):
        row = placement[r] if r < len(placement) else []
        for c in range(n_cols):
            tok = row[c] if c < len(row) else None
            if tok in EMPTY_TOKENS:
                continue
            tok = str(tok)
            if tok not in windows:
                continue
            cell = get_cell(facade, r, c, mirror_x=mirror_x)
            ratios = type_ratios.get(tok) or {}
            wr = float(ratios.get("width_ratio", default_wr))
            hr = float(ratios.get("height_ratio", default_hr))
            ir = fit_window_ir_to_cell(windows[tok], cell, width_ratio=wr, height_ratio=hr)
            params = (ir.get("boundary") or {}).get("params") or {}
            ww = float(params.get("width", cell["w"] * wr))
            hh = float(params.get("height", cell["h"] * hr))
            ox = float(cell["cx"]) - ww / 2.0
            oz = float(cell["z0"]) + float(cell["h"]) * bottom_m
            if oz + hh > float(cell["z1"]) - 0.02:
                oz = max(float(cell["z0"]) + 0.02, float(cell["z1"]) - hh - 0.02)
            out.append(
                {
                    "row": r,
                    "col": c,
                    "name": tok,
                    "type_id": type_id_from_name(tok),
                    "x0": ox,
                    "z0": oz,
                    "x1": ox + ww,
                    "z1": oz + hh,
                }
            )
    return out


def world_xz_to_pixel(
    x: float,
    z: float,
    *,
    bounds: tuple[float, float, float, float],
    res_x: int,
    res_y: int,
    ortho_zoom: float,
    padding_base: float = 1.45,
) -> tuple[float, float]:
    """Map façade world (X,Z) → image pixel (u,v), matching setup_camera_for_facade.

    dsl8 camera on +Y looking −Y mirrors world +X onto the *left* of the image
    (same as ``structure_lines`` ``flip_u=True``), so we negate X for U.
    """
    xmin, xmax, zmin, zmax = bounds
    w = max(1e-6, xmax - xmin)
    h = max(1e-6, zmax - zmin)
    cx = 0.5 * (xmin + xmax)
    cz = 0.5 * (zmin + zmax)
    aspect = res_x / max(res_y, 1)
    pad = padding_base / max(ortho_zoom, 1e-3)
    ortho_scale = max(h, w / aspect) * pad
    if aspect >= 1.0:
        view_w = ortho_scale
        view_h = ortho_scale / aspect
    else:
        view_h = ortho_scale
        view_w = ortho_scale * aspect
    # Camera mirror: world +X → image left
    u = (cx - x) / view_w * res_x + res_x * 0.5
    v = (cz - z) / view_h * res_y + res_y * 0.5
    return u, v


def overlay_clusters_on_render(
    *,
    facade: dict,
    render_path: Path,
    out_path: Path,
    ortho_zoom: float,
) -> Path | None:
    if not render_path.is_file():
        return None
    from PIL import Image, ImageDraw, ImageFont

    im = Image.open(render_path).convert("RGBA")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    total_w, total_h = total_grid_size(facade["grid"])
    bounds = (-total_w / 2.0, total_w / 2.0, 0.0, total_h)
    wins = planned_windows(facade)
    for w in wins:
        color = type_color(w["type_id"])
        u0, v1 = world_xz_to_pixel(
            w["x0"], w["z0"], bounds=bounds, res_x=im.width, res_y=im.height, ortho_zoom=ortho_zoom
        )
        u1, v0 = world_xz_to_pixel(
            w["x1"], w["z1"], bounds=bounds, res_x=im.width, res_y=im.height, ortho_zoom=ortho_zoom
        )
        x0, x1 = sorted((u0, u1))
        y0, y1 = sorted((v0, v1))
        draw.rectangle([x0, y0, x1, y1], fill=(*color, 55), outline=(*color, 230), width=3)
        draw.rectangle([x0, y0, x0 + 28, y0 + 16], fill=(*color, 220))
        draw.text((x0 + 3, y0 + 1), f"T{w['type_id']}", fill=(255, 255, 255, 255), font=font_sm)

    types = sorted({w["type_id"] for w in wins})
    lx, ly = 12, 12
    for tid in types:
        color = type_color(tid)
        draw.rectangle([lx, ly, lx + 18, ly + 18], fill=(*color, 230), outline=(255, 255, 255, 200))
        draw.text((lx + 24, ly), f"type {tid}", fill=(255, 255, 255, 255), font=font)
        ly += 24

    out = Image.alpha_composite(im, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    return out_path


def overlay_clusters_on_photo(*, recovery: dict, out_path: Path) -> Path | None:
    photo = recovery.get("meta", {}).get("image")
    if not photo or not Path(photo).is_file():
        return None
    from PIL import Image, ImageDraw, ImageFont

    im = Image.open(photo).convert("RGBA")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font_sm = ImageFont.load_default()

    for inst in recovery.get("instances") or []:
        tid = int(inst["type_id"])
        color = type_color(tid)
        x0, y0, x1, y1 = [float(v) for v in inst["box_xyxy"]]
        draw.rectangle([x0, y0, x1, y1], outline=(*color, 230), width=3)
        draw.rectangle([x0, y0, x0 + 28, y0 + 16], fill=(*color, 220))
        draw.text((x0 + 3, y0 + 1), f"T{tid}", fill=(255, 255, 255, 255), font=font_sm)

    out = Image.alpha_composite(im, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    return out_path


def write_photo_compare(
    *,
    photo_path: Path | None,
    render_path: Path,
    out_path: Path,
    title: str = "",
    target_h: int = 900,
) -> Path | None:
    if photo_path is None or not photo_path.is_file() or not render_path.is_file():
        return None
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    def fit_h(im: Image.Image, h: int) -> Image.Image:
        w = max(1, int(round(im.width * h / im.height)))
        return im.resize((w, h), Image.Resampling.LANCZOS)

    a = fit_h(Image.open(photo_path).convert("RGB"), target_h)
    b = fit_h(Image.open(render_path).convert("RGB"), target_h)
    gap, label_h = 16, 44
    canvas = Image.new("RGB", (a.width + gap + b.width, label_h + target_h), (28, 28, 28))
    d = ImageDraw.Draw(canvas)
    prefix = f"{title}  ·  " if title else ""
    d.text((12, 12), f"{prefix}photo (clusters)", fill=(230, 230, 230), font=font)
    d.text(
        (a.width + gap + 12, 12),
        f"{prefix}dsl8 blender (clusters)",
        fill=(230, 230, 230),
        font=font,
    )
    canvas.paste(a, (0, label_h))
    canvas.paste(b, (a.width + gap, label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def render_facade(
    *,
    blender: str,
    facade_json: Path,
    png_path: Path,
    res: str,
    samples: int,
    ortho_zoom: float | None = 0.85,
) -> None:
    if not Path(blender).is_file():
        raise SystemExit(f"blender not found: {blender}")
    if not (DSL8 / "main.py").is_file():
        raise SystemExit(f"facade_dsl8 not found: {DSL8}")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["FACADE_DSL_RENDER_IMAGE"] = str(png_path.resolve())
    env["FACADE_DSL_RENDER_RES"] = res
    env["FACADE_DSL_SAMPLES"] = str(samples)
    if ortho_zoom is not None:
        env["FACADE_DSL_ORTHO_ZOOM"] = str(ortho_zoom)
    cmd = [
        blender,
        "-b",
        "-P",
        str(DSL8 / "main.py"),
        "--",
        str(facade_json.resolve()),
        "--render-image",
        str(png_path.resolve()),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(DSL8), env=env, check=True)


def main() -> None:
    args = parse_args()
    recovery_path = resolve_recovery(args.recovery)
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    out_dir = (args.out_dir or recovery_path.parent / "blender").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.render:
        blender = resolve_blender(args.blender)
        if blender is None:
            print(
                "warn: Blender not found — wrote facade_dsl8.json but skipped render. "
                "Pass --blender or set $BLENDER."
            )
            args.render = False
        else:
            args.blender = blender

    facade = normalize_facade_spec(
        recovery,
        storey_height=args.storey_height,
        facade_width=args.facade_width,
    )
    facade_path = out_dir / "facade_dsl8.json"
    facade_path.write_text(json.dumps(facade, indent=2) + "\n", encoding="utf-8")
    n_win = sum(
        1
        for row in facade.get("placement") or []
        for tok in row
        if tok not in (None, "", "empty")
    )
    print(
        f"wrote {facade_path}  "
        f"types={len(facade.get('windows') or {})}  placed={n_win}"
    )

    render_png = out_dir / "facade_render.png"
    if args.render:
        if render_png.is_file() and not args.force and render_png.stat().st_size > 1000:
            print(f"skip existing {render_png} (pass --force)")
        else:
            render_facade(
                blender=args.blender,
                facade_json=facade_path,
                png_path=render_png,
                res=args.res,
                samples=args.samples,
                ortho_zoom=args.ortho_zoom,
            )
            print(f"façade render → {render_png}")
    else:
        print(
            "skip Blender (pass --render). Manual:\n"
            f"  {args.blender} -b -P {DSL8 / 'main.py'} -- {facade_path} "
            f"--render-image {render_png}"
        )

    fid = str(
        recovery.get("meta", {}).get("facade_id")
        or facade.get("meta", {}).get("facade_id")
        or ""
    )
    render_clusters = out_dir / "facade_render_clusters.png"
    photo_clusters = out_dir / "photo_clusters.png"
    if render_png.is_file():
        wrote_r = overlay_clusters_on_render(
            facade=facade,
            render_path=render_png,
            out_path=render_clusters,
            ortho_zoom=args.ortho_zoom,
        )
        if wrote_r:
            print(f"render clusters → {wrote_r}")
    wrote_p = overlay_clusters_on_photo(recovery=recovery, out_path=photo_clusters)
    if wrote_p:
        print(f"photo clusters → {wrote_p}")

    compare_path = out_dir / "compare_photo_vs_render.png"
    left = photo_clusters if photo_clusters.is_file() else None
    if left is None and recovery.get("meta", {}).get("image"):
        left = Path(recovery["meta"]["image"])
    right = render_clusters if render_clusters.is_file() else render_png
    if left and right.is_file():
        wrote = write_photo_compare(
            photo_path=left,
            render_path=right,
            out_path=compare_path,
            title=f"facade {fid}" if fid else "",
        )
        if wrote:
            print(f"compare → {wrote}")

    summary = {
        "recovery": str(recovery_path),
        "renderer": "facade_dsl8",
        "facade_dsl8": str(facade_path),
        "render": str(render_png) if render_png.is_file() else None,
        "render_clusters": str(render_clusters) if render_clusters.is_file() else None,
        "photo_clusters": str(photo_clusters) if photo_clusters.is_file() else None,
        "compare": str(compare_path) if compare_path.is_file() else None,
        "n_types": len(facade.get("windows") or {}),
        "n_placed": n_win,
        "grid": {
            "rows": len(facade["grid"]["rows"]),
            "cols": len(facade["grid"]["cols"]),
        },
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"done → {out_dir}")


if __name__ == "__main__":
    main()
