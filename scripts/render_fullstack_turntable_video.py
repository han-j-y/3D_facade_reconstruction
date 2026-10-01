#!/usr/bin/env python3
"""Turntable videos for the ``symmetry_from_sam3_fullstack`` Blender renders.

For every stem in the run this renders a moving-camera pass over the repaired
façade (``blender/facade_compile.json``) and encodes:

  <stem>/turntable_3d.mp4         bare 3D pass
  <stem>/turntable_fullstack.mp4  clusters BEFORE | clusters AFTER | 3D pass
  turntable_grid.mp4              all stems tiled

The still ortho render in ``fullstack_strip.png`` reads as a flat drawing; the
parallax here shows the wall slab, the cut openings and the recessed frames.

Example:
  python scripts/render_fullstack_turntable_video.py --frames 96 --samples 64
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BLENDER_DEFAULT = Path(os.environ.get("BLENDER") or shutil.which("blender") or "blender")
TURNTABLE_SCRIPT = ROOT / "scripts" / "blender_facade_turntable.py"
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir", type=Path, default=ROOT / "runs" / "symmetry_from_sam3_fullstack"
    )
    ap.add_argument("--stems", default="", help="comma stems; default = all in run-dir")
    ap.add_argument("--blender", type=Path, default=BLENDER_DEFAULT)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument(
        "--res", default="auto", help="WxH, or 'auto' to match each façade aspect"
    )
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--motion", choices=("sweep", "orbit"), default="sweep")
    ap.add_argument("--yaw-deg", type=float, default=52.0)
    ap.add_argument("--elev-deg", type=float, default=9.0)
    ap.add_argument("--lens", type=float, default=42.0)
    ap.add_argument(
        "--wall-depth",
        type=float,
        default=None,
        help="override wall slab depth in metres for the 3D pass",
    )
    ap.add_argument("--panel-height", type=int, default=520)
    ap.add_argument("--grid-cell-width", type=int, default=480)
    ap.add_argument("--no-panels", action="store_true")
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-render existing frames")
    return ap.parse_args()


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()


def fit_height(im: Image.Image, h: int) -> Image.Image:
    if im.height == h:
        return im
    w = max(1, round(im.width * h / im.height))
    return im.resize((w, h), Image.Resampling.LANCZOS)


def label_bar(text: str, width: int, *, height: int = 28, bold: bool = False) -> Image.Image:
    bar = Image.new("RGB", (width, height), (24, 24, 28))
    ImageDraw.Draw(bar).text(
        (8, (height - 18) // 2), text, fill=(235, 235, 235), font=font(14, bold=bold)
    )
    return bar


def discover_stems(run_dir: Path) -> list[str]:
    return sorted(
        p.name
        for p in run_dir.iterdir()
        if p.is_dir() and (p / "blender" / "facade_compile.json").is_file()
    )


def render_frames(
    spec: Path, frames_dir: Path, args: argparse.Namespace
) -> None:
    existing = sorted(frames_dir.glob("frame_*.png"))
    if len(existing) == args.frames and not args.force:
        print(f"  frames cached ({len(existing)})")
        return
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.blender),
        "-b",
        "-P",
        str(TURNTABLE_SCRIPT),
        "--",
        str(spec),
        "--out",
        str(frames_dir),
        "--frames",
        str(args.frames),
        "--res",
        args.res,
        "--samples",
        str(args.samples),
        "--motion",
        args.motion,
        "--yaw-deg",
        str(args.yaw_deg),
        "--elev-deg",
        str(args.elev_deg),
        "--lens",
        str(args.lens),
    ]
    if args.wall_depth is not None:
        cmd.extend(["--wall-depth", str(args.wall_depth)])
    log_path = frames_dir.parent / "blender_turntable.log"
    with log_path.open("w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)


def encode(frames_dir: Path, out_mp4: Path, fps: int) -> Path:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
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
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)
    return out_mp4


def header_text(stem: str, result: dict) -> str:
    if not result:
        return f"{stem}  ·  Sym-repaired façade"
    return (
        f"{stem}  ·  Sym repair  moves={result.get('label_moves', '?')}  "
        f"k {result.get('k_before', '?')}→{result.get('k_after', '?')}  "
        f"obj {result.get('obj_before', 0):.4f}→{result.get('obj_after', 0):.4f}"
    )


def compose_panel_frames(
    stem_dir: Path, frames_dir: Path, panel_dir: Path, *, height: int, stem: str
) -> Path | None:
    """Static cluster panels beside the moving 3D pass, one composite per frame."""
    before = stem_dir / "clusters_before.png"
    after = stem_dir / "clusters_after.png"
    if not (before.is_file() and after.is_file()):
        return None
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        return None

    left = [
        fit_height(Image.open(before).convert("RGB"), height),
        fit_height(Image.open(after).convert("RGB"), height),
    ]
    probe = fit_height(Image.open(frames[0]).convert("RGB"), height)
    widths = [im.width for im in left] + [probe.width]
    captions = [
        "clusters BEFORE",
        "clusters AFTER (Sym repair)",
        "Blender 3D (Sym-repaired) — moving camera",
    ]
    gap, cap_h, hdr_h = 10, 28, 30
    total_w = sum(widths) + gap * (len(widths) - 1)

    result_path = stem_dir / "result.json"
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    hdr = label_bar(header_text(stem, result), total_w, height=hdr_h, bold=True)
    caps = [label_bar(c, w, height=cap_h) for c, w in zip(captions, widths)]

    panel_dir.mkdir(parents=True, exist_ok=True)
    for i, fp in enumerate(frames, start=1):
        canvas = Image.new("RGB", (total_w, hdr_h + cap_h + height), (16, 16, 18))
        canvas.paste(hdr, (0, 0))
        cols = left + [fit_height(Image.open(fp).convert("RGB"), height)]
        x = 0
        for cap, im in zip(caps, cols):
            canvas.paste(cap, (x, hdr_h))
            canvas.paste(im, (x, hdr_h + cap_h))
            x += im.width + gap
        # libx264 yuv420p needs even dimensions
        w, h = canvas.size
        if w % 2 or h % 2:
            canvas = canvas.crop((0, 0, w - w % 2, h - h % 2))
        canvas.save(panel_dir / f"frame_{i:04d}.png")
    return panel_dir


def compose_grid_frames(
    stems: list[str], frame_dirs: dict[str, Path], grid_dir: Path, *, cell_w: int
) -> Path | None:
    """Tile every stem's pass into one overview clip (cells are letterboxed)."""
    usable = [s for s in stems if sorted(frame_dirs[s].glob("frame_*.png"))]
    if not usable:
        return None
    per_stem = {s: sorted(frame_dirs[s].glob("frame_*.png")) for s in usable}
    n_frames = min(len(v) for v in per_stem.values())
    cols = math.ceil(math.sqrt(len(usable)))
    rows = math.ceil(len(usable) / cols)

    # Stems render at their own aspect, so size cells by the tallest one.
    aspects = [Image.open(per_stem[s][0]).size for s in usable]
    cell_h = max(round(cell_w * h / w) for w, h in aspects)
    label_h = 24
    grid_w = cols * cell_w
    grid_h = rows * (cell_h + label_h)
    grid_w -= grid_w % 2
    grid_h -= grid_h % 2

    grid_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        canvas = Image.new("RGB", (grid_w, grid_h), (16, 16, 18))
        d = ImageDraw.Draw(canvas)
        for j, stem in enumerate(usable):
            r, c = divmod(j, cols)
            x, y = c * cell_w, r * (cell_h + label_h)
            im = Image.open(per_stem[stem][i]).convert("RGB")
            scale = min(cell_w / im.width, cell_h / im.height)
            im = im.resize(
                (max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                Image.Resampling.LANCZOS,
            )
            d.rectangle([x, y, x + cell_w, y + label_h], fill=(24, 24, 28))
            d.text((x + 8, y + 4), stem, fill=(235, 235, 235), font=font(14, bold=True))
            canvas.paste(
                im,
                (x + (cell_w - im.width) // 2, y + label_h + (cell_h - im.height) // 2),
            )
        canvas.save(grid_dir / f"frame_{i + 1:04d}.png")
    return grid_dir


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run dir not found: {run_dir}")
    if not Path(args.blender).is_file():
        raise SystemExit(f"blender not found: {args.blender}")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not on PATH")

    stems = (
        [s.strip() for s in args.stems.split(",") if s.strip()]
        if args.stems.strip()
        else discover_stems(run_dir)
    )
    if not stems:
        raise SystemExit(f"no stems with blender/facade_compile.json under {run_dir}")
    print(f"stems={len(stems)}  frames={args.frames}  res={args.res}  out={run_dir}")

    frame_dirs: dict[str, Path] = {}
    outputs: list[dict] = []
    for stem in stems:
        stem_dir = run_dir / stem
        spec = stem_dir / "blender" / "facade_compile.json"
        print(f"\n=== {stem} ===")
        frames_dir = stem_dir / "turntable" / "frames"
        render_frames(spec, frames_dir, args)
        frame_dirs[stem] = frames_dir

        row = {"stem": stem}
        bare = encode(frames_dir, stem_dir / "turntable_3d.mp4", args.fps)
        row["turntable_3d"] = str(bare)
        print(f"  {bare}")

        if not args.no_panels:
            panel_dir = stem_dir / "turntable" / "panels"
            made = compose_panel_frames(
                stem_dir,
                frames_dir,
                panel_dir,
                height=args.panel_height,
                stem=stem,
            )
            if made is not None:
                full = encode(made, stem_dir / "turntable_fullstack.mp4", args.fps)
                row["turntable_fullstack"] = str(full)
                print(f"  {full}")
                shutil.rmtree(panel_dir, ignore_errors=True)
        outputs.append(row)

    reel = [Path(r["turntable_3d"]) for r in outputs if r.get("turntable_3d")]
    if len(reel) > 1:
        list_path = run_dir / "turntable_concat.txt"
        list_path.write_text("".join(f"file '{p}'\n" for p in reel))
        showcase = run_dir / "turntable_orbit.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(showcase),
            ],
            check=True,
        )
        print(f"\n=== showcase ===\n  {showcase}")
        list_path.unlink(missing_ok=True)

    if not args.no_grid and len(stems) > 1:
        print("\n=== grid ===")
        grid_dir = run_dir / "turntable_grid_frames"
        made = compose_grid_frames(
            stems, frame_dirs, grid_dir, cell_w=args.grid_cell_width
        )
        if made is not None:
            grid = encode(made, run_dir / "turntable_grid.mp4", args.fps)
            print(f"  {grid}")
            shutil.rmtree(grid_dir, ignore_errors=True)

    if not args.keep_frames:
        for d in frame_dirs.values():
            shutil.rmtree(d.parent, ignore_errors=True)

    (run_dir / "turntable_summary.json").write_text(
        json.dumps(
            {
                "frames": args.frames,
                "fps": args.fps,
                "res": args.res,
                "samples": args.samples,
                "motion": args.motion,
                "elev_deg": args.elev_deg,
                "lens": args.lens,
                "wall_depth": args.wall_depth,
                "stems": outputs,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {run_dir / 'turntable_summary.json'}")


if __name__ == "__main__":
    main()
