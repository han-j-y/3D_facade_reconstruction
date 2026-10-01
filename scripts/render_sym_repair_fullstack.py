#!/usr/bin/env python3
"""Full-stack Sym repair → patched DSL → Blender 3D, with before/after strips.

Reads frozen ``e2e_sam3_cluster_*`` (box GMM types), applies the same Sym repair
as ``run_symmetry_from_sam3_e2e.py`` (chosen_cx + size-mismatch geo force),
writes a patched ``facade_dsl.json``, re-renders with Blender, and composes:

  photo+clusters BEFORE | photo+clusters AFTER | Blender BEFORE | Blender AFTER

Does **not** overwrite ``runs/e2e_sam3_cluster_*``; outputs under
``runs/symmetry_from_sam3_fullstack/``.

Example:
  python scripts/render_sym_repair_fullstack.py \\
    --stems cmp_b0020,cmp_b0250,cmp_b0010,cmp_b0083 --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent
E2E = ROOT
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from scripts.overlay_facade_asset_clusters import draw_overlay  # noqa: E402
from scripts.run_symmetry_from_sam3_e2e import (  # noqa: E402
    TITLE_BAR_H,
    _draw_axis,
    diagnose_matches,
    embed_roi,
    load_sam3_baseline,
    resolve_axis,
)
from scripts.walkthrough_symmetry_repair import (  # noqa: E402
    _load_script,
    facade_sym,
    repair_all_clusters,
)
from window_ast.symmetry import bounding_box  # noqa: E402

BLENDER_DEFAULT = Path(os.environ.get("BLENDER") or shutil.which("blender") or "blender")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sam3-root",
        type=Path,
        default=E2E / "runs",
    )
    ap.add_argument(
        "--axis-root",
        type=Path,
        default=ROOT / "runs" / "symmetry_axis_sweep" / "ghost_continuous",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "runs" / "symmetry_from_sam3_fullstack",
    )
    ap.add_argument("--stems", default="", help="comma stems; default = repaired set")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dino", default="dinov2_vits14")
    ap.add_argument("--facade-max-side", type=int, default=896)
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--match-iou", type=float, default=0.08)
    ap.add_argument("--match-center", type=float, default=0.55)
    ap.add_argument("--twin-mirror-min", type=float, default=0.30)
    ap.add_argument("--size-rel-tol", type=float, default=0.40)
    ap.add_argument("--size-mismatch-sym-gain", type=float, default=0.004)
    ap.add_argument("--size-mismatch-partner-iou", type=float, default=0.45)
    ap.add_argument("--size-mismatch-contain", type=float, default=0.85)
    ap.add_argument(
        "--blender",
        type=Path,
        default=BLENDER_DEFAULT if BLENDER_DEFAULT.is_file() else None,
    )
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument(
        "--blender-before",
        action="store_true",
        help="also show the baseline e2e render (default: repaired render only)",
    )
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--force-render", action="store_true")
    return ap.parse_args()


def default_stems(sam3_root: Path, repair_root: Path) -> list[str]:
    """Prefer stems that already have Sym moves + baseline Blender."""
    preferred = [
        "cmp_b0020",
        "cmp_b0250",
        "cmp_b0010",
        "cmp_b0083",
        "cmp_b0120",
        "cmp_b0100",
        "cmp_b0350",
        "cmp_b0001",
        "cmp_b0220",
        "cmp_b0102",
        "cmp_b0131",
        "cmp_b0270",
        "cmp_b0310",
        "cmp_b0170",
    ]
    out = []
    for stem in preferred:
        run = sam3_root / f"e2e_sam3_cluster_{stem}"
        if (run / "facade_dsl.json").is_file():
            out.append(stem)
    if out:
        return out
    return sorted(
        p.name.replace("e2e_sam3_cluster_", "")
        for p in sam3_root.glob("e2e_sam3_cluster_cmp_*")
        if (p / "facade_dsl.json").is_file()
    )[:12]


def type_lookup(dsl: dict) -> dict[int, dict]:
    return {int(t["type_id"]): t for t in dsl.get("window_types", [])}


def patch_dsl_types(dsl: dict, labels: list[int]) -> dict:
    """Rewrite window type_ids + structure IR from destination type exemplars."""
    out = copy.deepcopy(dsl)
    wins = [u for u in out["instances"] if u.get("kind") != "door"]
    if len(wins) != len(labels):
        raise RuntimeError(
            f"label/unit mismatch: {len(labels)} labels vs {len(wins)} windows"
        )
    by_tid = type_lookup(out)
    for u, tid in zip(wins, labels):
        tid = int(tid)
        u["type_id"] = tid
        src = by_tid.get(tid)
        if src is None:
            continue
        if src.get("structure_ir") is not None:
            u["structure_ir"] = copy.deepcopy(src["structure_ir"])
        if src.get("structure_tokens") is not None:
            u["structure_tokens_voted"] = list(src["structure_tokens"])
        u["asset"] = src.get("exemplar_asset") or u.get("asset")

    # refresh placement tokens from instance floor/bay
    placement = out.get("layout", {}).get("placement")
    if placement:
        for u, tid in zip(wins, labels):
            fl = u.get("floor")
            bay = u.get("bay")
            if fl is None or bay is None:
                continue
            fl, bay = int(fl), int(bay)
            if 0 <= fl < len(placement) and 0 <= bay < len(placement[fl]):
                placement[fl][bay] = f"win_type_{int(tid):02d}"

    # update window_types counts; drop empty types from meta only
    counts = Counter(int(x) for x in labels)
    kept = []
    for t in out.get("window_types", []):
        tid = int(t["type_id"])
        n = int(counts.get(tid, 0))
        t["n_instances"] = n
        if n > 0:
            kept.append(t)
    out["window_types"] = kept
    out["meta"]["n_types"] = len(kept)
    out["meta"]["n_units"] = len(wins)
    out["meta"]["notes"] = (
        (out["meta"].get("notes") or "")
        + " | Sym-repair patched types (structure IR from dest type vote)."
    )
    return out


def draw_clusters(
    facade: Image.Image,
    boxes: list[list[float]],
    labels: list[int],
    title: str,
    *,
    cx: float | None = None,
    axis_source: str = "",
    changed: set[int] | None = None,
    doors: list[tuple[list[float], str, tuple[int, int, int]]] | None = None,
) -> Image.Image:
    items = [{"box_xyxy": b, "id": f"u{i:03d}"} for i, b in enumerate(boxes)]
    panel = draw_overlay(facade, items, np.asarray(labels), title=title)
    if doors:
        d = ImageDraw.Draw(panel)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
        for (x0, y0, x1, y1), tag, color in doors:
            y0, y1 = y0 + TITLE_BAR_H, y1 + TITLE_BAR_H
            d.rectangle([x0, y0, x1, y1], outline=color, width=3)
            tw = 8 * len(tag) + 8
            top = max(TITLE_BAR_H, y0 - 18)
            d.rectangle([x0, top, x0 + tw, top + 18], fill=color)
            d.text((x0 + 3, top + 1), tag, fill=(255, 255, 255), font=font)
    if changed:
        d = ImageDraw.Draw(panel)
        for i in changed:
            x0, y0, x1, y1 = boxes[i]
            d.rectangle(
                [x0, y0 + TITLE_BAR_H, x1, y1 + TITLE_BAR_H],
                outline=(255, 220, 40),
                width=3,
            )
    if cx is not None:
        _draw_axis(panel, cx, label=f"axis cx={cx:.0f} ({axis_source})")
    return panel


def hstack(imgs: list[Image.Image], *, gap: int = 8, bg=(20, 20, 22)) -> Image.Image:
    imgs = [im.convert("RGB") for im in imgs]
    h = max(im.height for im in imgs)
    padded = []
    for im in imgs:
        if im.height == h:
            padded.append(im)
        else:
            c = Image.new("RGB", (im.width, h), bg)
            c.paste(im, (0, 0))
            padded.append(c)
    w = sum(im.width for im in padded) + gap * (len(padded) - 1)
    canvas = Image.new("RGB", (w, h), bg)
    x = 0
    for im in padded:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    return canvas


def mark_render_clusters(
    render_path: Path,
    compile_json: Path,
    labels: list[int],
    *,
    rf,
    base,
    ortho_zoom: float = 0.95,
    box_margin: float = 4.0,
) -> Image.Image | None:
    """Draw type-cluster boxes on the ortho render using the photo panel palette.

    ``render_facade`` has its own ``T#`` palette keyed by type_id; we re-draw with
    ``cluster_palette`` + dense ``c#`` ranks so colors match the photo panels.
    """
    if not render_path.is_file() or not compile_json.is_file():
        return None
    facade = json.loads(compile_json.read_text())
    im = Image.open(render_path).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    uniq = sorted({int(x) for x in labels})
    remap = {c: i for i, c in enumerate(uniq)}
    pal = base.cluster_palette(len(uniq))
    to_px = rf.render_pixel_mapper(
        render_path, facade, res_x=im.width, res_y=im.height, ortho_zoom=ortho_zoom
    )

    for w in rf.planned_windows(facade):
        tid = int(w["type_id"])
        if w.get("kind") == "door":
            color = rf.type_color(tid, "door")
            tag = f"D{tid:02d}"
        elif tid in remap:
            cid = remap[tid]
            color = pal[cid]
            tag = f"c{cid}"
        else:
            continue
        u0, v1 = to_px(w["x0"], w["z0"])
        u1, v0 = to_px(w["x1"], w["z1"])
        x0, x1 = sorted((u0, u1))
        y0, y1 = sorted((v0, v1))
        x0, y0, x1, y1 = x0 - box_margin, y0 - box_margin, x1 + box_margin, y1 + box_margin
        d.rectangle([x0, y0, x1, y1], outline=color + (255,), width=3)
        d.rectangle([x0, y0, x1, y1], fill=color + (40,))
        tw = 8 * len(tag) + 8
        d.rectangle([x0, max(0.0, y0 - 18), x0 + tw, y0], fill=color + (225,))
        ink = (255, 255, 255) if w.get("kind") == "door" else (0, 0, 0)
        d.text((x0 + 3, max(0.0, y0 - 17)), tag, fill=ink, font=font)

    return Image.alpha_composite(im, ov).convert("RGB")


def crop_render(
    im: Image.Image, *, pad: int = 12, ref: Image.Image | None = None
) -> Image.Image:
    """Trim the flat ortho background so the façade fills the panel.

    ``ref`` supplies the bbox (use the un-annotated render so the cluster legend
    does not widen the crop).
    """
    rgb = im.convert("RGB")
    arr = np.asarray((ref or rgb).convert("RGB")).astype(np.int16)
    # background = modal corner color of the ortho plate
    corners = np.stack(
        [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    ).astype(np.int16)
    bg = np.median(corners, axis=0)
    diff = np.abs(arr - bg).sum(axis=2)
    mask = diff > 18
    if not mask.any():
        return rgb
    ys, xs = np.nonzero(mask)
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(rgb.width, int(xs.max()) + 1 + pad)
    y1 = min(rgb.height, int(ys.max()) + 1 + pad)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return rgb
    return rgb.crop((x0, y0, min(x1, rgb.width), min(y1, rgb.height)))


def fit_height(im: Image.Image, h: int) -> Image.Image:
    if im.height == h:
        return im
    w = max(1, int(im.width * h / im.height))
    return im.resize((w, h), Image.Resampling.LANCZOS)


def label_bar(text: str, width: int, *, fill=(230, 230, 230)) -> Image.Image:
    bar = Image.new("RGB", (width, 28), (24, 24, 28))
    d = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except OSError:
        font = ImageFont.load_default()
    d.text((8, 6), text, fill=fill, font=font)
    return bar


def run_blender(dsl_path: Path, out_dir: Path, *, blender: Path, samples: int, force: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    render_png = out_dir / "facade_render.png"
    if render_png.is_file() and not force:
        return render_png
    env = os.environ.copy()
    compiler = E2E / "vendor" / "window_compiler"
    if (compiler / "main.py").is_file():
        env["FACADE_COMPILER_ROOT"] = str(compiler)
    cmd = [
        sys.executable,
        str(E2E / "scripts" / "render_facade.py"),
        "--recovery",
        str(dsl_path),
        "--out-dir",
        str(out_dir),
        "--blender",
        str(blender),
        "--render",
        "--samples",
        str(samples),
        "--force",
    ]
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=str(E2E))
    return render_png


def main() -> None:
    args = parse_args()
    stems = (
        [s.strip() for s in args.stems.split(",") if s.strip()]
        if args.stems.strip()
        else default_stems(args.sam3_root, ROOT / "runs" / "symmetry_from_sam3")
    )
    if not stems:
        raise SystemExit("no stems")
    if not args.skip_render and (args.blender is None or not Path(args.blender).is_file()):
        raise SystemExit(
            f"Blender not found ({args.blender}); pass --blender or --skip-render"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}  stems={len(stems)}  out={args.out_dir}")

    patch = _load_script(ROOT / "scripts" / "overlay_facade_patch_layout.py", "fs_patch")
    base = _load_script(ROOT / "scripts" / "overlay_facade_asset_clusters.py", "fs_base")
    # facade_e2e/scripts/render_facade.py: ortho projection + placement mirror
    os.environ.setdefault("FACADE_COMPILER_ROOT", str(E2E / "vendor" / "window_compiler"))
    rf = _load_script(E2E / "scripts" / "render_facade.py", "fs_render_facade")
    print(f"load {args.dino}…")
    model = torch.hub.load("facebookresearch/dinov2", args.dino, pretrained=True)
    model = model.to(device).eval()

    summary = []
    strips: list[Image.Image] = []

    for stem in stems:
        run_dir = args.sam3_root / f"e2e_sam3_cluster_{stem}"
        if not (run_dir / "facade_dsl.json").is_file():
            print(f"skip missing {run_dir}")
            continue
        print(f"\n=== {stem} ===")
        facade, boxes, labels_before, meta = load_sam3_baseline(run_dir)
        n = len(boxes)
        if n < 2:
            continue
        iw, ih = facade.size
        cx, cy, extent, axis_source = resolve_axis(
            stem, boxes, (iw, ih), args.axis_root
        )
        feats = embed_roi(
            facade,
            boxes,
            model=model,
            patch=patch,
            base=base,
            device=device,
            facade_max_side=args.facade_max_side,
            pca_dim=args.pca_dim,
            seed=args.seed,
        )
        windows = [
            {"id": f"u{i:03d}", "box_xyxy": boxes[i], "cluster": int(labels_before[i])}
            for i in range(n)
        ]
        labels_raw, log, _ = repair_all_clusters(
            boxes=boxes,
            labels=list(labels_before),
            feats=feats,
            windows=windows,
            facade=facade,
            axis="v",
            cx=cx,
            cy=cy,
            facade_extent=list(extent),
            img_size=(iw, ih),
            extent_mode="boxes",
            twin_mirror_min=args.twin_mirror_min,
            match_iou=args.match_iou,
            match_center=args.match_center,
            allow_private=False,
            size_rel_tol=args.size_rel_tol,
            size_mismatch_sym_gain=args.size_mismatch_sym_gain,
            size_mismatch_partner_iou=args.size_mismatch_partner_iou,
            size_mismatch_contain=args.size_mismatch_contain,
        )
        # keep original type-id space for DSL / structure IR lookup
        labels_after = [int(x) for x in labels_raw]
        changed = {i for i in range(n) if int(labels_before[i]) != labels_after[i]}
        k0 = len(set(labels_before))
        k1 = len(set(labels_after))
        s0 = facade_sym(
            boxes, labels_before, axis="v", extent_mode="boxes", image_size=(iw, ih)
        ).normalized
        s1 = facade_sym(
            boxes, labels_after, axis="v", extent_mode="boxes", image_size=(iw, ih)
        ).normalized
        print(
            f"  repair k {k0}→{k1}  moves={len(changed)}  "
            f"obj {s0:.4f}→{s1:.4f}  axis={cx:.1f}({axis_source})"
        )

        stem_dir = args.out_dir / stem
        stem_dir.mkdir(parents=True, exist_ok=True)
        dsl0 = json.loads((run_dir / "facade_dsl.json").read_text())
        dsl1 = patch_dsl_types(dsl0, labels_after)
        dsl_path = stem_dir / "facade_dsl_repaired.json"
        dsl_path.write_text(json.dumps(dsl1, indent=2) + "\n")
        door_marks = [
            (
                [float(v) for v in u["box_xyxy"]],
                f"D{int(u.get('type_id', 0)):02d}",
                rf.type_color(int(u.get("type_id", 0)), "door"),
            )
            for u in dsl0.get("instances", [])
            if u.get("kind") == "door"
        ]

        before_cl = draw_clusters(
            facade,
            boxes,
            labels_before,
            f"{stem} clusters BEFORE  k={k0}",
            cx=cx,
            axis_source=axis_source,
            doors=door_marks,
        )
        after_cl = draw_clusters(
            facade,
            boxes,
            labels_after,
            f"{stem} clusters AFTER  k={k1} moves={len(changed)}",
            cx=cx,
            axis_source=axis_source,
            changed=changed,
            doors=door_marks,
        )
        before_cl.save(stem_dir / "clusters_before.png")
        after_cl.save(stem_dir / "clusters_after.png")

        # baseline blender (from e2e) if present
        base_blender = run_dir / "blender" / "facade_render.png"
        base_compare = run_dir / "blender" / "compare_photo_vs_render.png"
        if not args.skip_render:
            blender_out = stem_dir / "blender"
            try:
                render_png = run_blender(
                    dsl_path,
                    blender_out,
                    blender=Path(args.blender),
                    samples=args.samples,
                    force=args.force_render,
                )
            except Exception as e:
                print(f"  blender FAILED: {e}")
                render_png = None
        else:
            render_png = stem_dir / "blender" / "facade_render.png"
            if not render_png.is_file():
                render_png = None

        # compose full-stack strip
        row_imgs: list[Image.Image] = [before_cl, after_cl]
        captions = ["clusters BEFORE", "clusters AFTER (Sym repair)"]
        if args.blender_before and base_blender.is_file():
            row_imgs.append(crop_render(Image.open(base_blender)))
            captions.append("Blender BEFORE (e2e box GMM)")
        if render_png is not None and Path(render_png).is_file():
            plain = Image.open(render_png)
            marked = mark_render_clusters(
                Path(render_png),
                Path(render_png).with_name("facade_compile.json"),
                labels_after,
                rf=rf,
                base=base,
            )
            if marked is not None:
                marked.save(stem_dir / "blender_clusters.png")
                row_imgs.append(crop_render(marked, ref=plain))
                captions.append("Blender 3D + clusters (Sym-repaired)")
            else:
                row_imgs.append(crop_render(plain))
                captions.append("Blender 3D (Sym-repaired types)")
        elif base_compare.is_file() and render_png is None:
            # still show baseline compare if repair render missing
            pass

        target_h = 420
        fitted = [fit_height(im, target_h) for im in row_imgs]
        # caption bars
        caps = [
            label_bar(c, fitted[i].width)
            for i, c in enumerate(captions)
        ]
        cols = []
        for cap, im in zip(caps, fitted):
            col = Image.new("RGB", (im.width, cap.height + im.height), (20, 20, 22))
            col.paste(cap, (0, 0))
            col.paste(im, (0, cap.height))
            cols.append(col)
        strip = hstack(cols, gap=10)
        hdr = label_bar(
            f"{stem}  full-stack  moves={len(changed)}  "
            f"k {k0}→{k1}  obj {s0:.4f}→{s1:.4f}  axis={cx:.0f}",
            strip.width,
            fill=(250, 250, 250),
        )
        full = Image.new("RGB", (strip.width, hdr.height + strip.height), (16, 16, 18))
        full.paste(hdr, (0, 0))
        full.paste(strip, (0, hdr.height))
        full.save(stem_dir / "fullstack_strip.png")
        strips.append(full)

        row = {
            "stem": stem,
            "k_before": k0,
            "k_after": k1,
            "label_moves": len(changed),
            "changed_ids": sorted(changed),
            "labels_before": [int(x) for x in labels_before],
            "labels_after": labels_after,
            "axis_cx": cx,
            "axis_source": axis_source,
            "obj_before": s0,
            "obj_after": s1,
            "dsl_repaired": str(dsl_path),
            "blender_after": str(render_png) if render_png else None,
            "blender_before": str(base_blender) if base_blender.is_file() else None,
            "log_tail": log[-8:],
        }
        (stem_dir / "result.json").write_text(json.dumps(row, indent=2) + "\n")
        (stem_dir / "log.txt").write_text("\n".join(log) + "\n")
        summary.append(row)

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if strips:
        # contact of all strips
        W = min(1600, max(s.width for s in strips))
        resized = []
        for s in strips:
            h = int(s.height * W / s.width)
            resized.append(s.resize((W, h), Image.Resampling.LANCZOS))
        gap = 16
        H = sum(im.height for im in resized) + gap * (len(resized) - 1) + 48
        contact = Image.new("RGB", (W, H), (14, 14, 16))
        d = ImageDraw.Draw(contact)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
            )
        except OSError:
            font = ImageFont.load_default()
        d.text(
            (12, 12),
            "Full-stack: SAM3 + box GMM → Sym repair → Blender",
            fill=(240, 240, 240),
            font=font,
        )
        y = 48
        for im in resized:
            contact.paste(im, (0, y))
            y += im.height + gap
        contact.save(args.out_dir / "contact_fullstack.png")
        print(f"wrote {args.out_dir / 'contact_fullstack.png'}")

    print("\n==== SUMMARY ====")
    for r in summary:
        print(
            f"{r['stem']:12} moves={r['label_moves']:2}  "
            f"k {r['k_before']}→{r['k_after']}  "
            f"blender={'yes' if r['blender_after'] else 'no'}"
        )
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
