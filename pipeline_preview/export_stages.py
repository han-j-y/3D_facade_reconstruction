#!/usr/bin/env python3
"""Write one image per pipeline stage (1–7) without editing run_pipeline.py.

  pipeline_preview\\.venv\\Scripts\\python pipeline_preview\\export_stages.py ^
    --image pipeline_preview\\input_cmp_b0168.png --from-index pipeline_preview\\out\\index_raw_input_cmp_b0168.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from facade_recovery.paths import (  # noqa: E402
    default_structure_ckpt,
    resolve_blender,
    resolve_compiler_root,
)
import run_pipeline as rp  # noqa: E402
import run_stages as rs  # noqa: E402

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


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _type_color(tid: int) -> tuple[int, int, int]:
    return TYPE_PALETTE[int(tid) % len(TYPE_PALETTE)]


def _save(im: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    print(f"  -> {path}")
    return path


def draw_cluster(facade: Image.Image, clustered: dict) -> Image.Image:
    labels = np.asarray(clustered["labels"], dtype=np.int32)
    items = [{"box_xyxy": b} for b in clustered["merged_boxes"]]
    n = len(set(int(x) for x in labels.tolist()))
    return rp.base.draw_overlay(
        facade, items, labels, title=f"3. Cluster window types  types={n} units={len(items)}"
    )


def draw_assetize(
    facade: Image.Image,
    units: list[dict],
    types: list[dict],
    out_dir: Path,
) -> Image.Image:
    labels = np.array([int(u["type_id"]) for u in units], dtype=np.int32)
    items = [{"box_xyxy": u["box_xyxy"]} for u in units]
    ov = rp.base.draw_overlay(
        facade,
        items,
        labels,
        title=f"4. Assetize crops  types={len(types)} units={len(units)}",
    )
    cell = 96
    strip_h = cell + 32
    n_types = max(1, len(types))
    strip = Image.new("RGB", (max(ov.width, n_types * (cell + 8)), strip_h), (18, 18, 18))
    d = ImageDraw.Draw(strip)
    font = _font(12)
    for t in types:
        tid = int(t["type_id"])
        x = tid * (cell + 8) + 4
        p = out_dir / t["exemplar_asset"] if not Path(t["exemplar_asset"]).is_file() else Path(t["exemplar_asset"])
        if not p.is_file():
            p = out_dir / t["exemplar_asset"]
        if p.is_file():
            im = Image.open(p).convert("RGB")
            im.thumbnail((cell, cell), Image.Resampling.LANCZOS)
            strip.paste(im, (x, 22))
        d.text((x, 4), f"T{tid} n={t['n_instances']}", fill=(230, 230, 230), font=font)
    canvas = Image.new("RGB", (max(ov.width, strip.width), ov.height + strip_h + 8), (12, 12, 12))
    canvas.paste(ov, (0, 0))
    canvas.paste(strip, (0, ov.height + 8))
    return canvas


def draw_structure(types: list[dict], out_dir: Path, note: str) -> Image.Image:
    cell = 160
    rows = max(1, len(types))
    w, h = 1100, 48 + rows * (cell + 16)
    im = Image.new("RGB", (w, h), (18, 18, 18))
    d = ImageDraw.Draw(im)
    d.text((12, 10), f"5. Predict structure  {note}", fill=(240, 240, 240), font=_font(18))
    font = _font(13)
    for i, t in enumerate(types):
        y = 44 + i * (cell + 16)
        tid = int(t["type_id"])
        color = _type_color(tid)
        d.rectangle([8, y, 8 + cell, y + cell], outline=color, width=3)
        p = out_dir / t.get("exemplar_asset", "")
        if p.is_file():
            crop = Image.open(p).convert("RGB")
            crop.thumbnail((cell - 8, cell - 8), Image.Resampling.LANCZOS)
            im.paste(crop, (12, y + 4))
        ir = t.get("structure_ir")
        tokens = t.get("structure_tokens") or []
        vote = t.get("structure_vote") or {}
        shape = None
        if isinstance(ir, dict):
            shape = (ir.get("boundary") or {}).get("shape") or (ir.get("boundary") or {}).get("params", {}).get("shape")
            params = (ir.get("boundary") or {}).get("params") or {}
            shape = shape or params.get("shape") or ir.get("type")
        tok_s = " ".join(str(x) for x in tokens[:18])
        if len(tokens) > 18:
            tok_s += " …"
        lines = [
            f"T{tid}  n={t.get('n_instances')}  shape={shape}",
            f"vote {vote.get('winner_count', 0)}/{vote.get('n_valid', 0)} unique={vote.get('n_unique', 0)}",
            tok_s or "(no IR — missing structure_best.pt)",
        ]
        tx = 8 + cell + 16
        for li, line in enumerate(lines):
            d.text((tx, y + 8 + li * 22), line[:90], fill=(220, 220, 220), font=font)
    return im


def draw_dsl(facade: Image.Image, dsl: dict, units: list[dict]) -> Image.Image:
    layout = dsl["layout"]
    floors = layout["floors"]
    bays = layout["bays"]
    placement = layout["placement"]
    n_f, n_b = len(floors), len(bays)
    cell_w, cell_h = 72, 56
    pad = 48
    grid_w = pad + n_b * cell_w + 16
    grid_h = pad + n_f * cell_h + 16
    grid = Image.new("RGB", (grid_w, grid_h), (24, 24, 24))
    d = ImageDraw.Draw(grid)
    d.text((8, 8), "6. Convert to DSL  floor x bay", fill=(240, 240, 240), font=_font(16))
    font = _font(11)
    for r, fl in enumerate(floors):
        for c, bay in enumerate(bays):
            x0 = pad + c * cell_w
            y0 = pad + r * cell_h
            tok = placement[r][c] if c < len(placement[r]) else None
            if tok:
                tid = int(str(tok).rsplit("_", 1)[-1])
                col = _type_color(tid)
                d.rectangle([x0, y0, x0 + cell_w - 4, y0 + cell_h - 4], fill=col, outline=(255, 255, 255))
                d.text((x0 + 4, y0 + 4), str(tok).replace("win_type_", "T"), fill=(0, 0, 0), font=font)
            else:
                d.rectangle([x0, y0, x0 + cell_w - 4, y0 + cell_h - 4], outline=(80, 80, 80))
                d.text((x0 + 8, y0 + 16), "empty", fill=(90, 90, 90), font=font)
            if r == 0:
                d.text((x0, pad - 16), fl["name"] if False else bay["name"], fill=(180, 180, 180), font=font)
        d.text((4, pad + r * cell_h + 16), fl["name"], fill=(180, 180, 180), font=font)

    labels = np.array([int(u["type_id"]) for u in units], dtype=np.int32)
    items = [{"box_xyxy": u["box_xyxy"]} for u in units]
    ov = rp.base.draw_overlay(
        facade,
        items,
        labels,
        title=f"6. DSL instances on photo  floors={n_f} bays={n_b}",
    )
    w = ov.width + grid.width + 12
    h = max(ov.height, grid.height)
    canvas = Image.new("RGB", (w, h), (12, 12, 12))
    canvas.paste(ov, (0, 0))
    canvas.paste(grid, (ov.width + 8, 0))
    return canvas


def draw_compile_schematic(dsl: dict) -> Image.Image:
    layout = dsl["layout"]
    floors = layout["floors"]
    bays = layout["bays"]
    placement = layout["placement"]
    types = {t["name"]: t for t in dsl.get("window_types") or []}
    w, h = 900, 1100
    im = Image.new("RGB", (w, h), (210, 198, 170))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w, 40], fill=(30, 30, 30))
    d.text((12, 10), "7. Compile the DSL  (2D schematic — Blender not required)", fill=(0, 220, 255), font=_font(16))
    margin = 40
    inner_w, inner_h = w - 2 * margin, h - margin - 56
    y = 56
    for r, fl in enumerate(floors):
        rh = max(48, int(inner_h * float(fl.get("h_norm", 1.0) / max(0.05, sum(f.get("h_norm", 1) for f in floors)))))
        x = margin
        total_w = sum(float(b.get("w_norm", 1)) for b in bays) or 1.0
        for c, bay in enumerate(bays):
            cw = max(24, int(inner_w * float(bay.get("w_norm", 1)) / total_w))
            tok = placement[r][c] if r < len(placement) and c < len(placement[r]) else None
            d.rectangle([x, y, x + cw - 3, y + rh - 3], outline=(120, 100, 70), width=1)
            if tok:
                tid = int(str(tok).rsplit("_", 1)[-1])
                col = _type_color(tid)
                inset = 8
                d.rectangle(
                    [x + inset, y + inset, x + cw - 3 - inset, y + rh - 3 - inset],
                    fill=(40, 60, 80),
                    outline=col,
                    width=3,
                )
                ir = (types.get(tok) or {}).get("structure_ir")
                shape = ""
                if isinstance(ir, dict):
                    shape = ((ir.get("boundary") or {}).get("params") or {}).get("shape") or ""
                d.text((x + inset + 4, y + inset + 4), f"T{tid} {shape}", fill=col, font=_font(11))
            x += cw
        y += rh
    return im


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=HERE / "out")
    ap.add_argument("--from-index", type=Path, default=None)
    ap.add_argument("--prompt", default="window")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--min-side", type=int, default=24)
    ap.add_argument("--max-side-frac", type=float, default=0.55)
    ap.add_argument("--col-tol", type=float, default=0.04)
    ap.add_argument("--row-tol", type=float, default=0.055)
    ap.add_argument("--spatial-strength", type=float, default=1.8)
    ap.add_argument("--unary-weight", type=float, default=0.9)
    ap.add_argument("--dino", default="dinov2_vits14")
    ap.add_argument("--facade-max-side", type=int, default=896)
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--device", default="cuda" if rp.torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = rp.torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    facade = Image.open(args.image).convert("RGB")

    # --- 1 Detect ---
    print("=== 1 Detect Windows ===")
    if args.from_index is not None:
        facade, raw_windows, _ = rp.load_from_index(args.from_index)
        print(f"reusing index: {len(raw_windows)}")
    else:
        raw_windows = rp.detect_windows(
            facade,
            prompt=args.prompt,
            threshold=args.threshold,
            min_side=args.min_side,
            max_side_frac=args.max_side_frac,
            device=device,
        )
    _save(
        rs.draw_detect(facade, raw_windows, len(raw_windows)),
        out_dir / f"s1_detect_windows_{stem}.png",
    )
    raw_boxes = [w["box_xyxy"] for w in raw_windows]

    # --- 2 Unitize ---
    print("=== 2 Unitize merge boxes ===")
    unitized = rs.unitize_boxes(
        facade, raw_boxes, row_tol=args.row_tol, col_tol=args.col_tol
    )
    _save(
        rs.draw_unitize(facade, raw_boxes, unitized),
        out_dir / f"s2_unitize_merge_boxes_{stem}.png",
    )

    # --- 3 Cluster ---
    print(f"=== 3 Cluster the window types ({args.dino}) ===")
    dino = rp.torch.hub.load("facebookresearch/dinov2", args.dino, pretrained=True)
    dino = dino.to(device).eval()
    clustered = rp.cluster_units(
        facade,
        raw_boxes,
        model=dino,
        device=device,
        facade_max_side=args.facade_max_side,
        pca_dim=args.pca_dim,
        seed=args.seed,
        k_max=args.k_max,
        col_tol=args.col_tol,
        row_tol=args.row_tol,
        spatial_strength=args.spatial_strength,
        unary_weight=args.unary_weight,
    )
    del dino
    rp.torch.cuda.empty_cache()
    _save(draw_cluster(facade, clustered), out_dir / f"s3_cluster_window_types_{stem}.png")

    medoids = rp.pick_medoids(clustered["feats"], clustered["labels"])
    assets_dir = out_dir / "assets" / "types"
    assets_dir.mkdir(parents=True, exist_ok=True)

    units: list[dict] = []
    for ui, box in enumerate(clustered["merged_boxes"]):
        tid = int(clustered["labels"][ui])
        type_dir = assets_dir / f"type_{tid:02d}"
        type_dir.mkdir(parents=True, exist_ok=True)
        crop_path = type_dir / f"unit_{ui:03d}.png"
        facade.crop(tuple(box)).save(crop_path)
        units.append(
            {
                "unit_id": ui,
                "box_xyxy": box,
                "floor": int(clustered["floor"][ui]),
                "bay": int(clustered["bay"][ui]),
                "type_id": tid,
                "member_raw_idxs": clustered["members"][ui],
                "asset": str(crop_path.relative_to(out_dir)),
                "is_exemplar": ui == medoids[tid],
            }
        )

    types_out: list[dict] = []
    for tid, med_i in sorted(medoids.items()):
        exemplar = units[med_i]
        canon = assets_dir / f"type_{tid:02d}" / "exemplar.png"
        shutil.copy(out_dir / exemplar["asset"], canon)
        member_units = [u for u in units if int(u["type_id"]) == int(tid)]
        types_out.append(
            {
                "type_id": tid,
                "name": f"win_type_{tid:02d}",
                "n_instances": len(member_units),
                "exemplar_unit": med_i,
                "exemplar_asset": str(canon.relative_to(out_dir)),
                "structure_ir": None,
            }
        )

    print("=== 4 Assetize crops ===")
    _save(
        draw_assetize(facade, units, types_out, out_dir),
        out_dir / f"s4_assetize_crops_{stem}.png",
    )

    # --- 5 Structure ---
    print("=== 5 Predict structure ===")
    ckpt = default_structure_ckpt()
    struct_note = "no checkpoint"
    if ckpt.is_file():
        print(f"loading {ckpt}")
        pred = rp.StructurePredictor(ckpt, device, structure_only=True)
        struct_note = ckpt.name
        for t in types_out:
            tid = int(t["type_id"])
            member_units = [u for u in units if int(u["type_id"]) == tid]
            med_i = int(t["exemplar_unit"])
            member_preds = []
            for u in member_units:
                p = pred.predict(out_dir / u["asset"])
                member_preds.append({"unit_id": int(u["unit_id"]), **p})
            voted = rp.vote_cluster_ir(member_preds, prefer_unit_id=med_i)
            t["structure_ir"] = voted["structure_ir"]
            t["structure_tokens"] = voted["structure_tokens"]
            t["structure_vote"] = voted["vote"]
            for u in member_units:
                u["structure_ir"] = voted["structure_ir"]
        del pred
        rp.torch.cuda.empty_cache()
    else:
        print(f"skip structure IR (missing {ckpt})")
        struct_note = f"skipped ({ckpt.name} missing)"
    _save(
        draw_structure(types_out, out_dir, struct_note),
        out_dir / f"s5_predict_structure_{stem}.png",
    )

    # --- 6 DSL ---
    print("=== 6 Convert to DSL ===")
    dsl = rp.build_facade_dsl(
        facade_id=stem,
        image_path=str(Path(args.image).resolve()),
        image_size=facade.size,
        units=units,
        types=types_out,
    )
    dsl_path = out_dir / f"facade_dsl_{stem}.json"
    dsl_path.write_text(json.dumps(dsl, indent=2) + "\n", encoding="utf-8")
    print(f"  dsl -> {dsl_path}")
    _save(draw_dsl(facade, dsl, units), out_dir / f"s6_convert_to_dsl_{stem}.png")

    # --- 7 Compile ---
    print("=== 7 Compile the DSL ===")
    blender = resolve_blender(None)
    compile_path = out_dir / f"s7_compile_dsl_{stem}.png"
    if blender:
        try:
            import os
            import subprocess

            blender_out = out_dir / "blender"
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            compiler = resolve_compiler_root(None)
            if compiler:
                env["FACADE_COMPILER_ROOT"] = str(compiler)
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "render_facade.py"),
                "--recovery",
                str(dsl_path),
                "--out-dir",
                str(blender_out),
                "--blender",
                str(blender),
                "--render",
                "--force",
            ]
            print("  blender render…")
            proc = subprocess.run(cmd, env=env)
            rendered = blender_out / "facade_render.png"
            if not rendered.is_file():
                rendered = next(blender_out.glob("*.png"), None)
            if rendered and Path(rendered).is_file():
                Image.open(rendered).convert("RGB").save(compile_path)
                print(f"  -> {compile_path}")
            else:
                print(f"  blender exit {proc.returncode}; writing 2D schematic")
                _save(draw_compile_schematic(dsl), compile_path)
        except Exception as exc:
            print(f"  blender failed ({exc}); writing 2D schematic")
            _save(draw_compile_schematic(dsl), compile_path)
    else:
        print("  Blender not found; writing 2D schematic")
        _save(draw_compile_schematic(dsl), compile_path)

    print("done: 7 stage images in", out_dir)


if __name__ == "__main__":
    main()
