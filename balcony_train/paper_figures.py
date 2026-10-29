"""Qualitative figures for the paper, composed from pipeline outputs.

* ``fig_results.jpg``  photo | recovered layout (types, balconies, enclosures) | 3D render
* ``fig_stages.jpg``   window and balcony stages for one facade
* ``fig_edits.jpg``    original render and each edit from ``paper_edit_demo.py``
* ``fig_failures.jpg`` enclosure misses, false positives and wrong rectangles

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/paper_figures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.enclosed_grid_store import (  # noqa: E402
    DEFAULT_EVAL_LABELS_JSONL,
    DEFAULT_EVAL_PHOTO_DIR,
    find_photo,
    load_labels,
)

PAPER = ROOT / "runs" / "paper_eval"
BATCH = ROOT / "runs" / "batch"
ENC = ROOT / "runs" / "cell_clf" / "compare_clf_rules_gate_ext_relabel"
OUT = PAPER / "figures"
H = 420  # row height of every tile
COLORS = [(230, 57, 70), (69, 123, 157), (42, 157, 143), (233, 196, 106), (155, 93, 229), (244, 162, 97),
          (29, 53, 87), (106, 153, 78), (188, 108, 37), (0, 150, 199), (217, 4, 41), (128, 128, 128)]


def font(size: int) -> ImageFont.ImageFont:
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit(im: Image.Image, h: int = H) -> Image.Image:
    return im.resize((max(1, round(im.width * h / im.height)), h), Image.LANCZOS)


def crop_render(im: Image.Image, pad: int = 8) -> Image.Image:
    a = np.asarray(im.convert("RGB")).astype(int)
    bg = a[2, 2]
    ys, xs = np.where(np.abs(a - bg).sum(-1) > 25)
    if len(xs) == 0:
        return im
    return im.crop((max(0, xs.min() - pad), max(0, ys.min() - pad),
                    min(im.width, xs.max() + pad), min(im.height, ys.max() + pad)))


def caption(im: Image.Image, text: str, size: int = 22) -> Image.Image:
    bar = 34
    out = Image.new("RGB", (im.width, im.height + bar), "white")
    out.paste(im, (0, bar))
    ImageDraw.Draw(out).text((6, 5), text, fill=(20, 20, 20), font=font(size))
    return out


def hstack(tiles: list[Image.Image], gap: int = 10) -> Image.Image:
    w = sum(t.width for t in tiles) + gap * (len(tiles) - 1)
    h = max(t.height for t in tiles)
    out = Image.new("RGB", (w, h), "white")
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width + gap
    return out


def vstack(rows: list[Image.Image], gap: int = 14) -> Image.Image:
    w = max(r.width for r in rows)
    h = sum(r.height for r in rows) + gap * (len(rows) - 1)
    out = Image.new("RGB", (w, h), "white")
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height + gap
    return out


def layout_overlay(stem: str) -> Image.Image:
    """Photo with window types (colored by type), open balconies (white) and enclosed cells (cyan)."""
    photo = Image.open(DEFAULT_EVAL_PHOTO_DIR / f"{stem}.jpg").convert("RGB")
    dsl = json.loads((PAPER / stem / "balcony" / "facade_dsl_with_balconies.json").read_text(encoding="utf-8"))
    over = Image.new("RGBA", photo.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    lw = max(2, photo.width // 300)
    for inst in dsl.get("instances", []):
        c = (139, 69, 19) if inst.get("kind") == "door" else COLORS[int(inst.get("type_id", 0)) % len(COLORS)]
        d.rectangle(inst["box_xyxy"], outline=(*c, 255), fill=(*c, 70), width=lw)
    for b in dsl["layout"].get("balconies", []):
        enclosed = b.get("type") == "balc_enclosed"
        c = (0, 229, 255) if enclosed else (255, 255, 255)
        d.rectangle(b["box_xyxy"], outline=(*c, 255), fill=(*c, 60 if enclosed else 0), width=lw + 1)
    return Image.alpha_composite(photo.convert("RGBA"), over).convert("RGB")


def fig_results(stems: list[str]) -> Image.Image:
    rows = []
    for s in stems:
        render = PAPER / s / "blender" / "facade_render.png"
        if not render.is_file():
            continue
        photo = Image.open(DEFAULT_EVAL_PHOTO_DIR / f"{s}.jpg").convert("RGB")
        tiles = [
            caption(fit(photo), f"{s}  photo"),
            caption(fit(layout_overlay(s)), "recovered layout (types / balconies / enclosures)"),
            caption(fit(crop_render(Image.open(render).convert("RGB"))), "compiled 3D (DSL)"),
        ]
        rows.append(hstack(tiles))
    return vstack(rows)


def fig_stages(stem: str) -> Image.Image:
    b = PAPER / stem / "balcony"
    items = [
        (BATCH / stem / "raw_windows.png", "1 SAM3 windows"),
        (BATCH / stem / "overview.png", "2-5 unitize, grid, types, vote"),
        (b / f"s1b_filtered_balconies_{stem}.png", "6 balconies (filtered)"),
        (b / f"s2b_enclosed_snap_{stem}.png", "7 enclosure cells"),
        (b / f"s3_cluster_balcony_types_{stem}.png", "8 balcony types"),
    ]
    tiles = [caption(fit(Image.open(p).convert("RGB"), 360), t) for p, t in items if p.is_file()]
    render = PAPER / stem / "blender" / "facade_render.png"
    if render.is_file():
        tiles.append(caption(fit(crop_render(Image.open(render).convert("RGB")), 360), "9 compiled 3D"))
    return vstack([hstack(tiles[:3]), hstack(tiles[3:])])


def fig_edits(stem: str) -> Image.Image | None:
    edits_json = PAPER / "edits" / stem / "edits.json"
    if not edits_json.is_file():
        return None
    edits = json.loads(edits_json.read_text(encoding="utf-8"))
    base = Image.open(PAPER / stem / "blender" / "facade_render.png").convert("RGB")
    a = np.asarray(base).astype(int)
    ys, xs = np.where(np.abs(a - a[2, 2]).sum(-1) > 25)
    box = (xs.min() - 8, ys.min() - 8, xs.max() + 8, ys.max() + 8)
    tiles = [caption(fit(base.crop(box)), "original", 20)]
    for e in edits:
        p = PAPER / "edits" / stem / e["edit"] / "blender" / "facade_render.png"
        if not p.is_file():
            continue
        label = f"{e['edit']}: {e['dsl_fields']} fields -> {e['instances_changed']} instances"
        tiles.append(caption(fit(Image.open(p).convert("RGB").crop(box)), label, 18))
    return hstack(tiles)


def fig_failures() -> Image.Image:
    picks = ["missed__cmp_x0074.jpg", "missed__cmp_x0081.jpg", "false_pos__cmp_x0073.jpg",
             "wrong__cmp_x0116.jpg", "wrong__cmp_x0127.jpg", "wrong__cmp_x0006.jpg"]
    tiles = [fit(Image.open(ENC / p).convert("RGB"), 380) for p in picks if (ENC / p).is_file()]
    return vstack([hstack(tiles[:3]), hstack(tiles[3:])])


def fig_pipeline_failures(items: list[tuple[str, str]]) -> Image.Image:
    """Photo with recovered layout next to the compiled render, for whole-pipeline failures."""
    rows = []
    for stem, why in items:
        render = PAPER / stem / "blender" / "facade_render.png"
        tiles = [caption(fit(layout_overlay(stem), 360), f"{stem}: {why}", 20)]
        if render.is_file():
            tiles.append(caption(fit(crop_render(Image.open(render).convert("RGB")), 360), "compiled 3D", 20))
        rows.append(hstack(tiles))
    return vstack(rows)


def fig_all(stems: list[str], per_row: int = 6) -> Image.Image:
    """Every eval photo: photo above its compiled render (supplementary)."""
    tiles = []
    for s in stems:
        render = PAPER / s / "blender" / "facade_render.png"
        if not render.is_file():
            continue
        top = fit(Image.open(DEFAULT_EVAL_PHOTO_DIR / f"{s}.jpg").convert("RGB"), 220)
        bot = fit(crop_render(Image.open(render).convert("RGB")), 220)
        w = max(top.width, bot.width)
        top = top.resize((w, round(top.height * w / top.width))) if top.width < w else top
        tiles.append(caption(vstack([top, bot], 4), s, 18))
    rows = [hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)]
    return vstack(rows)


def save(im: Image.Image, path: Path, max_w: int = 3600) -> None:
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=90)
    print(f"-> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # Chosen by per-photo compiled-window F1: best (0147, 0137), median (0118, 0028), plus 0005 (curtain-wall oriels).
    ap.add_argument("--stems", default="cmp_x0147,cmp_x0137,cmp_x0005,cmp_x0118,cmp_x0028")
    ap.add_argument("--stage-stem", default="cmp_x0005")
    ap.add_argument("--edit-stem", default="cmp_x0005")
    args = ap.parse_args()
    save(fig_results(args.stems.split(",")), OUT / "fig_results.jpg")
    save(fig_stages(args.stage_stem), OUT / "fig_stages.jpg")
    edits = fig_edits(args.edit_stem)
    if edits is not None:
        save(edits, OUT / "fig_edits.jpg")
    save(fig_failures(), OUT / "fig_failures.jpg")
    save(fig_pipeline_failures([
        ("cmp_x0054", "two buildings: grid inferred wrong, compiled layout breaks"),
        ("cmp_x0128", "SAM3 finds 6 of ~90 windows"),
    ]), OUT / "fig_failures_pipeline.jpg")
    stems = [
        stem
        for stem in sorted(load_labels(DEFAULT_EVAL_LABELS_JSONL))
        if find_photo(DEFAULT_EVAL_PHOTO_DIR, stem) is not None
    ]
    save(fig_all(stems), OUT / "fig_all_eval.jpg", max_w=4800)


if __name__ == "__main__":
    main()
