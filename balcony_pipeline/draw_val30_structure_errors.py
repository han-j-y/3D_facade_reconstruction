r"""Contact sheets of the railing classifier's structure errors on val30 (per unit, before any vote).

Ground truth: ``data/val30/type_gt.jsonl`` (``balcony_train/type_gt_ui.py``). Each unit's crop
(``runs/val30/<stem>/balcony/<asset>``, the crop the pipeline classifies) goes through the pipeline's railing
classifier. A tile shows the box with context around it (``--context`` x the box size on each side) and the
box in red, with the val30 id, unit index, GT group and the classifier's P(juliet). One sheet per error
direction (``juliet_as_projecting``, ``projecting_as_juliet``), paged; plus ``correct_*`` sheets for
comparison when ``--with-correct`` is given.

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_pipeline/draw_val30_structure_errors.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from draw_test100_vote_before_after import font  # noqa: E402
from run_balcony import juliet_prob  # noqa: E402

from balcony_train.model import try_load_predictor  # noqa: E402
from balcony_train.paths import DEFAULT_STRUCTURE_CKPT  # noqa: E402

TILE_W, TILE_H, CAP = 340, 260, 46


def tile(photo: Image.Image, box: list[int], ctx: float, lines: list[str], color) -> Image.Image:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    m = max(w, h) * ctx
    cx0, cy0 = max(0, int(x0 - m)), max(0, int(y0 - m))
    cx1, cy1 = min(photo.width, int(x1 + m)), min(photo.height, int(y1 + m))
    crop = photo.crop((cx0, cy0, cx1, cy1))
    s = min(TILE_W / crop.width, (TILE_H - CAP) / crop.height)
    crop = crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s))), Image.LANCZOS)
    d = ImageDraw.Draw(crop)
    d.rectangle(((x0 - cx0) * s, (y0 - cy0) * s, (x1 - cx0) * s, (y1 - cy0) * s), outline=(255, 40, 40), width=3)
    t = Image.new("RGB", (TILE_W, TILE_H), (30, 30, 30))
    t.paste(crop, ((TILE_W - crop.width) // 2, CAP + (TILE_H - CAP - crop.height) // 2))
    dt = ImageDraw.Draw(t)
    f = font(16)
    for k, line in enumerate(lines):
        dt.text((6, 3 + 21 * k), line, font=f, fill=color if k == 0 else (230, 230, 230))
    return t


def sheets(tiles: list[Image.Image], title: str, out: Path, cols: int, per_page: int) -> list[Path]:
    paths = []
    pages = [tiles[i:i + per_page] for i in range(0, len(tiles), per_page)] or [[]]
    for pi, page in enumerate(pages, 1):
        rows = max(1, (len(page) + cols - 1) // cols)
        sheet = Image.new("RGB", (cols * (TILE_W + 6) + 6, 50 + rows * (TILE_H + 6)), (15, 15, 15))
        ImageDraw.Draw(sheet).text((10, 12), f"{title}   page {pi}/{len(pages)}", font=font(22), fill=(255, 255, 255))
        for k, t in enumerate(page):
            sheet.paste(t, (6 + (k % cols) * (TILE_W + 6), 50 + (k // cols) * (TILE_H + 6)))
        p = out.with_name(f"{out.stem}_p{pi}.jpg")
        sheet.save(p, quality=88)
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=ROOT / "data" / "val30" / "type_gt.jsonl")
    ap.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "val30")
    ap.add_argument("--photos", type=Path, default=ROOT / "data" / "val30")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "val30" / "_structure_errors")
    ap.add_argument("--context", type=float, default=0.6, help="context around the box, x its larger side")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--per-page", type=int, default=36)
    ap.add_argument("--with-correct", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pred = try_load_predictor(None, torch.device(args.device), structure_ckpt=DEFAULT_STRUCTURE_CKPT)
    rows = [json.loads(l) for l in args.gt.read_text(encoding="utf-8").splitlines() if l.strip()]
    groups: dict[str, list] = {"juliet_as_projecting": [], "projecting_as_juliet": [],
                               "correct_juliet": [], "correct_projecting": []}
    listing = []
    for r in rows:
        bdir = args.run_dir / r["stem"] / "balcony"
        units = json.loads((bdir / "balcony_units.json").read_text(encoding="utf-8"))["units"]
        if [[int(v) for v in u["box_xyxy"]] for u in units] != [u["box"] for u in r["units"]]:
            print(f"skip {r['id']}: run boxes differ from the labeled boxes")
            continue
        photo = Image.open(next(p for p in (args.photos / f"{r['stem']}.jpg", args.photos / f"{r['stem']}.png")
                                if p.is_file())).convert("RGB")
        for k, (u, g) in enumerate(zip(units, r["units"])):
            if g["not_balcony"] or g["structure"] is None:
                continue
            pj = juliet_prob(pred, Image.open(bdir / u["asset"]).convert("RGB"))
            p_lab = "juliet" if pj > 0.5 else "projecting"
            key = (f"{g['structure']}_as_{p_lab}" if p_lab != g["structure"] else f"correct_{g['structure']}")
            listing.append({"id": r["id"], "stem": r["stem"], "unit": k, "group": g["group"], "box": g["box"],
                            "truth": g["structure"], "pred": p_lab, "p_juliet": round(pj, 3)})
            groups[key].append((pj, r["id"], k, photo, g["box"], g["group"]))
    (args.out_dir / "structure_val30.json").write_text(json.dumps(listing, indent=1), encoding="utf-8")
    titles = {"juliet_as_projecting": ("GT juliet, classifier projecting", (255, 120, 120)),
              "projecting_as_juliet": ("GT projecting, classifier juliet", (255, 120, 120)),
              "correct_juliet": ("GT juliet, classifier juliet (correct)", (120, 230, 120)),
              "correct_projecting": ("GT projecting, classifier projecting (correct)", (120, 230, 120))}
    for key, items in groups.items():
        if key.startswith("correct") and not args.with_correct:
            continue
        items.sort(key=lambda x: (x[1], x[2]))
        tiles = [tile(ph, box, args.context, [f"{vid} u{k} g{g}", f"P(juliet) {pj:.2f}"], titles[key][1])
                 for pj, vid, k, ph, box, g in items]
        paths = sheets(tiles, f"val30 structure: {titles[key][0]}  ({len(items)} units)",
                       args.out_dir / f"{key}.jpg", args.cols, args.per_page)
        print(f"{key}: {len(items)} units -> {', '.join(p.name for p in paths)}")
    print(f"listing -> {args.out_dir / 'structure_val30.json'}")


if __name__ == "__main__":
    main()
