"""Each of the user's three post-rules alone vs the baseline, on test100.

Rules (``postfilter.USER_RULES``): ``side_wall`` (farther than the main wall,
or cut off at the left / right border and taller than wide), ``solid_juliet``, ``no_window_near`` (roof / no window just
above, top floor exempt). Scored without ignore regions (default); ``--use-ignore``
adds the supplementary rows. Images: one row per photo, panels
ground truth | baseline | each rule; boxes a rule removed are drawn as blue dashes.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_pipeline/score_test100_user_rules.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from postfilter import USER_RULES, apply_rules  # noqa: E402
from score_test100_gt_boxes import prf, score_photo  # noqa: E402
from score_test100_postrules import DEFAULT_CKPT, DEFAULT_LABELS, classify  # noqa: E402

T100 = ROOT / "runs" / "test100"
TITLES = {"side_wall": "Rule 1: side wall (depth / edge)", "edge_depth": "Rule 1 (v1): edge / depth", "solid_juliet": "Rule 2: solid juliet",
          "no_window_near": "Rule 3: no window above"}


def dashed(d: ImageDraw.ImageDraw, b, color, w: int, dash: int = 12) -> None:
    x0, y0, x1, y1 = b
    for ax, ay, bx, by in ((x0, y0, x1, y0), (x0, y1, x1, y1), (x0, y0, x0, y1), (x1, y0, x1, y1)):
        n = max(1, int(max(abs(bx - ax), abs(by - ay)) // dash))
        for k in range(0, n, 2):
            t0, t1 = k / n, min(1.0, (k + 1) / n)
            d.line([(ax + (bx - ax) * t0, ay + (by - ay) * t0), (ax + (bx - ax) * t1, ay + (by - ay) * t1)],
                   fill=color, width=w)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=T100 / "balcony_gt_boxes.jsonl")
    ap.add_argument("--cache", type=Path, default=ROOT / "runs" / "batch" / "_rows" / "test100_sam3_stage_boxes.json")
    ap.add_argument("--dsl-dir", type=Path, default=T100 / "batch_grid2")
    ap.add_argument("--depth-dir", type=Path, default=T100 / "da3_depth")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--label-cache", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--out", type=Path, default=T100 / "balcony_user_rules_score.json")
    ap.add_argument("--img-dir", type=Path, default=T100 / "compare_rules")
    ap.add_argument("--use-ignore", action="store_true")
    ap.add_argument("--panel-height", type=int, default=760)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    gt = {json.loads(l)["stem"]: json.loads(l) for l in args.gt.read_text(encoding="utf-8").splitlines() if l.strip()}
    stems = sorted(gt)
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    labels = classify(cache, stems, args.ckpt, args.label_cache, args.device)

    variants = [("baseline", ())] + [(r, (r,)) for r in USER_RULES]
    kept: dict[str, dict[str, list]] = {n: {} for n, _ in variants}
    removed: dict[str, dict[str, list]] = {n: {} for n, _ in variants}
    for s in stems:
        dsl = json.loads((args.dsl_dir / s / "facade_dsl.json").read_text(encoding="utf-8"))
        dp = args.depth_dir / f"{s}_depth.npy"
        depth = np.load(dp) if dp.is_file() else None
        for name, rules in variants:
            k, log = apply_rules(cache[s]["filtered"], tuple(cache[s]["size"]), dsl, labels=labels[s],
                                 depth=depth, rules=rules)
            kept[name][s] = k
            removed[name][s] = [e["box"] for e in log]

    subsets = {"all": lambda s: True, "cmp": lambda s: s.startswith("cmp_"), "lsaa": lambda s: not s.startswith("cmp_")}
    results: dict = {}
    per_photo: dict = {s: {} for s in stems}
    print(f"{'variant':16s} {'subset':5s} IoU   pred   TP   FP   FN     P      R     F1   removed")
    for thr in (0.5, 0.3):
        for name, _ in variants:
            for sub, f in subsets.items():
                rows = []
                for s in stems:
                    if not f(s):
                        continue
                    ign = gt[s]["ignore"] if args.use_ignore else []
                    sc = score_photo(gt[s]["boxes"], ign, kept[name][s], thr, True)
                    rows.append(sc)
                    if thr == 0.5:
                        per_photo[s][name] = sc
                t = prf(rows)
                nrem = sum(len(removed[name][s]) for s in stems if f(s))
                results[f"{name}|{sub}|{thr}"] = {**t, "removed": nrem}
                print(f"{name:16s} {sub:5s} {thr:<4}  {t['pred']:4d} {t['tp']:4d} {t['fp']:4d} {t['fn']:4d}  "
                      f"{t['precision']:.3f}  {t['recall']:.3f}  {t['f1']:.3f}   {nrem}")
    args.out.write_text(json.dumps({"use_ignore": args.use_ignore, "results": results, "photos": per_photo,
                                    "removed": removed}, indent=0), encoding="utf-8")
    print(f"wrote {args.out}")

    # side-by-side images
    args.img_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("arial.ttf", 22)
    for s in stems:
        base = Image.open(ROOT / "data" / "test100" / "photos" / f"{s}.jpg").convert("RGB")
        W, H = base.size
        lw = max(3, W // 200)
        panels = []
        g = base.copy()
        dg = ImageDraw.Draw(g)
        for b in gt[s]["boxes"]:
            dg.rectangle(b, outline=(255, 0, 0), width=lw)
        panels.append((g, [f"Ground truth: {len(gt[s]['boxes'])}"]))
        for name, _ in variants:
            im = base.copy()
            d = ImageDraw.Draw(im)
            for b in kept[name][s]:
                d.rectangle(b, outline=(255, 0, 0), width=lw)
            for b in removed[name][s]:
                dashed(d, b, (0, 140, 255), lw)
            sc = per_photo[s][name]
            title = "Baseline" if name == "baseline" else TITLES[name]
            sub = f"TP {sc['tp']} FP {sc['fp']} FN {sc['fn']}"
            if name != "baseline":
                sub += f" | removed {len(removed[name][s])}"
            panels.append((im, [title, sub]))
        scale = args.panel_height / H
        pw = max(1, round(W * scale))
        head = 64
        out = Image.new("RGB", (len(panels) * (pw + 12), args.panel_height + head), "white")
        do = ImageDraw.Draw(out)
        for k, (im, lines) in enumerate(panels):
            x = k * (pw + 12)
            out.paste(im.resize((pw, args.panel_height), Image.BILINEAR), (x, head))
            for j, line in enumerate(lines):
                do.text((x + 6, 4 + 28 * j), line, fill="black", font=font)
        out.save(args.img_dir / f"{s}.jpg", quality=86)
    print(f"images -> {args.img_dir}")


if __name__ == "__main__":
    main()
