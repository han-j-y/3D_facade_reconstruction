r"""Rule 3 (roof / no window above) on top of rule 1, on test100.

Base: the pipeline boxes (test100 SAM3 ``filtered`` cache) with rule 1
(``side_wall``), as ``run_balcony.py`` does now. Rule 3 is
``postfilter.no_window_near``: drop a box unless a detected window is centred
over its x-range, lies above its centre, and has its bottom at most half a
window height above the box top. Boxes centred in the top floor band are kept.
Windows and floors come from ``runs/test100/batch_grid2/<stem>/facade_dsl.json``.

``--rule window_below`` is the hood rule (``postfilter.window_right_below``).
``--rule oriel`` drops an open box on an oriel cell (same floor, shared bay;
``enclosed.drop_open_under_enclosed``), with ``--oriel-cells pred`` (held-out cell
model, 89 photos) or ``gt`` (``oriel_labels.jsonl``, upper bound).
``--rule below_floor`` (rule 3 v2, user 2026-11-01) uses the floor bands
instead: drop a box whose top edge is below the bottom of its floor's window row
(``postfilter.below_floor``; rows shaded yellow in the images).

Scored against ``runs/test100/balcony_gt_boxes.jsonl`` without ignore regions
(group rule, IoU 0.5 and 0.3). Images: GT | pipeline | + rule 3; removed boxes
blue dashed, detected windows thin green, top floor band shaded (``no_window_near``)
or every floor band shaded (``below_floor``).

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_pipeline/score_test100_rule3.py
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

from postfilter import apply_rules, window_boxes  # noqa: E402
from score_sam3_label_iou import box_iou  # noqa: E402
from score_test100_gt_boxes import prf, score_photo  # noqa: E402
from score_test100_user_rules import dashed  # noqa: E402
from snap import _bands_from_window_instances, _bands_from_windows_dsl  # noqa: E402

T100 = ROOT / "runs" / "test100"
ORIEL_PRED = ROOT / "runs" / "oriel_grid_compare" / "grid2_test100" / "preds_heldout.json"
ORIEL_GT = T100 / "oriel_labels.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=T100 / "balcony_gt_boxes.jsonl")
    ap.add_argument("--cache", type=Path, default=ROOT / "runs" / "batch" / "_rows" / "test100_sam3_stage_boxes.json")
    ap.add_argument("--dsl-dir", type=Path, default=T100 / "batch_grid2")
    ap.add_argument("--depth-dir", type=Path, default=T100 / "da3_depth")
    ap.add_argument("--rule", default="no_window_near", choices=("no_window_near", "below_floor", "window_below", "oriel"))
    ap.add_argument("--oriel-cells", choices=("pred", "gt"), default="pred",
                    help="oriel rule: held-out cell model predictions, or the hand labels (upper bound)")
    ap.add_argument("--img-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--panel-height", type=int, default=800)
    args = ap.parse_args()
    sfx = {"no_window_near": "", "below_floor": "_below_floor", "window_below": "_window_below",
           "oriel": f"_oriel_{args.oriel_cells}"}[args.rule]
    oriel: dict[str, list] = {}
    if args.rule == "oriel":
        if args.oriel_cells == "pred":
            oriel = json.loads(ORIEL_PRED.read_text(encoding="utf-8"))
        else:
            oriel = {json.loads(l)["stem"]: json.loads(l)["cells"]
                     for l in ORIEL_GT.read_text(encoding="utf-8").splitlines() if l.strip()}
    args.img_dir = args.img_dir or T100 / f"compare_rule3{sfx}"
    args.out = args.out or T100 / f"balcony_rule3{sfx}_score.json"

    gt = {json.loads(l)["stem"]: json.loads(l) for l in args.gt.read_text(encoding="utf-8").splitlines() if l.strip()}
    stems = sorted(gt)
    cache = json.loads(args.cache.read_text(encoding="utf-8"))

    base, kept, removed, dsls = {}, {}, {}, {}
    for s in stems:
        dsl = json.loads((args.dsl_dir / s / "facade_dsl.json").read_text(encoding="utf-8"))
        dsls[s] = dsl
        dp = args.depth_dir / f"{s}_depth.npy"
        depth = np.load(dp) if dp.is_file() else None
        size = tuple(cache[s]["size"])
        base[s], _ = apply_rules(cache[s]["filtered"], size, dsl, depth=depth, rules=("side_wall",))
        kept[s], log = apply_rules(base[s], size, dsl, rules=(args.rule,),
                                   oriel_cells=[tuple(c) for c in oriel.get(s, [])])
        removed[s] = [e["box"] for e in log]

    subsets = {"all": lambda s: True, "cmp": lambda s: s.startswith("cmp_"), "lsaa": lambda s: not s.startswith("cmp_")}
    results: dict = {}
    per_photo: dict = {s: {} for s in stems}
    print(f"{'variant':10s} {'subset':5s} IoU   pred   TP   FP   FN     P      R     F1   removed  real")
    for thr in (0.5, 0.3):
        for name, boxes in (("pipeline", base), ("+rule3", kept)):
            for sub, f in subsets.items():
                rows = []
                for s in stems:
                    if f(s):
                        sc = score_photo(gt[s]["boxes"], [], boxes[s], thr, True)
                        rows.append(sc)
                        if thr == 0.5:
                            per_photo[s][name] = sc
                t = prf(rows)
                rem = [(s, b) for s in stems if f(s) for b in (removed[s] if name == "+rule3" else [])]
                real = sum(any(box_iou(b, g) >= 0.5 for g in gt[s]["boxes"]) for s, b in rem)
                results[f"{name}|{sub}|{thr}"] = {**t, "removed": len(rem), "removed_real": real}
                print(f"{name:10s} {sub:5s} {thr:<4}  {t['pred']:4d} {t['tp']:4d} {t['fp']:4d} {t['fn']:4d}  "
                      f"{t['precision']:.3f}  {t['recall']:.3f}  {t['f1']:.3f}   {len(rem):5d} {real:5d}")
    args.out.write_text(json.dumps({"results": results, "photos": per_photo, "removed": removed}, indent=0),
                        encoding="utf-8")

    args.img_dir.mkdir(parents=True, exist_ok=True)
    log = {s: [{"box": b, "real_balcony": any(box_iou(b, g) >= 0.5 for g in gt[s]["boxes"])} for b in removed[s]]
           for s in stems if removed[s]}
    (args.img_dir / "_removed.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    font = ImageFont.truetype("arial.ttf", 22)
    for s in stems:
        im0 = Image.open(ROOT / "data" / "test100" / "photos" / f"{s}.jpg").convert("RGB")
        W, H = im0.size
        lw = max(3, W // 200)
        g, a, c = im0.copy(), im0.copy(), im0.copy()
        for b in gt[s]["boxes"]:
            ImageDraw.Draw(g).rectangle(b, outline=(255, 0, 0), width=lw)
        # rule-3 panel: top floor band shaded, windows thin green
        floor_y, _bay = _bands_from_windows_dsl(dsls[s])
        if floor_y:
            bands = list(_bands_from_window_instances(dsls[s].get("instances") or [])[0].values())                 if args.rule == "below_floor" else [] if args.rule == "window_below" else [
                min(floor_y.values(), key=lambda v: min(v))]
            ov = Image.new("RGBA", c.size, (0, 0, 0, 0))
            for band in bands:
                ImageDraw.Draw(ov).rectangle([0, min(band), W, max(band)], fill=(255, 200, 0, 60))
            c = Image.alpha_composite(c.convert("RGBA"), ov).convert("RGB")
        if args.rule == "oriel" and oriel.get(s):
            fy, bx = _bands_from_windows_dsl(dsls[s])
            ov = Image.new("RGBA", c.size, (0, 0, 0, 0))
            for f, b in oriel[s]:
                if f in fy and b in bx:
                    ImageDraw.Draw(ov).rectangle([min(bx[b]), min(fy[f]), max(bx[b]), max(fy[f])],
                                                 fill=(200, 0, 200, 70), outline=(200, 0, 200, 255), width=lw)
            c = Image.alpha_composite(c.convert("RGBA"), ov).convert("RGB")
        dc = ImageDraw.Draw(c)
        for w in window_boxes(dsls[s]):
            dc.rectangle(w["box_xyxy"], outline=(0, 200, 0), width=max(2, lw // 2))
        for b in base[s]:
            ImageDraw.Draw(a).rectangle(b, outline=(255, 0, 0), width=lw)
        for b in kept[s]:
            dc.rectangle(b, outline=(255, 0, 0), width=lw)
        for b in removed[s]:
            dashed(dc, b, (0, 140, 255), lw)
        p0, p1 = per_photo[s]["pipeline"], per_photo[s]["+rule3"]
        heads = [[f"Ground truth: {len(gt[s]['boxes'])}", ""],
                 ["Pipeline (rule 1)", f"TP {p0['tp']} FP {p0['fp']} FN {p0['fn']}"],
                 [{"no_window_near": "+ Rule 3: no window above (green = windows)",
                   "below_floor": "+ Rule 3 v2: top below its window row (yellow = rows)",
                   "window_below": "+ Hood rule: sits on top of a window (green = windows)",
                   "oriel": f"+ Oriel rule ({'predicted' if args.oriel_cells == 'pred' else 'hand-labeled'}"
                            " oriel cells, purple)"}[args.rule],
                  f"TP {p1['tp']} FP {p1['fp']} FN {p1['fn']} | removed {len(removed[s])}"]]
        ph = args.panel_height
        pw = max(1, round(W * ph / H))
        head = 64
        out = Image.new("RGB", (3 * (pw + 12), ph + head), "white")
        do = ImageDraw.Draw(out)
        for k, (im, lines) in enumerate(zip((g, a, c), heads)):
            x = k * (pw + 12)
            out.paste(im.resize((pw, ph), Image.BILINEAR), (x, head))
            for j, line in enumerate(lines):
                do.text((x + 6, 4 + 28 * j), line, fill="black", font=font)
        out.save(args.img_dir / f"{s}.jpg", quality=86)
        if removed[s]:  # photos with removals, copied apart for review
            (args.img_dir / "removed_only").mkdir(exist_ok=True)
            out.save(args.img_dir / "removed_only" / f"{s}.jpg", quality=86)
    print(f"images -> {args.img_dir}")


if __name__ == "__main__":
    main()
