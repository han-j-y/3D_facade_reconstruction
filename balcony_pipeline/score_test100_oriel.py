r"""Oriel rule alone on top of the current open-balcony pipeline, on test100.

Base ("pipeline"): SAM3 ``filtered`` boxes + rule 1 (``side_wall``) +
``masonry_juliet_split`` with the ``railing_test100_v2.pt`` labels (cached
``runs/test100/railing_preds_v2.json``), as ``run_balcony.py`` now does before
snapping. "+ oriel": drop an open box on an oriel cell (same floor, shared bay;
``enclosed.drop_open_under_enclosed``). ``--oriel-cells pred`` (default) uses the
held-out cell model predictions (``runs/oriel_grid_compare/grid2_test100/preds_heldout.json``,
89 photos); ``gt`` the hand labels (``oriel_labels.jsonl``, upper bound).

Scored against ``runs/test100/balcony_gt_boxes.jsonl`` without ignore regions
(group rule, IoU 0.5 and 0.3). Images: two panels, pipeline | + oriel; removed
boxes blue dashed, oriel cells purple. Photos with removals also in ``removed_only/``.

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_pipeline/score_test100_oriel.py
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

from postfilter import apply_rules  # noqa: E402
from score_sam3_label_iou import box_iou  # noqa: E402
from score_test100_gt_boxes import prf, score_photo  # noqa: E402
from score_test100_user_rules import dashed  # noqa: E402
from snap import _bands_from_windows_dsl  # noqa: E402

T100 = ROOT / "runs" / "test100"
ORIEL_PRED = ROOT / "runs" / "oriel_grid_compare" / "grid2_test100" / "preds_heldout.json"
ORIEL_GT = T100 / "oriel_labels.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=T100 / "balcony_gt_boxes.jsonl")
    ap.add_argument("--cache", type=Path, default=ROOT / "runs" / "batch" / "_rows" / "test100_sam3_stage_boxes.json")
    ap.add_argument("--labels", type=Path, default=T100 / "railing_preds_v2.json")
    ap.add_argument("--dsl-dir", type=Path, default=T100 / "batch_grid2")
    ap.add_argument("--depth-dir", type=Path, default=T100 / "da3_depth")
    ap.add_argument("--oriel-cells", choices=("pred", "gt"), default="pred")
    ap.add_argument("--oriel-preds", type=Path, default=ORIEL_PRED,
                    help="predicted oriel cells per stem (pred mode), e.g. final_457_test100/preds_official.json")
    ap.add_argument("--img-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--panel-height", type=int, default=900)
    args = ap.parse_args()
    args.img_dir = args.img_dir or T100 / f"compare_oriel_{args.oriel_cells}"
    args.out = args.out or T100 / f"balcony_oriel_{args.oriel_cells}_score.json"

    gt = {json.loads(l)["stem"]: json.loads(l) for l in args.gt.read_text(encoding="utf-8").splitlines() if l.strip()}
    stems = sorted(gt)
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    labels_all = json.loads(args.labels.read_text(encoding="utf-8"))
    if args.oriel_cells == "pred":
        oriel = json.loads(args.oriel_preds.read_text(encoding="utf-8"))
    else:
        oriel = {json.loads(l)["stem"]: json.loads(l)["cells"]
                 for l in ORIEL_GT.read_text(encoding="utf-8").splitlines() if l.strip()}

    base, kept, removed, dsls = {}, {}, {}, {}
    for s in stems:
        dsl = json.loads((args.dsl_dir / s / "facade_dsl.json").read_text(encoding="utf-8"))
        dsls[s] = dsl
        dp = args.depth_dir / f"{s}_depth.npy"
        depth = np.load(dp) if dp.is_file() else None
        filt = cache[s]["filtered"]
        size = tuple(cache[s]["size"])
        sw, _ = apply_rules(filt, size, dsl, depth=depth, rules=("side_wall",))
        left = list(range(len(filt)))
        labs = []
        for b in sw:
            j = next(j for j in left if filt[j] == b)
            left.remove(j)
            labs.append(labels_all[s][j])
        base[s], _ = apply_rules(sw, size, dsl, labels=labs, rules=("masonry_juliet_split",))
        kept[s], log = apply_rules(base[s], size, dsl, oriel_cells=[tuple(c) for c in oriel.get(s, [])],
                                   rules=("oriel",))
        removed[s] = [e["box"] for e in log]

    subsets = {"all": lambda s: True, "cmp": lambda s: s.startswith("cmp_"), "lsaa": lambda s: not s.startswith("cmp_")}
    results: dict = {}
    per_photo: dict = {s: {} for s in stems}
    print(f"{'variant':10s} {'subset':5s} IoU   pred   TP   FP   FN     P      R     F1   removed  real")
    for thr in (0.5, 0.3):
        for name, boxes in (("pipeline", base), ("+oriel", kept)):
            for sub, f in subsets.items():
                rows = []
                for s in stems:
                    if f(s):
                        sc = score_photo(gt[s]["boxes"], [], boxes[s], thr, True)
                        rows.append(sc)
                        if thr == 0.5:
                            per_photo[s][name] = sc
                t = prf(rows)
                rem = [(s, b) for s in stems if f(s) for b in (removed[s] if name == "+oriel" else [])]
                real = sum(any(box_iou(b, g) >= 0.5 for g in gt[s]["boxes"]) for s, b in rem)
                results[f"{name}|{sub}|{thr}"] = {**t, "removed": len(rem), "removed_real": real}
                print(f"{name:10s} {sub:5s} {thr:<4}  {t['pred']:4d} {t['tp']:4d} {t['fp']:4d} {t['fn']:4d}  "
                      f"{t['precision']:.3f}  {t['recall']:.3f}  {t['f1']:.3f}   {len(rem):5d} {real:5d}")
    args.out.write_text(json.dumps({"oriel_cells": args.oriel_cells, "results": results, "photos": per_photo,
                                    "removed": removed}, indent=0), encoding="utf-8")

    args.img_dir.mkdir(parents=True, exist_ok=True)
    log = {s: [{"box": b, "real_balcony": any(box_iou(b, g) >= 0.5 for g in gt[s]["boxes"])} for b in removed[s]]
           for s in stems if removed[s]}
    (args.img_dir / "_removed.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    font = ImageFont.truetype("arial.ttf", 24)
    src = "predicted" if args.oriel_cells == "pred" else "hand-labeled"
    for s in stems:
        im0 = Image.open(ROOT / "data" / "test100" / "photos" / f"{s}.jpg").convert("RGB")
        W, H = im0.size
        lw = max(3, W // 200)
        a, c = im0.copy(), im0.copy()
        for b in base[s]:
            ImageDraw.Draw(a).rectangle(b, outline=(255, 0, 0), width=lw)
        if oriel.get(s):
            fy, bx = _bands_from_windows_dsl(dsls[s])
            ov = Image.new("RGBA", c.size, (0, 0, 0, 0))
            for fl, bay in oriel[s]:
                if fl in fy and bay in bx:
                    ImageDraw.Draw(ov).rectangle([min(bx[bay]), min(fy[fl]), max(bx[bay]), max(fy[fl])],
                                                 fill=(200, 0, 200, 60), outline=(200, 0, 200, 255), width=lw)
            c = Image.alpha_composite(c.convert("RGBA"), ov).convert("RGB")
        dc = ImageDraw.Draw(c)
        for b in kept[s]:
            dc.rectangle(b, outline=(255, 0, 0), width=lw)
        for b in removed[s]:
            dashed(dc, b, (0, 140, 255), lw)
        p0, p1 = per_photo[s]["pipeline"], per_photo[s]["+oriel"]
        heads = [["Pipeline", f"TP {p0['tp']} FP {p0['fp']} FN {p0['fn']}"],
                 [f"+ Remove balconies on oriels ({src} oriel cells, purple)",
                  f"TP {p1['tp']} FP {p1['fp']} FN {p1['fn']} | removed {len(removed[s])}"]]
        ph = args.panel_height
        pw = max(1, round(W * ph / H))
        head = 64
        out = Image.new("RGB", (2 * pw + 12, ph + head), "white")
        do = ImageDraw.Draw(out)
        for k, (pim, lines) in enumerate(zip((a, c), heads)):
            x = k * (pw + 12)
            out.paste(pim.resize((pw, ph), Image.BILINEAR), (x, head))
            for j, line in enumerate(lines):
                do.text((x + 6, 4 + 28 * j), line, fill="black", font=font)
        out.save(args.img_dir / f"{s}.jpg", quality=86)
        if removed[s]:
            (args.img_dir / "removed_only").mkdir(exist_ok=True)
            out.save(args.img_dir / "removed_only" / f"{s}.jpg", quality=86)
    print(f"images -> {args.img_dir}")


if __name__ == "__main__":
    main()
