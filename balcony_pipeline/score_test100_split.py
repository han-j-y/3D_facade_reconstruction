r"""Split wide boxes into one box per window, on top of rule 1, on test100.

Base: SAM3 ``filtered`` boxes + rule 1 (``side_wall``). Variants:

* ``juliet_split`` - a box the railing classifier (``railing_test100_v2.pt``,
  cached ``runs/test100/railing_preds_v2.json``) calls ``juliet`` and that has
  two or more windows just above it becomes one box per window (window x-range,
  box y-range).
* ``masonry_juliet_split`` - only boxes called ``juliet`` + ``open_work`` +
  ``masonry`` (baluster balustrades).
* ``wide_split`` - the same for any box, without the classifier condition.

``--labels hand`` uses the hand labels of the test crops instead of the
classifier (``crops/labels.jsonl`` via ``crops/manifest.jsonl``; supplementary
upper bound).

Scored against ``runs/test100/balcony_gt_boxes.jsonl`` without ignore regions
(group rule, IoU 0.5 and 0.3). Images: GT | pipeline | the ``--show`` variants (default
masonry_juliet_split only); split boxes are drawn as their pieces in blue, other boxes red.
Photos where a box was split are also copied to ``removed_only/``.

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_pipeline/score_test100_split.py
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

T100 = ROOT / "runs" / "test100"
VARIANTS = ("juliet_split", "masonry_juliet_split", "wide_split")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=T100 / "balcony_gt_boxes.jsonl")
    ap.add_argument("--cache", type=Path, default=ROOT / "runs" / "batch" / "_rows" / "test100_sam3_stage_boxes.json")
    ap.add_argument("--labels", default="classifier", choices=("classifier", "hand"))
    ap.add_argument("--pred-labels", type=Path, default=T100 / "railing_preds_v2.json")
    ap.add_argument("--dsl-dir", type=Path, default=T100 / "batch_grid2")
    ap.add_argument("--depth-dir", type=Path, default=T100 / "da3_depth")
    ap.add_argument("--img-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--panel-height", type=int, default=760)
    ap.add_argument("--show", nargs="+", default=["masonry_juliet_split"], choices=VARIANTS,
                    help="variants drawn after GT | pipeline (default: masonry_juliet_split only, 3 panels)")
    args = ap.parse_args()

    gt = {json.loads(l)["stem"]: json.loads(l) for l in args.gt.read_text(encoding="utf-8").splitlines() if l.strip()}
    stems = sorted(gt)
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    sfx = "" if args.labels == "classifier" else "_hand"
    args.img_dir = args.img_dir or T100 / f"compare_split{sfx}"
    args.out = args.out or T100 / f"balcony_split{sfx}_score.json"
    labels_all = json.loads(args.pred_labels.read_text(encoding="utf-8"))
    man = {}
    for line in (T100 / "crops" / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            man[(r["photo"][:-4], tuple(r["box_xyxy"]))] = r["crop"]
    hand = {json.loads(l)["path"]: json.loads(l)
            for l in (T100 / "crops" / "labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

    base: dict[str, list] = {}
    res: dict[str, dict[str, list]] = {v: {} for v in VARIANTS}
    logs: dict[str, dict[str, list]] = {v: {} for v in VARIANTS}
    for s in stems:
        dsl = json.loads((args.dsl_dir / s / "facade_dsl.json").read_text(encoding="utf-8"))
        dp = args.depth_dir / f"{s}_depth.npy"
        depth = np.load(dp) if dp.is_file() else None
        filt = cache[s]["filtered"]
        size = tuple(cache[s]["size"])
        base[s], _ = apply_rules(filt, size, dsl, depth=depth, rules=("side_wall",))
        left = list(range(len(filt)))
        labs = []
        for b in base[s]:
            j = next(j for j in left if filt[j] == b)
            left.remove(j)
            labs.append(labels_all[s][j] if args.labels == "classifier" else hand.get(man.get((s, tuple(b)))))
        for v in VARIANTS:
            res[v][s], logs[v][s] = apply_rules(base[s], size, dsl, labels=labs, rules=(v,))

    subsets = {"all": lambda s: True, "cmp": lambda s: s.startswith("cmp_"), "lsaa": lambda s: not s.startswith("cmp_")}
    results: dict = {}
    per_photo: dict = {s: {} for s in stems}
    print(f"{'variant':20s} {'subset':5s} IoU   pred   TP   FP   FN     P      R     F1   split  pieces  pieces_TP")
    for thr in (0.5, 0.3):
        for name, boxes in (("pipeline", base), *((v, res[v]) for v in VARIANTS)):
            for sub, f in subsets.items():
                rows = []
                for s in stems:
                    if f(s):
                        sc = score_photo(gt[s]["boxes"], [], boxes[s], thr, True)
                        rows.append(sc)
                        if thr == 0.5:
                            per_photo[s][name] = sc
                t = prf(rows)
                lg = [(s, e) for s in stems if f(s) for e in logs.get(name, {}).get(s, [])]
                pieces = [(s, q) for s, e in lg for q in e["into"]]
                ptp = sum(any(box_iou(q, g) >= thr for g in gt[s]["boxes"]) for s, q in pieces)
                results[f"{name}|{sub}|{thr}"] = {**t, "split": len(lg), "pieces": len(pieces), "pieces_tp": ptp}
                print(f"{name:20s} {sub:5s} {thr:<4}  {t['pred']:4d} {t['tp']:4d} {t['fp']:4d} {t['fn']:4d}  "
                      f"{t['precision']:.3f}  {t['recall']:.3f}  {t['f1']:.3f}  {len(lg):5d}  {len(pieces):6d}  {ptp:6d}")
    args.out.write_text(json.dumps({"results": results, "photos": per_photo, "logs": logs}, indent=0),
                        encoding="utf-8")

    args.img_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("arial.ttf", 22)
    for s in stems:
        im0 = Image.open(T100 / "photos" / f"{s}.jpg").convert("RGB")
        W, H = im0.size
        lw = max(3, W // 200)
        panels = []
        g = im0.copy()
        for b in gt[s]["boxes"]:
            ImageDraw.Draw(g).rectangle(b, outline=(255, 0, 0), width=lw)
        panels.append((g, [f"Ground truth: {len(gt[s]['boxes'])}", ""]))
        a = im0.copy()
        for b in base[s]:
            ImageDraw.Draw(a).rectangle(b, outline=(255, 0, 0), width=lw)
        sc = per_photo[s]["pipeline"]
        panels.append((a, ["Pipeline (rule 1)", f"TP {sc['tp']} FP {sc['fp']} FN {sc['fn']}"]))
        for v in args.show:
            im = im0.copy()
            d = ImageDraw.Draw(im)
            pieces = [q for e in logs[v][s] for q in e["into"]]
            for b in res[v][s]:
                d.rectangle(b, outline=(0, 140, 255) if b in pieces else (255, 0, 0), width=lw)
            sc = per_photo[s][v]
            title = {"juliet_split": "+ Split juliet boxes", "masonry_juliet_split": "+ Split juliet masonry open_work",
                     "wide_split": "+ Split any wide box"}[v]
            panels.append((im, [title, f"TP {sc['tp']} FP {sc['fp']} FN {sc['fn']} | split {len(logs[v][s])}"]))
        ph = args.panel_height
        pw = max(1, round(W * ph / H))
        head = 64
        out = Image.new("RGB", (len(panels) * (pw + 12), ph + head), "white")
        do = ImageDraw.Draw(out)
        for k, (pim, lines) in enumerate(panels):
            x = k * (pw + 12)
            out.paste(pim.resize((pw, ph), Image.BILINEAR), (x, head))
            for j, line in enumerate(lines):
                do.text((x + 6, 4 + 28 * j), line, fill="black", font=font)
        out.save(args.img_dir / f"{s}.jpg", quality=86)
        if any(logs[v][s] for v in args.show):
            (args.img_dir / "removed_only").mkdir(exist_ok=True)
            out.save(args.img_dir / "removed_only" / f"{s}.jpg", quality=86)
    print(f"images -> {args.img_dir}")


if __name__ == "__main__":
    main()
