r"""Side-by-side images on test100: ground truth (left) | final pipeline (right).

Final pipeline = SAM3 ``filtered`` boxes + rule 1 (``side_wall``) +
``masonry_juliet_split`` (``railing_test100_v2.pt`` labels, cached
``runs/test100/railing_preds_v2.json``), as ``run_balcony.py`` does before snapping. All boxes plain red. Header: box
counts, and TP / FP / FN at IoU 0.5 for the pipeline (group rule, no ignore regions).

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_pipeline/draw_test100_final.py
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
from score_test100_gt_boxes import score_photo  # noqa: E402

T100 = ROOT / "runs" / "test100"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=T100 / "balcony_gt_boxes.jsonl")
    ap.add_argument("--cache", type=Path, default=ROOT / "runs" / "batch" / "_rows" / "test100_sam3_stage_boxes.json")
    ap.add_argument("--dsl-dir", type=Path, default=T100 / "batch_grid2")
    ap.add_argument("--depth-dir", type=Path, default=T100 / "da3_depth")
    ap.add_argument("--out-dir", type=Path, default=T100 / "compare_gt_final")
    ap.add_argument("--labels", type=Path, default=T100 / "railing_preds_v2.json")
    ap.add_argument("--panel-height", type=int, default=900)
    args = ap.parse_args()

    gt = {json.loads(l)["stem"]: json.loads(l) for l in args.gt.read_text(encoding="utf-8").splitlines() if l.strip()}
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    labels_all = json.loads(args.labels.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("arial.ttf", 24)
    for s in sorted(gt):
        dsl = json.loads((args.dsl_dir / s / "facade_dsl.json").read_text(encoding="utf-8"))
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
        pred, _ = apply_rules(sw, size, dsl, labels=labs, rules=("masonry_juliet_split",))
        g = gt[s]["boxes"]
        sc = score_photo(g, [], pred, 0.5, True)

        im = Image.open(ROOT / "data" / "test100" / "photos" / f"{s}.jpg").convert("RGB")
        W, H = im.size
        lw = max(3, W // 200)
        left, right = im.copy(), im.copy()
        dl, dr = ImageDraw.Draw(left), ImageDraw.Draw(right)
        for b in g:
            dl.rectangle(b, outline=(255, 0, 0), width=lw)
        for b in pred:
            dr.rectangle(b, outline=(255, 0, 0), width=lw)
        heads = [f"Ground truth: {len(g)}", f"Pipeline: {len(pred)}  TP {sc['tp']} FP {sc['fp']} FN {sc['fn']}"]
        ph = args.panel_height
        pw = max(1, round(W * ph / H))
        head = 40
        out = Image.new("RGB", (2 * pw + 12, ph + head), "white")
        do = ImageDraw.Draw(out)
        for k, (pim, text) in enumerate(zip((left, right), heads)):
            x = k * (pw + 12)
            out.paste(pim.resize((pw, ph), Image.BILINEAR), (x, head))
            do.text((x + 6, 6), text, fill="black", font=font)
        out.save(args.out_dir / f"{s}.jpg", quality=86)
    print(f"{len(gt)} images -> {args.out_dir}")


if __name__ == "__main__":
    main()
