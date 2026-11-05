r"""Three panels per test100 photo: ground truth | previous pipeline | current pipeline.

- Ground truth: ``runs/test100/balcony_gt_boxes.jsonl``.
- Previous pipeline (old main row, F1 0.854): SAM3 ``filtered`` + ``side_wall`` +
  ``masonry_juliet_split`` with the cached ``railing_test100_v2.pt`` labels, no oriel removal
  (as ``draw_test100_final.py``).
- Current pipeline (main row since 2026-11-03): the ``final`` boxes of
  ``score_test100_pipeline.py`` (``runs/test100/balcony_pipeline_score.json``): v4 split,
  merge + snap, then oriel removal with the 457 oriel model. Boxes removed by the oriel step
  are drawn blue dashed (not counted).

Headers give box counts and TP / FP / FN at IoU 0.5 (group rule, no ignore regions). Photos
whose current boxes differ from the previous ones are also copied to ``changed_only/``.

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_pipeline/draw_test100_pipeline3.py
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
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from postfilter import apply_rules  # noqa: E402
from score_test100_gt_boxes import score_photo  # noqa: E402
from score_test100_user_rules import dashed  # noqa: E402

T100 = ROOT / "runs" / "test100"


def previous_boxes(s: str, cache: dict, labels_all: dict, dsl_dir: Path, depth_dir: Path) -> list:
    dsl = json.loads((dsl_dir / s / "facade_dsl.json").read_text(encoding="utf-8"))
    dp = depth_dir / f"{s}_depth.npy"
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
    return [list(b) for b in pred]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=T100 / "balcony_gt_boxes.jsonl")
    ap.add_argument("--cache", type=Path, default=ROOT / "runs" / "batch" / "_rows" / "test100_sam3_stage_boxes.json")
    ap.add_argument("--labels", type=Path, default=T100 / "railing_preds_v2.json")
    ap.add_argument("--current", type=Path, default=T100 / "balcony_pipeline_score.json")
    ap.add_argument("--dsl-dir", type=Path, default=T100 / "batch_grid2")
    ap.add_argument("--depth-dir", type=Path, default=T100 / "da3_depth")
    ap.add_argument("--out-dir", type=Path, default=T100 / "compare_gt_prev_current")
    ap.add_argument("--panel-height", type=int, default=900)
    args = ap.parse_args()

    gt = {json.loads(l)["stem"]: json.loads(l) for l in args.gt.read_text(encoding="utf-8").splitlines() if l.strip()}
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    labels_all = json.loads(args.labels.read_text(encoding="utf-8"))
    cur = json.loads(args.current.read_text(encoding="utf-8"))
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    changed_dir = args.out_dir / "changed_only"
    changed_dir.mkdir(parents=True)
    font = ImageFont.truetype("arial.ttf", 24)
    n_changed = 0
    for s in sorted(gt):
        g = gt[s]["boxes"]
        prev = previous_boxes(s, cache, labels_all, args.dsl_dir, args.depth_dir)
        now = [list(b) for b in cur["boxes"]["final"][s]]
        removed = [list(b) for b in cur["oriel_dropped"].get(s, [])]
        sp = score_photo(g, [], prev, 0.5, True)
        sc = score_photo(g, [], now, 0.5, True)

        im = Image.open(T100 / "photos" / f"{s}.jpg").convert("RGB")
        W, H = im.size
        lw = max(3, W // 200)
        panels = [im.copy(), im.copy(), im.copy()]
        draws = [ImageDraw.Draw(p) for p in panels]
        for b in g:
            draws[0].rectangle(b, outline=(255, 0, 0), width=lw)
        for b in prev:
            draws[1].rectangle(b, outline=(255, 0, 0), width=lw)
        for b in now:
            draws[2].rectangle(b, outline=(255, 0, 0), width=lw)
        for b in removed:
            dashed(draws[2], b, (40, 90, 255), lw)
        heads = [
            f"Ground truth: {len(g)}",
            f"Previous: {len(prev)}  TP {sp['tp']} FP {sp['fp']} FN {sp['fn']}",
            f"Current: {len(now)}  TP {sc['tp']} FP {sc['fp']} FN {sc['fn']}"
            + (f"  (oriel removed {len(removed)})" if removed else ""),
        ]
        ph = args.panel_height
        pw = max(1, round(W * ph / H))
        head = 40
        out = Image.new("RGB", (3 * pw + 24, ph + head), "white")
        do = ImageDraw.Draw(out)
        for k, (pim, text) in enumerate(zip(panels, heads)):
            x = k * (pw + 12)
            out.paste(pim.resize((pw, ph), Image.BILINEAR), (x, head))
            do.text((x + 6, 6), text, fill="black", font=font)
        out.save(args.out_dir / f"{s}.jpg", quality=86)
        if sorted(map(tuple, prev)) != sorted(map(tuple, now)):
            n_changed += 1
            shutil.copy2(args.out_dir / f"{s}.jpg", changed_dir / f"{s}.jpg")
    print(f"{len(gt)} images -> {args.out_dir}  (changed {n_changed} -> {changed_dir})")


if __name__ == "__main__":
    main()
