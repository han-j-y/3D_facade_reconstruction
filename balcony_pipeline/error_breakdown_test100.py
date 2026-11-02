r"""Error breakdown of the final open-balcony boxes on test100 (IoU 0.5, group rule, no ignore).

Final pipeline = SAM3 ``filtered`` boxes + rule 1 (``side_wall``) +
``masonry_juliet_split`` (``railing_test100_v2.pt`` labels, cached
``runs/test100/railing_preds_v2.json``; ``--no-split`` leaves it out). Matching as in
``score_test100_gt_boxes.py``. Every unmatched GT unit (FN) and unmatched
prediction (FP) gets reproducible, automatic tags.

FN, by stage where it was lost (first that applies):
  ``merged``         a final box covers >= 70% of the GT unit but IoU < 0.5: detected
                     as part of a larger box (e.g. one box over a continuous balustrade)
  ``sam3_miss``      no raw SAM3 box with IoU >= 0.1 (not detected at all)
  ``sam3_extent``    a raw SAM3 box overlaps (0.1 <= IoU < 0.5): wrong extent at detection
  ``overlap_filter`` a raw box matched (IoU >= 0.5) but no filtered box did
  ``side_wall``      a box removed by rule 1 matched
  ``localization``   a final box overlaps (0.1 <= IoU < 0.5) - detected, wrong extent
  ``other``          a filtered box matched, but greedy one-to-one gave it to another GT
FN attributes (may overlap): ``overlapping`` (intersects another GT box by >= 10%
of the smaller), ``on_oriel`` (centre in a hand-labeled oriel cell,
``oriel_labels.jsonl``), ``large`` (area >= 2x the photo's median GT area).

FP:
  ``localization``   overlaps a GT box (0.1 <= IoU < 0.5): split / merged / shifted
  ``no_balcony``     IoU < 0.1 with every GT box, sub-tagged by position (first
                     that applies): ``on_oriel``, ``below_window`` (a window just
                     above, ``postfilter.windows_just_above``: sill-like),
                     ``above_window`` (sits on a window, ``window_right_below``:
                     hood / roof-like), ``other``.

Position tags rely on detected windows, which miss some windows; they are a proxy.

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_pipeline/error_breakdown_test100.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from postfilter import apply_rules, window_boxes, window_right_below, windows_just_above  # noqa: E402
from score_sam3_label_iou import box_iou  # noqa: E402
from score_test100_gt_boxes import greedy, gt_units  # noqa: E402
from snap import _bands_from_windows_dsl  # noqa: E402

T100 = ROOT / "runs" / "test100"


def best(b, boxes) -> float:
    return max((box_iou(b, x) for x in boxes), default=0.0)


def area(b) -> float:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def inter(a, b) -> float:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=T100 / "balcony_gt_boxes.jsonl")
    ap.add_argument("--cache", type=Path, default=ROOT / "runs" / "batch" / "_rows" / "test100_sam3_stage_boxes.json")
    ap.add_argument("--dsl-dir", type=Path, default=T100 / "batch_grid2")
    ap.add_argument("--depth-dir", type=Path, default=T100 / "da3_depth")
    ap.add_argument("--oriel", type=Path, default=T100 / "oriel_labels.jsonl")
    ap.add_argument("--out", type=Path, default=T100 / "balcony_error_breakdown.json")
    ap.add_argument("--labels", type=Path, default=T100 / "railing_preds_v2.json")
    ap.add_argument("--no-split", action="store_true")
    args = ap.parse_args()

    gt = {json.loads(l)["stem"]: json.loads(l) for l in args.gt.read_text(encoding="utf-8").splitlines() if l.strip()}
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    labels_all = json.loads(args.labels.read_text(encoding="utf-8"))
    oriel = {json.loads(l)["stem"]: json.loads(l)["cells"]
             for l in args.oriel.read_text(encoding="utf-8").splitlines() if l.strip()}

    fn_stage: Counter = Counter()
    fn_attr: Counter = Counter()
    fp_kind: Counter = Counter()
    by_sub = {"cmp": Counter(), "lsaa": Counter()}
    rows = []
    n_gt = n_pred = n_tp = 0
    for s in sorted(gt):
        sub = "cmp" if s.startswith("cmp_") else "lsaa"
        dsl = json.loads((args.dsl_dir / s / "facade_dsl.json").read_text(encoding="utf-8"))
        dp = args.depth_dir / f"{s}_depth.npy"
        depth = np.load(dp) if dp.is_file() else None
        filt = cache[s]["filtered"]
        raw = cache[s]["raw"]
        pred, log = apply_rules(filt, tuple(cache[s]["size"]), dsl, depth=depth, rules=("side_wall",))
        removed = [e["box"] for e in log]
        if not args.no_split:
            left = list(range(len(filt)))
            labs = []
            for b in pred:
                j = next(j for j in left if filt[j] == b)
                left.remove(j)
                labs.append(labels_all[s][j])
            pred, _ = apply_rules(pred, tuple(cache[s]["size"]), dsl, labels=labs, rules=("masonry_juliet_split",))
        g = gt[s]["boxes"]
        units = gt_units(g, pred, 0.5, True)
        matched_p = greedy(units, pred, 0.5)
        matched_u = greedy(pred, units, 0.5)  # indices of units matched
        n_gt += len(units)
        n_pred += len(pred)
        n_tp += len(matched_p)
        windows = window_boxes(dsl)
        fy, bx = _bands_from_windows_dsl(dsl)
        cells = [(min(bx[b]), min(fy[f]), max(bx[b]), max(fy[f])) for f, b in oriel.get(s, []) if f in fy and b in bx]
        med = statistics.median([area(b) for b in g]) if g else 0

        def on_oriel(b) -> bool:
            cx, cy = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
            return any(c[0] <= cx <= c[2] and c[1] <= cy <= c[3] for c in cells)

        for k, u in enumerate(units):
            if k in matched_u:
                continue
            if any(inter(u, x) >= 0.7 * area(u) for x in pred):
                stage = "merged"
            elif best(u, raw) < 0.1:
                stage = "sam3_miss"
            elif best(u, raw) < 0.5:
                stage = "sam3_extent"
            elif best(u, filt) < 0.5:
                stage = "overlap_filter"
            elif best(u, removed) >= 0.5:
                stage = "side_wall"
            elif best(u, pred) >= 0.1:
                stage = "localization"
            else:
                stage = "other"
            attrs = []
            if any(o is not u and o != u and inter(u, o) >= 0.1 * min(area(u), area(o)) for o in g):
                attrs.append("overlapping")
            if on_oriel(u):
                attrs.append("on_oriel")
            if med and area(u) >= 2 * med:
                attrs.append("large")
            fn_stage[stage] += 1
            by_sub[sub][f"fn:{stage}"] += 1
            for a in attrs:
                fn_attr[a] += 1
            rows.append({"stem": s, "type": "FN", "box": u, "stage": stage, "attrs": attrs})

        for j, b in enumerate(pred):
            if j in matched_p:
                continue
            if best(b, g) >= 0.1:
                kind = "localization"
            elif on_oriel(b):
                kind = "no_balcony:on_oriel"
            elif windows_just_above(b, windows):
                kind = "no_balcony:below_window"
            elif window_right_below(b, windows):
                kind = "no_balcony:above_window"
            else:
                kind = "no_balcony:other"
            fp_kind[kind] += 1
            by_sub[sub][f"fp:{kind}"] += 1
            rows.append({"stem": s, "type": "FP", "box": b, "kind": kind})

    print(f"units {n_gt}  pred {n_pred}  TP {n_tp}  FN {n_gt - n_tp}  FP {n_pred - n_tp}")
    print("FN by stage:", dict(fn_stage.most_common()))
    print("FN attributes:", dict(fn_attr.most_common()))
    print("FP:", dict(fp_kind.most_common()))
    for k, v in by_sub.items():
        print(k, dict(sorted(v.items())))
    photos_fn = Counter(r["stem"] for r in rows if r["type"] == "FN")
    photos_fp = Counter(r["stem"] for r in rows if r["type"] == "FP")
    print("photos with no error:", sum(1 for s in gt if s not in photos_fn and s not in photos_fp))
    print("top FP photos:", photos_fp.most_common(5))
    print("top FN photos:", photos_fn.most_common(5))
    args.out.write_text(json.dumps({"fn_stage": fn_stage, "fn_attr": fn_attr, "fp": fp_kind,
                                    "by_subset": by_sub, "rows": rows}, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
