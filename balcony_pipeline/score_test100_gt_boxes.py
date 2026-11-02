"""Score open-balcony boxes on test100 against the hand-made box ground truth.

Ground truth: ``runs/test100/balcony_gt_boxes.jsonl`` (``balcony_train/gt_box_ui.py``),
balcony ``boxes`` plus ``ignore`` regions. Predictions: the test100 SAM3 stage
cache (``raw`` and the pipeline ``filtered`` boxes); no SAM3 run.

Matching, as in the cited CMP box score: one-to-one, greedy from the highest
IoU, IoU >= threshold is a TP. ``group`` (main row) also applies the CMP
side-by-side group rule of ``score_sam3_either_gt`` (each group counts whole or
as its boxes, whichever gives more TP); ``plain`` matches the boxes as drawn.

Ignore regions are OFF by default (user, 2026-11-01: they were drawn while
seeing the pipeline boxes). ``--use-ignore`` gives the supplementary score: an
unmatched prediction with at least half of its own area inside one ignore
region is dropped (neither TP nor FP); ignore regions are never FN.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_pipeline/score_test100_gt_boxes.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from score_sam3_either_gt import MAX_EXHAUSTIVE_GROUPS, bbox, row_groups  # noqa: E402
from score_sam3_label_iou import box_iou  # noqa: E402

DEFAULT_GT = ROOT / "runs" / "test100" / "balcony_gt_boxes.jsonl"
DEFAULT_CACHE = ROOT / "runs" / "batch" / "_rows" / "test100_sam3_stage_boxes.json"
DEFAULT_OUT = ROOT / "runs" / "test100" / "balcony_box_score.json"  # with --use-ignore: pass --out
STAGES = ("raw", "filtered")


def greedy(gt: list[list[int]], pred: list[list[int]], thr: float) -> set[int]:
    """Indices of predictions matched one-to-one (greedy by IoU)."""
    pairs = sorted(
        ((box_iou(g, p), i, j) for i, g in enumerate(gt) for j, p in enumerate(pred)),
        reverse=True,
    )
    used_g: set[int] = set()
    used_p: set[int] = set()
    for iou, i, j in pairs:
        if iou < thr:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
    return used_p


def gt_units(gt: list[list[int]], pred: list[list[int]], thr: float, group: bool) -> list[list[int]]:
    if not group:
        return gt
    groups = row_groups(gt)
    whole = [bbox([gt[k] for k in g]) for g in groups]
    multi = [i for i, g in enumerate(groups) if len(g) > 1]

    def units(chosen: set[int]) -> list[list[int]]:
        out: list[list[int]] = []
        for i, g in enumerate(groups):
            out.extend([whole[i]] if len(g) == 1 or i in chosen else [gt[k] for k in g])
        return out

    best: set[int] = set()
    best_tp = len(greedy(units(best), pred, thr))
    if len(multi) <= MAX_EXHAUSTIVE_GROUPS:
        for mask in range(1, 1 << len(multi)):
            chosen = {g for bit, g in enumerate(multi) if mask >> bit & 1}
            tp = len(greedy(units(chosen), pred, thr))
            if tp > best_tp or (tp == best_tp and len(chosen) < len(best)):
                best, best_tp = chosen, tp
    return units(best)


def inside_frac(p: list[int], r: list[int]) -> float:
    iw = max(0, min(p[2], r[2]) - max(p[0], r[0]))
    ih = max(0, min(p[3], r[3]) - max(p[1], r[1]))
    area = max(1, (p[2] - p[0]) * (p[3] - p[1]))
    return iw * ih / area


def score_photo(gt, ignore, pred, thr: float, group: bool) -> dict:
    units = gt_units(gt, pred, thr, group)
    matched = greedy(units, pred, thr)
    dropped = [j for j in range(len(pred)) if j not in matched
               and any(inside_frac(pred[j], r) >= 0.5 for r in ignore)]
    tp = len(matched)
    return {"pred": len(pred) - len(dropped), "gt": len(units), "tp": tp,
            "fp": len(pred) - len(dropped) - tp, "fn": len(units) - tp, "ignored_pred": len(dropped)}


def prf(rows: list[dict]) -> dict:
    t = {k: sum(r[k] for r in rows) for k in ("pred", "gt", "tp", "fp", "fn", "ignored_pred")}
    p = t["tp"] / t["pred"] if t["pred"] else 0.0
    r = t["tp"] / t["gt"] if t["gt"] else 0.0
    t.update(photos=len(rows), precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--use-ignore", action="store_true", help="supplementary: drop predictions in ignore regions")
    args = ap.parse_args()

    gt_rows = [json.loads(l) for l in args.gt.read_text(encoding="utf-8").splitlines() if l.strip()]
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    subsets = {
        "all": lambda s: True,
        "cmp": lambda s: s.startswith(("cmp_b", "cmp_x")),
        "lsaa": lambda s: not s.startswith(("cmp_b", "cmp_x")),
    }
    per_photo: dict[str, dict] = {}
    totals: dict[str, dict] = {}
    for stage in STAGES:
        for mode in ("group", "plain"):
            for thr in (0.5, 0.3):
                key = f"{stage}/{mode}/{thr}"
                rows = []
                for g in gt_rows:
                    ign = g.get("ignore", []) if args.use_ignore else []
                    sc = score_photo(g["boxes"], ign, cache[g["stem"]][stage], thr, mode == "group")
                    sc["stem"] = g["stem"]
                    rows.append(sc)
                    per_photo.setdefault(g["stem"], {})[key] = sc
                for name, f in subsets.items():
                    totals[f"{name}/{key}"] = prf([r for r in rows if f(r["stem"])])

    # localisation: best IoU of any pipeline box per GT box
    best = {name: [] for name in subsets}
    for g in gt_rows:
        pred = cache[g["stem"]]["filtered"]
        for b in g["boxes"]:
            v = max((box_iou(b, p) for p in pred), default=0.0)
            for name, f in subsets.items():
                if f(g["stem"]):
                    best[name].append(v)
    loc = {n: {"n": len(v), "mean": statistics.mean(v), "median": statistics.median(v),
               "lt_0.1": sum(x < 0.1 for x in v), "ge_0.5": sum(x >= 0.5 for x in v)} for n, v in best.items() if v}

    print(f"{'subset/stage/mode/IoU':32s} photos  pred   gt   TP   FP   FN  ign     P      R     F1")
    for k, t in totals.items():
        print(f"{k:32s} {t['photos']:5d} {t['pred']:5d} {t['gt']:4d} {t['tp']:4d} {t['fp']:4d} {t['fn']:4d} "
              f"{t['ignored_pred']:4d}  {t['precision']:.3f}  {t['recall']:.3f}  {t['f1']:.3f}")
    for n, v in loc.items():
        print(f"best IoU per GT box, pipeline, {n}: {v}")
    args.out.write_text(json.dumps({"use_ignore": args.use_ignore, "totals": totals, "localisation": loc,
                                    "photos": per_photo}, indent=1),
                        encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
