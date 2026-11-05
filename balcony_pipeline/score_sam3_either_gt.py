"""One-to-one IoU score where a side-by-side balcony may count whole or in fragments.

Ground truth is the balcony components of the CMP label map (palette index 7),
as in ``score_sam3_label_iou.py``. Two components belong to one group when they
share a row and sit side by side:

* vertical overlap is at least half the shorter box height, and
* the horizontal gap is at most the shorter box height.

Grouping is transitive over the label components. A group's merged box is the
bounding box of its components.

Three ground-truth sets are scored against the same raw SAM3 boxes, one-to-one
greedy from the highest IoU, IoU >= 0.5 is a true positive:

* fragments: the label components as drawn;
* merged: every group as one box;
* either: each group is one box or its fragments, whichever gives more true
  positives in that image. A tie keeps the fragments, so a group is only
  counted whole when a SAM3 box matches the whole balcony. Recall uses the
  ground-truth count of the chosen units.

SAM3 boxes come from ``--boxes`` (stem -> list of xyxy boxes), the cache written
by ``score_sam3_merged_gt.py``.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_pipeline/score_sam3_either_gt.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from balcony_train.crop_balconies_xml import balcony_pixel_boxes, scale_box  # noqa: E402
from score_sam3_label_iou import match_iou  # noqa: E402

ROWS = ROOT / "runs" / "batch" / "_rows"
DEFAULT_DIR = ROOT / "data" / "extended_selected"
DEFAULT_BOXES = ROWS / "extended_selected_sam3_boxes.json"
DEFAULT_OUT = ROWS / "extended_selected_sam3_either_gt.json"
EXCLUDED = (
    "cmp_x0002 cmp_x0003 cmp_x0006 cmp_x0008 cmp_x0011 cmp_x0012 cmp_x0014 cmp_x0016 cmp_x0021 cmp_x0036 "
    "cmp_x0043 cmp_x0044 cmp_x0067 cmp_x0107 cmp_x0116 cmp_x0127 cmp_x0162 cmp_x0173 cmp_x0176 cmp_x0227"
)
MAX_EXHAUSTIVE_GROUPS = 12


def same_row_neighbors(a: list[int], b: list[int]) -> bool:
    short = min(a[3] - a[1], b[3] - b[1])
    if short <= 0:
        return False
    if min(a[3], b[3]) - max(a[1], b[1]) < 0.5 * short:
        return False
    return max(a[0], b[0]) - min(a[2], b[2]) <= short


def row_groups(boxes: list[list[int]]) -> list[list[int]]:
    """Indices of transitively side-by-side components, one list per group."""
    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if same_row_neighbors(boxes[i], boxes[j]):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: (boxes[g[0]][1], boxes[g[0]][0]))


def bbox(boxes: list[list[int]]) -> list[int]:
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def score_image(gt: list[list[int]], det: list[list[int]], iou: float) -> dict:
    groups = row_groups(gt)
    whole = [bbox([gt[k] for k in g]) for g in groups]
    multi = [i for i, g in enumerate(groups) if len(g) > 1]

    def units(chosen: set[int]) -> list[list[int]]:
        out: list[list[int]] = []
        for i, g in enumerate(groups):
            if len(g) == 1 or i in chosen:
                out.append(whole[i])
            else:
                out.extend(gt[k] for k in g)
        return out

    def tp_of(chosen: set[int]) -> int:
        return match_iou(units(chosen), det, threshold=iou)[0]

    best: set[int] = set()
    best_tp = tp_of(best)
    exhaustive = len(multi) <= MAX_EXHAUSTIVE_GROUPS
    if exhaustive:
        for mask in range(1, 1 << len(multi)):
            chosen = {g for bit, g in enumerate(multi) if mask >> bit & 1}
            tp = tp_of(chosen)
            if tp > best_tp or (tp == best_tp and len(chosen) < len(best)):
                best, best_tp = chosen, tp
    else:
        improved = True
        while improved:
            improved = False
            for g in multi:
                if g not in best and tp_of(best | {g}) > best_tp:
                    best = best | {g}
                    best_tp = tp_of(best)
                    improved = True

    return {
        "sam3": len(det),
        "fragments": {"gt": len(gt), "tp": match_iou(gt, det, threshold=iou)[0]},
        "merged": {"gt": len(whole), "tp": match_iou(whole, det, threshold=iou)[0]},
        "either": {"gt": len(units(best)), "tp": best_tp},
        "groups": [g for g in groups if len(g) > 1],
        "groups_counted_whole": [groups[i] for i in sorted(best)],
        "exhaustive": exhaustive,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--boxes", type=Path, default=DEFAULT_BOXES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--exclude", default=EXCLUDED)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    excluded = set(args.exclude.replace(",", " ").split())
    preds = json.loads(args.boxes.read_text(encoding="utf-8"))
    images = sorted(p for p in args.image_dir.glob("*.jpg") if p.stem not in excluded)
    missing = [p.stem for p in images if p.stem not in preds]
    if missing:
        raise SystemExit(f"no cached SAM3 boxes for {missing}")

    rows = []
    for path in images:
        label_path = args.label_dir / f"{path.stem}.png"
        if not label_path.is_file():
            print(f"skip {path.stem}: missing {label_path}", flush=True)
            continue
        with Image.open(path) as photo:
            width, height = photo.size
        label = Image.open(label_path)
        gt = [
            list(scale_box(b, src_w=label.size[0], src_h=label.size[1], dst_w=width, dst_h=height))
            for b in balcony_pixel_boxes(label)
        ]
        rows.append({"stem": path.stem, **score_image(gt, preds[path.stem], args.iou)})

    sam_n = sum(r["sam3"] for r in rows)
    total: dict = {"images": len(rows), "sam3": sam_n, "iou_threshold": args.iou}
    for key in ("fragments", "merged", "either"):
        gt_n = sum(r[key]["gt"] for r in rows)
        tp_n = sum(r[key]["tp"] for r in rows)
        p = tp_n / sam_n if sam_n else 0.0
        r = tp_n / gt_n if gt_n else 0.0
        total[key] = {
            "gt": gt_n,
            "tp": tp_n,
            "fp": sam_n - tp_n,
            "fn": gt_n - tp_n,
            "precision": p,
            "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
        }
        print(
            f"{key:9s} gt={gt_n}  sam3={sam_n}  tp={tp_n}  fp={sam_n - tp_n}  fn={gt_n - tp_n}  "
            f"P={p:.3f}  R={r:.3f}  F1={total[key]['f1']:.3f}",
            flush=True,
        )
    total["groups"] = sum(len(r["groups"]) for r in rows)
    total["groups_counted_whole"] = sum(len(r["groups_counted_whole"]) for r in rows)
    total["not_exhaustive"] = [r["stem"] for r in rows if not r["exhaustive"]]
    print(
        f"groups={total['groups']}  counted whole={total['groups_counted_whole']}  "
        f"not exhaustive={total['not_exhaustive']}",
        flush=True,
    )
    args.out.write_text(json.dumps({"total": total, "images": rows}, indent=1), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
