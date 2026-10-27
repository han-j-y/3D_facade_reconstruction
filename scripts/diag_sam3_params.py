#!/usr/bin/env python3
"""Offline sweep of SAM3 detection parameters on cached detections (``cache_sam3_detections.py``).

Per setting: score threshold, prompt, container policy (``union`` = pipeline: nested boxes merge into their envelope;
``children`` = drop a box that contains >= 2 other detections before the merge, i.e. keep the individual panes) and
object-mask exclusion. The detections then go through the pipeline's unitize (floors, spanning drop, ast merge,
per-floor split) and the units are scored against the CMP ground-truth windows (test split COCO, IoU >= 0.5,
one-to-one) and against the labelled windows of real_labels_test (how many are matched by a unit).

    python scripts/diag_sam3_params.py --cache ../window_ast_predictor/runs/clean_eval/testset/sam3_cache/samoff
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_pipeline as rp  # noqa: E402

EXP = ROOT.parent
COCO = EXP / "sam3_ft" / "data" / "cmp_windows" / "test" / "_annotations.coco.json"
LABELS = EXP / "window_ast_predictor" / "data" / "real_labels_test" / "labels.jsonl"


def iou_matrix(a: list, b: list) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)))
    A, B = np.asarray(a, float)[:, None], np.asarray(b, float)[None]
    iw = np.clip(np.minimum(A[..., 2], B[..., 2]) - np.maximum(A[..., 0], B[..., 0]), 0, None)
    ih = np.clip(np.minimum(A[..., 3], B[..., 3]) - np.maximum(A[..., 1], B[..., 1]), 0, None)
    inter = iw * ih
    area = lambda X: (X[..., 2] - X[..., 0]) * (X[..., 3] - X[..., 1])  # noqa: E731
    return inter / (area(A) + area(B) - inter)


def n_matched(pred: list, gt: list, thr: float = 0.5) -> int:
    m = iou_matrix(pred, gt)
    if m.size == 0:
        return 0
    r, c = linear_sum_assignment(-m)
    return int((m[r, c] >= thr).sum())


def contained(o: list, b: list, inside: float = 0.8) -> bool:
    iw = max(0, min(b[2], o[2]) - max(b[0], o[0]))
    ih = max(0, min(b[3], o[3]) - max(b[1], o[1]))
    ao = (o[2] - o[0]) * (o[3] - o[1])
    return ao < (b[2] - b[0]) * (b[3] - b[1]) and iw * ih >= inside * ao


def drop_containers(boxes: list, min_children: int = 2) -> list:
    return [b for i, b in enumerate(boxes)
            if sum(contained(o, b) for j, o in enumerate(boxes) if j != i) < min_children]


def prefer_inner(boxes: list, low: list, min_frac: float) -> list:
    """Replace a box by the largest lower-score detection inside it covering >= min_frac of its area (CMP glazing
    convention: SAM3 often boxes the whole opening or surround); containers of >= 2 kept boxes are dropped."""
    boxes = drop_containers(boxes)
    out = []
    for b in boxes:
        ab = (b[2] - b[0]) * (b[3] - b[1])
        kids = [o for o in low if contained(o, b) and (o[2] - o[0]) * (o[3] - o[1]) >= min_frac * ab]
        out.append(max(kids, key=lambda o: (o[2] - o[0]) * (o[3] - o[1])) if kids else b)
    return out


def unitize(boxes: list, iw: int, ih: int, col_tol: float, row_tol: float) -> list:
    if not boxes:
        return []
    boxes = [list(b) for b in boxes]
    cy = np.array([0.5 * (b[1] + b[3]) / ih for b in boxes])
    floors = rp.assign_floors(cy, row_tol)
    spanning = set(rp.spanning_box_indices(boxes, floors))
    if spanning:
        boxes = [b for i, b in enumerate(boxes) if i not in spanning]
        cy = np.array([0.5 * (b[1] + b[3]) / ih for b in boxes])
        floors = rp.assign_floors(cy, row_tol)
    cx = np.array([0.5 * (b[0] + b[2]) / iw for b in boxes])
    ast = rp._load(ROOT / "scripts" / "overlay_facade_merge_boxes_ast.py", "ast_merge")
    merged, members, _, _ = ast.merge_adjacent_boxes(boxes, cx, cy, row_tol=row_tol, adj_gap=1.0, merge_bays=False,
                                                     col_tol=col_tol, merge_vertical=False, adjacent=False)
    merged, _ = rp.split_member_groups_by_floor(boxes, merged, members, floors)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.45, 0.5, 0.55, 0.6, 0.65])
    ap.add_argument("--prompts", nargs="+", default=["window"])
    ap.add_argument("--policies", nargs="+", default=["union", "children"])
    ap.add_argument("--excludes", nargs="+", default=["car", "car+storefront", "car+shop window"],
                    help="'+'-joined mask prompts from the cache")
    ap.add_argument("--exclude-min-cover", type=float, default=0.5)
    ap.add_argument("--inner-min-score", type=float, default=0.3,
                    help="policy inner<frac>: lowest score of the inner detections that may replace a box")
    ap.add_argument("--col-tol", type=float, default=0.045)
    ap.add_argument("--row-tol", type=float, default=0.055)
    ap.add_argument("--ref-runs", nargs="*", default=["runs_tree_scratch_bal_b_samftema19_ms12",
                                                       "runs_tree_scratch_bal_b_samoff_ms12"],
                    help="pipeline runs whose facade_dsl units are scored the same way, for reference")
    ap.add_argument("--per-facade", nargs="*", default=[], help="settings 'thr,prompt,policy,exclude' to list per facade")
    ap.add_argument("--dump", type=Path, default=None, help="write units of --per-facade settings here (json)")
    args = ap.parse_args()

    coco = json.loads(COCO.read_text())
    gt: dict[str, list] = {}
    for im in coco["images"]:
        gt[im["file_name"].removesuffix(".jpg")] = []
    id2 = {im["id"]: im["file_name"].removesuffix(".jpg") for im in coco["images"]}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        gt[id2[a["image_id"]]].append([x, y, x + w, y + h])
    lab: dict[str, list] = {}
    for line in LABELS.open():
        r = json.loads(line)
        if r.get("status") == "done":
            lab.setdefault("cmp_" + r["id"].split("_")[0], []).append(r["e2e_box_xyxy"])

    stems = sorted(p.stem for p in args.cache.glob("cmp_*.json"))
    cache = {s: json.loads((args.cache / f"{s}.json").read_text()) for s in stems}
    masks = {}
    for s in stems:
        z = np.load(args.cache / f"{s}_masks.npz")
        h, w = z["shape"]
        masks[s] = {k: np.unpackbits(z[k])[: h * w].reshape(h, w).astype(bool) for k in z.files if k != "shape"}

    def score(units_by_stem: dict) -> tuple:
        tp = npred = ngt = lm = ln = 0
        for s, u in units_by_stem.items():
            tp += n_matched(u, gt[s])
            npred += len(u)
            ngt += len(gt[s])
            lm += n_matched(u, lab.get(s, []))
            ln += len(lab.get(s, []))
        p, r = tp / max(npred, 1), tp / max(ngt, 1)
        return p, r, 2 * p * r / max(p + r, 1e-9), lm, ln

    def run(thr, prompt, policy, excl) -> dict:
        out = {}
        for s in stems:
            iw, ih = cache[s]["image_size"]
            dets = [d for d in cache[s]["detections"][prompt] if d["score"] >= thr]
            m = np.zeros((ih, iw), bool)
            for k in excl.split("+"):
                m |= masks[s][k]
            dets, _ = rp.drop_boxes_on_mask(dets, m, min_cover=args.exclude_min_cover)
            boxes = [d["box_xyxy"] for d in dets]
            if policy == "children":
                boxes = drop_containers(boxes)
            elif policy.startswith("inner"):
                low = [d["box_xyxy"] for d in cache[s]["detections"][prompt]
                       if args.inner_min_score <= d["score"] < thr]
                boxes = prefer_inner(boxes, low, float(policy.removeprefix("inner") or 0.4))
            out[s] = unitize(boxes, iw, ih, args.col_tol, args.row_tol)
        return out

    print(f"{'setting':44s} {'units':>5s} {'P':>5s} {'R':>5s} {'F1':>5s}  labelled matched")
    for run_name in args.ref_runs:
        units = {}
        for s in stems:
            dsl = json.loads((ROOT / run_name / f"e2e_sam3_cluster_{s}" / "facade_dsl.json").read_text())
            units[s] = [u["box_xyxy"] for u in dsl["instances"] if u.get("kind", "window") == "window"]
        p, r, f, lm, ln = score(units)
        print(f"{'[run] ' + run_name.replace('runs_tree_scratch_bal_b_', ''):44s} "
              f"{sum(map(len, units.values())):5d} {p:5.2f} {r:5.2f} {f:5.2f}  {lm}/{ln}")
    results = {}
    for thr, prompt, policy, excl in product(args.thresholds, args.prompts, args.policies, args.excludes):
        units = run(thr, prompt, policy, excl)
        key = f"{thr},{prompt},{policy},{excl}"
        results[key] = units
        p, r, f, lm, ln = score(units)
        print(f"{key:44s} {sum(map(len, units.values())):5d} {p:5.2f} {r:5.2f} {f:5.2f}  {lm}/{ln}")
    dump = {}
    for key in args.per_facade:
        thr, prompt, policy, excl = key.split(",")
        units = results.get(key) or run(float(thr), prompt, policy, excl)
        print(f"\n{key}: per facade units / GT matched / labelled matched")
        for s in stems:
            print(f"  {s}: {len(units[s]):3d} units  GT {n_matched(units[s], gt[s])}/{len(gt[s])}  "
                  f"labelled {n_matched(units[s], lab.get(s, []))}/{len(lab.get(s, []))}")
        dump[key] = units
    if args.dump:
        args.dump.write_text(json.dumps({"images": {s: cache[s]["image"] for s in stems}, "gt": gt,
                                         "settings": dump}))


if __name__ == "__main__":
    main()
