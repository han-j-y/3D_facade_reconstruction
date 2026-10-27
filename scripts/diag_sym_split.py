#!/usr/bin/env python3
"""Symmetry-aware structure splits inside box groups (offline, on finished e2e runs).

Mirror partners: each window box reflected about the run's chosen symmetry axis (``axis/<stem>/summary.json``,
``chosen_cx``) is paired with the other window it overlaps most (IoU >= ``--pair-iou``); windows reflecting onto
themselves or onto nothing are unconstrained. Two ways to use them:

* ``gate``: walk each box group's average-linkage tree (cosine on the descriptor) top-down; a node is split only if
  its merge height is above the cut and the split separates at most ``eps`` of the mirror pairs inside the node,
* ``sym``: replace every descriptor by the mean of itself and its partner's, then cut as usual.

Reports b0001 (3 box groups; should stay 3) and b0150 (arched units should be split from plain ones) and the pooled
13-facade within-box-group same->1 / diff->split.

Example:
  python scripts/diag_sym_split.py --runs runs_tree_scratch_bal_b_samwds_ms12
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_pipeline as rp  # noqa: E402
from diag_cluster_features import TREE, TRAIN_FACADES, iou  # noqa: E402
from diag_cut_window import ARCHED_B0150  # noqa: E402
from diag_split_sweep import describe_all  # noqa: E402
from diag_sam3_params import unitize  # noqa: E402


def mirror_partners(boxes: list[list[int]], cx: float, pair_iou: float) -> list[int | None]:
    out: list[int | None] = []
    for i, b in enumerate(boxes):
        m = [2 * cx - b[2], b[1], 2 * cx - b[0], b[3]]
        if iou(m, b) >= 0.5:
            out.append(None)
            continue
        j = max((k for k in range(len(boxes)) if k != i), key=lambda k: iou(m, boxes[k]), default=None)
        out.append(j if j is not None and iou(m, boxes[j]) >= pair_iou else None)
    return out


def refine_axis(boxes: list, cx: float, pair_iou: float, radius: float) -> float:
    """Shift ``cx`` within +-radius px to the position pairing the most boxes (ties: nearest to ``cx``); narrow
    windows lose their partners at a few px of axis error."""
    if radius <= 0:
        return cx
    grid = np.arange(cx - radius, cx + radius + 0.5, 1.0)
    n = [sum(p is not None for p in mirror_partners(boxes, float(c), pair_iou)) for c in grid]
    best = max(n)
    return float(min((c for c, k in zip(grid, n) if k == best), key=lambda c: abs(c - cx)))


def gate_split(feat: np.ndarray, idx: np.ndarray, partner: list[int | None], thr: float, eps: float) -> np.ndarray:
    """Top-down average-linkage splits of units ``idx``, refusing splits that break > eps of mirror pairs."""
    if len(idx) == 1:
        return np.zeros(1, dtype=np.int64)
    Z = linkage(pdist(feat[idx], metric="cosine"), method="average")
    n = len(idx)
    leaves: dict[int, list[int]] = {i: [i] for i in range(n)}
    for k, (a, b, _, _) in enumerate(Z):
        leaves[n + k] = leaves[int(a)] + leaves[int(b)]
    lab = np.zeros(n, dtype=np.int64)
    nxt = [0]

    def visit(node: int) -> None:
        if node < n:
            lab[leaves[node]] = nxt[0]
            nxt[0] += 1
            return
        a, b, h, _ = Z[node - n]
        a, b = int(a), int(b)
        ok = h > thr
        if ok:
            side = {u: 0 for u in leaves[a]} | {u: 1 for u in leaves[b]}
            glob = {int(idx[u]): s for u, s in side.items()}
            pairs = [(i, partner[i]) for i in glob if partner[i] is not None and partner[i] in glob and i < partner[i]]
            broken = sum(glob[i] != glob[j] for i, j in pairs)
            ok = not pairs or broken / len(pairs) <= eps
        if ok:
            visit(a)
            visit(b)
        else:
            lab[leaves[node]] = nxt[0]
            nxt[0] += 1

    visit(2 * n - 2)
    return lab


def cluster(feat, groups, partner, thr, mode, eps) -> np.ndarray:
    if mode == "sym":
        feat = np.stack([feat[i] if p is None else 0.5 * (feat[i] + feat[p]) for i, p in enumerate(partner)])
    labels = np.zeros(len(groups), dtype=np.int64)
    nxt = 0
    for g in sorted(set(groups.tolist())):
        idx = np.where(groups == g)[0]
        if mode == "gate":
            sub = gate_split(feat, idx, partner, thr, eps)
        else:
            sub = np.zeros(1, dtype=np.int64) if len(idx) == 1 else \
                fcluster(linkage(pdist(feat[idx], metric="cosine"), method="average"), t=thr, criterion="distance") - 1
        _, sub = np.unique(sub, return_inverse=True)
        labels[idx] = sub + nxt
        nxt += int(sub.max()) + 1
    return labels


def overlap_small(a: list, b: list) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih / max(min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])), 1e-9)


def two_tier(dets: list, anchor: float, max_overlap: float) -> list:
    keep = [d for d in dets if d["score"] >= anchor]
    for d in sorted((d for d in dets if d["score"] < anchor), key=lambda d: -d["score"]):
        if all(overlap_small(d["box_xyxy"], k["box_xyxy"]) < max_overlap for k in keep):
            keep.append(d)
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, nargs="+", default=[ROOT / "runs_tree_scratch_bal_b_samwds_ms12"])
    ap.add_argument("--desc", default="heads")
    ap.add_argument("--pair-iou", type=float, default=0.3)
    ap.add_argument("--cuts", type=float, nargs="+", default=[0.1, 0.16, 0.2, 0.25, 0.3])
    ap.add_argument("--labels", type=Path, default=TREE / "data" / "real_labels_test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dump-dir", type=Path, default=None,
                    help="per run: boxes, axis, GT groups and the labels of every setting at --dump-cut")
    ap.add_argument("--dump-cut", type=float, default=0.16)
    ap.add_argument("--min-side", type=int, default=13, help="as run_pipeline.py --min-side")
    ap.add_argument("--raw-min-score", type=float, default=None,
                    help="rebuild the units from the run's raw detections with score >= this (unitize as the "
                    "pipeline) instead of facade_dsl.json, e.g. to emulate a higher --threshold")
    ap.add_argument("--raw-anchor-score", type=float, default=None,
                    help="two-tier: keep raw detections >= this; add those in [--raw-min-score, this) in score "
                    "order only if they overlap no kept box (intersection / smaller area < --fill-max-overlap)")
    ap.add_argument("--fill-max-overlap", type=float, default=0.2)
    ap.add_argument("--axis-refine", type=float, default=0.0,
                    help="shift the axis within +- this fraction of the image width to pair the most boxes (0 = off)")
    ap.add_argument("--col-tol", type=float, default=0.045)
    ap.add_argument("--row-tol", type=float, default=0.055)
    ap.add_argument("--include-train", action="store_true", help="also dump b0100 (excluded from the pooled scores)")
    ap.add_argument("--dump-name", default=None, help="dump file stem (default: the run directory name)")
    ap.add_argument("--min-rel-width", type=float, default=0.0, help="as run_pipeline.py --min-rel-width")
    ap.add_argument("--dump-facades", nargs="+", default=["b0001", "b0150", "b0083"])
    args = ap.parse_args()
    pred = rp.TreeStructurePredictor(TREE / "runs" / "scratch_bal_b" / "best.pt", torch.device(args.device),
                                     tree_root=TREE)
    with pred.pkgs:
        from scripts.train_tree_adapter import topo_ok

    gt_rows: dict[str, list] = {}
    for line in (args.labels / "labels.jsonl").open():
        r = json.loads(line)
        if r.get("status") == "done":
            r["gt"] = json.loads((args.labels / r["id"] / "tokens.json").read_text())["tokens"]
            gt_rows.setdefault(r["id"].split("_")[0], []).append(r)

    settings = [("plain", None)] + [("gate", e) for e in (0.0, 0.1, 0.25)] + [("sym", None)]
    for run in args.runs:
        data, meta = {}, {}
        for fac in sorted(gt_rows):
            if fac in TRAIN_FACADES and not args.include_train:
                continue
            dsl = json.loads((run / f"e2e_sam3_cluster_cmp_{fac}" / "facade_dsl.json").read_text())
            boxes = [u["box_xyxy"] for u in dsl["instances"] if u.get("kind", "window") == "window"]
            if args.raw_min_score is not None:
                raw = json.loads((run / f"e2e_sam3_cluster_cmp_{fac}" / "index_raw.json").read_text())["windows"]
                iw, ih = dsl["meta"]["image_size"]
                raw = [w for w in raw if w["score"] >= args.raw_min_score
                       and min(w["box_xyxy"][2] - w["box_xyxy"][0], w["box_xyxy"][3] - w["box_xyxy"][1])
                       >= args.min_side]
                if args.raw_anchor_score is not None:
                    raw = two_tier(raw, args.raw_anchor_score, args.fill_max_overlap)
                boxes = unitize([w["box_xyxy"] for w in raw], iw, ih, args.col_tol, args.row_tol)
            keep = [k and min(b[2] - b[0], b[3] - b[1]) >= args.min_side
                    for b, k in zip(boxes, rp.narrow_box_keep(boxes, args.min_rel_width))]
            if not all(keep):
                print(f"{fac}: dropped small/narrow {[b for b, k in zip(boxes, keep) if not k]}")
            boxes = [b for b, k in zip(boxes, keep) if k]
            groups, _ = rp.cluster_box_geometry_pipeline(
                boxes, tuple(dsl["meta"]["image_size"]), rp.base, method="gmm_diag", k_max=6, seed=42,
                conservative_k=True, conservative_k_wh_tol=0.008, merge_near=False, snap_wh=True)
            cx = json.loads((run / "axis" / f"cmp_{fac}" / "summary.json").read_text())["chosen_cx"]
            cx = refine_axis(boxes, cx, args.pair_iou, args.axis_refine * dsl["meta"]["image_size"][0])
            photo = Image.open(ROOT / dsl["meta"]["image"]).convert("RGB")
            feat = describe_all(pred, [photo.crop(tuple(int(v) for v in b)) for b in boxes])[args.desc]
            matched = []
            for r in gt_rows[fac]:
                j = max(range(len(boxes)), key=lambda k: iou(boxes[k], r["e2e_box_xyxy"]))
                if iou(boxes[j], r["e2e_box_xyxy"]) >= 0.5:
                    matched.append((j, r["gt"]))
            partner = mirror_partners(boxes, cx, args.pair_iou)
            data[fac] = (boxes, np.asarray(groups), feat, partner, matched)
            meta[fac] = {"image": str(ROOT / dsl["meta"]["image"]), "axis_x": cx}
        arched = [i for i, b in enumerate(data["b0150"][0]) if any(iou(b, a) >= 0.5 for a in ARCHED_B0150)]
        print(f"\n== {run.name} ({args.desc}); b0001 mirror-paired {sum(p is not None for p in data['b0001'][3])}/"
              f"{len(data['b0001'][3])}, b0150 {sum(p is not None for p in data['b0150'][3])}/"
              f"{len(data['b0150'][3])}; b0150 arched {arched}")
        print(f"{'setting':16s} {'cut':>5s}  {'b0001 clusters':>14s}  {'b0150 clusters':>14s}  {'arched split?':>13s}"
              f"   13-facade within-group same->1 / diff->split")
        dumps: dict[str, dict[str, list[int]]] = {}
        for mode, eps in settings:
            for thr in args.cuts:
                res = {}
                for fac, (boxes, groups, feat, partner, matched) in data.items():
                    res[fac] = cluster(feat, groups, partner, thr, mode, eps if eps is not None else 0.0)
                l2 = res["b0150"]
                sep = all(l2[a] != l2[p] for a in arched for p in range(len(l2))
                          if p not in arched and data["b0150"][1][p] == data["b0150"][1][a])
                si = sn = do = dn = 0
                for fac, (boxes, groups, feat, partner, matched) in data.items():
                    if fac in TRAIN_FACADES:
                        continue
                    lab = res[fac]
                    for (a, ga), (b, gb) in combinations(matched, 2):
                        if groups[a] != groups[b]:
                            continue
                        if topo_ok(ga, gb):
                            sn += 1
                            si += lab[a] == lab[b]
                        else:
                            dn += 1
                            do += lab[a] != lab[b]
                name = mode if eps is None else f"{mode} eps {eps}"
                print(f"{name:16s} {thr:5.2f}  {len(set(res['b0001'].tolist())):14d}  "
                      f"{len(set(l2.tolist())):14d}  {'yes' if sep else 'no':>13s}   {si / sn:5.0%} / {do / dn:5.0%}")
                if args.dump_dir and abs(thr - args.dump_cut) < 1e-9:
                    for fac, lab in res.items():
                        dumps.setdefault(fac, {})[f"{name} @{thr}"] = lab.tolist()
        if args.dump_dir:
            out = {}
            for fac in args.dump_facades:
                boxes, groups, _, _, matched = data[fac]
                gt_groups: list[list[str]] = []
                gt = {}
                for j, toks in matched:
                    k = next((i for i, g in enumerate(gt_groups) if topo_ok(g, toks)), None)
                    if k is None:
                        gt_groups.append(toks)
                        k = len(gt_groups) - 1
                    gt[str(j)] = k
                out[fac] = {**meta[fac], "boxes": boxes, "gt": gt,
                            "labels": {"box only": groups.tolist(), **dumps[fac]}}
            args.dump_dir.mkdir(parents=True, exist_ok=True)
            (args.dump_dir / f"{args.dump_name or run.name}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
