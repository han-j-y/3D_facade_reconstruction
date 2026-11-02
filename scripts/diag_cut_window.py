#!/usr/bin/env python3
"""Is there one structure-split cut that keeps b0001's box groups whole and separates b0150's arched windows?

For each run and descriptor, the average-linkage dendrogram (cosine) of every box group gives:

* b0001 ``keep``: the cut must be >= the highest merge in every box group (otherwise a group splits),
* b0150 ``separate``: the cut must be < the lowest merge that joins an arched unit with a plain one in any box group;
  ``clean`` is the highest merge inside the arched-only and plain-only parts (at or above it each side is one
  cluster).

Arched units of b0150 are the units overlapping (IoU >= 0.5) the reference arched boxes below (from the W+D+S FT
run). The window of valid cuts is [keep, separate); ``clusters`` lists b0150's cluster count at the window's middle.

Example:
  python scripts/diag_cut_window.py --runs runs_tree_scratch_bal_b_samwds_ms12 runs_tree_scratch_bal_b_samoff_ms12
"""

from __future__ import annotations

import argparse
import json
import sys
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
from diag_cluster_features import TREE, iou  # noqa: E402
from diag_split_sweep import describe_all  # noqa: E402

ARCHED_B0150 = [[397, 617, 429, 693], [440, 617, 474, 692], [548, 617, 608, 693]]


def merges(feat: np.ndarray, flags: list[bool]) -> tuple[float, float, float]:
    """(top merge, lowest arched/plain merge, highest arched-only or plain-only merge) of one group."""
    if len(feat) < 2:
        return 0.0, np.inf, 0.0
    Z = linkage(pdist(feat, metric="cosine"), method="average")
    n = len(feat)
    kind = {i: {flags[i]} for i in range(n)}
    mixed, pure = np.inf, 0.0
    for k, (a, b, h, _) in enumerate(Z):
        s = kind[int(a)] | kind[int(b)]
        kind[n + k] = s
        if len(s) == 2 and len(kind[int(a)]) == 1 and len(kind[int(b)]) == 1 and kind[int(a)] != kind[int(b)]:
            mixed = min(mixed, h)
        elif len(s) == 1:
            pure = max(pure, h)
    return float(Z[-1, 2]), float(mixed), float(pure)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, nargs="+", default=[ROOT / "runs_tree_scratch_bal_b_samwds_ms12"])
    ap.add_argument("--descs", nargs="+", default=["heads", "heads_tta", "heads_t2", "z", "bars+heads"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    pred = rp.TreeStructurePredictor(TREE / "checkpoints" / "structure_best.pt", torch.device(args.device),
                                     tree_root=TREE)

    def load(run: Path, fac: str):
        dsl = json.loads((run / f"e2e_sam3_cluster_cmp_{fac}" / "facade_dsl.json").read_text())
        boxes = [u["box_xyxy"] for u in dsl["instances"] if u.get("kind", "window") == "window"]
        groups, _ = rp.cluster_box_geometry_pipeline(
            boxes, tuple(dsl["meta"]["image_size"]), rp.base, method="gmm_diag", k_max=6, seed=42,
            conservative_k=True, conservative_k_wh_tol=0.008, merge_near=False, snap_wh=True)
        photo = Image.open(ROOT / dsl["meta"]["image"]).convert("RGB")
        return boxes, np.asarray(groups), describe_all(pred, [photo.crop(tuple(int(v) for v in b)) for b in boxes])

    for run in args.runs:
        b1, g1, f1 = load(run, "b0001")
        b2, g2, f2 = load(run, "b0150")
        arched = [any(iou(b, a) >= 0.5 for a in ARCHED_B0150) for b in b2]
        print(f"\n== {run.name}: b0001 box groups {np.bincount(g1).tolist()}; b0150 box groups "
              f"{np.bincount(g2).tolist()}, arched units {[i for i, a in enumerate(arched) if a]} in groups "
              f"{sorted({int(g2[i]) for i, a in enumerate(arched) if a})}")
        print(f"{'descriptor':11s} {'b0001 keep >=':>14s} {'b0150 separate <':>17s} {'b0150 clean >=':>15s}   window")
        for d in args.descs:
            keep = max(merges(f1[d][g1 == g], [False] * int((g1 == g).sum()))[0] for g in set(g1.tolist()))
            sep, clean = np.inf, 0.0
            for g in set(g2.tolist()):
                idx = np.where(g2 == g)[0]
                _, m, p = merges(f2[d][idx], [arched[i] for i in idx])
                sep, clean = min(sep, m), max(clean, p)
            ok = keep < sep
            note = f"[{keep:.3f}, {sep:.3f})" if ok else "none"
            if ok:
                t = 0.5 * (keep + sep)
                lab = np.zeros(len(b2), dtype=np.int64)
                nxt = 0
                for g in sorted(set(g2.tolist())):
                    idx = np.where(g2 == g)[0]
                    sub = np.zeros(1, dtype=np.int64) if len(idx) == 1 else fcluster(
                        linkage(pdist(f2[d][idx], metric="cosine"), method="average"), t=t, criterion="distance") - 1
                    lab[idx] = sub + nxt
                    nxt += int(sub.max()) + 1
                note += f"  at {t:.3f}: b0150 {len(set(lab.tolist()))} clusters"
            print(f"{d:11s} {keep:14.3f} {sep:17.3f} {clean:15.3f}   {note}")


if __name__ == "__main__":
    main()
