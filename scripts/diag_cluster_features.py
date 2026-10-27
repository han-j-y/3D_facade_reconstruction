#!/usr/bin/env python3
"""Which window descriptor should split the box-geometry groups into types? (e2e test facades)

Takes the window units of a ``shape_then_struct`` run and its box groups (from the matching ``--cluster-mode box``
run, or recomputed with the pipeline's box-clustering defaults), describes every unit with the pane-tree structure model on the same
tight crop the pipeline uses, and re-clusters each facade offline. Scored on the labelled test windows
(``window_ast_predictor/data/real_labels_test``, matched at IoU >= 0.5 as in ``eval_e2e_testset.py``):

* pair AUC: P(cosine distance of a different-structure pair > that of a same-structure pair), within a facade
  and within a box group,
* same→1 / diff→split of the clusters (pooled over facades, pre-symmetry-repair types) for a sweep of the
  average-linkage cosine cut, plus a leave-one-facade-out pick of the cut.

Descriptors: ``z`` root node embedding (what ``shape_then_struct`` uses), ``g`` global image embedding,
``bars`` bar presence along x / y, ``heads`` prefix / type / axis / V / H probabilities, ``bars+heads``,
``pred`` the member's predicted structure (same topology = one type), ``dino`` DINOv2 ViT-S/14 facade patch
features RoI-pooled over the box (``shape_then_feat``'s feature).

Example:
  python scripts/diag_cluster_features.py --run runs_tree_scratch_bal_b_samwds_ms12 \
      --box-run runs_tree_scratch_bal_b_samwds_box_ms12
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_pipeline as rp  # noqa: E402

TREE = ROOT.parent / "window_ast_predictor"
TRAIN_FACADES = {"b0100"}
NAMES = ["z", "g", "bars", "heads", "bars+heads", "dino"]
THRS = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5]


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    p, n = np.asarray(pos), np.asarray(neg)
    return float((n[:, None] > p[None]).mean() + 0.5 * (n[:, None] == p[None]).mean())


@torch.no_grad()
def describe(pred: rp.TreeStructurePredictor, crops: list[Image.Image]) -> dict[str, np.ndarray]:
    x = torch.stack([pred.tfm(pred.place(c.convert("RGB"), [0, 0, 1, 1], pred.place_scale)[0])
                     for c in crops]).to(pred.device)
    with pred.pkgs:
        d = pred.model.describe(x)
        _, g, _ = pred.model.encode(x)
    out = {k: d[k].float().cpu().numpy() for k in ("z", "bars", "heads")}
    out["g"] = g.float().cpu().numpy()
    out["bars+heads"] = np.concatenate([out["bars"], out["heads"]], 1)
    return out


def split(feat: np.ndarray, groups: np.ndarray, thr: float) -> np.ndarray:
    labels = np.zeros(len(groups), dtype=np.int64)
    nxt = 0
    for g in sorted(set(groups.tolist())):
        idx = np.where(groups == g)[0]
        sub = np.zeros(1, dtype=np.int64) if len(idx) == 1 else \
            fcluster(linkage(pdist(feat[idx], metric="cosine"), method="average"), t=thr, criterion="distance") - 1
        labels[idx] = sub + nxt
        nxt += int(sub.max()) + 1
    return labels


def pair_counts(lab: np.ndarray, matched: list[tuple[int, list[str]]], same_fn) -> tuple[int, int, int, int]:
    si = sn = do = dn = 0
    for (a, ga), (b, gb) in combinations(matched, 2):
        if same_fn(ga, gb):
            sn += 1
            si += lab[a] == lab[b]
        else:
            dn += 1
            do += lab[a] != lab[b]
    return si, sn, do, dn


def summarize(counts: list[tuple[int, int, int, int]]) -> tuple[float, float, float]:
    si, sn, do, dn = (sum(c[i] for c in counts) for i in range(4))
    s, d = si / max(1, sn), do / max(1, dn)
    return s, d, 0.5 * (s + d)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=ROOT / "runs_tree_scratch_bal_b_samwds_ms12")
    ap.add_argument("--box-run", type=Path, default=None,
                    help="matching --cluster-mode box run; default: recompute the box groups with the pipeline defaults")
    ap.add_argument("--structure-ckpt", type=Path, default=TREE / "runs" / "scratch_bal_b" / "best.pt")
    ap.add_argument("--labels", type=Path, default=TREE / "data" / "real_labels_test")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dump-labels", type=Path, default=None,
                    help="per-facade unit boxes, cluster labels (box only / current / z whole facade) and GT groups")
    ap.add_argument("--dump-thr", type=float, default=0.2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    pred = rp.TreeStructurePredictor(args.structure_ckpt, device, tree_root=TREE)
    with pred.pkgs:
        from scripts.train_tree_adapter import topo_ok
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True).to(device).eval()

    labels = defaultdict(list)
    for line in (args.labels / "labels.jsonl").open():
        r = json.loads(line)
        if r.get("status") == "done":
            r["gt"] = json.loads((args.labels / r["id"] / "tokens.json").read_text())["tokens"]
            labels[r["id"].split("_")[0]].append(r)

    fac_data = {}
    for fac, rows in sorted(labels.items()):
        d = args.run / f"e2e_sam3_cluster_cmp_{fac}"
        if not (d / "facade_dsl.json").is_file():
            continue
        dsl = json.loads((d / "facade_dsl.json").read_text())
        units = [u for u in dsl["instances"] if u.get("kind", "window") == "window"]
        boxes = [u["box_xyxy"] for u in units]
        if args.box_run is not None:
            box_dsl = json.loads((args.box_run / d.name / "facade_dsl.json").read_text())
            box_of = {tuple(u["box_xyxy"]): int(u["type_id"]) for u in box_dsl["instances"]
                      if u.get("kind", "window") == "window"}
        else:
            box_lab, _ = rp.cluster_box_geometry_pipeline(
                boxes, tuple(dsl["meta"]["image_size"]), rp.base, method="gmm_diag", k_max=6, seed=42,
                conservative_k=True, conservative_k_wh_tol=0.008, merge_near=False, snap_wh=True)
            box_of = {tuple(b): int(k) for b, k in zip(boxes, box_lab.tolist())}
        photo = Image.open(ROOT / dsl["meta"]["image"]).convert("RGB")
        feats = describe(pred, [photo.crop(tuple(int(v) for v in b)) for b in boxes])
        spatial, meta = rp.patch.facade_patch_spatial(dino, photo, device=device, max_side=896)
        feats["dino"] = rp.patch.roi_pool_patches(spatial, boxes, meta)
        with pred.pkgs:
            fps = [rp.structure_fingerprint(u["structure_ir_member"]) for u in units]
        feats["pred"] = np.array([fps.index(f) for f in fps])
        matched = []
        for r in rows:
            j = max(range(len(units)), key=lambda k: iou(boxes[k], r["e2e_box_xyxy"]))
            if iou(boxes[j], r["e2e_box_xyxy"]) >= 0.5:
                matched.append((j, r["gt"]))
        fac_data[fac] = {"feats": feats, "groups": np.array([box_of[tuple(b)] for b in boxes]),
                         "current": np.array([int(u["type_id"]) for u in units]), "matched": matched}
        print(f"{fac}: {len(units)} units, {len(set(box_of.values()))} box groups, {len(matched)} labelled matched")

    test = [f for f in fac_data if f not in TRAIN_FACADES]
    report: dict = {}

    print("\n== pair AUC on matched labelled windows, 13 test facades (higher = descriptor separates better)")
    print(f"{'descriptor':11s} {'within facade':>14s} {'within box group':>17s}   median dist same / diff")
    for n in NAMES:
        pos, neg, pos_g, neg_g = [], [], [], []
        for f in test:
            v = fac_data[f]
            F = v["feats"][n]
            for (a, ga), (b, gb) in combinations(v["matched"], 2):
                dist = 1.0 - float(F[a] @ F[b] / (np.linalg.norm(F[a]) * np.linalg.norm(F[b]) + 1e-9))
                same = topo_ok(ga, gb)
                (pos if same else neg).append(dist)
                if v["groups"][a] == v["groups"][b]:
                    (pos_g if same else neg_g).append(dist)
        report.setdefault("auc", {})[n] = {"facade": auc(pos, neg), "box_group": auc(pos_g, neg_g)}
        print(f"{n:11s} {auc(pos, neg):14.2f} {auc(pos_g, neg_g):17.2f}   {np.median(pos):.3f} / {np.median(neg):.3f}")

    def counts(f: str, lab: np.ndarray):
        return pair_counts(lab, fac_data[f]["matched"], topo_ok)

    print("\n== clustering, 13 test facades (pre-repair types): same→1 / diff→split / mean")
    ref = {
        "box only": {f: fac_data[f]["groups"] for f in test},
        "current run (z @ 0.16)": {f: fac_data[f]["current"] for f in test},
        "box + pred structure": {f: split(np.eye(len(fac_data[f]["groups"]))[fac_data[f]["feats"]["pred"]],
                                          fac_data[f]["groups"], 0.5) for f in test},
        "pred structure only": {f: fac_data[f]["feats"]["pred"] for f in test},
    }
    for name, labs in ref.items():
        s, d, m = summarize([counts(f, labs[f]) for f in test])
        n_types = np.mean([len(set(labs[f].tolist())) for f in test])
        report.setdefault("reference", {})[name] = {"same": s, "diff": d, "mean": m, "types": n_types}
        print(f"  {name:28s} {s:5.0%} / {d:5.0%} / {m:5.0%}   avg types {n_types:.1f}")

    for scope in ("box", "facade"):
        print(f"\n-- split {'within box groups' if scope == 'box' else 'whole facade (no box stage)'}"
              f": mean of same→1 and diff→split per cut; LOFO = cut picked on the other 12 facades")
        print(f"{'descriptor':11s} " + " ".join(f"{t:>5.2f}" for t in THRS) + "     LOFO same / diff / mean")
        for n in NAMES:
            per = {}
            for t in THRS:
                per[t] = {f: counts(f, split(fac_data[f]["feats"][n],
                                             fac_data[f]["groups"] if scope == "box"
                                             else np.zeros(len(fac_data[f]["groups"]), dtype=np.int64), t))
                          for f in test}
            lofo = []
            for f in test:
                best = max(THRS, key=lambda t: summarize([per[t][o] for o in test if o != f])[2])
                lofo.append(per[best][f])
            ls, ld, lm = summarize(lofo)
            row = {t: summarize(list(per[t].values())) for t in THRS}
            report.setdefault(scope, {})[n] = {"sweep": {str(t): row[t] for t in THRS}, "lofo": (ls, ld, lm)}
            print(f"{n:11s} " + " ".join(f"{row[t][2]:5.0%}" for t in THRS) + f"     {ls:4.0%} / {ld:4.0%} / {lm:4.0%}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))

    if args.dump_labels:
        dump = {}
        for f, v in fac_data.items():
            gt_groups: list[list[str]] = []
            gt = {}
            for j, toks in v["matched"]:
                k = next((i for i, g in enumerate(gt_groups) if topo_ok(g, toks)), None)
                if k is None:
                    gt_groups.append(toks)
                    k = len(gt_groups) - 1
                gt[j] = k
            n = len(v["groups"])
            dump[f] = {
                "image": str(ROOT / json.loads((args.run / f"e2e_sam3_cluster_cmp_{f}" / "facade_dsl.json")
                                               .read_text())["meta"]["image"]),
                "boxes": [u["box_xyxy"] for u in json.loads((args.run / f"e2e_sam3_cluster_cmp_{f}" /
                                                             "facade_dsl.json").read_text())["instances"]
                          if u.get("kind", "window") == "window"],
                "labels": {
                    "box only": v["groups"].tolist(),
                    "current: box then z @0.16": v["current"].tolist(),
                    f"z, whole facade @{args.dump_thr}": split(v["feats"]["z"], np.zeros(n, dtype=np.int64),
                                                               args.dump_thr).tolist(),
                },
                "gt": {str(j): k for j, k in gt.items()},
            }
        args.dump_labels.parent.mkdir(parents=True, exist_ok=True)
        args.dump_labels.write_text(json.dumps(dump))


if __name__ == "__main__":
    main()
