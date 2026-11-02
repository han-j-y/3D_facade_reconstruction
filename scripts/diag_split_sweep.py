#!/usr/bin/env python3
"""How to split box-geometry groups by structure with fewer same-structure splits (``shape_then_struct`` knobs).

For each run (window units + member structure predictions of a finished e2e run), the box groups are recomputed
with the pipeline defaults and every group is split by a descriptor of the pane-tree model with average- or
single-linkage cosine cuts; sub-clusters smaller than ``min size`` are folded back into the nearest larger one.
Scored on the 13 labelled test facades, before symmetry repair:

* within-group same->1: same-structure pairs inside one box group that stay together (what over-splitting breaks),
* within-group diff->split: different-structure pairs inside one box group that get separated,
* voted: share of labelled windows whose cluster's majority-voted structure is right (pipeline voting rule),
* clusters: mean number of clusters per facade.

Example:
  python scripts/diag_split_sweep.py --runs runs_tree_scratch_bal_b_samwds_ms12 runs_tree_scratch_bal_b_samftema19_ms12
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist, pdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_pipeline as rp  # noqa: E402
from diag_cluster_features import TREE, TRAIN_FACADES, iou  # noqa: E402

CUTS = [0.05, 0.08, 0.1, 0.12, 0.16, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
MIN_SIZES = [1, 2, 3]
HEADS = ("type", "axis", "V", "H")


@torch.no_grad()
def describe_all(pred: rp.TreeStructurePredictor, crops: list[Image.Image]) -> dict[str, np.ndarray]:
    """Root-node descriptors: ``z``, ``hidden`` (node MLP layer before z), ``zproj`` (z on the heads' weight rows,
    orthonormalised), ``logits`` (prefix + type / axis / V / H), ``heads`` (their softmaxes), ``heads_t2``
    (softmax at temperature 2), ``heads_tta`` (softmax of logits averaged with the mirrored crop), ``bars+heads``."""
    m = pred.model
    x = torch.stack([pred.tfm(pred.place(c.convert("RGB"), [0, 0, 1, 1], pred.place_scale)[0])
                     for c in crops]).to(pred.device)
    hidden = []
    hook = m.node_mlp[1].register_forward_hook(lambda _m, _i, o: hidden.append(o))
    with pred.pkgs:
        m.eval()
        fmap, g, out = m.encode(x)
        bi = torch.arange(len(x), device=x.device)
        p = m.nodes(fmap, g, bi, out["root_box"])
        hook.remove()
        fmap2, g2, out2 = m.encode(x.flip(-1))
        tta = m.nodes_tta((fmap, fmap2), (g, g2), bi, out["root_box"])
    logits = torch.cat([out["prefix"]] + [p[k] for k in HEADS], 1)
    soft = lambda ls, t=1.0: torch.cat([(l / t).softmax(1) for l in ls], 1)  # noqa: E731
    w = torch.cat([getattr(m, f"{k}_head").weight for k in ("type", "axis", "v", "h")], 0)
    q, _ = torch.linalg.qr(w.T.float())
    f = {
        "z": p["z"], "hidden": hidden[0], "zproj": p["z"].float() @ q, "logits": logits,
        "heads": soft([out["prefix"]] + [p[k] for k in HEADS]),
        "heads_t2": soft([out["prefix"]] + [p[k] for k in HEADS], 2.0),
        "heads_tta": soft([(out["prefix"] + out2["prefix"]) / 2] + [tta[k] for k in HEADS]),
        "bars+heads": torch.cat([torch.sigmoid(p["profile"][:, :, 0]).flatten(1),
                                 soft([out["prefix"]] + [p[k] for k in HEADS])], 1),
    }
    return {k: v.float().cpu().numpy() for k, v in f.items()}


def split(feat: np.ndarray, groups: np.ndarray, thr: float, method: str, min_size: int) -> np.ndarray:
    labels = np.zeros(len(groups), dtype=np.int64)
    nxt = 0
    for g in sorted(set(groups.tolist())):
        idx = np.where(groups == g)[0]
        if len(idx) == 1:
            sub = np.zeros(1, dtype=np.int64)
        else:
            sub = fcluster(linkage(pdist(feat[idx], metric="cosine"), method=method), t=thr,
                           criterion="distance") - 1
            sizes = Counter(sub.tolist())
            big = [s for s, n in sizes.items() if n >= min_size]
            if not big:
                sub[:] = 0
            elif len(big) < len(sizes):
                cent = np.stack([feat[idx][sub == s].mean(0) for s in big])
                for s, n in sizes.items():
                    if n < min_size:
                        m = sub == s
                        sub[m] = big[int(cdist(feat[idx][m].mean(0)[None], cent, metric="cosine").argmin())]
            _, sub = np.unique(sub, return_inverse=True)
        labels[idx] = sub + nxt
        nxt += int(sub.max()) + 1
    return labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, nargs="+", default=[ROOT / "runs_tree_scratch_bal_b_samwds_ms12"])
    ap.add_argument("--structure-ckpt", type=Path, default=TREE / "checkpoints" / "structure_best.pt")
    ap.add_argument("--labels", type=Path, default=TREE / "data" / "real_labels_test")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--descs", nargs="+", default=["z", "heads", "bars+heads"],
                    choices=["z", "hidden", "zproj", "logits", "heads", "heads_t2", "heads_tta", "bars+heads"])
    ap.add_argument("--methods", nargs="+", default=["average", "single", "complete"])
    ap.add_argument("--min-sizes", nargs="+", type=int, default=MIN_SIZES)
    ap.add_argument("--dump-dir", type=Path, default=None,
                    help="per run: unit boxes, GT groups and labels of box only / box then z @0.16 / heads @0.2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    pred = rp.TreeStructurePredictor(args.structure_ckpt, device, tree_root=TREE)
    with pred.pkgs:
        from scripts.train_tree_adapter import topo_ok
        from window_ast.structure import ir_to_structure_tokens

    labels = defaultdict(list)
    for line in (args.labels / "labels.jsonl").open():
        r = json.loads(line)
        if r.get("status") == "done":
            r["gt"] = json.loads((args.labels / r["id"] / "tokens.json").read_text())["tokens"]
            labels[r["id"].split("_")[0]].append(r)

    data = {}
    for run in args.runs:
        for fac, rows in sorted(labels.items()):
            if fac in TRAIN_FACADES:
                continue
            dsl = json.loads((run / f"e2e_sam3_cluster_cmp_{fac}" / "facade_dsl.json").read_text())
            units = [u for u in dsl["instances"] if u.get("kind", "window") == "window"]
            boxes = [u["box_xyxy"] for u in units]
            groups, _ = rp.cluster_box_geometry_pipeline(
                boxes, tuple(dsl["meta"]["image_size"]), rp.base, method="gmm_diag", k_max=6, seed=42,
                conservative_k=True, conservative_k_wh_tol=0.008, merge_near=False, snap_wh=True)
            photo = Image.open(ROOT / dsl["meta"]["image"]).convert("RGB")
            feats = describe_all(pred, [photo.crop(tuple(int(v) for v in b)) for b in boxes])
            with pred.pkgs:
                keys = [rp.structure_fingerprint(u["structure_ir_member"]) for u in units]
                toks = [ir_to_structure_tokens(u["structure_ir_member"]) for u in units]
            matched = []
            for r in rows:
                j = max(range(len(units)), key=lambda k: iou(boxes[k], r["e2e_box_xyxy"]))
                if iou(boxes[j], r["e2e_box_xyxy"]) >= 0.5:
                    matched.append((j, r["gt"]))
            data[(run.name, fac)] = {"feats": feats, "groups": np.asarray(groups), "keys": keys, "toks": toks,
                                     "matched": matched, "n_lab": len(rows), "boxes": boxes,
                                     "image": str(ROOT / dsl["meta"]["image"])}
        print(f"loaded {run.name}")

    def score(lab_of) -> dict[str, float]:
        si = sn = do = dn = ok = n = 0
        ncl = []
        for key, v in data.items():
            lab = lab_of(v)
            ncl.append(len(set(lab.tolist())))
            winner = {}
            for c in set(lab.tolist()):
                ks = Counter(v["keys"][i] for i in np.where(lab == c)[0])
                best = sorted(ks.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                winner[c] = v["toks"][v["keys"].index(best)]
            for j, gt in v["matched"]:
                ok += topo_ok(winner[lab[j]], gt)
            n += v["n_lab"]
            for (a, ga), (b, gb) in combinations(v["matched"], 2):
                if v["groups"][a] != v["groups"][b]:
                    continue
                if topo_ok(ga, gb):
                    sn += 1
                    si += lab[a] == lab[b]
                else:
                    dn += 1
                    do += lab[a] != lab[b]
        return {"same": si / max(1, sn), "diff": do / max(1, dn), "voted": ok / max(1, n), "clusters": float(np.mean(ncl)),
                "n_same": sn, "n_diff": dn}

    rows_out = []
    base = score(lambda v: v["groups"])
    print(f"\nruns: {', '.join(r.name for r in args.runs)}  (13 test facades each, before symmetry repair)")
    print(f"{'setting':34s} {'same->1':>8s} {'diff->split':>12s} {'voted':>6s} {'clusters':>9s}   (within box groups)")
    print(f"{'box only':34s} {base['same']:8.0%} {base['diff']:12.0%} {base['voted']:6.0%} {base['clusters']:9.1f}"
          f"   pairs: {base['n_same']} same, {base['n_diff']} different")
    rows_out.append({"setting": "box only", **base})
    for desc, method, ms, cut in product(args.descs, args.methods, args.min_sizes, CUTS):
        s = score(lambda v: split(v["feats"][desc], v["groups"], cut, method, ms))
        name = f"{desc} {method} cut {cut} min {ms}"
        rows_out.append({"setting": name, "desc": desc, "method": method, "cut": cut, "min_size": ms, **s})
    for desc in args.descs:
        for method in args.methods:
            print(f"-- {desc}, {method} linkage")
            for ms in args.min_sizes:
                for r in rows_out:
                    if r.get("desc") == desc and r.get("method") == method and r.get("min_size") == ms:
                        print(f"{r['setting']:34s} {r['same']:8.0%} {r['diff']:12.0%} {r['voted']:6.0%} "
                              f"{r['clusters']:9.1f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows_out, indent=1))

    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        for run in args.runs:
            dump = {}
            for (rn, fac), v in data.items():
                if rn != run.name:
                    continue
                gt_groups: list[list[str]] = []
                gt = {}
                for j, toks in v["matched"]:
                    k = next((i for i, g in enumerate(gt_groups) if topo_ok(g, toks)), None)
                    if k is None:
                        gt_groups.append(toks)
                        k = len(gt_groups) - 1
                    gt[str(j)] = k
                dump[fac] = {"image": v["image"], "boxes": v["boxes"], "gt": gt, "labels": {
                    "box only": v["groups"].tolist(),
                    "box then z @0.16": split(v["feats"]["z"], v["groups"], 0.16, "average", 1).tolist(),
                    "box then heads @0.2": split(v["feats"]["heads"], v["groups"], 0.2, "average", 1).tolist(),
                }}
            (args.dump_dir / f"{run.name}.json").write_text(json.dumps(dump))


if __name__ == "__main__":
    main()
