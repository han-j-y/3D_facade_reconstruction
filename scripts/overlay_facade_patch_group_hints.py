#!/usr/bin/env python3
"""Patch-ROI grouping hints: same-bay + adjacent panes, then merge similar groups.

Uses full-facade DINO patch ROI only (no structure embed yet). Goal: combine
windows that layout says belong together (vertical stacks, multi-pane units)
when appearance still roughly agrees — e.g. facade 221 left stack.

Variants
  - patch_roi          : flat spectral_rbf (baseline)
  - patch_roi_hints    : adj same-floor panes + soft same-bay + centroid merge
  - patch_roi_force_bay: force each bay to one label, then merge bays by centroid

Example:
  python scripts/overlay_facade_patch_group_hints.py \\
    --random-sample --max-facades 20 --device cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


base = _load(ROOT / "scripts" / "overlay_facade_asset_clusters.py", "ov_base")
patch = _load(ROOT / "scripts" / "overlay_facade_patch_layout.py", "ov_patch")
lay = _load(ROOT / "scripts" / "overlay_facade_layout_first.py", "ov_lay")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--assets-dir",
        type=Path,
        default=EXP / "data" / "facades" / "assets" / "windows",
    )
    ap.add_argument(
        "--facade-dir",
        type=Path,
        default=EXP / "data" / "facades" / "train_up",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "runs" / "facade_asset_overlays_patch_hints",
    )
    ap.add_argument("--dino", default="dinov2_vits14")
    ap.add_argument("--facade-max-side", type=int, default=896)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-windows", type=int, default=4)
    ap.add_argument("--max-facades", type=int, default=20)
    ap.add_argument("--facade-ids", type=str, default=None)
    ap.add_argument("--random-sample", action="store_true")
    ap.add_argument("--match-thr", type=float, default=0.55)
    ap.add_argument("--col-tol", type=float, default=0.045)
    ap.add_argument("--row-tol", type=float, default=0.055)
    ap.add_argument(
        "--within-bay-sim",
        type=float,
        default=0.45,
        help="soft same-bay merge threshold (lower = more combining)",
    )
    ap.add_argument(
        "--adj-gap",
        type=float,
        default=0.35,
        help="max horizontal gap as fraction of median window width for same-floor pane merge",
    )
    ap.add_argument(
        "--adj-sim",
        type=float,
        default=0.35,
        help="min patch sim to merge adjacent same-floor panes (low: geometry-led)",
    )
    ap.add_argument("--merge-sim", type=float, default=0.70)
    ap.add_argument("--same-floor-bonus", type=float, default=0.05)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument(
        "--variants",
        type=str,
        default="patch_roi,patch_roi_hints,patch_roi_force_bay",
    )
    return ap.parse_args()


def box_width(b: list[int]) -> float:
    return max(1.0, float(b[2] - b[0]))


def horizontally_adjacent(
    boxes: list[list[int]],
    i: int,
    j: int,
    *,
    gap_frac: float,
    med_w: float,
) -> bool:
    a, b = boxes[i], boxes[j]
    # vertical overlap
    y0 = max(a[1], b[1])
    y1 = min(a[3], b[3])
    if y1 - y0 <= 0.25 * min(a[3] - a[1], b[3] - b[1]):
        return False
    if a[2] <= b[0]:
        gap = b[0] - a[2]
    elif b[2] <= a[0]:
        gap = a[0] - b[2]
    else:
        gap = 0.0  # overlap in x
    return gap <= gap_frac * med_w


def cluster_members(labels: np.ndarray) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for i, lab in enumerate(labels.tolist()):
        out.setdefault(int(lab), []).append(i)
    return out


def centroid(feats: np.ndarray, idxs: list[int]) -> np.ndarray:
    v = feats[idxs].mean(axis=0)
    return v / max(1e-8, float(np.linalg.norm(v)))


def greedy_centroid_merge(
    uf: lay.UnionFind,
    feats: np.ndarray,
    floor: np.ndarray,
    *,
    merge_sim: float,
    same_floor_bonus: float,
) -> np.ndarray:
    labels = uf.labels()
    while True:
        members = cluster_members(labels)
        cids = sorted(members)
        if len(cids) <= 1:
            break
        cents = {c: centroid(feats, members[c]) for c in cids}
        floors_of = {c: {int(floor[i]) for i in members[c]} for c in cids}
        best = None
        for a in range(len(cids)):
            for b in range(a + 1, len(cids)):
                c1, c2 = cids[a], cids[b]
                s = float(cents[c1] @ cents[c2])
                if floors_of[c1] & floors_of[c2]:
                    s += same_floor_bonus
                if s >= merge_sim and (best is None or s > best[0]):
                    best = (s, c1, c2)
        if best is None:
            break
        _s, c1, c2 = best
        root = members[c1][0]
        for i in members[c2]:
            uf.union(root, i)
        labels = uf.labels()
    return labels


def patch_roi_hints_labels(
    feats: np.ndarray,
    boxes: list[list[int]],
    cx: np.ndarray,
    cy: np.ndarray,
    *,
    col_tol: float,
    row_tol: float,
    within_bay_sim: float,
    adj_gap: float,
    adj_sim: float,
    merge_sim: float,
    same_floor_bonus: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grouping hints from patch ROI + layout (no force-bay)."""
    n = len(feats)
    bay = lay.assign_bays(cx, col_tol)
    floor = lay.assign_bays(cy, row_tol)
    sim = lay.cosine_sim_matrix(feats)
    uf = lay.UnionFind(n)
    med_w = float(np.median([box_width(b) for b in boxes]))

    # A) same-floor adjacent panes → multi-pane unit (geometry-led, soft appearance check)
    for i in range(n):
        for j in range(i + 1, n):
            if int(floor[i]) != int(floor[j]):
                continue
            if not horizontally_adjacent(boxes, i, j, gap_frac=adj_gap, med_w=med_w):
                continue
            if sim[i, j] >= adj_sim:
                uf.union(i, j)

    # B) same bay vertical stack
    for b in sorted(set(int(v) for v in bay.tolist())):
        idxs = [i for i in range(n) if int(bay[i]) == b]
        for a in range(len(idxs)):
            for c in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[c]
                if sim[i, j] >= within_bay_sim:
                    uf.union(i, j)

    labels = greedy_centroid_merge(
        uf,
        feats,
        floor,
        merge_sim=merge_sim,
        same_floor_bonus=same_floor_bonus,
    )
    return labels, bay, floor


def patch_roi_force_bay_labels(
    feats: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    *,
    col_tol: float,
    row_tol: float,
    merge_sim: float,
    same_floor_bonus: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hard: every window in a bay shares a label; then merge bay-groups by centroid."""
    n = len(feats)
    bay = lay.assign_bays(cx, col_tol)
    floor = lay.assign_bays(cy, row_tol)
    uf = lay.UnionFind(n)
    for b in sorted(set(int(v) for v in bay.tolist())):
        idxs = [i for i in range(n) if int(bay[i]) == b]
        for i in idxs[1:]:
            uf.union(idxs[0], i)
    labels = greedy_centroid_merge(
        uf,
        feats,
        floor,
        merge_sim=merge_sim,
        same_floor_bonus=same_floor_bonus,
    )
    return labels, bay, floor


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    # Prefer same random set as layout_first_rand if no ids given + random
    default_rand = "252,108,121,17,15,400,270,112,263,22,189,221,335,328,388,204,283,339,47,74"
    if args.facade_ids:
        facade_ids = [s.strip() for s in args.facade_ids.split(",") if s.strip()]
    elif args.random_sample:
        counts = []
        for p in args.facade_dir.glob("*.png"):
            fid = p.stem
            n = len(list(args.assets_dir.glob(f"{fid}_*.png")))
            if n >= args.min_windows:
                counts.append((n, fid))
        rng = random.Random(args.seed)
        rng.shuffle(counts)
        facade_ids = [fid for _n, fid in counts[: args.max_facades]]
    else:
        facade_ids = [s.strip() for s in default_rand.split(",") if s.strip()][
            : args.max_facades
        ]

    # Always include 221 for the motivating example
    if "221" not in facade_ids:
        facade_ids = ["221"] + facade_ids[:-1]

    print(f"facades={len(facade_ids)} ids={facade_ids}")
    print(f"loading {args.dino}…")
    model = torch.hub.load("facebookresearch/dinov2", args.dino, pretrained=True)
    model = model.to(device).eval()

    all_summary: dict[str, list[dict]] = {v: [] for v in variants}

    for fi, fid in enumerate(facade_ids):
        facade_path = args.facade_dir / f"{fid}.png"
        if not facade_path.is_file():
            continue
        items = base.load_facade_items(args.assets_dir, fid)
        if len(items) < args.min_windows:
            continue

        facade_im = Image.open(facade_path).convert("RGB")
        facade_rgb = np.array(facade_im)
        iw, ih = facade_im.size
        kept: list[dict[str, Any]] = []
        crops_pil: list[Image.Image] = []
        for it in items:
            crop = np.array(Image.open(it["image"]).convert("RGB"))
            box, score = base.find_box(facade_rgb, crop)
            if box is None or score < args.match_thr:
                continue
            rec = dict(it)
            rec["box_xyxy"] = list(box)
            rec["match_score"] = float(score)
            kept.append(rec)
            crops_pil.append(Image.fromarray(crop))
        if len(kept) < args.min_windows:
            continue

        boxes = [it["box_xyxy"] for it in kept]
        cx = np.array([0.5 * (b[0] + b[2]) / iw for b in boxes], dtype=np.float64)
        cy = np.array([0.5 * (b[1] + b[3]) / ih for b in boxes], dtype=np.float64)

        spatial, meta = patch.facade_patch_spatial(
            model, facade_im, device=device, max_side=args.facade_max_side
        )
        patch_feats = patch.roi_pool_patches(spatial, boxes, meta)
        feats = base.apply_pca(
            patch_feats, min(args.pca_dim, max(2, len(kept) - 1)), args.seed
        )

        print(f"[{fi+1}/{len(facade_ids)}] facade {fid}: n={len(kept)}")

        label_sets: dict[str, np.ndarray] = {}
        bay = lay.assign_bays(cx, args.col_tol)
        floor = lay.assign_bays(cy, args.row_tol)

        if "patch_roi" in variants:
            k = base.select_k(feats, "spectral_rbf", args.k_max, args.seed)
            label_sets["patch_roi"] = base.cluster_features(
                feats, "spectral_rbf", k, args.seed
            )
        if "patch_roi_hints" in variants:
            labs, bay, floor = patch_roi_hints_labels(
                feats,
                boxes,
                cx,
                cy,
                col_tol=args.col_tol,
                row_tol=args.row_tol,
                within_bay_sim=args.within_bay_sim,
                adj_gap=args.adj_gap,
                adj_sim=args.adj_sim,
                merge_sim=args.merge_sim,
                same_floor_bonus=args.same_floor_bonus,
            )
            label_sets["patch_roi_hints"] = labs
        if "patch_roi_force_bay" in variants:
            labs, bay, floor = patch_roi_force_bay_labels(
                feats,
                cx,
                cy,
                col_tol=args.col_tol,
                row_tol=args.row_tol,
                merge_sim=args.merge_sim,
                same_floor_bonus=args.same_floor_bonus,
            )
            label_sets["patch_roi_force_bay"] = labs

        for variant, labels in label_sets.items():
            n_clust = len(set(int(c) for c in labels.tolist()))
            title = (
                f"facade {fid}  {variant}  k={n_clust}  "
                f"bays={len(set(bay.tolist()))} floors={len(set(floor.tolist()))}"
            )
            overlay = base.draw_overlay(facade_im, kept, labels, title=title)
            combo = base.make_side_by_side(overlay, crops_pil, labels)
            facade_out = out_dir / variant / f"facade_{fid}"
            facade_out.mkdir(parents=True, exist_ok=True)
            overlay.save(facade_out / "overlay.png")
            combo.save(facade_out / "overlay_with_crops.png")
            (facade_out / "clusters.json").write_text(
                json.dumps(
                    {
                        "facade_id": fid,
                        "variant": variant,
                        "k": n_clust,
                        "within_bay_sim": args.within_bay_sim,
                        "merge_sim": args.merge_sim,
                        "adj_gap": args.adj_gap,
                        "adj_sim": args.adj_sim,
                        "windows": [
                            {
                                "id": kept[i]["id"],
                                "box_xyxy": kept[i]["box_xyxy"],
                                "cluster": int(labels[i]),
                                "bay": int(bay[i]),
                                "floor": int(floor[i]),
                            }
                            for i in range(len(kept))
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            all_summary[variant].append(
                {"facade_id": fid, "n_windows": len(kept), "k": n_clust}
            )
            print(f"    {variant}: k={n_clust}")

    done_ids = [s["facade_id"] for s in next(iter(all_summary.values()), [])]
    for variant, summary in all_summary.items():
        mdir = out_dir / variant
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        base.build_method_overview(out_dir, variant, [s["facade_id"] for s in summary])
        print(f"overview → {mdir / 'overview_overlays.png'}")

    patch.build_compare_strip(out_dir, done_ids or facade_ids, variants, max_facades=16)
    # spotlight 221
    if (out_dir / "patch_roi" / "facade_221" / "overlay.png").is_file():
        imgs = []
        for v in variants:
            p = out_dir / v / "facade_221" / "overlay.png"
            if p.is_file():
                im = Image.open(p).convert("RGB")
                im.thumbnail((480, 640), Image.Resampling.LANCZOS)
                imgs.append(im)
        if imgs:
            w = sum(im.width for im in imgs) + 8 * (len(imgs) - 1)
            h = max(im.height for im in imgs)
            strip = Image.new("RGB", (w, h), (12, 12, 12))
            x = 0
            for im in imgs:
                strip.paste(im, (x, 0))
                x += im.width + 8
            strip.save(out_dir / "facade_221_compare.png")
            print(f"221 spotlight → {out_dir / 'facade_221_compare.png'}")

    (out_dir / "summary_all.json").write_text(
        json.dumps({"facade_ids": facade_ids, "results": all_summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
