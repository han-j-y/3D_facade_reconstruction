#!/usr/bin/env python3
"""Layout-first facade clustering: floors/bays first, then merge when similar.

Pipeline per facade:
  1. Match asset crops → boxes on ``train_up``
  2. Full-facade DINO patch ROI embeddings
  3. Split windows into floors (y) and bays (x) via 1D gap clustering
  4. Within each bay, merge windows whose patch features are similar
  5. Across bays, merge cluster centroids when still similar (same-floor preferred)
  6. Overlay results; also write floor/bay debug overlays

Compares against flat ``patch_roi`` spectral clustering.

Example:
  python scripts/overlay_facade_layout_first.py --max-facades 20 --device cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

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


base = _load(ROOT / "scripts" / "overlay_facade_asset_clusters.py", "overlay_base")
patch = _load(ROOT / "scripts" / "overlay_facade_patch_layout.py", "overlay_patch")


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
        default=ROOT / "runs" / "facade_asset_overlays_layout_first",
    )
    ap.add_argument("--dino", default="dinov2_vits14")
    ap.add_argument("--facade-max-side", type=int, default=896)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-windows", type=int, default=4)
    ap.add_argument("--max-facades", type=int, default=20)
    ap.add_argument("--facade-ids", type=str, default=None)
    ap.add_argument(
        "--random-sample",
        action="store_true",
        help="sample facades uniformly at random (instead of largest-first)",
    )
    ap.add_argument("--match-thr", type=float, default=0.55)
    ap.add_argument("--col-tol", type=float, default=0.045, help="bay gap in normalized x")
    ap.add_argument("--row-tol", type=float, default=0.055, help="floor gap in normalized y")
    ap.add_argument(
        "--within-bay-sim",
        type=float,
        default=0.62,
        help="cosine sim to merge windows inside the same bay",
    )
    ap.add_argument(
        "--merge-sim",
        type=float,
        default=0.78,
        help="cosine sim of centroids to merge across bays",
    )
    ap.add_argument(
        "--same-floor-bonus",
        type=float,
        default=0.06,
        help="extra sim credit when merging clusters that share a floor",
    )
    ap.add_argument("--k-max", type=int, default=8, help="for flat patch_roi baseline")
    ap.add_argument(
        "--variants",
        type=str,
        default="patch_roi,layout_grid,layout_first",
        help="layout_grid=color by (floor,bay) cell only; layout_first=merge",
    )
    return ap.parse_args()


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

    def labels(self) -> np.ndarray:
        roots = [self.find(i) for i in range(len(self.p))]
        remap = {r: i for i, r in enumerate(sorted(set(roots)))}
        return np.array([remap[r] for r in roots], dtype=np.int32)


def assign_bays(cx: np.ndarray, tol: float) -> np.ndarray:
    """Greedy 1D clustering of normalized centers (floors if cy, bays if cx)."""
    order = np.argsort(cx)
    bay = np.full(len(cx), -1, dtype=np.int32)
    cur = -1
    bay_cx = None
    bay_n = 0
    for i in order:
        xi = float(cx[i])
        if bay_cx is None or abs(xi - bay_cx) > tol:
            cur += 1
            bay_cx = xi
            bay_n = 1
        else:
            bay_n += 1
            bay_cx += (xi - bay_cx) / bay_n
        bay[i] = cur
    return bay


def cosine_sim_matrix(x: np.ndarray) -> np.ndarray:
    x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)
    return (x @ x.T).astype(np.float64)


def layout_first_labels(
    feats: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    *,
    col_tol: float,
    row_tol: float,
    within_bay_sim: float,
    merge_sim: float,
    same_floor_bonus: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split into floors/bays, merge within bay, then merge similar centroids."""
    n = len(feats)
    bay = assign_bays(cx, col_tol)
    floor = assign_bays(cy, row_tol)
    sim = cosine_sim_matrix(feats)
    uf = UnionFind(n)

    # 1) within-bay merges (strong layout prior: stacked windows often same type)
    for b in sorted(set(int(v) for v in bay.tolist())):
        idxs = [i for i in range(n) if int(bay[i]) == b]
        for a in range(len(idxs)):
            for c in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[c]
                if sim[i, j] >= within_bay_sim:
                    uf.union(i, j)

    # 2) across-bay: greedily merge cluster pairs by centroid similarity
    def cluster_members(labels: np.ndarray) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i, lab in enumerate(labels.tolist()):
            out.setdefault(int(lab), []).append(i)
        return out

    def centroid(idxs: list[int]) -> np.ndarray:
        v = feats[idxs].mean(axis=0)
        return v / max(1e-8, float(np.linalg.norm(v)))

    labels = uf.labels()
    while True:
        members = cluster_members(labels)
        cids = sorted(members)
        if len(cids) <= 1:
            break
        cents = {c: centroid(members[c]) for c in cids}
        floors_of = {
            c: set(int(floor[i]) for i in members[c]) for c in cids
        }
        best = None  # (score, c1, c2)
        for a in range(len(cids)):
            for b in range(a + 1, len(cids)):
                c1, c2 = cids[a], cids[b]
                s = float(cents[c1] @ cents[c2])
                if floors_of[c1] & floors_of[c2]:
                    s += same_floor_bonus
                thr = merge_sim
                if s >= thr and (best is None or s > best[0]):
                    best = (s, c1, c2)
        if best is None:
            break
        _s, c1, c2 = best
        # union all members of c2 into c1's root
        root = members[c1][0]
        for i in members[c2]:
            uf.union(root, i)
        labels = uf.labels()

    return labels, bay, floor


def cell_labels(bay: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """One cluster id per (floor, bay) cell — pure layout split, no appearance."""
    keys = [(int(floor[i]), int(bay[i])) for i in range(len(bay))]
    remap = {k: i for i, k in enumerate(sorted(set(keys)))}
    return np.array([remap[k] for k in keys], dtype=np.int32)


def draw_layout_debug(
    facade: Image.Image,
    items: list[dict[str, Any]],
    bay: np.ndarray,
    floor: np.ndarray,
) -> Image.Image:
    """Left: color by bay; right: color by floor."""
    iw, ih = facade.size
    left = base.draw_overlay(
        facade,
        items,
        bay,
        title="bays (vertical stacks)",
    )
    right = base.draw_overlay(
        facade,
        items,
        floor,
        title="floors (horizontal bands)",
    )
    # add center ticks
    for im, vals, name in ((left, bay, "bay"), (right, floor, "floor")):
        pass
    h = max(left.height, right.height)
    out = Image.new("RGB", (left.width + 8 + right.width, h), (12, 12, 12))
    out.paste(left, (0, 0))
    out.paste(right, (left.width + 8, 0))
    # small legend text
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(out)
    d.text(
        (12, h - 22),
        f"n_bays={len(set(bay.tolist()))}  n_floors={len(set(floor.tolist()))}  "
        f"size={iw}x{ih}",
        fill=(220, 220, 220),
        font=font,
    )
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    if args.facade_ids:
        facade_ids = [s.strip() for s in args.facade_ids.split(",") if s.strip()]
    else:
        counts = []
        for p in args.facade_dir.glob("*.png"):
            fid = p.stem
            n = len(list(args.assets_dir.glob(f"{fid}_*.png")))
            if n >= args.min_windows:
                counts.append((n, fid))
        if args.random_sample:
            import random

            rng = random.Random(args.seed)
            rng.shuffle(counts)
            facade_ids = [fid for _n, fid in counts[: args.max_facades]]
        else:
            counts.sort(reverse=True)
            facade_ids = [fid for _n, fid in counts[: args.max_facades]]
    print(f"facades={len(facade_ids)} ids={facade_ids} variants={variants}")

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
        mean_match = float(np.mean([it["match_score"] for it in kept]))

        spatial, meta = patch.facade_patch_spatial(
            model, facade_im, device=device, max_side=args.facade_max_side
        )
        patch_feats = patch.roi_pool_patches(spatial, boxes, meta)
        feats_pca = base.apply_pca(
            patch_feats, min(args.pca_dim, max(2, len(kept) - 1)), args.seed
        )

        bay = assign_bays(cx, args.col_tol)
        floor = assign_bays(cy, args.row_tol)
        lf_labels, bay2, floor2 = layout_first_labels(
            feats_pca,
            cx,
            cy,
            col_tol=args.col_tol,
            row_tol=args.row_tol,
            within_bay_sim=args.within_bay_sim,
            merge_sim=args.merge_sim,
            same_floor_bonus=args.same_floor_bonus,
        )
        bay, floor = bay2, floor2

        print(
            f"[{fi+1}/{len(facade_ids)}] facade {fid}: n={len(kept)} "
            f"bays={len(set(bay.tolist()))} floors={len(set(floor.tolist()))} "
            f"match={mean_match:.3f}"
        )

        # debug split visualization (once per facade)
        dbg_dir = out_dir / "layout_split" / f"facade_{fid}"
        dbg_dir.mkdir(parents=True, exist_ok=True)
        draw_layout_debug(facade_im, kept, bay, floor).save(dbg_dir / "floors_bays.png")
        (dbg_dir / "layout.json").write_text(
            json.dumps(
                {
                    "facade_id": fid,
                    "windows": [
                        {
                            "id": kept[i]["id"],
                            "box_xyxy": kept[i]["box_xyxy"],
                            "bay": int(bay[i]),
                            "floor": int(floor[i]),
                            "cx": float(cx[i]),
                            "cy": float(cy[i]),
                        }
                        for i in range(len(kept))
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        label_sets: dict[str, np.ndarray] = {}
        if "patch_roi" in variants:
            k = base.select_k(feats_pca, "spectral_rbf", args.k_max, args.seed)
            label_sets["patch_roi"] = base.cluster_features(
                feats_pca, "spectral_rbf", k, args.seed
            )
        if "layout_grid" in variants:
            label_sets["layout_grid"] = cell_labels(bay, floor)
        if "layout_first" in variants:
            label_sets["layout_first"] = lf_labels

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
                {
                    "facade_id": fid,
                    "n_windows": len(kept),
                    "k": n_clust,
                    "n_bays": len(set(bay.tolist())),
                    "n_floors": len(set(floor.tolist())),
                }
            )
            print(f"    {variant}: k={n_clust}")

    done_ids: list[str] = []
    for summary in all_summary.values():
        if summary:
            done_ids = [s["facade_id"] for s in summary]
            break
    for variant, summary in all_summary.items():
        mdir = out_dir / variant
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        base.build_method_overview(out_dir, variant, [s["facade_id"] for s in summary])
        print(f"overview → {mdir / 'overview_overlays.png'}")

    patch.build_compare_strip(out_dir, done_ids or facade_ids, variants)
    (out_dir / "summary_all.json").write_text(
        json.dumps(
            {
                "within_bay_sim": args.within_bay_sim,
                "merge_sim": args.merge_sim,
                "col_tol": args.col_tol,
                "row_tol": args.row_tol,
                "results": all_summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
