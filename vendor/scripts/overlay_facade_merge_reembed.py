#!/usr/bin/env python3
"""Merge window boxes, recompute DINO on combined boxes, then neighbor prior.

Pipeline:
  1. Union multi-pane / bay-face boxes
  2. Re-embed each union with full-facade DINO patch ROI
  3. Cluster (spectral_rbf)
  4. Neighbor consistency (Potts ICM): if nearby same-bay/floor windows
     agree, outliers flip toward them — ``nearby same ⇒ I should be same``

Example:
  python scripts/overlay_facade_merge_reembed.py \\
    --in-dir runs/facade_cluster_221 --device cuda
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


base = _load(ROOT / "scripts" / "overlay_facade_asset_clusters.py", "ov_base")
patch = _load(ROOT / "scripts" / "overlay_facade_patch_layout.py", "ov_patch")
merge = _load(ROOT / "scripts" / "overlay_facade_merge_boxes.py", "ov_merge")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-dir",
        type=Path,
        default=ROOT / "runs" / "facade_cluster_221",
        help="dir with index.json from cluster_facade_windows.py",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: <in-dir>/merge_reembed",
    )
    ap.add_argument("--dino", default="dinov2_vits14")
    ap.add_argument("--facade-max-side", type=int, default=896)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--col-tol", type=float, default=0.04)
    ap.add_argument("--row-tol", type=float, default=0.055)
    ap.add_argument("--fixed-k", type=int, default=None)
    ap.add_argument(
        "--spatial-strength",
        type=float,
        default=1.8,
        help="neighbor Potts strength (0=off). 'nearby same → I should be same'",
    )
    ap.add_argument("--spatial-iters", type=int, default=12)
    ap.add_argument(
        "--unary-weight",
        type=float,
        default=0.9,
        help="how strongly to stick to initial DINO cluster vs neighbors",
    )
    return ap.parse_args()


def spatial_affinity_units(
    boxes: list[list[int]],
    bay: list[int],
    floor: np.ndarray,
    *,
    iw: int,
    ih: int,
    col_tol: float,
    row_tol: float,
) -> np.ndarray:
    """Neighbor weights: same bay ≫ same floor ≫ nearby diagonal."""
    n = len(boxes)
    cx = np.array([0.5 * (b[0] + b[2]) / iw for b in boxes], dtype=np.float64)
    cy = np.array([0.5 * (b[1] + b[3]) / ih for b in boxes], dtype=np.float64)
    W = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(float(cx[i]) - float(cx[j]))
            dy = abs(float(cy[i]) - float(cy[j]))
            if int(bay[i]) == int(bay[j]):
                # vertical stack in one bay — strongest "I should match neighbors"
                w = 1.6 * float(np.exp(-dy / 0.40))
            elif int(floor[i]) == int(floor[j]) or dy <= row_tol:
                w = 0.45 * float(np.exp(-dx / 0.28))
            else:
                dist = float(np.hypot(dx, dy))
                w = 0.10 * float(np.exp(-dist / 0.22))
            if w >= 0.05:
                W[i, j] = W[j, i] = w
    return W


def neighbor_consistency_labels(
    labels0: np.ndarray,
    W: np.ndarray,
    feats: np.ndarray,
    *,
    strength: float,
    iters: int,
    unary_weight: float,
) -> np.ndarray:
    """ICM Potts: prefer neighbors' labels; unary sticks to DINO init.

    Inductive bias: if nearby windows agree, flip outliers toward them —
    especially when feature similarity to that neighbor is also high.
    """
    if strength <= 0 or len(labels0) == 0:
        return labels0.copy()
    labs = labels0.copy()
    classes = sorted(set(int(x) for x in labs.tolist()))
    # feature sim softens pairwise: more pull from similar neighbors
    fn = feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8, None)
    sim = np.clip(fn @ fn.T, 0.0, 1.0)
    pair = float(strength) * 1.25
    for _ in range(iters):
        changed = False
        for i in range(len(labs)):
            best_c = int(labs[i])
            best_cost = None
            for c in classes:
                cost = unary_weight * (0.0 if c == int(labels0[i]) else 1.0)
                for j in range(len(labs)):
                    if W[i, j] <= 0:
                        continue
                    # disagree with neighbor j
                    if int(labs[j]) != c:
                        # similar-looking neighbors count more
                        cost += pair * W[i, j] * (0.5 + 0.5 * float(sim[i, j]))
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_c = c
            if best_c != int(labs[i]):
                labs[i] = best_c
                changed = True
        if not changed:
            break
    uniq = sorted(set(int(x) for x in labs.tolist()))
    remap = {u: i for i, u in enumerate(uniq)}
    return np.array([remap[int(x)] for x in labs], dtype=np.int32)


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir / "merge_reembed"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    index = json.loads((in_dir / "index.json").read_text())
    facade = Image.open(index["facade_path"]).convert("RGB")
    iw, ih = facade.size
    boxes = [[int(v) for v in w["box_xyxy"]] for w in index["windows"]]
    cx = np.array([0.5 * (b[0] + b[2]) / iw for b in boxes], dtype=np.float64)
    cy = np.array([0.5 * (b[1] + b[3]) / ih for b in boxes], dtype=np.float64)
    n_raw = len(boxes)
    print(f"facade {index['facade_id']}: raw boxes={n_raw}")

    # --- 1) merge boxes ---
    merged_boxes, members, bay, floor = merge.merge_adjacent_boxes(
        boxes,
        cx,
        cy,
        row_tol=args.row_tol,
        adj_gap=1.0,
        merge_bays=False,
        col_tol=args.col_tol,
    )
    n_m = len(merged_boxes)
    print(f"merged boxes: {n_raw} → {n_m}")

    bay_u = []
    for mem in members:
        labs = [int(bay[i]) for i in mem]
        bay_u.append(max(set(labs), key=labs.count))

    # --- 2) DINO on raw panes (baseline) + on merged boxes (recomputed) ---
    print(f"loading {args.dino}…")
    model = torch.hub.load("facebookresearch/dinov2", args.dino, pretrained=True)
    model = model.to(device).eval()

    spatial, meta = patch.facade_patch_spatial(
        model, facade, device=device, max_side=args.facade_max_side
    )

    feats_raw = patch.roi_pool_patches(spatial, boxes, meta)
    feats_raw_pca = base.apply_pca(
        feats_raw, min(args.pca_dim, max(2, n_raw - 1)), args.seed
    )

    print("recomputing DINO patch-ROI on combined boxes…")
    feats_m = patch.roi_pool_patches(spatial, merged_boxes, meta)
    feats_m_pca = base.apply_pca(
        feats_m, min(args.pca_dim, max(2, n_m - 1)), args.seed
    )

    # also: mean of member raw feats (what people sometimes do instead of re-embed)
    feats_pooled = []
    for mem in members:
        v = feats_raw[mem].mean(axis=0)
        v = v / max(1e-8, float(np.linalg.norm(v)))
        feats_pooled.append(v)
    feats_pooled = np.stack(feats_pooled, 0).astype(np.float32)
    feats_pooled_pca = base.apply_pca(
        feats_pooled, min(args.pca_dim, max(2, n_m - 1)), args.seed
    )

    def cluster(feats: np.ndarray) -> np.ndarray:
        n = len(feats)
        if n < 2:
            return np.zeros(n, dtype=np.int32)
        if args.fixed_k is not None:
            k = min(args.fixed_k, n - 1) if n > 2 else 2
        else:
            k = base.select_k(feats, "spectral_rbf", min(args.k_max, max(2, n - 1)), args.seed)
        k = max(2, min(k, n - 1)) if n > 2 else 1
        if n == 2:
            return base.cluster_features(feats, "spectral_rbf", 2, args.seed)
        return base.cluster_features(feats, "spectral_rbf", k, args.seed)

    labels_raw = cluster(feats_raw_pca)
    labels_pooled = cluster(feats_pooled_pca)
    labels_reembed = cluster(feats_m_pca)

    # floor ids on merged units
    mcy = np.array(
        [0.5 * (b[1] + b[3]) / ih for b in merged_boxes], dtype=np.float64
    )
    floor_m = merge.lay.assign_bays(mcy, args.row_tol)

    W = spatial_affinity_units(
        merged_boxes,
        bay_u,
        floor_m,
        iw=iw,
        ih=ih,
        col_tol=args.col_tol,
        row_tol=args.row_tol,
    )
    labels_neighbor = neighbor_consistency_labels(
        labels_reembed,
        W,
        feats_m_pca,
        strength=args.spatial_strength,
        iters=args.spatial_iters,
        unary_weight=args.unary_weight,
    )
    print(
        f"neighbor prior: k {len(set(labels_reembed.tolist()))} → "
        f"{len(set(labels_neighbor.tolist()))}  "
        f"(spatial_strength={args.spatial_strength})"
    )

    # bay-colored union boxes (geometry only)
    uniq = sorted(set(bay_u))
    remap = {u: i for i, u in enumerate(uniq)}
    labels_bay = np.array([remap[b] for b in bay_u], dtype=np.int32)

    items_raw = [{"box_xyxy": b} for b in boxes]
    items_m = [{"box_xyxy": b} for b in merged_boxes]
    crops_m = [facade.crop(tuple(b)) for b in merged_boxes]

    variants = [
        ("raw_boxes_raw_feats", items_raw, labels_raw, boxes),
        ("merged_reembed_feats", items_m, labels_reembed, merged_boxes),
        ("merged_reembed_neighbor", items_m, labels_neighbor, merged_boxes),
        ("merged_bay_color", items_m, labels_bay, merged_boxes),
    ]

    thumbs = []
    for name, items, labels, _boxlist in variants:
        k = len(set(int(c) for c in labels.tolist()))
        title = f"{index['facade_id']}  {name}  n={len(items)}  k={k}"
        ov = base.draw_overlay(facade, items, labels, title=title)
        ov.save(out_dir / f"overlay_{name}.png")
        if name == "merged_reembed_neighbor":
            base.make_side_by_side(ov, crops_m, labels).save(
                out_dir / "overlay_reembed_neighbor_with_crops.png"
            )
        im = ov.copy()
        im.thumbnail((480, 640), Image.Resampling.LANCZOS)
        thumbs.append((name, im, k))
        print(f"  {name}: k={k}")

    # compare strip
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13
        )
        font_b = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except OSError:
        font = font_b = ImageFont.load_default()
    cell_w = max(t.width for _, t, _ in thumbs)
    cell_h = max(t.height for _, t, _ in thumbs) + 36
    grid = Image.new("RGB", (len(thumbs) * cell_w + 8, cell_h + 8), (14, 14, 14))
    draw = ImageDraw.Draw(grid)
    for i, (name, im, k) in enumerate(thumbs):
        x = 4 + i * cell_w
        draw.text((x + 4, 4), name, fill=(240, 220, 120), font=font_b)
        draw.text((x + 4, 20), f"k={k}", fill=(200, 200, 200), font=font)
        grid.paste(im, (x, 36))
    grid.save(out_dir / "compare_reembed.png")

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "facade_id": index["facade_id"],
                "n_raw": n_raw,
                "n_merged": n_m,
                "dino": args.dino,
                "feature": "patch_roi_on_combined_box",
                "spatial_strength": args.spatial_strength,
                "unary_weight": args.unary_weight,
                "k_reembed": len(set(int(c) for c in labels_reembed.tolist())),
                "k_neighbor": len(set(int(c) for c in labels_neighbor.tolist())),
                "units": [
                    {
                        "box_xyxy": merged_boxes[i],
                        "members": [index["windows"][j]["idx"] for j in members[i]],
                        "bay": int(bay_u[i]),
                        "floor": int(floor_m[i]),
                        "cluster_reembed": int(labels_reembed[i]),
                        "cluster_neighbor": int(labels_neighbor[i]),
                    }
                    for i in range(n_m)
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    np.save(out_dir / "feats_merged_reembed.npy", feats_m)
    np.save(out_dir / "feats_merged_reembed_pca.npy", feats_m_pca)
    print(f"compare → {out_dir / 'compare_reembed.png'}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
