#!/usr/bin/env python3
"""A/B: crop-CLS vs full-facade DINO patch ROI (+ layout coords + spatial Potts).

For each facade:
  1. Match asset crops onto ``train_up/{id}.png``
  2. Embed with several variants and cluster (spectral_rbf, silhouette k)
  3. Write overlays under ``runs/facade_asset_overlays_patch_layout/{variant}/``

Variants
  - crop_cls              : per-crop DINO CLS (previous baseline)
  - patch_roi             : mean-pool patch tokens inside each box (full facade)
  - patch_roi_layout      : patch_roi + normalized (cx, cy, log-area, aspect)
  - patch_roi_layout_potts: patch_roi_layout + light bay/floor ICM Potts

Example:
  python scripts/overlay_facade_patch_layout.py --max-facades 20 --device cuda
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
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_overlay_base():
    path = ROOT / "scripts" / "overlay_facade_asset_clusters.py"
    spec = importlib.util.spec_from_file_location("overlay_facade_asset_clusters", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


base = _load_overlay_base()


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
        default=ROOT / "runs" / "facade_asset_overlays_patch_layout",
    )
    ap.add_argument("--dino", default="dinov2_vits14")
    ap.add_argument("--crop-size", type=int, default=224, help="CLS crop resize")
    ap.add_argument(
        "--facade-max-side",
        type=int,
        default=896,
        help="max side for full-facade DINO forward (multiple of 14 enforced)",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--method", default="spectral_rbf")
    ap.add_argument("--min-windows", type=int, default=4)
    ap.add_argument("--max-facades", type=int, default=20)
    ap.add_argument("--facade-ids", type=str, default=None)
    ap.add_argument("--match-thr", type=float, default=0.55)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--fixed-k", type=int, default=None)
    ap.add_argument(
        "--layout-weight",
        type=float,
        default=0.35,
        help="scale of layout dims relative to unit-norm visual feats",
    )
    ap.add_argument("--col-tol", type=float, default=0.045)
    ap.add_argument("--row-tol", type=float, default=0.055)
    ap.add_argument("--spatial-strength", type=float, default=0.55)
    ap.add_argument("--spatial-iters", type=int, default=8)
    ap.add_argument(
        "--variants",
        type=str,
        default="crop_cls,patch_roi,patch_roi_layout,patch_roi_layout_potts",
    )
    return ap.parse_args()


def round_to_patch(n: int, patch: int = 14) -> int:
    return max(patch, int(round(n / patch) * patch))


def prepare_facade_tensor(
    facade: Image.Image,
    max_side: int,
    patch: int = 14,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Aspect-preserving resize + center-pad to patch-aligned canvas."""
    w0, h0 = facade.size
    scale = min(1.0, float(max_side) / max(w0, h0))
    cw = max(patch, int(round(w0 * scale)))
    ch = max(patch, int(round(h0 * scale)))
    w1 = round_to_patch(cw, patch)
    h1 = round_to_patch(ch, patch)
    content = facade.resize((cw, ch), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w1, h1), (128, 128, 128))
    ox = (w1 - cw) // 2
    oy = (h1 - ch) // 2
    canvas.paste(content, (ox, oy))
    tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    meta = {
        "w0": float(w0),
        "h0": float(h0),
        "cw": float(cw),
        "ch": float(ch),
        "w1": float(w1),
        "h1": float(h1),
        "ox": float(ox),
        "oy": float(oy),
        "patch": float(patch),
    }
    return tf(canvas), meta


def box_to_patch_slices(
    box: list[int],
    meta: dict[str, float],
) -> tuple[slice, slice]:
    """Map original-image xyxy box → patch-grid slices on the DINO feature map."""
    x0, y0, x1, y1 = [float(v) for v in box]
    w0, h0 = meta["w0"], meta["h0"]
    sx = meta["cw"] / w0
    sy = meta["ch"] / h0
    # in padded canvas coords
    xa = meta["ox"] + x0 * sx
    xb = meta["ox"] + x1 * sx
    ya = meta["oy"] + y0 * sy
    yb = meta["oy"] + y1 * sy
    p = meta["patch"]
    hp = int(meta["h1"] / p)
    wp = int(meta["w1"] / p)
    c0 = int(np.clip(np.floor(xa / p), 0, wp - 1))
    c1 = int(np.clip(np.ceil(xb / p), c0 + 1, wp))
    r0 = int(np.clip(np.floor(ya / p), 0, hp - 1))
    r1 = int(np.clip(np.ceil(yb / p), r0 + 1, hp))
    return slice(r0, r1), slice(c0, c1)


@torch.no_grad()
def facade_patch_spatial(model, facade: Image.Image, device: torch.device, max_side: int) -> tuple[torch.Tensor, dict[str, float]]:
    """Return patch feature map [C, Hp, Wp] and resize meta."""
    x, meta = prepare_facade_tensor(facade, max_side=max_side)
    x = x.unsqueeze(0).to(device)
    feats = model.forward_features(x)
    if isinstance(feats, dict) and feats.get("x_norm_patchtokens") is not None:
        tokens = feats["x_norm_patchtokens"]  # [1, N, C]
    else:
        tokens = model.get_intermediate_layers(x, n=1)[0]
    b, n, c = tokens.shape
    hp = int(meta["h1"] / meta["patch"])
    wp = int(meta["w1"] / meta["patch"])
    if hp * wp != n:
        side = int(round(n**0.5))
        hp = wp = side
    spatial = tokens.transpose(1, 2).reshape(b, c, hp, wp)[0]
    return spatial, meta


def roi_pool_patches(
    spatial: torch.Tensor,
    boxes: list[list[int]],
    meta: dict[str, float],
) -> np.ndarray:
    """Mean-pool patch tokens in each box → [N, C] L2-normalized."""
    vecs = []
    for box in boxes:
        rs, cs = box_to_patch_slices(box, meta)
        patch = spatial[:, rs, cs]
        if patch.numel() == 0:
            # fallback: center patch
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            rs, cs = box_to_patch_slices([cx, cy, cx + 1, cy + 1], meta)
            patch = spatial[:, rs, cs]
        v = patch.mean(dim=(1, 2))
        v = F.normalize(v.float(), dim=0)
        vecs.append(v.cpu().numpy())
    return np.stack(vecs, 0).astype(np.float32)


def layout_features(boxes: list[list[int]], iw: int, ih: int) -> np.ndarray:
    """Normalized (cx, cy, log-area, aspect) per window."""
    rows = []
    for x0, y0, x1, y1 in boxes:
        w = max(1.0, float(x1 - x0))
        h = max(1.0, float(y1 - y0))
        rows.append(
            [
                0.5 * (x0 + x1) / max(1, iw),
                0.5 * (y0 + y1) / max(1, ih),
                np.log(w * h),
                w / h,
            ]
        )
    x = np.asarray(rows, dtype=np.float32)
    # z-score within facade so layout is relative
    x = (x - x.mean(0, keepdims=True)) / np.clip(x.std(0, keepdims=True), 1e-6, None)
    # unit-norm each row so layout_weight is comparable to visual
    x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)
    return x.astype(np.float32)


def concat_layout(visual: np.ndarray, layout: np.ndarray, weight: float) -> np.ndarray:
    if weight <= 0:
        return visual
    return np.concatenate([visual, weight * layout], axis=1).astype(np.float32)


def spatial_affinity(cx: np.ndarray, cy: np.ndarray, *, col_tol: float, row_tol: float) -> np.ndarray:
    n = len(cx)
    W = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(float(cx[i]) - float(cx[j]))
            dy = abs(float(cy[i]) - float(cy[j]))
            if dx <= col_tol:
                w = 1.35 * float(np.exp(-dy / 0.35))
            elif dy <= row_tol:
                w = 0.35 * float(np.exp(-dx / 0.30))
            else:
                dist = float(np.hypot(dx, dy))
                w = 0.08 * float(np.exp(-dist / 0.22))
            if w >= 0.05:
                W[i, j] = W[j, i] = w
    return W


def spatial_refine_labels(
    labels: np.ndarray,
    W: np.ndarray,
    *,
    strength: float,
    iters: int,
) -> np.ndarray:
    """ICM Potts: unary prefers initial label; pairwise prefer spatial neighbors."""
    if strength <= 0 or len(labels) == 0:
        return labels.copy()
    labs = labels.copy()
    classes = sorted(set(int(x) for x in labs.tolist()))
    pair = float(strength) * 1.25
    unary_w = 2.5
    for _ in range(iters):
        changed = False
        for i in range(len(labs)):
            best_c = int(labs[i])
            best_cost = None
            for c in classes:
                cost = unary_w * (0.0 if c == int(labels[i]) else 1.0)
                for j in range(len(labs)):
                    if W[i, j] <= 0:
                        continue
                    cost += pair * W[i, j] * (0.0 if int(labs[j]) == c else 1.0)
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


def pca_fit(x: np.ndarray, dim: int, seed: int) -> np.ndarray:
    return base.apply_pca(x, dim, seed)


def build_compare_strip(
    out_dir: Path,
    facade_ids: list[str],
    variants: list[str],
    *,
    max_facades: int = 12,
) -> None:
    """Grid: rows=facades, cols=variants."""
    fids = facade_ids[:max_facades]
    cells: list[list[Image.Image | None]] = []
    for fid in fids:
        row = []
        for v in variants:
            p = out_dir / v / f"facade_{fid}" / "overlay.png"
            if p.is_file():
                im = Image.open(p).convert("RGB")
                im.thumbnail((360, 360), Image.Resampling.LANCZOS)
                row.append(im)
            else:
                row.append(None)
        cells.append(row)
    if not any(any(c) for c in cells):
        return
    cell_w = max((im.width for row in cells for im in row if im), default=360)
    cell_h = max((im.height for row in cells for im in row if im), default=360) + 20
    header = 28
    grid = Image.new(
        "RGB",
        (len(variants) * cell_w + 8, header + len(fids) * cell_h + 8),
        (14, 14, 14),
    )
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        font_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except OSError:
        font = font_b = ImageFont.load_default()
    draw = ImageDraw.Draw(grid)
    for ci, v in enumerate(variants):
        draw.text((8 + ci * cell_w, 6), v, fill=(240, 220, 120), font=font_b)
    for ri, fid in enumerate(fids):
        y0 = header + ri * cell_h
        draw.text((8, y0), f"facade {fid}", fill=(200, 200, 200), font=font)
        for ci, im in enumerate(cells[ri]):
            if im is None:
                continue
            grid.paste(im, (4 + ci * cell_w, y0 + 18))
    grid.save(out_dir / "compare_variants.png")
    print(f"compare → {out_dir / 'compare_variants.png'}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    method = args.method

    if args.facade_ids:
        facade_ids = [s.strip() for s in args.facade_ids.split(",") if s.strip()]
    else:
        counts = []
        for p in args.facade_dir.glob("*.png"):
            fid = p.stem
            n = len(list(args.assets_dir.glob(f"{fid}_*.png")))
            if n >= args.min_windows:
                counts.append((n, fid))
        counts.sort(reverse=True)
        facade_ids = [fid for _n, fid in counts[: args.max_facades]]
    print(f"facades={len(facade_ids)} variants={variants} method={method}")

    print(f"loading {args.dino}…")
    model = torch.hub.load("facebookresearch/dinov2", args.dino, pretrained=True)
    model = model.to(device).eval()

    all_summary: dict[str, list[dict]] = {v: [] for v in variants}

    for fi, fid in enumerate(facade_ids):
        facade_path = args.facade_dir / f"{fid}.png"
        if not facade_path.is_file():
            print(f"skip {fid}: missing facade")
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
            print(f"skip {fid}: only {len(kept)} matched")
            continue

        boxes = [it["box_xyxy"] for it in kept]
        mean_match = float(np.mean([it["match_score"] for it in kept]))
        print(f"[{fi+1}/{len(facade_ids)}] facade {fid}: n={len(kept)} match={mean_match:.3f}")

        # --- visual embeddings ---
        crop_feats = base.embed_paths(
            [it["image"] for it in kept],
            model=model,
            image_size=args.crop_size,
            batch_size=args.batch_size,
            device=device,
        )
        spatial, meta = facade_patch_spatial(
            model, facade_im, device=device, max_side=args.facade_max_side
        )
        patch_feats = roi_pool_patches(spatial, boxes, meta)
        lay = layout_features(boxes, iw, ih)

        cx = np.array([0.5 * (b[0] + b[2]) / iw for b in boxes], dtype=np.float64)
        cy = np.array([0.5 * (b[1] + b[3]) / ih for b in boxes], dtype=np.float64)
        W = spatial_affinity(cx, cy, col_tol=args.col_tol, row_tol=args.row_tol)

        feat_by_variant: dict[str, np.ndarray] = {}
        if "crop_cls" in variants:
            feat_by_variant["crop_cls"] = crop_feats
        if "patch_roi" in variants:
            feat_by_variant["patch_roi"] = patch_feats
        if "patch_roi_layout" in variants or "patch_roi_layout_potts" in variants:
            combined = concat_layout(patch_feats, lay, args.layout_weight)
            if "patch_roi_layout" in variants:
                feat_by_variant["patch_roi_layout"] = combined
            if "patch_roi_layout_potts" in variants:
                feat_by_variant["patch_roi_layout_potts"] = combined

        for variant in variants:
            raw = feat_by_variant[variant]
            feats = pca_fit(raw, min(args.pca_dim, max(2, len(kept) - 1)), args.seed)
            if args.fixed_k is not None:
                k = args.fixed_k
            else:
                k = base.select_k(feats, method, args.k_max, args.seed)
            labels = base.cluster_features(feats, method, k, args.seed)
            if variant.endswith("_potts"):
                labels = spatial_refine_labels(
                    labels,
                    W,
                    strength=args.spatial_strength,
                    iters=args.spatial_iters,
                )
            n_clust = len(set(int(c) for c in labels.tolist()))
            title = (
                f"facade {fid}  {variant}  method={method}  k={n_clust}  "
                f"windows={len(kept)}  layout_w={args.layout_weight}"
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
                        "method": method,
                        "k": n_clust,
                        "layout_weight": args.layout_weight,
                        "spatial_strength": args.spatial_strength if variant.endswith("_potts") else 0.0,
                        "windows": [
                            {
                                "id": it["id"],
                                "box_xyxy": it["box_xyxy"],
                                "match_score": it["match_score"],
                                "cluster": int(labels[i]),
                                "cx": float(cx[i]),
                                "cy": float(cy[i]),
                            }
                            for i, it in enumerate(kept)
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
                    "mean_match": mean_match,
                }
            )
            print(f"    {variant}: k={n_clust} → {facade_out}")

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

    build_compare_strip(out_dir, done_ids or facade_ids, variants)
    (out_dir / "summary_all.json").write_text(
        json.dumps(
            {
                "variants": variants,
                "method": method,
                "layout_weight": args.layout_weight,
                "spatial_strength": args.spatial_strength,
                "facade_max_side": args.facade_max_side,
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
