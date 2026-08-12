#!/usr/bin/env python3
"""Overlay within-facade DINO clusters on original train_up facade images.

For each facade id:
  1. Load crops from ``assets/windows/{id}_*.png``
  2. Recover boxes on ``train_up/{id}.png`` via template matching
  3. Cluster DINO features (default: spectral_rbf, k via silhouette)
  4. Draw colored boxes on the original facade

Example:
  python scripts/overlay_facade_asset_clusters.py \\
    --method spectral_rbf --max-facades 20 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "facade_asset_overlays")
    ap.add_argument("--dino", default="dinov2_vits14")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--method",
        default=None,
        help="single method (legacy); prefer --methods",
    )
    ap.add_argument(
        "--methods",
        type=str,
        default="spectral_rbf,spectral_nn,kmeans,agglo_ward,agglo_average_cosine,gmm_diag",
        help="comma-separated methods to overlay",
    )
    ap.add_argument("--min-windows", type=int, default=4)
    ap.add_argument("--max-facades", type=int, default=24)
    ap.add_argument("--facade-ids", type=str, default=None, help="comma ids, else largest first")
    ap.add_argument("--match-thr", type=float, default=0.55, help="min NCC to accept a box")
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--fixed-k", type=int, default=None, help="if set, skip silhouette k select")
    ap.add_argument(
        "--features-cache",
        type=Path,
        default=None,
        help="optional .npy from cluster_facade_dino_sweep (assets)",
    )
    return ap.parse_args()


def light_normalize(im: Image.Image, size: int = 224) -> Image.Image:
    im = im.convert("RGB")
    im = ImageOps.autocontrast(im, cutoff=1)
    w, h = im.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (128, 128, 128))
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def cluster_palette(n: int) -> list[tuple[int, int, int]]:
    # Distinct HSV-ish palette
    cols = []
    for i in range(max(n, 1)):
        hue = (i * 0.61803398875) % 1.0
        import colorsys

        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        cols.append((int(r * 255), int(g * 255), int(b * 255)))
    return cols


def find_box(fac_rgb: np.ndarray, crop_rgb: np.ndarray) -> tuple[tuple[int, int, int, int] | None, float]:
    fac_g = cv2.cvtColor(fac_rgb, cv2.COLOR_RGB2GRAY)
    crop_g = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    if crop_g.shape[0] > fac_g.shape[0] or crop_g.shape[1] > fac_g.shape[1]:
        return None, -1.0
    res = cv2.matchTemplate(fac_g, crop_g, cv2.TM_CCOEFF_NORMED)
    _minv, maxv, _minl, maxl = cv2.minMaxLoc(res)
    x, y = maxl
    h, w = crop_g.shape
    return (int(x), int(y), int(x + w), int(y + h)), float(maxv)


def load_facade_items(assets_dir: Path, facade_id: str) -> list[dict[str, Any]]:
    crops = sorted(
        assets_dir.glob(f"{facade_id}_*.png"),
        key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else p.stem,
    )
    return [{"id": p.stem, "image": str(p), "facade_id": facade_id} for p in crops]


@torch.no_grad()
def embed_paths(
    paths: list[str],
    *,
    model,
    image_size: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    tf = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    feats = []
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        imgs = [tf(light_normalize(Image.open(p), size=image_size)) for p in batch]
        x = torch.stack(imgs, 0).to(device)
        h = F.normalize(model(x).float(), dim=-1)
        feats.append(h.cpu().numpy())
    return np.concatenate(feats, 0).astype(np.float32)


def apply_pca(x: np.ndarray, dim: int, seed: int) -> np.ndarray:
    if dim <= 0 or dim >= x.shape[1]:
        return x
    from sklearn.decomposition import PCA

    dim = min(dim, x.shape[0] - 1, x.shape[1])
    z = PCA(n_components=dim, random_state=seed).fit_transform(x)
    z = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-8, None)
    return z.astype(np.float32)


def cluster_features(x: np.ndarray, method: str, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
    from sklearn.mixture import GaussianMixture

    k = int(max(2, min(k, len(x) - 1)))
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(x)
    if method == "spectral_rbf":
        return SpectralClustering(
            n_clusters=k, affinity="rbf", random_state=seed, assign_labels="kmeans"
        ).fit_predict(x)
    if method == "spectral_nn":
        return SpectralClustering(
            n_clusters=k,
            affinity="nearest_neighbors",
            n_neighbors=min(10, max(3, len(x) // 3)),
            random_state=seed,
            assign_labels="kmeans",
        ).fit_predict(x)
    if method == "agglo_ward":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x)
    if method == "agglo_average_cosine":
        return AgglomerativeClustering(n_clusters=k, linkage="average", metric="cosine").fit_predict(x)
    if method == "gmm_diag":
        return GaussianMixture(
            n_components=k, covariance_type="diag", random_state=seed, n_init=3
        ).fit(x).predict(x)
    raise ValueError(method)


def select_k(x: np.ndarray, method: str, k_max: int, seed: int) -> int:
    from sklearn import metrics

    n = len(x)
    if n < 4:
        return 2
    best_k, best_sil = 2, -1.0
    for k in range(2, min(k_max, n - 1) + 1):
        try:
            labels = cluster_features(x, method, k, seed)
        except Exception:
            continue
        if len(set(labels.tolist())) < 2:
            continue
        try:
            sil = float(metrics.silhouette_score(x, labels, metric="cosine"))
        except Exception:
            continue
        if sil > best_sil:
            best_sil, best_k = sil, k
    return best_k


def draw_overlay(
    facade: Image.Image,
    items: list[dict[str, Any]],
    labels: np.ndarray,
    *,
    title: str,
) -> Image.Image:
    im = facade.convert("RGB").copy()
    draw = ImageDraw.Draw(im, "RGBA")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = font_sm = ImageFont.load_default()

    n_clust = len(set(int(c) for c in labels.tolist()))
    palette = cluster_palette(n_clust)
    # map label ids to 0..C-1 dense
    uniq = sorted(set(int(c) for c in labels.tolist()))
    remap = {c: i for i, c in enumerate(uniq)}

    for i, item in enumerate(items):
        box = item.get("box_xyxy")
        if not box:
            continue
        x0, y0, x1, y1 = [int(v) for v in box]
        cid = remap[int(labels[i])]
        color = palette[cid]
        # translucent fill + solid border
        draw.rectangle([x0, y0, x1, y1], outline=color + (255,), width=3)
        draw.rectangle([x0, y0, x1, y1], fill=color + (40,))
        tag = f"c{cid}"
        tw = 8 * len(tag) + 8
        draw.rectangle([x0, max(0, y0 - 18), x0 + tw, y0], fill=color + (220,))
        draw.text((x0 + 3, max(0, y0 - 17)), tag, fill=(0, 0, 0), font=font_sm)

    # title bar
    bar_h = 36
    canvas = Image.new("RGB", (im.width, im.height + bar_h), (20, 20, 20))
    canvas.paste(im, (0, bar_h))
    d2 = ImageDraw.Draw(canvas)
    d2.text((12, 8), title, fill=(240, 240, 240), font=font)
    # legend
    x = im.width - 12
    for cid in range(n_clust - 1, -1, -1):
        label = f"c{cid}"
        tw = 10 * len(label) + 28
        x -= tw + 8
        color = palette[cid]
        d2.rectangle([x, 6, x + 18, 24], fill=color)
        d2.text((x + 22, 8), label, fill=(230, 230, 230), font=font_sm)
    return canvas


def make_side_by_side(overlay: Image.Image, crops: list[Image.Image], labels: np.ndarray) -> Image.Image:
    """Overlay | cluster crop strips."""
    cell = 72
    by = defaultdict(list)
    for i, lab in enumerate(labels.tolist()):
        by[int(lab)].append(i)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    strips = []
    pad, lh = 4, 16
    uniq = sorted(by)
    remap = {c: i for i, c in enumerate(uniq)}
    palette = cluster_palette(len(uniq))
    for raw_cid in uniq:
        cid = remap[raw_cid]
        idxs = by[raw_cid][:16]
        w = len(idxs) * (cell + pad) + pad
        strip = Image.new("RGB", (max(w, 120), cell + lh + pad), (24, 24, 24))
        d = ImageDraw.Draw(strip)
        d.rectangle([0, 0, 8, strip.height], fill=palette[cid])
        d.text((12, 1), f"c{cid} n={len(by[raw_cid])}", fill=(230, 230, 230), font=font)
        for j, i in enumerate(idxs):
            thumb = ImageOps.contain(crops[i].convert("RGB"), (cell, cell))
            strip.paste(thumb, (pad + j * (cell + pad), lh))
        strips.append(strip)
    panel_w = max((s.width for s in strips), default=200)
    panel_h = sum(s.height + 4 for s in strips) - 4 if strips else 200
    right = Image.new("RGB", (panel_w, max(panel_h, overlay.height)), (16, 16, 16))
    y = 0
    for s in strips:
        right.paste(s, (0, y))
        y += s.height + 4
    # scale overlay to reasonable width
    max_w = 1200
    ov = overlay
    if ov.width > max_w:
        nh = int(ov.height * max_w / ov.width)
        ov = ov.resize((max_w, nh), Image.Resampling.LANCZOS)
    h = max(ov.height, right.height)
    out = Image.new("RGB", (ov.width + 8 + right.width, h), (12, 12, 12))
    out.paste(ov, (0, 0))
    out.paste(right, (ov.width + 8, 0))
    return out


def build_method_overview(out_dir: Path, method: str, facade_ids: list[str]) -> None:
    thumbs = []
    for fid in facade_ids:
        p = out_dir / method / f"facade_{fid}" / "overlay.png"
        if not p.is_file():
            continue
        im = Image.open(p).convert("RGB")
        im.thumbnail((480, 480), Image.Resampling.LANCZOS)
        thumbs.append((fid, im))
    if not thumbs:
        return
    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    cell_w = max(t.width for _, t in thumbs)
    cell_h = max(t.height for _, t in thumbs) + 22
    grid = Image.new("RGB", (cols * cell_w + 8, rows * cell_h + 8), (16, 16, 16))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(grid)
    for i, (fid, im) in enumerate(thumbs):
        r, c = divmod(i, cols)
        x, y = 4 + c * cell_w, 4 + r * cell_h
        draw.text((x + 4, y), f"facade {fid}", fill=(230, 230, 230), font=font)
        grid.paste(im, (x, y + 18))
    grid.save(out_dir / method / "overview_overlays.png")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if args.method:
        methods = [args.method]
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    allowed = {
        "spectral_rbf",
        "spectral_nn",
        "kmeans",
        "agglo_ward",
        "agglo_average_cosine",
        "gmm_diag",
    }
    bad = [m for m in methods if m not in allowed]
    if bad:
        raise SystemExit(f"unknown methods {bad}; allowed={sorted(allowed)}")

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
    print(f"facades={len(facade_ids)} methods={methods}")

    print(f"loading {args.dino}…")
    model = torch.hub.load("facebookresearch/dinov2", args.dino, pretrained=True)
    model = model.to(device).eval()

    all_summary: dict[str, list[dict]] = {m: [] for m in methods}

    for fi, fid in enumerate(facade_ids):
        facade_path = args.facade_dir / f"{fid}.png"
        if not facade_path.is_file():
            print(f"skip {fid}: missing facade")
            continue
        items = load_facade_items(args.assets_dir, fid)
        if len(items) < args.min_windows:
            print(f"skip {fid}: only {len(items)} crops")
            continue

        facade_im = Image.open(facade_path).convert("RGB")
        facade_rgb = np.array(facade_im)
        kept = []
        crops_pil = []
        for it in items:
            crop = np.array(Image.open(it["image"]).convert("RGB"))
            box, score = find_box(facade_rgb, crop)
            if box is None or score < args.match_thr:
                continue
            it = dict(it)
            it["box_xyxy"] = list(box)
            it["match_score"] = score
            kept.append(it)
            crops_pil.append(Image.fromarray(crop))
        if len(kept) < args.min_windows:
            print(f"skip {fid}: only {len(kept)} matched boxes")
            continue

        feats = embed_paths(
            [it["image"] for it in kept],
            model=model,
            image_size=args.image_size,
            batch_size=args.batch_size,
            device=device,
        )
        feats = apply_pca(feats, min(args.pca_dim, max(2, len(kept) - 1)), args.seed)
        mean_match = float(np.mean([it["match_score"] for it in kept]))
        print(f"[{fi+1}/{len(facade_ids)}] facade {fid}: n={len(kept)} match={mean_match:.3f}")

        for method in methods:
            if args.fixed_k is not None:
                k = args.fixed_k
            else:
                k = select_k(feats, method, args.k_max, args.seed)
            labels = cluster_features(feats, method, k, args.seed)
            n_clust = len(set(int(c) for c in labels.tolist()))

            title = f"facade {fid}  method={method}  k={n_clust}  windows={len(kept)}"
            overlay = draw_overlay(facade_im, kept, labels, title=title)
            combo = make_side_by_side(overlay, crops_pil, labels)

            facade_out = out_dir / method / f"facade_{fid}"
            facade_out.mkdir(parents=True, exist_ok=True)
            overlay.save(facade_out / "overlay.png")
            combo.save(facade_out / "overlay_with_crops.png")
            (facade_out / "clusters.json").write_text(
                json.dumps(
                    {
                        "facade_id": fid,
                        "method": method,
                        "k": n_clust,
                        "windows": [
                            {
                                "id": it["id"],
                                "box_xyxy": it["box_xyxy"],
                                "match_score": it["match_score"],
                                "cluster": int(labels[i]),
                            }
                            for i, it in enumerate(kept)
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            all_summary[method].append(
                {
                    "facade_id": fid,
                    "n_windows": len(kept),
                    "k": n_clust,
                    "mean_match": mean_match,
                }
            )
            print(f"    {method}: k={n_clust} → {facade_out}")

    for method, summary in all_summary.items():
        mdir = out_dir / method
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        build_method_overview(out_dir, method, [s["facade_id"] for s in summary])
        print(f"overview → {mdir / 'overview_overlays.png'}")

    (out_dir / "summary_all.json").write_text(json.dumps(all_summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
