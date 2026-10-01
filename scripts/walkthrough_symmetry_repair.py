#!/usr/bin/env python3
"""Step-by-step symmetry repair on spectral clusters (worked example).

Default: CMP façades under ``data/facades/base/*.jpg`` + XML window GT
(not ``train_up`` / detector boxes).

Pipeline:
  1. Load base JPG + CMP XML windows (full façade)
  2. Merge adjacent same-floor panes → units
  3. DINOv2 patch-ROI embed + spectral_rbf on units
  4. Zhang facade symmetry: **in-box** (intra) + **global** (inter, same type)
     over the whole-facade extent; reflect about facade midline
  5. STEAL/JOIN gated by feature similarity; accept if facade objective improves

Example:
  python scripts/walkthrough_symmetry_repair.py --stem cmp_b0001
  python scripts/walkthrough_symmetry_repair.py --stem cmp_b0083 --sym-scope cluster
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.overlay_facade_asset_clusters import draw_overlay  # noqa: E402
from window_ast.symmetry import (  # noqa: E402
    bounding_box,
    cluster_mirror_score,
    facade_center_x,
    facade_symmetry,
    find_mirror_partner_index,
    normalized_integral_symmetry,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stem",
        default="cmp_b0001",
        help="CMP base id without extension (e.g. cmp_b0001)",
    )
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=EXP / "data" / "facades" / "base",
        help="CMP façades root with {stem}.jpg + {stem}.xml",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "runs" / "symmetry_repair_walkthrough",
    )
    ap.add_argument("--cluster", type=int, default=None, help="focus cluster id; default=auto")
    ap.add_argument("--axis", choices=("v", "h", "both"), default="v")
    ap.add_argument(
        "--sym-scope",
        choices=("facade", "cluster"),
        default="facade",
        help="score/reflect about whole-facade extent (paper) or focus-cluster AABB",
    )
    ap.add_argument(
        "--extent",
        choices=("boxes", "image"),
        default="boxes",
        help="facade extent = tight bbox of all windows, or full image (0,0,W,H)",
    )
    ap.add_argument("--match-iou", type=float, default=0.15)
    ap.add_argument("--match-center", type=float, default=0.35)
    ap.add_argument(
        "--merge-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="unitize adjacent panes, then spectral-cluster merged boxes (default on)",
    )
    ap.add_argument(
        "--sim-margin",
        type=float,
        default=0.0,
        help="STEAL/JOIN: mover must be at least this much closer (cosine) to "
        "destination cluster mean than to its current cluster mean",
    )
    ap.add_argument(
        "--twin-merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="if focus has a strong facade-mirror twin cluster, try MERGE (default on)",
    )
    ap.add_argument(
        "--twin-mirror-min",
        type=float,
        default=0.5,
        help="min cluster_mirror_score(focus, twin) to attempt MERGE",
    )
    ap.add_argument(
        "--twin-pair-sim-min",
        type=float,
        default=0.25,
        help="twin-only appearance gate for MERGE: min cosine(src, mirror twin). "
        "Use -1 to disable",
    )
    ap.add_argument(
        "--all-clusters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run twin-MERGE + leftover repair over every cluster (not one focus)",
    )
    ap.add_argument(
        "--cluster-feat",
        choices=("dino", "lineart", "lineart_dino"),
        default="dino",
        help="unit features for spectral clustering: RGB-DINO, ControlNet lineart "
        "ink maps, or DINO on lineart",
    )
    ap.add_argument("--dino", default="dinov2_vits14")
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--facade-max-side", type=int, default=896)
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--col-tol", type=float, default=0.045)
    ap.add_argument("--row-tol", type=float, default=0.055)
    ap.add_argument("--adj-gap", type=float, default=1.0)
    ap.add_argument(
        "--merge-vertical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="before clustering, union nearly-touching same-story vertical "
        "splits (sash/transom); does NOT stack across floors",
    )
    ap.add_argument(
        "--vert-gap",
        type=float,
        default=0.55,
        help="max vertical gap as fraction of median box height "
        "(default 0.55 ≈ same-story only)",
    )
    ap.add_argument(
        "--shape-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="after spectral, split clusters that mix very different "
        "widths or heights (default on)",
    )
    ap.add_argument(
        "--shape-gap",
        type=float,
        default=0.01,
        help="facade-relative step in sorted width/height to shape-split "
        "(same idea as floor/bay tol): consecutive w/W or h/H gap ≥ this "
        "(default 0.01 ≈ 1%% of facade)",
    )
    ap.add_argument(
        "--shape-ratio",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    return ap.parse_args()


def parse_cmp_windows(xml_path: Path) -> list[tuple[float, float, float, float]]:
    """Normalized (u0, v0, u1, v1) windows; CMP ``<x>``=row, ``<y>``=col."""
    text = xml_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text.startswith("<objects"):
        text = f"<objects>\n{text}\n</objects>"
    root = ET.fromstring(text)
    boxes: list[tuple[float, float, float, float]] = []
    for obj in root.iter("object"):
        name = (obj.findtext("labelname") or "").strip().lower()
        if name != "window":
            continue
        pts = obj.find("points")
        if pts is None:
            continue
        rows = [float(x.text) for x in pts.findall("x") if x.text]
        cols = [float(y.text) for y in pts.findall("y") if y.text]
        if len(rows) < 2 or len(cols) < 2:
            continue
        u0, u1 = min(cols), max(cols)
        v0, v1 = min(rows), max(rows)
        if u1 - u0 < 1e-4 or v1 - v0 < 1e-4:
            continue
        boxes.append((u0, v0, u1, v1))
    return boxes


def load_base_facade(base_dir: Path, stem: str) -> tuple[Image.Image, list[dict]]:
    """Load ``{stem}.jpg`` and pixel windows from ``{stem}.xml``."""
    jpg = base_dir / f"{stem}.jpg"
    if not jpg.is_file():
        png = base_dir / f"{stem}.png"
        if not png.is_file():
            raise FileNotFoundError(f"missing image for {stem} under {base_dir}")
        jpg = png
    xml = base_dir / f"{stem}.xml"
    if not xml.is_file():
        raise FileNotFoundError(f"missing XML for {stem}: {xml}")
    facade = Image.open(jpg).convert("RGB")
    w, h = facade.size
    raw: list[dict] = []
    for i, (u0, v0, u1, v1) in enumerate(parse_cmp_windows(xml)):
        box = [u0 * w, v0 * h, u1 * w, v1 * h]
        raw.append({"id": f"{stem}_w{i}", "box_xyxy": box, "cluster": -1})
    if not raw:
        raise RuntimeError(f"no window boxes in {xml}")
    return facade, raw


def _load_script(path: Path, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def embed_lineart_boxes(
    facade_lineart: Image.Image,
    boxes: list[list[int]] | list[list[float]],
    *,
    map_size: int = 48,
) -> np.ndarray:
    """Per-box ink embedding from a facade-wide ControlNet lineart image."""
    from window_ast.lineart_input import ink_channel

    feats: list[np.ndarray] = []
    w, h = facade_lineart.size
    for b in boxes:
        x0, y0, x1, y1 = [int(round(v)) for v in b[:4]]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, max(x0 + 1, x1)), min(h, max(y0 + 1, y1))
        crop = facade_lineart.crop((x0, y0, x1, y1))
        ink = ink_channel(crop, map_size).numpy()[0]  # [H,W] ink in [0,1]
        # 2D map + axis profiles (structure-ish, cheap)
        vp = ink.mean(axis=0)
        hp = ink.mean(axis=1)
        feat = np.concatenate([ink.reshape(-1), vp, hp]).astype(np.float32)
        feat /= max(float(np.linalg.norm(feat)), 1e-8)
        feats.append(feat)
    return np.stack(feats, 0) if feats else np.zeros((0, 1), dtype=np.float32)


def _max_sorted_step(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    s = np.sort(values)
    return float(np.max(s[1:] - s[:-1]))


def _split_on_big_steps(
    values: np.ndarray,
    thr: float,
    *,
    min_size: int = 2,
    singleton_thr: float | None = None,
) -> np.ndarray:
    """Partition by large gaps in the sorted 1D distribution.

    Cuts where a consecutive step is ≥ ``thr``. Both sides normally need
    ≥ ``min_size`` members (default 2) to avoid peeling tiny annotation
    jitter. If ``singleton_thr`` is set, gaps ≥ that value may peel a
    singleton — for clear outliers (e.g. one wide bay vs narrow siblings).
    """
    n = len(values)
    out = np.zeros(n, dtype=np.int32)
    if n < 2:
        return out
    if singleton_thr is None:
        singleton_thr = max(2.0 * thr, 0.02)
    order = np.argsort(values)
    sorted_v = values[order]
    gaps = sorted_v[1:] - sorted_v[:-1]
    cut_after: set[int] = set()
    for i, g in enumerate(gaps.tolist()):
        g = float(g)
        if g < thr:
            continue
        left = i + 1
        right = n - left
        need = 1 if g >= singleton_thr else min_size
        if left >= need and right >= need:
            cut_after.add(int(i))
    if not cut_after:
        return out
    g = 0
    sorted_groups = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if i > 0 and (i - 1) in cut_after:
            g += 1
        sorted_groups[i] = g
    out[order] = sorted_groups
    return out


def shape_split_labels(
    labels: np.ndarray | list[int],
    boxes: list[list[float]] | list[list[int]],
    *,
    image_size: tuple[int, int],
    gap_thr: float = 0.01,
    singleton_thr: float | None = None,
    max_passes: int = 6,
) -> np.ndarray:
    """Split clusters whose width or height distribution has a big step.

    Within each cluster, sort facade-normalized widths and heights
    (``w/W``, ``h/H``). If the largest consecutive gap exceeds ``gap_thr``,
    cut only at those large gaps (small within-band jitter is ignored).
    Very large gaps may peel a singleton outlier. Repeats until stable.
    """
    out = np.asarray(labels, dtype=np.int32).copy()
    if len(out) == 0:
        return out
    fw = max(1.0, float(image_size[0]))
    fh = max(1.0, float(image_size[1]))
    ws = np.array([max(1.0, float(b[2] - b[0])) / fw for b in boxes], dtype=np.float64)
    hs = np.array([max(1.0, float(b[3] - b[1])) / fh for b in boxes], dtype=np.float64)
    if singleton_thr is None:
        singleton_thr = max(2.0 * gap_thr, 0.02)

    def _reindex(arr: np.ndarray) -> np.ndarray:
        mapping: dict[int, int] = {}
        remapped = np.zeros_like(arr)
        for i, lab in enumerate(arr.tolist()):
            lab = int(lab)
            if lab not in mapping:
                mapping[lab] = len(mapping)
            remapped[i] = mapping[lab]
        return remapped

    for _ in range(max_passes):
        next_id = int(out.max()) + 1
        changed = False
        for c in sorted(set(int(x) for x in out.tolist())):
            idxs = [i for i, lab in enumerate(out) if int(lab) == c]
            if len(idxs) < 2:
                continue
            w = ws[idxs]
            h = hs[idxs]
            w_step = _max_sorted_step(w)
            h_step = _max_sorted_step(h)
            if w_step < gap_thr and h_step < gap_thr:
                continue
            vals = w if w_step >= h_step else h
            groups = _split_on_big_steps(
                vals, gap_thr, singleton_thr=singleton_thr
            )
            uniq = sorted(set(int(g) for g in groups.tolist()))
            if len(uniq) < 2:
                continue
            sizes = {g: int(np.sum(groups == g)) for g in uniq}
            keep = max(uniq, key=lambda g: (sizes[g], -g))
            for g in uniq:
                if g == keep:
                    continue
                for li, gi in enumerate(groups.tolist()):
                    if int(gi) == g:
                        out[idxs[li]] = next_id
                next_id += 1
            changed = True
        if not changed:
            break
    return _reindex(out)


def build_merged_units(
    *,
    facade_id: str,
    facade: Image.Image,
    raw_windows: list[dict],
    device: str,
    dino: str,
    cluster_feat: str,
    facade_max_side: int,
    pca_dim: int,
    k_max: int,
    seed: int,
    col_tol: float,
    row_tol: float,
    adj_gap: float,
    merge_vertical: bool = True,
    vert_gap: float = 0.55,
    shape_split: bool = True,
    shape_gap: float = 0.01,
) -> tuple[list[dict], list[int], np.ndarray, dict[str, Any], Image.Image]:
    """Unitize boxes (horizontal, optional same-story vertical) → embed → spectral."""
    import torch

    merge_mod = _load_script(ROOT / "scripts" / "overlay_facade_merge_boxes_ast.py", "merge_walk")
    patch = _load_script(ROOT / "scripts" / "overlay_facade_patch_layout.py", "patch_walk")
    base = _load_script(ROOT / "scripts" / "overlay_facade_asset_clusters.py", "base_walk")

    boxes = [list(map(float, w["box_xyxy"])) for w in raw_windows]
    iw, ih = facade.size
    cx = np.array([0.5 * (b[0] + b[2]) / iw for b in boxes], dtype=np.float64)
    cy = np.array([0.5 * (b[1] + b[3]) / ih for b in boxes], dtype=np.float64)
    int_boxes = [[int(round(v)) for v in b] for b in boxes]
    merged_boxes, members, _bay, _floor = merge_mod.merge_adjacent_boxes(
        int_boxes,
        cx,
        cy,
        row_tol=row_tol,
        adj_gap=adj_gap,
        merge_bays=False,
        col_tol=col_tol,
        merge_vertical=False,
    )
    n_h = len(merged_boxes)
    if merge_vertical and n_h >= 2:
        # Same-story vertical splits BEFORE clustering (no labels yet).
        dummy = np.zeros(n_h, dtype=np.int32)
        v_boxes, v_members, _ = merge_mod.merge_vertical_same_cluster(
            merged_boxes,
            dummy,
            members,
            row_tol=row_tol,
            vert_gap=vert_gap,
            image_size=facade.size,
        )
        if len(v_boxes) < n_h:
            print(
                f"unitize vertical (pre-cluster): {n_h} → {len(v_boxes)} "
                f"(vert_gap={vert_gap})"
            )
            merged_boxes, members = v_boxes, v_members

    n_m = len(merged_boxes)
    facade_for_embed = facade
    feat_tag = cluster_feat
    spatial = None
    patch_meta = None
    facade_la = None

    if cluster_feat in ("lineart", "lineart_dino"):
        from window_ast.lineart_input import compute_lineart, get_lineart_detector

        print(f"computing ControlNet lineart for clustering ({cluster_feat})…")
        det = get_lineart_detector(device)
        facade_la = compute_lineart(facade, detector=det, detect_res=512)
        if cluster_feat == "lineart":
            raw_x = embed_lineart_boxes(facade_la, int_boxes)
            mer_x = embed_lineart_boxes(facade_la, merged_boxes)
            feats_raw = base.apply_pca(raw_x, min(pca_dim, max(2, len(boxes) - 1)), seed)
            feats_m = base.apply_pca(mer_x, min(pca_dim, max(2, n_m - 1)), seed)
        else:
            facade_for_embed = facade_la
            print(f"loading {dino} on lineart…")
            model = torch.hub.load("facebookresearch/dinov2", dino, pretrained=True)
            model = model.to(device).eval()
            spatial, patch_meta = patch.facade_patch_spatial(
                model, facade_for_embed, device=torch.device(device), max_side=facade_max_side
            )
            feats_raw = base.apply_pca(
                patch.roi_pool_patches(spatial, int_boxes, patch_meta),
                min(pca_dim, max(2, len(boxes) - 1)),
                seed,
            )
            feats_m = base.apply_pca(
                patch.roi_pool_patches(spatial, merged_boxes, patch_meta),
                min(pca_dim, max(2, n_m - 1)),
                seed,
            )
    else:
        print(f"loading {dino} for merge→cluster…")
        model = torch.hub.load("facebookresearch/dinov2", dino, pretrained=True)
        model = model.to(device).eval()
        spatial, patch_meta = patch.facade_patch_spatial(
            model, facade, device=torch.device(device), max_side=facade_max_side
        )
        feats_raw = base.apply_pca(
            patch.roi_pool_patches(spatial, int_boxes, patch_meta),
            min(pca_dim, max(2, len(boxes) - 1)),
            seed,
        )
        feats_m = base.apply_pca(
            patch.roi_pool_patches(spatial, merged_boxes, patch_meta),
            min(pca_dim, max(2, n_m - 1)),
            seed,
        )

    k_raw = base.select_k(feats_raw, "spectral_rbf", min(k_max, len(boxes) - 1), seed)
    labels_raw = base.cluster_features(feats_raw, "spectral_rbf", k_raw, seed)

    if n_m >= 4:
        k_m = base.select_k(feats_m, "spectral_rbf", min(k_max, n_m - 1), seed)
        labels_m = base.cluster_features(feats_m, "spectral_rbf", k_m, seed)
    elif n_m >= 2:
        labels_m = base.cluster_features(feats_m, "spectral_rbf", 2, seed)
        k_m = 2
    else:
        labels_m = np.zeros(n_m, dtype=np.int32)
        k_m = 1

    labels_spectral = np.asarray(labels_m, dtype=np.int32).copy()
    k_spectral = int(len(set(int(x) for x in labels_spectral.tolist())))

    if shape_split and n_m >= 2:
        before_k = k_spectral
        labels_m = shape_split_labels(
            labels_m,
            [[float(x) for x in b] for b in merged_boxes],
            image_size=facade.size,
            gap_thr=shape_gap,
        )
        after_k = int(len(set(int(x) for x in labels_m.tolist())))
        if after_k != before_k:
            print(
                f"shape-split (step w/W|h/H≥{shape_gap}): k {before_k} → {after_k}"
            )
            k_m = after_k

    units = []
    for mi, mb in enumerate(merged_boxes):
        units.append(
            {
                "id": f"{facade_id}_u{mi}",
                "box_xyxy": [float(x) for x in mb],
                "cluster": int(labels_m[mi]),
                "n_members": len(members[mi]),
                "members": [raw_windows[j]["id"] for j in members[mi]],
            }
        )

    items_before = [{"box_xyxy": b} for b in boxes]
    items_after = [{"box_xyxy": b} for b in merged_boxes]
    panel = merge_mod.draw_before_after(
        facade,
        int_boxes,
        items_before,
        labels_raw,
        items_after,
        labels_m,
        fid=facade_id,
        n_merged_from=len(boxes),
    )
    fnt = load_font(16)
    bar = Image.new("RGB", (panel.width, 40), (18, 18, 20))
    ImageDraw.Draw(bar).text(
        (10, 10),
        f"PRE · CMP GT n={len(boxes)} k={k_raw} → unitized n={n_m} k={k_m}  "
        f"feat={feat_tag}"
        + (f"  vert_gap={vert_gap}" if merge_vertical else ""),
        fill=(230, 230, 235),
        font=fnt,
    )
    pre = Image.new("RGB", (panel.width, panel.height + 40), (8, 8, 10))
    pre.paste(bar, (0, 0))
    pre.paste(panel, (0, 40))

    meta_out = {
        "k": int(k_m),
        "k_spectral": k_spectral,
        "n_raw": len(boxes),
        "n_merged": n_m,
        "merge_first": True,
        "source": "facades/base",
        "cluster_feat": feat_tag,
        "merge_vertical": bool(merge_vertical),
        "vert_gap": float(vert_gap),
        "shape_split": bool(shape_split),
        "shape_gap": float(shape_gap),
        "labels_spectral": [int(x) for x in labels_spectral.tolist()],
    }
    return units, [int(x) for x in labels_m.tolist()], np.asarray(feats_m), meta_out, pre


def cosine_to_cluster(
    feats: np.ndarray, labels: list[int], idx: int, cid: int
) -> float | None:
    """Cosine similarity of unit ``idx`` to mean of cluster ``cid`` (excluding idx)."""
    idxs = [i for i, c in enumerate(labels) if c == cid and i != idx]
    if not idxs:
        return None
    mean = feats[idxs].mean(axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-8)
    v = feats[idx]
    v = v / (np.linalg.norm(v) + 1e-8)
    return float(np.dot(v, mean))


def sim_prefers_dest(
    feats: np.ndarray,
    labels: list[int],
    mover: int,
    src_cid: int,
    dest_cid: int,
    *,
    margin: float,
) -> tuple[bool, float | None, float | None]:
    """True if mover is closer (cosine) to dest than to src by ``margin``."""
    sim_src = cosine_to_cluster(feats, labels, mover, src_cid)
    sim_dst = cosine_to_cluster(feats, labels, mover, dest_cid)
    if sim_src is None or sim_dst is None:
        return True, sim_src, sim_dst
    return (sim_dst + 1e-9) >= (sim_src + margin), sim_src, sim_dst


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def box_center(b: list[float]) -> tuple[float, float]:
    return 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])


def box_iou(a: list[float], b: list[float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ba = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / max(aa + ba - inter, 1e-8))


def reflect_box(b: list[float], *, axis: str, cx: float, cy: float) -> list[float]:
    x0, y0, x1, y1 = b
    if axis == "v":
        return [2 * cx - x1, y0, 2 * cx - x0, y1]
    return [x0, 2 * cy - y1, x1, 2 * cy - y0]


def match_reflected(
    target: list[float],
    boxes: list[list[float]],
    *,
    iou_thr: float,
    center_thr: float,
    exclude: int | None = None,
) -> tuple[int | None, float]:
    """Return (index, score) of best match to ``target``; score = IoU + soft center."""
    tc = box_center(target)
    diag = max(1.0, float(np.hypot(target[2] - target[0], target[3] - target[1])))
    best_i, best_s = None, -1.0
    for i, b in enumerate(boxes):
        if exclude is not None and i == exclude:
            continue
        iou = box_iou(target, b)
        bc = box_center(b)
        dist = float(np.hypot(tc[0] - bc[0], tc[1] - bc[1])) / diag
        if iou < iou_thr and dist > center_thr:
            continue
        s = iou + max(0.0, 1.0 - dist)
        if s > best_s:
            best_s, best_i = s, i
    return best_i, best_s


def cluster_boxes(windows: list[dict], cid: int) -> list[list[float]]:
    return [list(map(float, w["box_xyxy"])) for w in windows if int(w["cluster"]) == cid]


def axis_weights(axis: str) -> tuple[float, float]:
    """Map walkthrough axis to Zhang I_h / I_v weights.

    ``axis='v'`` = reflect about a **vertical** line (left–right façade symmetry)
    → paper ``I_h`` → ``horizontal_weight=1``.
    ``axis='h'`` = reflect about a **horizontal** line (up–down) → ``I_v``.
    """
    return (1.0, 0.0) if axis == "v" else (0.0, 1.0)


def cluster_local_sym(boxes: list[list[float]], *, axis: str) -> float:
    """Per-cluster NIS about the cluster's own bbox (legacy local score)."""
    if len(boxes) < 1:
        return 0.0
    wh, wv = axis_weights(axis)
    return float(
        normalized_integral_symmetry(
            boxes, horizontal_weight=wh, vertical_weight=wv, n_samples=256
        )
    )


def facade_sym(
    boxes: list[list[float]],
    labels: list[int],
    *,
    axis: str,
    extent_mode: str,
    image_size: tuple[int, int],
):
    """Whole-facade in-box + global (inter) Zhang scores."""
    wh, wv = axis_weights(axis)
    extent = None
    img = None
    if extent_mode == "image":
        img = (float(image_size[0]), float(image_size[1]))
    return facade_symmetry(
        boxes,
        types=labels,
        image_size=img,
        n_samples=256,
        horizontal_weight=wh,
        vertical_weight=wv,
        extent=extent,
    )


def facade_axis_center(
    boxes: list[list[float]],
    *,
    extent_mode: str,
    image_size: tuple[int, int],
) -> tuple[float, float, list[float]]:
    """Midpoint of facade extent → reflection axis for witnesses."""
    if extent_mode == "image":
        w, h = image_size
        extent = [0.0, 0.0, float(w), float(h)]
    else:
        bb = bounding_box(boxes)
        assert bb is not None
        extent = list(bb)
    cx = 0.5 * (extent[0] + extent[2])
    cy = 0.5 * (extent[1] + extent[3])
    return cx, cy, extent


def pick_focus_cluster(
    windows: list[dict],
    axis: str,
    *,
    cx: float,
    cy: float,
) -> int:
    """Prefer clusters with case-1 witnesses about the given facade axis."""
    labels = sorted({int(w["cluster"]) for w in windows})
    boxes = [list(map(float, w["box_xyxy"])) for w in windows]
    clabs = [int(w["cluster"]) for w in windows]
    best: tuple[int, int, int, int] | None = None  # (n1, n, -n2, cid)
    for cid in labels:
        idxs = [i for i, c in enumerate(clabs) if c == cid]
        if len(idxs) < 2:
            continue
        n1 = n2 = 0
        for i in idxs:
            rb = reflect_box(boxes[i], axis=axis, cx=cx, cy=cy)
            j, _ = match_reflected(rb, boxes, iou_thr=0.15, center_thr=0.35, exclude=i)
            if j is None:
                n2 += 1
            elif clabs[j] != cid:
                n1 += 1
        key = (n1, len(idxs), -n2, cid)
        if best is None or key > best:
            best = key
    return int(best[3]) if best is not None else int(labels[0])


def draw_step_panel(
    facade: Image.Image,
    windows: list[dict],
    labels: list[int],
    *,
    focus: int,
    axis: str,
    title: str,
    highlights: list[dict[str, Any]] | None = None,
    cell_w: int = 520,
    axis_cx: float | None = None,
    axis_cy: float | None = None,
    facade_extent: list[float] | None = None,
) -> Image.Image:
    items = [{"box_xyxy": w["box_xyxy"]} for w in windows]
    lab = np.asarray(labels, dtype=np.int32)
    ov = draw_overlay(facade, items, lab, title=title)
    scale = cell_w / ov.width
    nh = max(1, int(ov.height * scale))
    ov = ov.resize((cell_w, nh), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(ov)
    fnt = load_font(14)
    fnt_sm = load_font(12)

    idxs = [i for i, c in enumerate(labels) if c == focus]
    if idxs:
        boxes = [list(map(float, windows[i]["box_xyxy"])) for i in idxs]
        bb = bounding_box(boxes)
        assert bb is not None
        x0, y0, x1, y1 = [v * scale for v in bb]
        d.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=2)
        d.text((x0 + 4, max(0, y0 - 16)), f"focus c{focus}", fill=(255, 255, 0), font=fnt_sm)

    # facade / global reflection axis
    if facade_extent is not None and axis_cx is not None and axis_cy is not None:
        ex0, ey0, ex1, ey1 = [v * scale for v in facade_extent]
        d.rectangle([ex0, ey0, ex1, ey1], outline=(180, 180, 255), width=1)
        if axis == "v":
            d.line(
                [(axis_cx * scale, ey0), (axis_cx * scale, ey1)],
                fill=(0, 220, 255),
                width=2,
            )
        else:
            d.line(
                [(ex0, axis_cy * scale), (ex1, axis_cy * scale)],
                fill=(0, 220, 255),
                width=2,
            )
        d.text(
            (ex0 + 4, min(ov.height - 14, ey1 + 2)),
            "facade axis",
            fill=(0, 220, 255),
            font=fnt_sm,
        )
    elif idxs and axis_cx is None:
        boxes = [list(map(float, windows[i]["box_xyxy"])) for i in idxs]
        bb = bounding_box(boxes)
        assert bb is not None
        x0, y0, x1, y1 = [v * scale for v in bb]
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        if axis == "v":
            d.line([(cx, y0), (cx, y1)], fill=(255, 255, 0), width=2)
        else:
            d.line([(x0, cy), (x1, cy)], fill=(255, 255, 0), width=2)

    if highlights:
        for h in highlights:
            b = [v * scale for v in h["box"]]
            col = h.get("color", (255, 0, 0))
            d.rectangle(b, outline=col, width=3)
            if "reflected" in h:
                rb = [v * scale for v in h["reflected"]]
                d.rectangle(rb, outline=(255, 180, 0), width=2)
                # dashed-ish link
                c1 = box_center(b)
                c2 = box_center(rb)
                d.line([c1, c2], fill=(255, 180, 0), width=1)
            tag = h.get("tag", "")
            d.text((b[0] + 2, b[1] + 2), tag, fill=col, font=fnt_sm)

    # caption bar
    bar = Image.new("RGB", (ov.width, 36), (20, 20, 22))
    ImageDraw.Draw(bar).text((8, 10), title, fill=(230, 230, 235), font=fnt)
    out = Image.new("RGB", (ov.width, ov.height + 36), (12, 12, 14))
    out.paste(bar, (0, 0))
    out.paste(ov, (0, 36))
    return out


def rank_clusters_for_repair(
    boxes: list[list[float]],
    labels: list[int],
    *,
    cx: float,
    cy: float,
    axis: str,
    match_iou: float,
    match_center: float = 0.35,
) -> list[tuple[int, int, int, float]]:
    """Return [(cid, n_case1, size, best_mirror), ...] sorted for repair order.

    Size-1 clusters are included (shape-split often leaves singleton L/R twins).
    """
    from collections import Counter

    labs = sorted({int(c) for c in labels})
    ranked: list[tuple[int, int, int, float]] = []
    for cid in labs:
        idxs = [i for i, c in enumerate(labels) if c == cid]
        if not idxs:
            continue
        twin_votes: Counter[int] = Counter()
        n1 = 0
        for i in idxs:
            rb = reflect_box(boxes[i], axis=axis, cx=cx, cy=cy)
            j, _ = match_reflected(
                rb, boxes, iou_thr=match_iou, center_thr=match_center, exclude=i
            )
            if j is not None and labels[j] != cid:
                n1 += 1
                twin_votes[int(labels[j])] += 1
        if n1 == 0:
            continue
        twin_cid = twin_votes.most_common(1)[0][0]
        boxes_f = [boxes[i] for i in idxs]
        boxes_t = [boxes[i] for i, c in enumerate(labels) if c == twin_cid]
        ms = cluster_mirror_score(boxes_f, boxes_t, cx, iou_thresh=match_iou)
        ranked.append((cid, n1, len(idxs), float(ms)))
    ranked.sort(key=lambda t: (t[3], t[1], t[2]), reverse=True)
    return ranked


def box_fits_cluster_size(
    box: list[float] | list[int],
    cluster_boxes: list[list[float]] | list[list[int]],
    *,
    rel_tol: float = 0.25,
    min_slack_px: float = 2.0,
) -> bool:
    """Whether ``box`` width/height sit within the cluster's size band.

    Slack is the cluster's own half-range (max−min)/2, floored by
    ``rel_tol × median`` and ``min_slack_px`` — so a wide bay cannot join a
    tight narrow cluster, but normal L/R twin jitter still matches.
    """
    if not cluster_boxes:
        return False
    bw = max(1.0, float(box[2] - box[0]))
    bh = max(1.0, float(box[3] - box[1]))
    ws = [max(1.0, float(b[2] - b[0])) for b in cluster_boxes]
    hs = [max(1.0, float(b[3] - b[1])) for b in cluster_boxes]

    def _ok(val: float, vals: list[float]) -> bool:
        med = float(np.median(vals))
        half = 0.5 * (max(vals) - min(vals)) if len(vals) >= 2 else 0.0
        slack = max(half, rel_tol * med, min_slack_px)
        return abs(val - med) <= slack + 1e-6

    return _ok(bw, ws) and _ok(bh, hs)


def mirror_pair_geo_strength(
    box_a: list[float],
    box_b: list[float],
    *,
    cx: float,
    axis: str = "v",
) -> dict[str, float]:
    """How strongly two boxes look like L/R twins despite size differences.

    ``partner_iou`` — IoU of A reflected about ``cx`` vs B.
    ``small_cover`` — fraction of the smaller box covered by the reflection of
    the larger (1.0 = pediment-vs-pane / crop truncation twin).
    """
    if axis != "v":
        # horizontal axis not used in current repair path
        rb = reflect_box(box_a, axis=axis, cx=cx, cy=0.0)
        return {"partner_iou": box_iou(rb, box_b), "small_cover": 0.0}

    rb_a = reflect_box(box_a, axis="v", cx=cx, cy=0.0)
    iou = float(box_iou(rb_a, box_b))

    def _area(b: list[float]) -> float:
        return max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))

    if _area(box_a) >= _area(box_b):
        r_big, small = rb_a, box_b
    else:
        r_big = reflect_box(box_b, axis="v", cx=cx, cy=0.0)
        small = box_a
    ix0 = max(small[0], r_big[0])
    iy0 = max(small[1], r_big[1])
    ix1 = min(small[2], r_big[2])
    iy1 = min(small[3], r_big[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    cover = float(inter / _area(small))
    return {"partner_iou": iou, "small_cover": cover}


def repair_all_clusters(
    *,
    boxes: list[list[float]],
    labels: list[int],
    feats: np.ndarray,
    windows: list[dict],
    facade: Image.Image,
    axis: str,
    cx: float,
    cy: float,
    facade_extent: list[float],
    img_size: tuple[int, int],
    extent_mode: str,
    twin_mirror_min: float,
    match_iou: float,
    match_center: float = 0.35,
    allow_private: bool = False,
    size_rel_tol: float = 0.35,
    size_mismatch_sym_gain: float = 0.004,
    size_mismatch_partner_iou: float = 0.45,
    size_mismatch_contain: float = 0.85,
    max_passes: int = 3,
) -> tuple[list[int], list[str], list[Image.Image]]:
    """Greedy twin-MERGE + leftover repair over all clusters with case-1 witnesses.

    ``allow_private=False`` (default) blocks leftover unify from inventing a new
    type — prefer joining an existing twin/main cluster instead.
    ``size_rel_tol`` is slightly looser than the standalone default (0.25) so
    SAM box jitter still pairs L/R twins.

    Size-mismatch leftovers can still force-relabel when any of:
      - facade Sym gain ≥ ``size_mismatch_sym_gain``
      - reflected partner IoU ≥ ``size_mismatch_partner_iou`` (crop truncation)
      - smaller box ⊂ reflection of larger ≥ ``size_mismatch_contain``
        (pediment-vs-pane / partial detection)
    and facade Sym does not drop.
    """
    from collections import Counter

    from window_ast.symmetry import integral_symmetry_parts

    labels_w = list(labels)
    log: list[str] = []
    panels: list[Image.Image] = []
    panel_kw = {
        "axis_cx": cx,
        "axis_cy": cy,
        "facade_extent": facade_extent,
    }

    def _fits(box, cluster_boxes) -> bool:
        return box_fits_cluster_size(box, cluster_boxes, rel_tol=size_rel_tol)

    def _obj(labs: list[int]) -> tuple[float, float]:
        fs = facade_sym(
            boxes,
            labs,
            axis=axis,
            extent_mode=extent_mode,
            image_size=img_size,
        )
        return fs.normalized, fs.normalized_global

    def _twin_sim(i: int, j: int) -> float:
        a = feats[i] / (np.linalg.norm(feats[i]) + 1e-8)
        b = feats[j] / (np.linalg.norm(feats[j]) + 1e-8)
        return float(np.dot(a, b))

    s0, g0 = _obj(labels_w)
    log.append(f"ALL-CLUSTERS start  facade obj={s0:.4f} glob={g0:.4f}")
    log.append(
        f"  size_mismatch_sym_gain={size_mismatch_sym_gain:.4f}  "
        f"partner_iou≥{size_mismatch_partner_iou:.2f}  "
        f"contain≥{size_mismatch_contain:.2f}  "
        f"size_rel_tol={size_rel_tol:.2f}"
    )

    for pass_i in range(max_passes):
        ranked = rank_clusters_for_repair(
            boxes,
            labels_w,
            cx=cx,
            cy=cy,
            axis=axis,
            match_iou=match_iou,
            match_center=match_center,
        )
        if not ranked:
            log.append(f"pass {pass_i}: no clusters with case-1 witnesses")
            break
        log.append(
            f"pass {pass_i}: candidates "
            + ", ".join(f"c{c}(n1={n1},m={ms:.2f})" for c, n1, _sz, ms in ranked)
        )
        progressed = False
        seen_pair: set[tuple[int, int]] = set()

        for focus, _n1, _sz, _ms0 in ranked:
            if focus not in {int(c) for c in labels_w}:
                continue
            focus_idxs = [i for i, c in enumerate(labels_w) if c == focus]
            if not focus_idxs:
                continue

            # witnesses / dominant twin
            twin_votes: Counter[int] = Counter()
            for i in focus_idxs:
                rb = reflect_box(boxes[i], axis=axis, cx=cx, cy=cy)
                j, _ = match_reflected(
                    rb, boxes, iou_thr=match_iou, center_thr=match_center, exclude=i
                )
                if j is not None and labels_w[j] != focus:
                    twin_votes[int(labels_w[j])] += 1
            if not twin_votes:
                continue
            twin_cid, n_votes = twin_votes.most_common(1)[0]
            pair_key = (min(focus, twin_cid), max(focus, twin_cid))
            if pair_key in seen_pair:
                continue
            seen_pair.add(pair_key)

            twin_idxs = [i for i, c in enumerate(labels_w) if c == twin_cid]
            boxes_f = [boxes[i] for i in focus_idxs]
            boxes_t = [boxes[i] for i in twin_idxs]
            mscore = cluster_mirror_score(
                boxes_f, boxes_t, cx, iou_thresh=match_iou
            )

            def _split_geo_leftover(
                src_idxs: list[int], dst_cid: int
            ) -> tuple[
                list[tuple[int, int, float]],
                list[tuple[int, int, int, float]],
            ]:
                geo: list[tuple[int, int, float]] = []
                left: list[tuple[int, int, int, float]] = []
                dst_boxes = [boxes[j] for j, c in enumerate(labels_w) if c == dst_cid]
                for i in src_idxs:
                    # Prefer soft center+IoU match (SAM-friendly) over pure IoU.
                    rb = reflect_box(boxes[i], axis=axis, cx=cx, cy=cy)
                    pj, _ = match_reflected(
                        rb,
                        boxes,
                        iou_thr=match_iou,
                        center_thr=match_center,
                        exclude=i,
                    )
                    if pj is None:
                        pj, _ = find_mirror_partner_index(
                            boxes, i, cx, iou_thresh=match_iou
                        )
                    if pj is None:
                        continue
                    pc = int(labels_w[pj])
                    sim = _twin_sim(i, pj)
                    if pc == dst_cid:
                        if _fits(boxes[i], dst_boxes):
                            geo.append((i, pj, sim))
                        else:
                            left.append((i, pj, pc, sim))
                    else:
                        left.append((i, pj, pc, sim))
                return geo, left

            # Geo twins either direction (larger→smaller can look weak even
            # when the smaller side is a clear geometric twin cluster).
            geo_f, left_f = _split_geo_leftover(focus_idxs, twin_cid)
            geo_t, left_t = _split_geo_leftover(twin_idxs, focus)
            frac_f = len(geo_f) / max(len(focus_idxs), 1)
            frac_t = len(geo_t) / max(len(twin_idxs), 1)
            # Require ≥2 geo hits when the source has ≥2 members; allow 1
            # for singleton L/R twins left by shape-split.
            def _maj(geo: list, src_n: int, frac: float) -> bool:
                need = 2 if src_n >= 2 else 1
                return frac >= 0.5 and len(geo) >= need

            maj_f = _maj(geo_f, len(focus_idxs), frac_f)
            maj_t = _maj(geo_t, len(twin_idxs), frac_t)

            # Absorb the smaller cluster into the larger when either side
            # has a majority of geometric twins into the other.
            if len(focus_idxs) <= len(twin_idxs):
                src_cid, dst_cid = focus, twin_cid
                merge_geo = geo_f
                geo_frac = frac_f
            else:
                src_cid, dst_cid = twin_cid, focus
                merge_geo = geo_t
                geo_frac = frac_t
            # Leftovers from both sides (partner not in the twin pair).
            leftovers = left_f + left_t
            seen_src: set[int] = set()
            leftovers_u: list[tuple[int, int, int, float]] = []
            for item in leftovers:
                if item[0] in seen_src:
                    continue
                seen_src.add(item[0])
                leftovers_u.append(item)
            leftovers = leftovers_u

            s_cur, g_cur = _obj(labels_w)
            majority_geo = maj_f or maj_t
            log.append(
                f"  try c{focus}↔c{twin_cid}  votes={n_votes}  mirror={mscore:.3f}  "
                f"geo_f={len(geo_f)}/{len(focus_idxs)} ({frac_f:.0%})  "
                f"geo_t={len(geo_t)}/{len(twin_idxs)} ({frac_t:.0%})  "
                f"absorb c{src_cid}→c{dst_cid} leftover={len(leftovers)}"
            )

            lab_cur = list(labels_w)
            moved = [i for i, _, _ in merge_geo]
            # Strong cluster-level mirror: absorb whole src cluster even if
            # per-box size gates emptied merge_geo (common with SAM jitter).
            if not moved and mscore >= twin_mirror_min:
                moved = list(
                    focus_idxs if src_cid == focus else twin_idxs
                )
                merge_geo = [(i, -1, 0.0) for i in moved]
            merged_ok = False
            allow_merge = (mscore >= twin_mirror_min or majority_geo) and moved
            if allow_merge:
                for i, _, _ in merge_geo:
                    lab_cur[i] = dst_cid
                sm, gm = _obj(lab_cur)
                reason = (
                    f"mirror={mscore:.2f}"
                    if mscore >= twin_mirror_min
                    else f"majority_geo f={frac_f:.0%} t={frac_t:.0%}"
                )
                log.append(
                    f"    MERGE geo c{src_cid}→c{dst_cid} ({reason}): "
                    f"obj {s_cur:.4f}→{sm:.4f} glob {g_cur:.4f}→{gm:.4f}"
                )
                if sm > s_cur + 1e-6:
                    merged_ok = True
                    labels_w = lab_cur
                    s_cur, g_cur = sm, gm
                    focus_survive = dst_cid
                    panels.append(
                        draw_step_panel(
                            facade,
                            windows,
                            labels_w,
                            focus=focus_survive,
                            axis=axis,
                            title=(
                                f"MERGE c{src_cid}→c{dst_cid} n={len(moved)}  "
                                f"obj→{sm:.3f}"
                            ),
                            highlights=[
                                {
                                    "box": boxes[i],
                                    "color": (0, 255, 120),
                                    "tag": "m",
                                }
                                for i in moved[:12]
                            ],
                            **panel_kw,
                        )
                    )
                    progressed = True
                else:
                    lab_cur = list(labels_w)
                    log.append("    MERGE rejected (Sym flat)")
            else:
                log.append(
                    f"    MERGE skip (mirror={mscore:.2f} min={twin_mirror_min} "
                    f"geo={len(moved)} frac={geo_frac:.0%})"
                )
                # MERGE skipped: still let geometric twin pairs try leftover unify
                # (appearance + Sym gate), including size-mismatch force path.
                seen_src_geo = {item[0] for item in leftovers}
                for i, pj, sim in merge_geo:
                    if pj < 0 or i in seen_src_geo:
                        continue
                    leftovers.append((i, pj, int(labels_w[pj]), sim))
                    seen_src_geo.add(i)

            # leftovers relative to original focus members still needing repair
            if leftovers:
                base_labs = list(labels_w)
                # recompute leftover partners under current labels
                for i, pj, pc, sim in leftovers:
                    # if src already moved into dst via geo merge, skip
                    if labels_w[i] == dst_cid and merged_ok:
                        continue
                    pair_boxes = [boxes[i], boxes[pj]]
                    hw = 1.0 if axis == "v" else 0.0
                    wv = 0.0 if axis == "v" else 1.0
                    _, inter_split = integral_symmetry_parts(
                        pair_boxes,
                        types=[0, 1],
                        horizontal_weight=hw,
                        vertical_weight=wv,
                        n_samples=256,
                        extent=tuple(facade_extent),
                    )
                    _, inter_same = integral_symmetry_parts(
                        pair_boxes,
                        types=[0, 0],
                        horizontal_weight=hw,
                        vertical_weight=wv,
                        n_samples=256,
                        extent=tuple(facade_extent),
                    )
                    if inter_same <= inter_split + 1e-6:
                        continue
                    # Size-mismatched geometric "twins" (crop truncation,
                    # pediment-vs-pane): force-relabel when partner geometry is
                    # strong or facade Sym gain is large — absolute ΔSym alone
                    # is often tiny on busy facades.
                    size_ok = _fits(boxes[i], [boxes[pj]]) and _fits(
                        boxes[pj], [boxes[i]]
                    )
                    main_c = dst_cid if merged_ok else src_cid
                    if not size_ok:
                        geo = mirror_pair_geo_strength(
                            boxes[i], boxes[pj], cx=cx, axis=axis
                        )
                        strong_iou = (
                            geo["partner_iou"] >= size_mismatch_partner_iou
                        )
                        strong_contain = (
                            geo["small_cover"] >= size_mismatch_contain
                        )
                        cand_dests = {
                            int(pc),
                            int(main_c),
                            int(labels_w[i]),
                            int(labels_w[pj]),
                        }
                        # Rank by (Sym gain, appearance to dest cluster).
                        ranked: list[
                            tuple[float, float, float, int, list[int]]
                        ] = []
                        area_i = max(
                            1.0,
                            (boxes[i][2] - boxes[i][0])
                            * (boxes[i][3] - boxes[i][1]),
                        )
                        area_j = max(
                            1.0,
                            (boxes[pj][2] - boxes[pj][0])
                            * (boxes[pj][3] - boxes[pj][1]),
                        )
                        larger_lab = (
                            int(labels_w[i])
                            if area_i >= area_j
                            else int(labels_w[pj])
                        )
                        for dest_c in sorted(cand_dests):
                            trial = list(base_labs)
                            trial[i] = dest_c
                            trial[pj] = dest_c
                            sm, gm = _obj(trial)
                            gain = sm - s_cur
                            sim_i = cosine_to_cluster(
                                feats, base_labs, i, dest_c
                            )
                            sim_j = cosine_to_cluster(
                                feats, base_labs, pj, dest_c
                            )
                            sims = [
                                x
                                for x in (sim_i, sim_j)
                                if x is not None
                            ]
                            app = (
                                float(sum(sims) / len(sims)) if sims else -1e9
                            )
                            # Pediment-vs-pane / crop: prefer the larger box's
                            # type when Sym ties (don't absorb ornate into edge).
                            size_pref = (
                                1.0 if int(dest_c) == larger_lab else 0.0
                            )
                            ranked.append(
                                (gain, size_pref, app, dest_c, trial)
                            )
                        ranked.sort(
                            key=lambda t: (t[0], t[1], t[2]), reverse=True
                        )
                        gain, _sp, _app, dest_c, trial = ranked[0]
                        # Containment twins (pediment↔pane): lock dest to the
                        # larger box's type if it doesn't hurt Sym.
                        if strong_contain:
                            for g, _s, _a, dc, tr in ranked:
                                if dc == larger_lab and g >= -1e-6:
                                    gain, dest_c, trial = g, dc, tr
                                    break
                        gm = _obj(trial)[1]
                        allow_force = False
                        reason = ""
                        if gain >= size_mismatch_sym_gain:
                            allow_force = True
                            reason = f"Δsym={gain:+.4f}"
                        elif (strong_iou or strong_contain) and gain >= -1e-6:
                            allow_force = True
                            reason = (
                                f"geo iou={geo['partner_iou']:.2f} "
                                f"contain={geo['small_cover']:.2f} "
                                f"Δsym={gain:+.4f}"
                            )
                        if allow_force:
                            sm = s_cur + gain
                            base_labs = trial
                            labels_w = trial
                            s_cur, g_cur = sm, gm
                            progressed = True
                            log.append(
                                f"    leftover {windows[i]['id']}↔{windows[pj]['id']} "
                                f"→ force_join(c{dest_c}) "
                                f"size_mismatch ({reason}) obj→{sm:.4f}"
                            )
                            panels.append(
                                draw_step_panel(
                                    facade,
                                    windows,
                                    labels_w,
                                    focus=dest_c
                                    if dest_c in set(labels_w)
                                    else main_c,
                                    axis=axis,
                                    title=(
                                        f"leftover {windows[i]['id']}[force]  "
                                        f"obj→{sm:.3f}"
                                    ),
                                    highlights=[
                                        {
                                            "box": boxes[i],
                                            "color": (255, 160, 40),
                                            "tag": "F",
                                        },
                                        {
                                            "box": boxes[pj],
                                            "color": (255, 160, 40),
                                            "tag": "F",
                                        },
                                    ],
                                    **panel_kw,
                                )
                            )
                        else:
                            log.append(
                                f"    leftover {windows[i]['id']}↔{windows[pj]['id']} "
                                f"skip (size mismatch Δsym={gain:+.4f} "
                                f"iou={geo['partner_iou']:.2f} "
                                f"contain={geo['small_cover']:.2f})"
                            )
                        continue
                    sim_i_pc = cosine_to_cluster(feats, base_labs, i, pc)
                    sim_j_pc = cosine_to_cluster(feats, base_labs, pj, pc)
                    sim_i_main = cosine_to_cluster(feats, base_labs, i, main_c)
                    sim_j_main = cosine_to_cluster(feats, base_labs, pj, main_c)

                    def _avg(a, b):
                        vals = [x for x in (a, b) if x is not None]
                        return float(sum(vals) / len(vals)) if vals else -1e9

                    score_pc = _avg(sim_i_pc, sim_j_pc)
                    score_main = _avg(sim_i_main, sim_j_main)
                    new_c = max(base_labs) + 1
                    dest_opts = [
                        (score_pc, "join_partner", pc),
                        (score_main, "join_main", main_c),
                    ]
                    if allow_private:
                        dest_opts.append((sim, "private", new_c))
                    # Drop destinations whose size band rejects either box.
                    filtered: list[tuple[float, str, int]] = []
                    for sc, name, dc in dest_opts:
                        if name == "private":
                            filtered.append((sc, name, dc))
                            continue
                        dest_boxes = [
                            boxes[j] for j, c in enumerate(base_labs) if int(c) == dc
                        ]
                        if not dest_boxes:
                            continue
                        if _fits(boxes[i], dest_boxes) and (
                            _fits(boxes[pj], dest_boxes)
                            or int(base_labs[pj]) == dc
                        ):
                            filtered.append((sc, name, dc))
                    if not filtered:
                        if allow_private:
                            filtered = [(sim, "private", new_c)]
                        else:
                            log.append(
                                f"    leftover {windows[i]['id']}↔{windows[pj]['id']} "
                                f"skip (no safe existing dest)"
                            )
                            continue
                    filtered.sort(key=lambda t: t[0], reverse=True)
                    _sc, dest_name, dest_c = filtered[0]
                    trial = list(base_labs)
                    trial[i] = dest_c
                    trial[pj] = dest_c
                    sm, gm = _obj(trial)
                    # accept leftover unify if facade Sym does not collapse
                    # (pair Sym is label-indifferent; allow small dips)
                    if sm + 1e-4 >= s_cur:
                        base_labs = trial
                        labels_w = trial
                        s_cur, g_cur = sm, gm
                        progressed = True
                        log.append(
                            f"    leftover {windows[i]['id']}↔{windows[pj]['id']} "
                            f"→ {dest_name}(c{dest_c}) obj→{sm:.4f}"
                        )
                        panels.append(
                            draw_step_panel(
                                facade,
                                windows,
                                labels_w,
                                focus=dest_c if dest_c in set(labels_w) else main_c,
                                axis=axis,
                                title=(
                                    f"leftover {windows[i]['id']}[{dest_name}]  "
                                    f"obj→{sm:.3f}"
                                ),
                                highlights=[
                                    {
                                        "box": boxes[i],
                                        "color": (0, 200, 255),
                                        "tag": "L",
                                    },
                                    {
                                        "box": boxes[pj],
                                        "color": (0, 200, 255),
                                        "tag": "L",
                                    },
                                ],
                                **panel_kw,
                            )
                        )
                    else:
                        log.append(
                            f"    leftover {windows[i]['id']} unify→{dest_name} "
                            f"rejected (obj {s_cur:.4f}→{sm:.4f})"
                        )

        if not progressed:
            log.append(f"pass {pass_i}: no further progress")
            break

    s1, g1 = _obj(labels_w)
    log.append(f"ALL-CLUSTERS done  facade obj={s0:.4f}→{s1:.4f}  glob={g0:.4f}→{g1:.4f}")
    return labels_w, log, panels


def main() -> None:
    args = parse_args()
    stem = args.stem
    facade, raw_windows = load_base_facade(args.base_dir, stem)

    pre_panel: Image.Image | None = None
    feats: np.ndarray | None = None
    if args.merge_first:
        windows, labels, feats, meta, pre_panel = build_merged_units(
            facade_id=stem,
            facade=facade,
            raw_windows=raw_windows,
            device=args.device,
            dino=args.dino,
            cluster_feat=args.cluster_feat,
            facade_max_side=args.facade_max_side,
            pca_dim=args.pca_dim,
            k_max=args.k_max,
            seed=args.seed,
            col_tol=args.col_tol,
            row_tol=args.row_tol,
            adj_gap=args.adj_gap,
            merge_vertical=args.merge_vertical,
            vert_gap=args.vert_gap,
            shape_split=args.shape_split,
            shape_gap=args.shape_gap,
        )
        tag = f"merged_{args.cluster_feat}"
    else:
        import torch

        patch = _load_script(ROOT / "scripts" / "overlay_facade_patch_layout.py", "patch_raw")
        base = _load_script(ROOT / "scripts" / "overlay_facade_asset_clusters.py", "base_raw")
        boxes_raw = [list(map(float, w["box_xyxy"])) for w in raw_windows]
        int_boxes = [[int(round(v)) for v in b] for b in boxes_raw]
        print(f"loading {args.dino}…")
        model = torch.hub.load("facebookresearch/dinov2", args.dino, pretrained=True)
        model = model.to(args.device).eval()
        spatial, pmeta = patch.facade_patch_spatial(
            model, facade, device=torch.device(args.device), max_side=args.facade_max_side
        )
        feats = base.apply_pca(
            patch.roi_pool_patches(spatial, int_boxes, pmeta),
            min(args.pca_dim, max(2, len(boxes_raw) - 1)),
            args.seed,
        )
        k = base.select_k(feats, "spectral_rbf", min(args.k_max, len(boxes_raw) - 1), args.seed)
        labels_np = base.cluster_features(feats, "spectral_rbf", k, args.seed)
        labels = [int(x) for x in labels_np.tolist()]
        windows = []
        for i, w in enumerate(raw_windows):
            windows.append(
                {
                    "id": w["id"],
                    "box_xyxy": w["box_xyxy"],
                    "cluster": labels[i],
                }
            )
        meta = {
            "k": int(k),
            "n_raw": len(windows),
            "n_merged": len(windows),
            "merge_first": False,
            "source": "facades/base",
        }
        tag = "raw"

    boxes = [list(map(float, w["box_xyxy"])) for w in windows]
    axis = "v" if args.axis in ("v", "both") else "h"
    img_size = facade.size
    assert feats is not None

    if args.sym_scope == "facade":
        cx, cy, facade_extent = facade_axis_center(
            boxes, extent_mode=args.extent, image_size=img_size
        )
        panel_axis_kw = {
            "axis_cx": cx,
            "axis_cy": cy,
            "facade_extent": facade_extent,
        }
    else:
        cx = cy = 0.0
        facade_extent = list(bounding_box(boxes) or (0, 0, *img_size))
        panel_axis_kw = {
            "axis_cx": 0.5 * (facade_extent[0] + facade_extent[2]),
            "axis_cy": 0.5 * (facade_extent[1] + facade_extent[3]),
            "facade_extent": facade_extent,
        }
        cx, cy = panel_axis_kw["axis_cx"], panel_axis_kw["axis_cy"]

    # ----- all-cluster lineart/dino repair path -----
    if args.all_clusters:
        labels_split = list(labels)
        boxes_all = [list(b) for b in boxes]
        labels_spectral = list(meta.get("labels_spectral") or labels_split)
        if len(labels_spectral) != len(boxes_all):
            labels_spectral = list(labels_split)

        fsym_spec = facade_sym(
            boxes_all,
            labels_spectral,
            axis=axis,
            extent_mode=args.extent if args.sym_scope == "facade" else "boxes",
            image_size=img_size,
        )
        fsym_split = facade_sym(
            boxes_all,
            labels_split,
            axis=axis,
            extent_mode=args.extent if args.sym_scope == "facade" else "boxes",
            image_size=img_size,
        )
        labels_after, log, repair_panels = repair_all_clusters(
            boxes=boxes,
            labels=labels_split,
            feats=feats,
            windows=windows,
            facade=facade,
            axis=axis,
            cx=cx,
            cy=cy,
            facade_extent=list(facade_extent),
            img_size=img_size,
            extent_mode=args.extent if args.sym_scope == "facade" else "boxes",
            twin_mirror_min=args.twin_mirror_min,
            match_iou=args.match_iou,
            match_center=args.match_center,
            allow_private=False,
        )
        fsym_repair = facade_sym(
            boxes,
            labels_after,
            axis=axis,
            extent_mode=args.extent if args.sym_scope == "facade" else "boxes",
            image_size=img_size,
        )
        out_dir = args.out_dir / f"{stem}_{tag}" / "all_clusters"
        out_dir.mkdir(parents=True, exist_ok=True)

        from scripts.overlay_facade_asset_clusters import draw_overlay as _dov

        def _sum_panel(labs: list[int], title: str) -> Image.Image:
            items = [{"box_xyxy": b} for b in boxes_all]
            ov = _dov(facade, items, np.asarray(labs, dtype=np.int32), title=title)
            bar_h = 36
            ImageDraw.Draw(ov).line(
                [(cx, bar_h), (cx, ov.height)], fill=(0, 220, 255), width=3
            )
            w = 720
            nh = max(1, int(ov.height * w / ov.width))
            return ov.resize((w, nh), Image.Resampling.LANCZOS)

        k_spec = len(set(labels_spectral))
        k_split = len(set(labels_split))
        k_rep = len(set(labels_after))
        step1 = _sum_panel(
            labels_spectral,
            f"1/3  {args.cluster_feat} spectral  k={k_spec}  "
            f"Sym={fsym_spec.normalized:.3f}",
        )
        step2 = _sum_panel(
            labels_split,
            f"2/3  shape-split (step/facade≥{args.shape_gap})  k={k_spec}→{k_split}  "
            f"Sym={fsym_split.normalized:.3f}",
        )
        step3 = _sum_panel(
            labels_after,
            f"3/3  symmetry repair  k={k_split}→{k_rep}  "
            f"Sym={fsym_split.normalized:.3f}→{fsym_repair.normalized:.3f}",
        )

        # step-by-step progress page
        steps = [step1, step2, step3]
        gap = 10
        W = max(p.width for p in steps)
        H = sum(p.height for p in steps) + gap * (len(steps) + 1)
        progress = Image.new("RGB", (W + 20, H), (8, 8, 10))
        y = gap
        for p in steps:
            progress.paste(p, ((W - p.width) // 2 + 10, y))
            y += p.height + gap
        progress_path = out_dir / "steps.png"
        progress.save(progress_path)

        # final: spectral → shape-split → repair (same three stages as steps.png)
        summary = out_dir / "final_before_after.png"
        progress.save(summary)

        # detailed walkthrough: unitize pre + three stages + repair panels
        panels: list[Image.Image] = []
        if pre_panel is not None:
            panels.append(pre_panel)
        panels.extend(steps)
        panels.extend(repair_panels)

        body = panels[1:] if pre_panel is not None else panels
        target_w = max((p.width for p in body), default=720)
        scaled: list[Image.Image] = []
        for p in panels:
            if p.width > target_w + 40:
                nh = max(1, int(p.height * target_w / p.width))
                scaled.append(p.resize((target_w, nh), Image.Resampling.LANCZOS))
            else:
                scaled.append(p)
        panels = scaled
        W = max(p.width for p in panels)
        H = sum(p.height for p in panels) + gap * (len(panels) + 1)
        page = Image.new("RGB", (W + 20, H), (8, 8, 10))
        y = gap
        for p in panels:
            page.paste(p, ((W - p.width) // 2 + 10, y))
            y += p.height + gap
        dest = out_dir / "walkthrough.png"
        page.save(dest)

        (out_dir / "log.txt").write_text("\n".join(log) + "\n")
        (out_dir / "result_labels.json").write_text(
            json.dumps(
                {
                    "stem": stem,
                    "cluster_feat": args.cluster_feat,
                    "all_clusters": True,
                    "meta": meta,
                    "sym_spectral": {
                        "total": fsym_spec.normalized,
                        "global": fsym_spec.normalized_global,
                        "in_box": fsym_spec.normalized_in_box,
                    },
                    "sym_split": {
                        "total": fsym_split.normalized,
                        "global": fsym_split.normalized_global,
                        "in_box": fsym_split.normalized_in_box,
                    },
                    "sym_repair": {
                        "total": fsym_repair.normalized,
                        "global": fsym_repair.normalized_global,
                        "in_box": fsym_repair.normalized_in_box,
                    },
                    "labels_spectral": labels_spectral,
                    "labels_split": labels_split,
                    "labels_after": labels_after,
                    "log": log,
                },
                indent=2,
            )
            + "\n"
        )
        print("\n".join(log))
        print(f"\nwrote {progress_path}")
        print(f"wrote {summary}")
        print(f"wrote {dest}")
        return

    # ----- single-focus walkthrough (legacy) -----
    if args.cluster is not None:
        focus = int(args.cluster)
    elif args.sym_scope == "facade":
        focus = pick_focus_cluster(windows, axis, cx=cx, cy=cy)
    else:
        best = None
        for cid in sorted({int(c) for c in labels}):
            idxs = [i for i, c in enumerate(labels) if c == cid]
            if len(idxs) < 2:
                continue
            bb0 = bounding_box([boxes[i] for i in idxs])
            if bb0 is None:
                continue
            cx0 = 0.5 * (bb0[0] + bb0[2])
            cy0 = 0.5 * (bb0[1] + bb0[3])
            n1 = n2 = 0
            for ii in idxs:
                rb = reflect_box(boxes[ii], axis=axis, cx=cx0, cy=cy0)
                jj, _ = match_reflected(rb, boxes, iou_thr=0.15, center_thr=0.35, exclude=ii)
                if jj is None:
                    n2 += 1
                elif labels[jj] != cid:
                    n1 += 1
            key = (n1, len(idxs), -n2, cid)
            if best is None or key > best:
                best = key
        focus = int(best[3]) if best is not None else int(labels[0])

    out_dir = args.out_dir / f"{stem}_{tag}" / f"cluster_{focus}"
    out_dir.mkdir(parents=True, exist_ok=True)

    focus_idxs = [i for i, c in enumerate(labels) if c == focus]
    focus_boxes = [boxes[i] for i in focus_idxs]
    bb = bounding_box(focus_boxes)
    assert bb is not None

    if args.sym_scope == "cluster":
        cx, cy = 0.5 * (bb[0] + bb[2]), 0.5 * (bb[1] + bb[3])
        facade_extent = list(bb)
        panel_axis_kw = {
            "axis_cx": cx,
            "axis_cy": cy,
            "facade_extent": facade_extent,
        }

    s_local0 = cluster_local_sym(focus_boxes, axis=axis)
    fsym0 = facade_sym(
        boxes,
        labels,
        axis=axis,
        extent_mode=args.extent if args.sym_scope == "facade" else "boxes",
        image_size=img_size,
    )
    # objective used for accept/reject
    if args.sym_scope == "facade":
        s0 = fsym0.normalized
        # prefer moves that raise global (inter) pairs; in-box is mostly fixed
        s0_global = fsym0.normalized_global
        s0_inbox = fsym0.normalized_in_box
    else:
        s0 = s_local0
        s0_global = s0_inbox = s_local0

    def score_labels(labs: list[int]) -> tuple[float, float, float, float]:
        """Return (objective, local_C, facade_norm, facade_global_norm)."""
        fb = [boxes[t] for t, c in enumerate(labs) if c == focus]
        local = cluster_local_sym(fb, axis=axis) if fb else 0.0
        fs = facade_sym(
            boxes,
            labs,
            axis=axis,
            extent_mode=args.extent if args.sym_scope == "facade" else "boxes",
            image_size=img_size,
        )
        if args.sym_scope == "facade":
            return fs.normalized, local, fs.normalized, fs.normalized_global
        return local, local, fs.normalized, fs.normalized_global

    # --- find witnesses about facade (or cluster) axis ---
    witnesses: list[dict[str, Any]] = []
    for i in focus_idxs:
        rb = reflect_box(boxes[i], axis=axis, cx=cx, cy=cy)
        j, sc = match_reflected(
            rb, boxes, iou_thr=args.match_iou, center_thr=args.match_center, exclude=i
        )
        if j is None:
            witnesses.append(
                {
                    "src": i,
                    "case": 2,
                    "reflected": rb,
                    "note": "empty at symmetry position",
                }
            )
        elif labels[j] != focus:
            witnesses.append(
                {
                    "src": i,
                    "case": 1,
                    "twin": j,
                    "twin_cluster": labels[j],
                    "reflected": rb,
                    "match": sc,
                    "note": f"twin in c{labels[j]}",
                }
            )

    case1 = [w for w in witnesses if w["case"] == 1]
    case2 = [w for w in witnesses if w["case"] == 2]

    log: list[str] = []
    log.append(
        f"stem={stem}  source=facades/base  merge_first={args.merge_first}  "
        f"raw={meta.get('n_raw')} → units={meta.get('n_merged')}  k={meta.get('k')}  "
        f"focus=c{focus}  axis={axis}  sym_scope={args.sym_scope}  extent={args.extent}"
    )
    log.append(
        f"cluster size={len(focus_idxs)}  clusterAABB={bb}  "
        f"reflect_center=({cx:.1f},{cy:.1f})  facade_extent={facade_extent}"
    )
    log.append(
        f"Sym local(C)={s_local0:.4f}  |  facade: in-box={fsym0.normalized_in_box:.4f}  "
        f"global(inter)={fsym0.normalized_global:.4f}  total={fsym0.normalized:.4f}  "
        f"(I_in={fsym0.in_box:.1f} I_glob={fsym0.global_pairs:.1f} I_blank={fsym0.blank:.1f})"
    )
    log.append(f"objective ({args.sym_scope}) before = {s0:.4f}")
    log.append(f"witnesses: case1={len(case1)}  case2={len(case2)}")
    for w in case1 + case2:
        log.append(
            f"  src={windows[w['src']]['id']} case={w['case']} {w['note']}"
            + (f" twin={windows[w['twin']]['id']}" if "twin" in w else "")
        )

    panels: list[Image.Image] = []
    if pre_panel is not None:
        panels.append(pre_panel)

    panels.append(
        draw_step_panel(
            facade,
            windows,
            labels,
            focus=focus,
            axis=axis,
            title=(
                f"Step 0 · spectral  focus=c{focus}  "
                f"facade Sym={fsym0.normalized:.3f} "
                f"(in={fsym0.normalized_in_box:.3f} glob={fsym0.normalized_global:.3f})  "
                f"local={s_local0:.3f}"
            ),
            **panel_axis_kw,
        )
    )

    hl = []
    for i in focus_idxs:
        rb = reflect_box(boxes[i], axis=axis, cx=cx, cy=cy)
        hl.append({"box": boxes[i], "reflected": rb, "color": (0, 255, 120), "tag": "b"})
    axis_name = "facade center" if args.sym_scope == "facade" else "cluster center"
    panels.append(
        draw_step_panel(
            facade,
            windows,
            labels,
            focus=focus,
            axis=axis,
            title=f"Step 1 · reflect c{focus} about {axis_name}",
            highlights=hl,
            **panel_axis_kw,
        )
    )

    hl2 = []
    for w in case1:
        hl2.append(
            {
                "box": boxes[w["src"]],
                "reflected": w["reflected"],
                "color": (255, 60, 60),
                "tag": "case1 src",
            }
        )
        hl2.append(
            {
                "box": boxes[w["twin"]],
                "color": (80, 160, 255),
                "tag": f"twin c{w['twin_cluster']}",
            }
        )
    for w in case2:
        hl2.append(
            {
                "box": boxes[w["src"]],
                "reflected": w["reflected"],
                "color": (255, 140, 0),
                "tag": "case2 empty",
            }
        )
    panels.append(
        draw_step_panel(
            facade,
            windows,
            labels,
            focus=focus,
            axis=axis,
            title=f"Step 2 · witnesses  case1={len(case1)}  case2={len(case2)}",
            highlights=hl2,
            **panel_axis_kw,
        )
    )

    # Step 3: twin-MERGE (twin-only appearance) → leftover Sym repair
    new_labels = list(labels)
    chosen = None
    assert feats is not None
    obj_before = s0
    glob_before = s0_global

    twin_cid = None
    if args.twin_merge and case1:
        from collections import Counter

        twin_votes = Counter(int(w["twin_cluster"]) for w in case1)
        twin_cid, n_votes = twin_votes.most_common(1)[0]
        boxes_f = [boxes[i] for i in focus_idxs]
        boxes_t = [boxes[i] for i, c in enumerate(labels) if c == twin_cid]
        mscore = cluster_mirror_score(
            boxes_f, boxes_t, cx, iou_thresh=args.match_iou
        )

        def _twin_sim(i: int, j: int) -> float:
            a = feats[i] / (np.linalg.norm(feats[i]) + 1e-8)
            b = feats[j] / (np.linalg.norm(feats[j]) + 1e-8)
            return float(np.dot(a, b))

        def _obj(labs: list[int]) -> tuple[float, float, float]:
            fs = facade_sym(
                boxes,
                labs,
                axis=axis,
                extent_mode=args.extent if args.sym_scope == "facade" else "boxes",
                image_size=img_size,
            )
            if args.sym_scope == "facade":
                return fs.normalized, fs.normalized_global, fs.normalized_in_box
            union = [boxes[t] for t, c in enumerate(labs) if c in (focus, twin_cid)]
            return (
                cluster_local_sym(union, axis=axis),
                fs.normalized_global,
                fs.normalized_in_box,
            )

        # Partition by geometric partner only (no appearance gate on merge)
        merge_geo: list[tuple[int, int, float]] = []  # (src, twin, sim) — logged only
        leftovers: list[tuple[int, int, int, float]] = []  # src, partner, pc, sim
        unmatched: list[int] = []
        for i in focus_idxs:
            pj, _ = find_mirror_partner_index(
                boxes, i, cx, iou_thresh=args.match_iou
            )
            if pj is None:
                unmatched.append(i)
                continue
            pc = int(labels[pj])
            sim = _twin_sim(i, pj)
            if pc == twin_cid:
                merge_geo.append((i, pj, sim))
            else:
                leftovers.append((i, pj, pc, sim))

        log.append("")
        log.append(
            f"Step 3 twin-MERGE (geometry→all, then Sym): focus=c{focus} ↔ c{twin_cid}  "
            f"votes={n_votes}/{len(case1)}  mirror={mscore:.3f}  "
            f"geo_twins={len(merge_geo)}  leftovers={len(leftovers)}  "
            f"unmatched={len(unmatched)}"
        )
        for i, pj, sim in merge_geo:
            log.append(
                f"  geo    {windows[i]['id']}↔{windows[pj]['id']}(c{twin_cid})  "
                f"twin_sim={sim:.3f}"
            )
        for i, pj, pc, sim in leftovers:
            log.append(
                f"  leftover {windows[i]['id']}↔{windows[pj]['id']}(c{pc})  "
                f"twin_sim={sim:.3f}"
            )

        # --- primary MERGE: all geometric twins → twin_cid; accept if Sym improves ---
        lab_cur = list(labels)
        moved = [i for i, _, _ in merge_geo]
        if mscore >= args.twin_mirror_min and moved:
            for i, _, _ in merge_geo:
                lab_cur[i] = twin_cid
            sm, gm, _ = _obj(lab_cur)
            log.append(
                f"  MERGE all geo → c{twin_cid}: obj {s0:.4f}→{sm:.4f}  "
                f"glob {s0_global:.4f}→{gm:.4f}"
            )
            if sm > s0 + 1e-6:
                panels.append(
                    draw_step_panel(
                        facade,
                        windows,
                        lab_cur,
                        focus=twin_cid,
                        axis=axis,
                        title=(
                            f"Step 3a · MERGE all geo c{focus}→c{twin_cid}  "
                            f"n={len(moved)}  obj {s0:.3f}→{sm:.3f}"
                        ),
                        highlights=[
                            {"box": boxes[i], "color": (0, 255, 120), "tag": "merged"}
                            for i in moved[:16]
                        ]
                        + [
                            {
                                "box": boxes[i],
                                "color": (255, 140, 0),
                                "tag": "leftover",
                            }
                            for i, _, _, _ in leftovers
                        ],
                        **panel_axis_kw,
                    )
                )
                s0, s0_global = sm, gm
                focus = twin_cid
                chosen = ("merge", list(lab_cur), sm, moved, twin_cid)
            else:
                log.append("  MERGE rejected (Sym did not improve) — revert")
                lab_cur = list(labels)
        elif mscore < args.twin_mirror_min:
            log.append(
                f"  MERGE skipped: mirror {mscore:.2f}<{args.twin_mirror_min}"
            )
        else:
            log.append("  MERGE skipped: no geometric twins to dominant cluster")

        # --- leftover repair: Sym only (appearance logged, not gating) ---
        if leftovers:
            # start from post-merge labeling (or original if merge failed)
            base_labs = list(lab_cur)
            s_base, g_base, _ = _obj(base_labs)

            # group leftovers by partner cluster
            by_pc: dict[int, list[tuple[int, int, float]]] = {}
            for i, pj, pc, sim in leftovers:
                by_pc.setdefault(pc, []).append((i, pj, sim))

            log.append("")
            log.append(
                f"Step 3b leftover Sym repair (from obj={s_base:.4f} glob={g_base:.4f}):"
            )

            final_labs = list(base_labs)
            leftover_moved: list[int] = []
            for pc, group in sorted(by_pc.items()):
                # Per leftover pair:
                #   Twin-pair global Sym is label-agnostic (c3 vs c1 indifferent).
                #   Decide unify vs leave from the PAIR's I_inter only; pick
                #   destination cluster by appearance.
                from window_ast.symmetry import integral_symmetry_parts

                for i, pj, sim in group:
                    pair_boxes = [boxes[i], boxes[pj]]
                    _, inter_split = integral_symmetry_parts(
                        pair_boxes,
                        types=[0, 1],
                        horizontal_weight=1.0 if axis == "v" else 0.0,
                        vertical_weight=0.0 if axis == "v" else 1.0,
                        n_samples=256,
                        extent=tuple(facade_extent) if facade_extent else None,
                    )
                    _, inter_same = integral_symmetry_parts(
                        pair_boxes,
                        types=[0, 0],
                        horizontal_weight=1.0 if axis == "v" else 0.0,
                        vertical_weight=0.0 if axis == "v" else 1.0,
                        n_samples=256,
                        extent=tuple(facade_extent) if facade_extent else None,
                    )
                    unify = inter_same > inter_split + 1e-6

                    log.append(
                        f"  leftover {windows[i]['id']}↔{windows[pj]['id']}(c{pc})  "
                        f"twin_sim={sim:.3f}  "
                        f"pair_I_inter split={inter_split:.0f} same={inter_same:.0f}  "
                        f"unify={unify}"
                    )

                    if not unify:
                        # optional drop if it helps full facade (src alone)
                        lab_drop = list(base_labs)
                        lab_drop[i] = max(lab_drop) + 1
                        s_drop, g_drop, _ = _obj(lab_drop)
                        s_leave, _, _ = _obj(base_labs)
                        if s_drop > s_leave + 1e-6:
                            final_labs = lab_drop
                            leftover_moved.append(i)
                            base_labs = list(final_labs)
                            s_base, g_base = s_drop, g_drop
                            log.append(f"    → drop_src obj {s_leave:.4f}→{s_drop:.4f}")
                            panels.append(
                                draw_step_panel(
                                    facade,
                                    windows,
                                    final_labs,
                                    focus=focus,
                                    axis=axis,
                                    title=f"Step 3b · {windows[i]['id']}[drop]  obj→{s_drop:.3f}",
                                    highlights=[
                                        {
                                            "box": boxes[i],
                                            "color": (255, 140, 0),
                                            "tag": "drop",
                                        }
                                    ],
                                    **panel_axis_kw,
                                )
                            )
                        else:
                            log.append("    → leave")
                        continue

                    # Destinations that share a label for the twin pair.
                    # partner_cid vs main twin cluster are Sym-indifferent for the pair;
                    # appearance breaks the tie. Skip depleted remnant src (e.g. c6).
                    main_c = int(focus)  # surviving twin cluster after geo MERGE
                    sim_i_pc = cosine_to_cluster(feats, base_labs, i, pc)
                    sim_j_pc = cosine_to_cluster(feats, base_labs, pj, pc)
                    sim_i_main = cosine_to_cluster(feats, base_labs, i, main_c)
                    sim_j_main = cosine_to_cluster(feats, base_labs, pj, main_c)

                    def _avg(a, b):
                        vals = [x for x in (a, b) if x is not None]
                        return float(sum(vals) / len(vals)) if vals else -1e9

                    score_pc = _avg(sim_i_pc, sim_j_pc)
                    score_main = _avg(sim_i_main, sim_j_main)
                    score_priv = sim
                    new_c = max(base_labs) + 1
                    dest_opts = [
                        (score_pc, "join_partner", pc),
                        (score_main, "join_main_twin", main_c),
                        (score_priv, "keep_private", new_c),
                    ]
                    dest_opts.sort(key=lambda t: t[0], reverse=True)
                    _score_d, dest_name, dest_c = dest_opts[0]

                    labs = list(base_labs)
                    labs[i] = dest_c
                    labs[pj] = dest_c
                    sm, gm, _ = _obj(labs)
                    log.append(
                        f"    dest appearance: partner={score_pc:.3f}  "
                        f"main_twin={score_main:.3f}  private={score_priv:.3f}  "
                        f"→ {dest_name}(c{dest_c})  facade_obj={sm:.4f} "
                        f"(pair Sym indifferent to label)"
                    )
                    final_labs = labs
                    leftover_moved.extend([i, pj])
                    base_labs = list(final_labs)
                    s_base, g_base = sm, gm
                    panels.append(
                        draw_step_panel(
                            facade,
                            windows,
                            final_labs,
                            focus=focus,
                            axis=axis,
                            title=(
                                f"Step 3b · {windows[i]['id']}[{dest_name}]  "
                                f"obj→{sm:.3f}"
                            ),
                            highlights=[
                                {
                                    "box": boxes[i],
                                    "color": (0, 200, 255),
                                    "tag": dest_name[:8],
                                },
                                {
                                    "box": boxes[pj],
                                    "color": (0, 200, 255),
                                    "tag": dest_name[:8],
                                },
                            ],
                            **panel_axis_kw,
                        )
                    )

            # if leftover repair changed labels beyond primary merge, update chosen
            if leftover_moved or (chosen is not None and final_labs != list(lab_cur)):
                sm, gm, _ = _obj(final_labs)
                all_moved = list(dict.fromkeys((moved if chosen else []) + leftover_moved))
                chosen = ("merge", final_labs, sm, all_moved, focus)
                s0, s0_global = sm, gm
                log.append(
                    f"  leftover repair done: obj→{sm:.4f} glob→{gm:.4f}"
                )

    if chosen is None and case1:
        w = case1[0]
        i, j = w["src"], w["twin"]
        dest = labels[j]
        lab_steal = list(labels)
        lab_steal[j] = focus
        s_steal, local_steal, fac_steal, glob_steal = score_labels(lab_steal)
        ok_steal, sim_twin_dest, sim_twin_focus = sim_prefers_dest(
            feats, labels, j, dest, focus, margin=args.sim_margin
        )
        lab_join = list(labels)
        lab_join[i] = dest
        s_join, local_join, fac_join, glob_join = score_labels(lab_join)
        ok_join, sim_src_focus, sim_src_dest = sim_prefers_dest(
            feats, labels, i, focus, dest, margin=args.sim_margin
        )

        def _fmt(a: float | None) -> str:
            return "n/a" if a is None else f"{a:.3f}"

        log.append("")
        log.append(f"Step 3 trials on src={windows[i]['id']} twin={windows[j]['id']} (c{dest}):")
        log.append(
            f"  STEAL twin→c{focus}:  obj {s0:.4f} → {s_steal:.4f}  Δ={s_steal - s0:+.4f}  "
            f"local {s_local0:.4f}→{local_steal:.4f}  "
            f"facade {fsym0.normalized:.4f}→{fac_steal:.4f} "
            f"(glob {s0_global:.4f}→{glob_steal:.4f})  "
            f"sim(twin|c{dest})={_fmt(sim_twin_dest)} sim(twin|c{focus})={_fmt(sim_twin_focus)}  "
            f"sim_ok={ok_steal}"
        )
        log.append(
            f"  JOIN  src→c{dest}:   obj {s0:.4f} → {s_join:.4f}  Δ={s_join - s0:+.4f}  "
            f"local {s_local0:.4f}→{local_join:.4f}  "
            f"facade {fsym0.normalized:.4f}→{fac_join:.4f} "
            f"(glob {s0_global:.4f}→{glob_join:.4f})  "
            f"sim(src|c{focus})={_fmt(sim_src_focus)} sim(src|c{dest})={_fmt(sim_src_dest)}  "
            f"sim_ok={ok_join}"
        )

        if s_steal > s0 + 1e-6 and ok_steal:
            chosen = ("steal", lab_steal, s_steal, i, j)
        elif s_join > s0 + 1e-6 and ok_join:
            chosen = ("join", lab_join, s_join, i, j)
        else:
            reasons = []
            if not (s_steal > s0 + 1e-6) and not (s_join > s0 + 1e-6):
                reasons.append("obj flat/worse")
            if s_steal > s0 + 1e-6 and not ok_steal:
                reasons.append("STEAL blocked by sim")
            if s_join > s0 + 1e-6 and not ok_join:
                reasons.append("JOIN blocked by sim")
            log.append(f"  REJECT: {', '.join(reasons) or 'no viable move'}")
            chosen = None

        panels.append(
            draw_step_panel(
                facade,
                windows,
                lab_steal,
                focus=focus,
                axis=axis,
                title=(
                    f"Step 3a · STEAL  obj {s0:.3f}→{s_steal:.3f}  "
                    f"glob {s0_global:.3f}→{glob_steal:.3f}  sim_ok={ok_steal}"
                ),
                highlights=[
                    {"box": boxes[i], "color": (255, 60, 60), "tag": "src"},
                    {"box": boxes[j], "color": (0, 255, 120), "tag": "stolen→c"},
                ],
                **panel_axis_kw,
            )
        )
        panels.append(
            draw_step_panel(
                facade,
                windows,
                lab_join,
                focus=focus,
                axis=axis,
                title=(
                    f"Step 3b · JOIN  obj {s0:.3f}→{s_join:.3f}  "
                    f"glob {s0_global:.3f}→{glob_join:.3f}  sim_ok={ok_join}"
                ),
                highlights=[
                    {"box": boxes[i], "color": (80, 160, 255), "tag": f"→c{dest}"},
                    {"box": boxes[j], "color": (255, 60, 60), "tag": "twin"},
                ],
                **panel_axis_kw,
            )
        )
    elif chosen is None and case2:
        w = case2[0]
        i = w["src"]
        lab_drop = list(labels)
        new_c = max(labels) + 1
        lab_drop[i] = new_c
        s_drop, local_drop, fac_drop, glob_drop = score_labels(lab_drop)
        log.append("")
        log.append(
            f"Step 3 case2 DROP src={windows[i]['id']}: "
            f"obj {s0:.4f} → {s_drop:.4f}  local {s_local0:.4f}→{local_drop:.4f}  "
            f"facade {fsym0.normalized:.4f}→{fac_drop:.4f}"
        )
        panels.append(
            draw_step_panel(
                facade,
                windows,
                lab_drop,
                focus=focus,
                axis=axis,
                title=f"Step 3 · DROP  obj {s0:.3f}→{s_drop:.3f}",
                highlights=[
                    {
                        "box": boxes[i],
                        "reflected": w["reflected"],
                        "color": (255, 140, 0),
                        "tag": "dropped",
                    }
                ],
                **panel_axis_kw,
            )
        )
        if s_drop > s0 + 1e-6:
            chosen = ("drop", lab_drop, s_drop, i, None)
        else:
            log.append("  DROP rejected (obj did not improve)")
            chosen = None

    if chosen:
        kind, new_labels, s_new, i, j = chosen
        log.append("")
        log.append(f"Step 4 ACCEPT: {kind}  obj {obj_before:.4f} → {s_new:.4f}")
        if kind == "merge":
            merge_idxs = i  # list of indices
            survive = j
            hl = [
                {"box": boxes[t], "color": (0, 255, 120), "tag": "merged"}
                for t in merge_idxs[:16]
            ]
            focus = int(survive)
        else:
            hl = [{"box": boxes[i], "color": (255, 60, 60), "tag": "src"}]
            if j is not None:
                hl.append({"box": boxes[j], "color": (0, 255, 120), "tag": "twin"})
        panels.append(
            draw_step_panel(
                facade,
                windows,
                new_labels,
                focus=focus,
                axis=axis,
                title=f"Step 4 · after {kind}  obj={s_new:.3f}",
                highlights=hl,
                **panel_axis_kw,
            )
        )

        if kind != "merge":
            labels2 = list(new_labels)
            focus_idxs2 = [t for t, c in enumerate(labels2) if c == focus]
            s_cur, _, _, _ = score_labels(labels2)
            improved = False
            for i2 in focus_idxs2:
                rb = reflect_box(boxes[i2], axis=axis, cx=cx, cy=cy)
                j2, _ = match_reflected(
                    rb, boxes, iou_thr=args.match_iou, center_thr=args.match_center, exclude=i2
                )
                if j2 is None or labels2[j2] == focus:
                    continue
                ok2, _, _ = sim_prefers_dest(
                    feats, labels2, j2, labels2[j2], focus, margin=args.sim_margin
                )
                if not ok2:
                    continue
                trial = list(labels2)
                trial[j2] = focus
                s_t, _, _, _ = score_labels(trial)
                if s_t > s_cur + 1e-6:
                    labels2 = trial
                    s_cur = s_t
                    improved = True
                    log.append(
                        f"Step 5 greedy STEAL {windows[j2]['id']}→c{focus}  obj→{s_cur:.4f}"
                    )
                    break
            if improved:
                panels.append(
                    draw_step_panel(
                        facade,
                        windows,
                        labels2,
                        focus=focus,
                        axis=axis,
                        title=f"Step 5 · second greedy steal  obj={s_cur:.3f}",
                        **panel_axis_kw,
                    )
                )
                new_labels = labels2
        else:
            new_labels = list(new_labels)

    # assemble page
    gap = 10
    body = [p for p in panels if p is not pre_panel]
    target_w = max((p.width for p in body), default=520)
    scaled: list[Image.Image] = []
    for p in panels:
        if p.width > target_w + 40:
            nh = max(1, int(p.height * target_w / p.width))
            scaled.append(p.resize((target_w, nh), Image.Resampling.LANCZOS))
        else:
            scaled.append(p)
    panels = scaled

    W = max(p.width for p in panels)
    H = sum(p.height for p in panels) + gap * (len(panels) + 1)
    page = Image.new("RGB", (W + 20, H), (8, 8, 10))
    y = gap
    for p in panels:
        page.paste(p, ((W - p.width) // 2 + 10, y))
        y += p.height + gap
    dest = out_dir / "walkthrough.png"
    page.save(dest)
    (out_dir / "log.txt").write_text("\n".join(log) + "\n")
    (out_dir / "result_labels.json").write_text(
        json.dumps(
            {
                "stem": stem,
                "source": "facades/base",
                "merge_first": args.merge_first,
                "sym_scope": args.sym_scope,
                "extent": args.extent,
                "meta": meta,
                "focus": focus,
                "axis": axis,
                "sym_local_before": s_local0,
                "facade_before": {
                    "in_box": fsym0.normalized_in_box,
                    "global": fsym0.normalized_global,
                    "total": fsym0.normalized,
                    "I_in_box": fsym0.in_box,
                    "I_global": fsym0.global_pairs,
                    "I_blank": fsym0.blank,
                    "extent": list(fsym0.extent),
                },
                "objective_before": obj_before,
                "sim_margin": args.sim_margin,
                "labels_before": labels,
                "labels_after": new_labels,
                "units": windows,
                "log": log,
            },
            indent=2,
        )
        + "\n"
    )
    print("\n".join(log))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
