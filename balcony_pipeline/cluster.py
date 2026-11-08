"""Open-balcony type clustering.

``appearance`` (default): DINOv2 ROI, spectral clustering, Potts, then the width rule.
k = 1 is allowed (``select_k_allow_one``), and no type keeps two units whose box widths differ
2x or more (``split_by_width``; heights are not used, they change with the view angle).
With ``links`` (``--cluster-links``, pipeline default since 2026-11-07): must-link atoms first (``link_atoms``: rule A same-floor row,
rule B adjacent-floor column that also sits the same way on its window), spectral on the atoms, and a
Potts whose strong same-bay weight needs the same window relation (``linked_labels``).
``box`` matches the window pipeline's ``--cluster-mode box``: z-scored width,
height, and aspect (GMM). DINOv2 features are still returned so symmetry
repair and the medoid can use them. Boxes are already unitized.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from facade_recovery.box_cluster import cluster_box_geometry_pipeline


def select_k_allow_one(
    feats_pca: np.ndarray,
    feats_raw: np.ndarray,
    *,
    base,
    k_max: int,
    seed: int,
    sil_min: float,
    pair_sim_min: float,
) -> tuple[int, np.ndarray, float | None]:
    """k for the spectral step, where k = 1 is allowed.

    n >= 3: best k in 2..k_max by cosine silhouette (as before); k = 1 when that best silhouette is below
    ``sil_min`` (no clear split). n = 2: no silhouette, so k = 1 when the two raw DINOv2 ROI vectors have
    cosine similarity >= ``pair_sim_min``. Returns (k, labels, silhouette or pair similarity).
    """
    from sklearn import metrics

    n = len(feats_pca)
    if n < 2:
        return 1, np.zeros(n, dtype=np.int64), None
    if n == 2:
        a, b = feats_raw[0], feats_raw[1]
        sim = float(a @ b / max(1e-8, float(np.linalg.norm(a) * np.linalg.norm(b))))
        if sim >= pair_sim_min:
            return 1, np.zeros(2, dtype=np.int64), sim
        return 2, np.array([0, 1], dtype=np.int64), sim
    k = base.select_k(feats_pca, "spectral_rbf", min(k_max, n - 1), seed)
    k = max(2, min(k, n - 1))
    labels = base.cluster_features(feats_pca, "spectral_rbf", k, seed)
    sil = None
    if len(set(labels.tolist())) >= 2:
        try:
            sil = float(metrics.silhouette_score(feats_pca, labels, metric="cosine"))
        except Exception:
            sil = None
    if sil is None or sil < sil_min:
        return 1, np.zeros(n, dtype=np.int64), sil
    return k, labels, sil


def split_by_width(
    labels: np.ndarray, boxes: list[list[int]], *, max_ratio: float, atoms: np.ndarray | None = None
) -> np.ndarray:
    """No type holds two units whose box widths differ by ``max_ratio`` or more.

    Only the width is used: looking up at a facade changes balcony heights from floor to floor much
    more than widths. A type that breaks the rule is cut at its largest gap in log width, recursively,
    until every part holds. ``max_ratio`` <= 1 turns the rule off. With ``atoms`` (must-link groups,
    already within the ratio) a cut never goes through an atom: atoms are sorted by mean log width.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if max_ratio <= 1 or len(labels) == 0:
        return labels.astype(np.int32)
    w = np.array([max(1.0, float(b[2] - b[0])) for b in boxes], dtype=np.float64)
    atoms = np.arange(len(labels)) if atoms is None else np.asarray(atoms)
    groups: list[list[int]] = []

    def cut(items: list[list[int]]) -> None:
        items = sorted(items, key=lambda it: float(np.mean(np.log(w[it]))))
        flat = [i for it in items for i in it]
        if len(items) == 1 or w[flat].max() / w[flat].min() < max_ratio:
            groups.append(flat)
            return
        g = int(np.argmax(np.diff([float(np.mean(np.log(w[it]))) for it in items])))
        cut(items[: g + 1])
        cut(items[g + 1:])

    for t in sorted(set(labels.tolist())):
        idx = [i for i in range(len(labels)) if labels[i] == t]
        by_atom: dict[int, list[int]] = {}
        for i in idx:
            by_atom.setdefault(int(atoms[i]), []).append(i)
        cut(list(by_atom.values()))
    # number types by their first unit, as the other steps do
    groups.sort(key=lambda g: min(g))
    out = np.zeros(len(labels), dtype=np.int32)
    for t, g in enumerate(groups):
        out[g] = t
    return out


def _tv(p, q) -> float:
    return 0.5 * float(np.abs(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)).sum())


def label_distance(a: dict, b: dict) -> float:
    """Distance of two units' railing-classifier outputs (``RailingPredictor.predict_probs``).

    Per head the total-variation distance of the two softmax vectors (0 = same, 1 = opposite certain
    answers); the result is the largest head. Material only counts as much as both units are open work
    (min of the two open_work probabilities). Floor: triangle is folded into rectangle, as in the vote.
    """
    d = [_tv(a["kind"], b["kind"])]
    if a.get("material") is not None and b.get("material") is not None:
        d.append(_tv(a["material"], b["material"]) * min(a["kind"][0], b["kind"][0]))
    if a.get("structure") is not None and b.get("structure") is not None:
        d.append(_tv(a["structure"], b["structure"]))
    if a.get("floor") is not None and b.get("floor") is not None:
        fa = [a["floor"][0] + a["floor"][1], a["floor"][2]]
        fb = [b["floor"][0] + b["floor"][1], b["floor"][2]]
        d.append(_tv(fa, fb))
    return max(d)


def _mean_probs(ps: list[dict]) -> dict:
    out = {}
    for k in ("kind", "material", "structure", "floor"):
        vals = [p[k] for p in ps if p.get(k) is not None]
        out[k] = np.mean(np.asarray(vals, dtype=np.float64), axis=0).tolist() if len(vals) == len(ps) else None
    return out


def split_by_labels(
    labels: np.ndarray,
    probs: list[dict],
    *,
    thr: float,
    atoms: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Inside each type, average-linkage clustering of the classifier outputs (``label_distance``), cut at
    ``thr`` (as the window track splits box groups by its structure model's head probabilities).

    Items are atoms (must-link groups, their mean probabilities) when given, else units, so a split never
    goes through an atom. ``thr`` <= 0 turns it off. Types are renumbered by first unit.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    if thr <= 0 or n == 0:
        return labels.astype(np.int32), []
    atoms = np.arange(n) if atoms is None else np.asarray(atoms)
    groups: list[list[int]] = []
    log: list[dict] = []
    for t in sorted(set(labels.tolist())):
        idx = [i for i in range(n) if labels[i] == t]
        by_atom: dict[int, list[int]] = {}
        for i in idx:
            by_atom.setdefault(int(atoms[i]), []).append(i)
        items = list(by_atom.values())
        if len(items) == 1:
            groups.append(idx)
            continue
        ip = [_mean_probs([probs[i] for i in it]) for it in items]
        cond = [label_distance(ip[a], ip[b]) for a in range(len(items)) for b in range(a + 1, len(items))]
        sub = fcluster(linkage(np.asarray(cond), method="average"), t=thr, criterion="distance")
        parts: dict[int, list[int]] = {}
        for it, s in zip(items, sub.tolist()):
            parts.setdefault(int(s), []).extend(it)
        groups.extend(parts.values())
        if len(parts) > 1:
            log.append({"type": int(t), "n": len(idx), "parts": len(parts)})
    groups.sort(key=lambda g: min(g))
    out = np.zeros(n, dtype=np.int32)
    for k, g in enumerate(groups):
        out[g] = k
    return out, log


def window_relations(
    boxes: list[list[int]],
    floors: list[int],
    windows: list[dict] | None,
) -> list[tuple[float, float] | None]:
    """Per balcony: (top_rel, bot_rel) to its window, or None when it has none.

    The window is the one on the same floor with the largest x overlap (kind ``window`` only).
    ``top_rel`` = (balcony top - window bottom) / window height, ``bot_rel`` the same for the
    balcony bottom. Dividing by the window height of the same floor keeps the numbers stable when
    the view angle changes box heights from floor to floor.
    """
    out: list[tuple[float, float] | None] = []
    wins = [w for w in (windows or []) if w.get("kind", "window") == "window" and "box_xyxy" in w]
    for b, f in zip(boxes, floors):
        best, best_ov = None, 0.0
        for w in wins:
            if int(w.get("floor", -1)) != int(f):
                continue
            wb = w["box_xyxy"]
            ov = min(b[2], wb[2]) - max(b[0], wb[0])
            if ov > best_ov:
                best, best_ov = wb, ov
        if best is None:
            out.append(None)
            continue
        wh = max(1.0, float(best[3] - best[1]))
        out.append(((b[1] - best[3]) / wh, (b[3] - best[3]) / wh))
    return out


def _window_match(ra, rb, tol: float) -> bool | None:
    """True / False when both balconies have a window, None when one has none."""
    if ra is None or rb is None:
        return None
    return abs(ra[0] - rb[0]) <= tol and abs(ra[1] - rb[1]) <= tol


def link_atoms(
    boxes: list[list[int]],
    floors: list[int],
    rel: list[tuple[float, float] | None],
    *,
    row_align_tol: float = 0.15,
    row_width_ratio: float = 1.35,
    col_center_tol: float = 0.25,
    col_width_ratio: float = 1.15,
    window_tol: float = 0.15,
    max_width_ratio: float = 2.0,
) -> tuple[np.ndarray, list[tuple[int, int, str]]]:
    """Must-link components ("atoms") from two geometric rules.

    A (row): same floor, top or bottom aligned within ``row_align_tol`` x the taller height, widths
    within ``row_width_ratio``.
    B (column): adjacent floors, centre x within ``col_center_tol`` x the narrower width, widths within
    ``col_width_ratio``; when both have a window, ``top_rel`` and ``bot_rel`` within ``window_tol``
    (a balcony below its window does not join a grille in front of a window). Heights are not used.
    Links are added from the closest widths first and skipped when the joined component would hold
    widths ``max_width_ratio`` or more apart (the width rule). Returns atom id per unit and the links.
    """
    n = len(boxes)
    w = [max(1.0, float(b[2] - b[0])) for b in boxes]
    h = [max(1.0, float(b[3] - b[1])) for b in boxes]
    cx = [0.5 * (b[0] + b[2]) for b in boxes]
    cand: list[tuple[float, int, int, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            r = max(w[i], w[j]) / min(w[i], w[j])
            fi, fj = int(floors[i]), int(floors[j])
            if fi == fj:
                tol = row_align_tol * max(h[i], h[j])
                aligned = abs(boxes[i][1] - boxes[j][1]) <= tol or abs(boxes[i][3] - boxes[j][3]) <= tol
                if aligned and r <= row_width_ratio:
                    cand.append((r, i, j, "row"))
            elif abs(fi - fj) == 1:
                if abs(cx[i] - cx[j]) <= col_center_tol * min(w[i], w[j]) and r <= col_width_ratio:
                    if _window_match(rel[i], rel[j], window_tol) is not False:
                        cand.append((r, i, j, "column"))
    parent = list(range(n))
    wmin, wmax = w[:], w[:]

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    links = []
    for r, i, j, kind in sorted(cand):
        a, b = find(i), find(j)
        if a == b:
            continue
        lo, hi = min(wmin[a], wmin[b]), max(wmax[a], wmax[b])
        if max_width_ratio > 1 and hi / lo >= max_width_ratio:
            continue
        parent[b] = a
        wmin[a], wmax[a] = lo, hi
        links.append((i, j, kind))
    roots = {}
    atoms = np.array([roots.setdefault(find(i), len(roots)) for i in range(n)], dtype=np.int32)
    return atoms, links


def balcony_affinity(
    boxes: list[list[int]],
    bays: list[int],
    floors: list[int],
    rel: list[tuple[float, float] | None],
    *,
    iw: int,
    ih: int,
    row_tol: float,
    window_tol: float = 0.15,
) -> np.ndarray:
    """``reemb.spatial_affinity_units`` with one change: the strong same-bay weight only when the two
    balconies sit the same way on their windows (or one has no window); otherwise the weak weight."""
    n = len(boxes)
    cx = np.array([0.5 * (b[0] + b[2]) / iw for b in boxes])
    cy = np.array([0.5 * (b[1] + b[3]) / ih for b in boxes])
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = abs(cx[i] - cx[j]), abs(cy[i] - cy[j])
            if int(bays[i]) == int(bays[j]) and _window_match(rel[i], rel[j], window_tol) is not False:
                wgt = 1.6 * float(np.exp(-dy / 0.40))
            elif int(floors[i]) == int(floors[j]) or dy <= row_tol:
                wgt = 0.45 * float(np.exp(-dx / 0.28))
            else:
                wgt = 0.10 * float(np.exp(-float(np.hypot(dx, dy)) / 0.22))
            if wgt >= 0.05:
                W[i, j] = W[j, i] = wgt
    return W


def linked_labels(
    boxes: list[list[int]],
    floors: list[int],
    bays: list[int],
    feats_raw: np.ndarray,
    windows: list[dict] | None,
    *,
    image_size: tuple[int, int],
    base,
    reemb,
    pca_dim: int,
    k_max: int,
    seed: int,
    row_tol: float,
    spatial_strength: float,
    unary_weight: float,
    k1_sil_min: float,
    k1_pair_sim_min: float,
    width_ratio: float,
) -> tuple[np.ndarray, dict]:
    """Atoms (rules A / B) → spectral on atom mean features (k = 1 allowed) → window-aware Potts on
    units → each atom takes its members' majority label → width rule over atoms."""
    n = len(boxes)
    rel = window_relations(boxes, floors, windows)
    atoms, links = link_atoms(boxes, floors, rel, max_width_ratio=width_ratio)
    info = {"atoms": atoms.tolist(), "links": links, "window_rel": rel}
    if n < 2:
        return np.zeros(n, dtype=np.int32), info
    na = int(atoms.max()) + 1
    fa = np.stack([feats_raw[atoms == a].mean(0) for a in range(na)])
    fa = fa / np.clip(np.linalg.norm(fa, axis=1, keepdims=True), 1e-8, None)
    if na == 1:
        return np.zeros(n, dtype=np.int32), info
    fa_pca = base.apply_pca(fa, min(pca_dim, max(2, na - 1)), seed)
    _k, la, _s = select_k_allow_one(fa_pca, fa, base=base, k_max=k_max, seed=seed,
                                    sil_min=k1_sil_min, pair_sim_min=k1_pair_sim_min)
    labels0 = np.asarray(la, dtype=np.int64)[atoms]
    if len(set(labels0.tolist())) > 1:
        iw, ih = image_size
        W = balcony_affinity(boxes, bays, floors, rel, iw=iw, ih=ih, row_tol=row_tol)
        fu = base.apply_pca(feats_raw, min(pca_dim, max(2, n - 1)), seed)
        lab = reemb.neighbor_consistency_labels(labels0, W, fu, strength=spatial_strength, iters=12,
                                                unary_weight=unary_weight)
        for a in range(na):  # atoms stay whole
            idx = np.flatnonzero(atoms == a)
            vals, cnt = np.unique(lab[idx], return_counts=True)
            lab[idx] = vals[int(np.argmax(cnt))]
    else:
        lab = labels0
    return split_by_width(lab, boxes, max_ratio=width_ratio, atoms=atoms), info


def appearance_labels(
    boxes: list[list[int]],
    floors: list[int],
    bays: list[int],
    feats_pca: np.ndarray,
    *,
    image_size: tuple[int, int],
    base,
    reemb,
    k_max: int,
    seed: int,
    col_tol: float,
    row_tol: float,
    spatial_strength: float,
    unary_weight: float,
    feats_raw: np.ndarray | None = None,
    allow_k1: bool = True,
    k1_sil_min: float = 0.25,
    k1_pair_sim_min: float = 0.80,
) -> np.ndarray:
    """DINOv2 spectral clustering (k = 1 allowed), then neighbor Potts."""
    n = len(boxes)
    if n < 2:
        return np.zeros(n, dtype=np.int32)
    iw, ih = image_size
    if allow_k1:
        k, labels0, _score = select_k_allow_one(
            feats_pca,
            feats_pca if feats_raw is None else feats_raw,
            base=base,
            k_max=k_max,
            seed=seed,
            sil_min=k1_sil_min,
            pair_sim_min=k1_pair_sim_min,
        )
        if k == 1:
            return np.zeros(n, dtype=np.int32)
    else:
        k = base.select_k(feats_pca, "spectral_rbf", min(k_max, n - 1), seed)
        k = max(2, min(k, n - 1))
        labels0 = base.cluster_features(feats_pca, "spectral_rbf", k, seed)
    affinity = reemb.spatial_affinity_units(
        boxes,
        [int(b) for b in bays],
        np.asarray(floors, dtype=np.int32),
        iw=iw,
        ih=ih,
        col_tol=col_tol,
        row_tol=row_tol,
    )
    return reemb.neighbor_consistency_labels(
        labels0,
        affinity,
        feats_pca,
        strength=spatial_strength,
        iters=12,
        unary_weight=unary_weight,
    )


def box_geometry_labels(
    boxes: list[list[int]],
    image_size: tuple[int, int],
    backend,
    *,
    method: str = "gmm_diag",
    k_max: int = 6,
    seed: int = 42,
    conservative_k: bool = True,
    conservative_k_wh_tol: float = 0.008,
    merge_near: bool = False,
    snap_wh: bool = True,
) -> np.ndarray:
    """Window stage-1 labels: z-scored (width, height, aspect)."""
    n = len(boxes)
    if n < 2:
        return np.zeros(n, dtype=np.int32)
    labels, _feats = cluster_box_geometry_pipeline(
        boxes,
        image_size,
        backend,
        method=method,
        k_max=k_max,
        seed=seed,
        conservative_k=conservative_k,
        conservative_k_wh_tol=conservative_k_wh_tol,
        merge_near=merge_near,
        snap_wh=snap_wh,
    )
    return np.asarray(labels, dtype=np.int32)


def cluster_balcony_boxes(
    facade: Image.Image,
    boxes: list[list[int]],
    floors: list[int],
    bays: list[int],
    *,
    model,
    device: torch.device,
    facade_max_side: int,
    pca_dim: int,
    seed: int,
    k_max: int,
    col_tol: float,
    row_tol: float,
    spatial_strength: float,
    unary_weight: float,
    patch,
    base,
    reemb,
    cluster_mode: str = "appearance",
    k_max_box: int = 6,
    box_method: str = "gmm_diag",
    box_conservative_k: bool = True,
    box_conservative_k_wh_tol: float = 0.008,
    box_merge_near: bool = False,
    box_snap_wh: bool = True,
    allow_k1: bool = True,
    k1_sil_min: float = 0.25,
    k1_pair_sim_min: float = 0.80,
    width_ratio: float = 2.0,
    links: bool = False,
    windows: list[dict] | None = None,
    label_probs: list[dict] | None = None,
    label_split_thr: float = 0.0,
) -> dict[str, Any]:
    """``label_probs`` + ``label_split_thr`` > 0: split every type by the railing classifier's outputs
    (``split_by_labels``) before the width rule."""
    iw, ih = facade.size
    n = len(boxes)
    spatial, meta = patch.facade_patch_spatial(
        model, facade, device=device, max_side=facade_max_side
    )
    feats = patch.roi_pool_patches(spatial, boxes, meta)
    feats_pca = base.apply_pca(feats, min(pca_dim, max(2, n - 1)), seed)

    if cluster_mode == "box":
        labels = box_geometry_labels(
            boxes,
            (iw, ih),
            base,
            method=box_method,
            k_max=k_max_box,
            seed=seed,
            conservative_k=box_conservative_k,
            conservative_k_wh_tol=box_conservative_k_wh_tol,
            merge_near=box_merge_near,
            snap_wh=box_snap_wh,
        )
        labels_box = labels.copy()
    elif cluster_mode == "appearance" and links:
        labels, link_info = linked_labels(
            boxes, floors, bays, feats, windows, image_size=(iw, ih), base=base, reemb=reemb,
            pca_dim=pca_dim, k_max=k_max, seed=seed, row_tol=row_tol, spatial_strength=spatial_strength,
            unary_weight=unary_weight, k1_sil_min=k1_sil_min, k1_pair_sim_min=k1_pair_sim_min,
            width_ratio=width_ratio,
        )
        label_log = []
        if label_probs is not None and label_split_thr > 0:
            atoms = np.asarray(link_info["atoms"])
            labels, label_log = split_by_labels(labels, label_probs, thr=label_split_thr, atoms=atoms)
            labels = split_by_width(labels, boxes, max_ratio=width_ratio, atoms=atoms)
        return {"labels": np.asarray(labels, dtype=np.int32), "labels_box": None, "feats": feats_pca,
                "cluster_mode": cluster_mode, "links": link_info, "label_split": label_log}
    elif cluster_mode == "appearance":
        labels = appearance_labels(
            boxes,
            floors,
            bays,
            feats_pca,
            image_size=(iw, ih),
            base=base,
            reemb=reemb,
            k_max=k_max,
            seed=seed,
            col_tol=col_tol,
            row_tol=row_tol,
            spatial_strength=spatial_strength,
            unary_weight=unary_weight,
            feats_raw=feats,
            allow_k1=allow_k1,
            k1_sil_min=k1_sil_min,
            k1_pair_sim_min=k1_pair_sim_min,
        )
        if label_probs is not None and label_split_thr > 0:
            labels, _log = split_by_labels(labels, label_probs, thr=label_split_thr)
        labels = split_by_width(labels, boxes, max_ratio=width_ratio)
        labels_box = None
    else:
        raise ValueError(f"unknown balcony cluster_mode: {cluster_mode}")
    return {
        "labels": np.asarray(labels, dtype=np.int32),
        "labels_box": labels_box,
        "feats": feats_pca,
        "cluster_mode": cluster_mode,
    }
