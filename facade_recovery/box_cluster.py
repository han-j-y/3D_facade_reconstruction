"""Z-scored box geometry clustering (stage 1) and optional appearance splits (stage 2).

Mirrors ``window_ast_predictor/scripts/cluster_shape_then_feat.py`` stage-1/2 logic
for use in the e2e pipeline without cross-package imports.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ClusterBackend(Protocol):
    def select_k(self, feats: np.ndarray, method: str, k_max: int, seed: int) -> int: ...
    def cluster_features(
        self, feats: np.ndarray, method: str, k: int, seed: int
    ) -> np.ndarray: ...


def box_geometry_feats(boxes: list[list[float]], image_size: tuple[int, int]) -> np.ndarray:
    fw, fh = max(1.0, float(image_size[0])), max(1.0, float(image_size[1]))
    ws = np.array([max(1.0, b[2] - b[0]) / fw for b in boxes], dtype=np.float64)
    hs = np.array([max(1.0, b[3] - b[1]) / fh for b in boxes], dtype=np.float64)
    asp = ws / np.clip(hs, 1e-6, None)
    x = np.stack([ws, hs, asp], axis=1)
    x = (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-8)
    return x.astype(np.float32)


def _box_cluster_medians(
    labels: np.ndarray,
    boxes: list[list[float]],
    image_size: tuple[int, int],
) -> dict[int, tuple[float, float]]:
    fw, fh = max(1.0, float(image_size[0])), max(1.0, float(image_size[1]))
    ws = np.array([max(1.0, b[2] - b[0]) / fw for b in boxes], dtype=np.float64)
    hs = np.array([max(1.0, b[3] - b[1]) / fh for b in boxes], dtype=np.float64)
    clusters = sorted(set(int(x) for x in labels.tolist()))
    return {
        c: (float(np.median(ws[labels == c])), float(np.median(hs[labels == c])))
        for c in clusters
    }


def has_mergeable_box_clusters(
    labels: np.ndarray,
    boxes: list[list[float]],
    image_size: tuple[int, int],
    *,
    wh_tol: float = 0.008,
) -> bool:
    if wh_tol <= 0:
        return False
    meds = _box_cluster_medians(labels, boxes, image_size)
    clusters = sorted(meds)
    for i, ci in enumerate(clusters):
        wi, hi = meds[ci]
        for cj in clusters[i + 1 :]:
            wj, hj = meds[cj]
            if abs(wi - wj) <= wh_tol and abs(hi - hj) <= wh_tol:
                return True
    return False


def select_k_conservative(
    x: np.ndarray,
    boxes: list[list[float]],
    image_size: tuple[int, int],
    method: str,
    k_max: int,
    seed: int,
    backend: ClusterBackend,
    *,
    wh_tol: float = 0.008,
) -> int:
    k = backend.select_k(x, method, k_max, seed)
    while k > 1:
        labels = backend.cluster_features(x, method, k, seed)
        if not has_mergeable_box_clusters(labels, boxes, image_size, wh_tol=wh_tol):
            return k
        k -= 1
    return 1


def cluster_boxes(
    boxes: list[list[float]],
    image_size: tuple[int, int],
    method: str,
    backend: ClusterBackend,
    *,
    k_max: int,
    seed: int,
    conservative_k: bool = True,
    conservative_k_wh_tol: float = 0.008,
) -> np.ndarray:
    x = box_geometry_feats(boxes, image_size)
    k_cap = min(k_max, len(x) - 1)
    if conservative_k:
        k = select_k_conservative(
            x,
            boxes,
            image_size,
            method,
            k_cap,
            seed,
            backend,
            wh_tol=conservative_k_wh_tol,
        )
    else:
        k = backend.select_k(x, method, k_cap, seed)
    if k <= 1:
        return np.zeros(len(boxes), dtype=np.int32)
    return backend.cluster_features(x, method, k, seed)


def merge_near_box_clusters(
    labels: np.ndarray,
    boxes: list[list[float]],
    image_size: tuple[int, int],
    *,
    wh_tol: float = 0.008,
) -> np.ndarray:
    out = labels.astype(np.int32).copy()
    if len(out) < 2 or wh_tol <= 0:
        return out
    fw, fh = max(1.0, float(image_size[0])), max(1.0, float(image_size[1]))
    ws = np.array([max(1.0, b[2] - b[0]) / fw for b in boxes], dtype=np.float64)
    hs = np.array([max(1.0, b[3] - b[1]) / fh for b in boxes], dtype=np.float64)
    changed = True
    while changed:
        changed = False
        clusters = sorted(set(int(x) for x in out.tolist()))
        medians = {
            c: (float(np.median(ws[out == c])), float(np.median(hs[out == c])))
            for c in clusters
        }
        for i, ci in enumerate(clusters):
            for cj in clusters[i + 1 :]:
                wi, hi = medians[ci]
                wj, hj = medians[cj]
                if abs(wi - wj) <= wh_tol and abs(hi - hj) <= wh_tol:
                    out[out == cj] = ci
                    changed = True
                    break
            if changed:
                break
    uniq = sorted(set(int(x) for x in out.tolist()))
    remap = {u: i for i, u in enumerate(uniq)}
    return np.array([remap[int(x)] for x in out.tolist()], dtype=np.int32)


def robustify_box_labels(
    labels: np.ndarray,
    box_feats: np.ndarray,
    *,
    z_thr: float = 2.5,
) -> np.ndarray:
    out = labels.astype(np.int32).copy()
    if len(out) < 3:
        return out
    for c in sorted(set(int(x) for x in out.tolist())):
        idxs = np.where(out == c)[0]
        if len(idxs) < 3:
            continue
        sub = box_feats[idxs]
        cent = sub.mean(axis=0)
        dists = np.linalg.norm(sub - cent, axis=1)
        med = float(np.median(dists))
        mad = float(np.median(np.abs(dists - med))) + 1e-8
        for li, gi in enumerate(idxs.tolist()):
            z = (float(dists[li]) - med) / mad
            if z <= z_thr:
                continue
            others = sorted(set(int(x) for x in out.tolist()) - {c})
            if not others:
                continue
            best_c, best_d = c, float(dists[li])
            for oc in others:
                oidx = np.where(out == oc)[0]
                d = float(np.linalg.norm(box_feats[gi] - box_feats[oidx].mean(axis=0)))
                if d < best_d:
                    best_d, best_c = d, oc
            if best_c != c:
                out[gi] = best_c
    uniq = sorted(set(int(x) for x in out.tolist()))
    remap = {u: i for i, u in enumerate(uniq)}
    return np.array([remap[int(x)] for x in out.tolist()], dtype=np.int32)


def snap_box_labels_wh(
    labels: np.ndarray,
    boxes: list[list[float]],
    image_size: tuple[int, int],
    *,
    min_gain: float = 1e-4,
    w_weight: float = 1.0,
    h_weight: float = 1.0,
    max_wh_ratio: float = 1.35,
) -> np.ndarray:
    out = labels.astype(np.int32).copy()
    if len(out) < 2:
        return out
    fw, fh = max(1.0, float(image_size[0])), max(1.0, float(image_size[1]))
    ws = np.array([max(1.0, b[2] - b[0]) / fw for b in boxes], dtype=np.float64)
    hs = np.array([max(1.0, b[3] - b[1]) / fh for b in boxes], dtype=np.float64)
    clusters = sorted(set(int(x) for x in out.tolist()))
    medians = {
        c: (float(np.median(ws[out == c])), float(np.median(hs[out == c])))
        for c in clusters
    }

    def _dist(i: int, c: int) -> float:
        return float(
            w_weight * (ws[i] - medians[c][0]) ** 2
            + h_weight * (hs[i] - medians[c][1]) ** 2
        )

    def _size_ok(i: int, c: int) -> bool:
        mw, mh = medians[c]
        if mw <= 0 or mh <= 0:
            return False
        wr = ws[i] / mw
        hr = hs[i] / mh
        if wr > max_wh_ratio or wr < 1.0 / max_wh_ratio:
            return False
        if hr > max_wh_ratio or hr < 1.0 / max_wh_ratio:
            return False
        return True

    for i in range(len(out)):
        cur = int(out[i])
        d_cur = _dist(i, cur)
        best_c = cur
        best_d = d_cur
        for c in clusters:
            if c == cur:
                continue
            if not _size_ok(i, c):
                continue
            d = _dist(i, c)
            if d < best_d:
                best_d, best_c = d, c
        if best_c != cur and best_d + min_gain < d_cur:
            out[i] = best_c

    uniq = sorted(set(int(x) for x in out.tolist()))
    remap = {u: i for i, u in enumerate(uniq)}
    return np.array([remap[int(x)] for x in out.tolist()], dtype=np.int32)


def refine_box_labels(
    labels: np.ndarray,
    boxes: list[list[float]],
    image_size: tuple[int, int],
    box_feats: np.ndarray,
    *,
    robustify_z: float = 2.5,
    merge_near: bool = False,
    merge_near_wh_tol: float = 0.008,
    snap_wh: bool = True,
    snap_wh_min_gain: float = 1e-4,
) -> np.ndarray:
    out = robustify_box_labels(labels, box_feats, z_thr=robustify_z)
    if merge_near:
        out = merge_near_box_clusters(out, boxes, image_size, wh_tol=merge_near_wh_tol)
    if snap_wh:
        out = snap_box_labels_wh(out, boxes, image_size, min_gain=snap_wh_min_gain)
    return out


def _pairwise_cosine_sims(feats: np.ndarray) -> np.ndarray:
    x = feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8, None)
    return (x @ x.T).astype(np.float64)


def _upper_triangle_sims(sim: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(sim.shape[0], k=1)
    return sim[i, j]


def subcluster_if_clear(
    feats: np.ndarray,
    backend: ClusterBackend,
    *,
    k_max: int,
    seed: int,
    pair_split_thr: float,
    cohesive_median_thr: float,
    min_silhouette: float,
    min_sil_gain: float,
) -> tuple[np.ndarray, int, str, dict[str, float]]:
    from sklearn import metrics

    m = len(feats)
    if m == 1:
        return np.zeros(1, dtype=np.int32), 1, "singleton", {"median_sim": 1.0}

    sim = _pairwise_cosine_sims(feats)
    tri = _upper_triangle_sims(sim)
    stats = {
        "median_sim": float(np.median(tri)),
        "min_sim": float(np.min(tri)),
        "max_sim": float(np.max(tri)),
    }

    if m == 2:
        s = float(tri[0])
        if s < pair_split_thr:
            return np.array([0, 1], dtype=np.int32), 2, "pair_split", stats
        return np.zeros(2, dtype=np.int32), 1, "pair_keep", stats

    if stats["median_sim"] >= cohesive_median_thr:
        return np.zeros(m, dtype=np.int32), 1, "cohesive", stats

    def _try_k(k: int) -> tuple[float, np.ndarray] | None:
        try:
            labels = backend.cluster_features(feats, "spectral_rbf", k, seed)
        except Exception:
            return None
        if len(set(labels.tolist())) < 2:
            return None
        try:
            sil = float(metrics.silhouette_score(feats, labels, metric="cosine"))
        except Exception:
            return None
        return sil, labels.astype(np.int32)

    k2 = _try_k(2)
    if k2 is None or k2[0] < min_silhouette:
        stats["best_silhouette"] = float(k2[0]) if k2 else -1.0
        stats["best_k"] = 1.0
        return np.zeros(m, dtype=np.int32), 1, "weak_split", stats

    best_sil, best_k, best_labels = k2[0], 2, k2[1]
    for k in range(3, min(k_max, m - 1) + 1):
        trial = _try_k(k)
        if trial is None:
            continue
        sil_k, labels_k = trial
        if sil_k >= min_silhouette and sil_k >= best_sil + min_sil_gain:
            best_sil, best_k, best_labels = sil_k, k, labels_k

    stats["best_silhouette"] = float(best_sil)
    stats["best_k"] = float(best_k)
    return best_labels, best_k, "silhouette", stats


def cluster_within_groups(
    feats: np.ndarray,
    box_labels: np.ndarray,
    backend: ClusterBackend,
    *,
    k_max_within: int,
    seed: int,
    pair_split_thr: float = 0.55,
    cohesive_median_thr: float = 0.60,
    min_silhouette: float = 0.28,
    min_sil_gain: float = 0.10,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    sub_labels = np.zeros(len(feats), dtype=np.int32)
    groups_info: list[dict[str, Any]] = []
    next_global = 0
    global_map: dict[tuple[int, int], int] = {}

    for box_c in sorted(set(int(x) for x in box_labels.tolist())):
        idxs = np.where(box_labels == box_c)[0]
        sub_feats = feats[idxs]
        sub, k_sub, reason, split_stats = subcluster_if_clear(
            sub_feats,
            backend,
            k_max=k_max_within,
            seed=seed,
            pair_split_thr=pair_split_thr,
            cohesive_median_thr=cohesive_median_thr,
            min_silhouette=min_silhouette,
            min_sil_gain=min_sil_gain,
        )

        for local_i, gi in enumerate(idxs):
            sc = int(sub[local_i])
            key = (box_c, sc)
            if key not in global_map:
                global_map[key] = next_global
                next_global += 1
            sub_labels[gi] = global_map[key]

        groups_info.append(
            {
                "box_cluster": int(box_c),
                "n": int(len(idxs)),
                "k_within": int(k_sub),
                "split_reason": reason,
                "split_stats": split_stats,
                "indices": [int(i) for i in idxs.tolist()],
                "sub_labels": [int(x) for x in sub.tolist()],
            }
        )

    return sub_labels, groups_info


def relabel_contiguous(labels: np.ndarray) -> np.ndarray:
    uniq = sorted({int(x) for x in labels.tolist()})
    remap = {u: i for i, u in enumerate(uniq)}
    return np.array([remap[int(x)] for x in labels.tolist()], dtype=np.int32)


def split_same_floor_cross_bay_labels(
    labels: np.ndarray,
    floors: list[int] | np.ndarray,
    bays: list[int] | np.ndarray,
    *,
    only_floors: set[int] | None = None,
    only_bays: set[int] | None = None,
) -> np.ndarray:
    """Give distinct cluster ids to different bays on the same floor.

    Prevents symmetric outer attic windows (and similar cases) from sharing one
    type when geometry/appearance clustering merges them.
    """
    out = labels.astype(np.int32).copy()
    if len(out) < 2:
        return out
    next_id = int(out.max()) + 1
    from collections import defaultdict

    by_lab_floor: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i in range(len(out)):
        by_lab_floor[(int(out[i]), int(floors[i]))].append(i)

    for (_lab, fl), idxs in by_lab_floor.items():
        if only_floors is not None and fl not in only_floors:
            continue
        if only_bays is not None:
            idxs = [i for i in idxs if int(bays[i]) in only_bays]
        if len(idxs) < 2:
            continue
        bay_groups: dict[int, list[int]] = defaultdict(list)
        for i in idxs:
            bay_groups[int(bays[i])].append(i)
        if len(bay_groups) <= 1:
            continue
        for bis in list(bay_groups.values())[1:]:
            for i in bis:
                out[i] = next_id
            next_id += 1

    return relabel_contiguous(out)


def cluster_box_geometry_pipeline(
    boxes: list[list[float]],
    image_size: tuple[int, int],
    backend: ClusterBackend,
    *,
    method: str = "gmm_diag",
    k_max: int = 6,
    seed: int = 42,
    conservative_k: bool = True,
    conservative_k_wh_tol: float = 0.008,
    merge_near: bool = False,
    snap_wh: bool = True,
    snap_wh_min_gain: float = 1e-4,
    robustify_z: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage-1 box z-score clustering with optional de-overcluster refinements."""
    labels_raw = cluster_boxes(
        boxes,
        image_size,
        method,
        backend,
        k_max=k_max,
        seed=seed,
        conservative_k=conservative_k,
        conservative_k_wh_tol=conservative_k_wh_tol,
    )
    box_feats = box_geometry_feats(boxes, image_size)
    labels = refine_box_labels(
        labels_raw,
        boxes,
        image_size,
        box_feats,
        robustify_z=robustify_z,
        merge_near=merge_near,
        merge_near_wh_tol=conservative_k_wh_tol,
        snap_wh=snap_wh,
        snap_wh_min_gain=snap_wh_min_gain,
    )
    return labels, box_feats
