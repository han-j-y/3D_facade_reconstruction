"""Open-balcony type clustering.

``appearance`` is the previous path: DINOv2 ROI, spectral clustering, Potts.
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
) -> np.ndarray:
    """DINOv2 spectral clustering, then neighbor Potts."""
    n = len(boxes)
    if n < 2:
        return np.zeros(n, dtype=np.int32)
    iw, ih = image_size
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
) -> dict[str, Any]:
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
        )
        labels_box = None
    else:
        raise ValueError(f"unknown balcony cluster_mode: {cluster_mode}")
    return {
        "labels": np.asarray(labels, dtype=np.int32),
        "labels_box": labels_box,
        "feats": feats_pca,
        "cluster_mode": cluster_mode,
    }
