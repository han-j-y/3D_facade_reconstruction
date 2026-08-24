"""Balcony appearance clustering — same DINO ROI + spectral + Potts as windows,
but boxes are already unitized/snapped (no second window merge).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image


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
) -> dict[str, Any]:
    iw, ih = facade.size
    n = len(boxes)
    spatial, meta = patch.facade_patch_spatial(
        model, facade, device=device, max_side=facade_max_side
    )
    feats = patch.roi_pool_patches(spatial, boxes, meta)
    feats_pca = base.apply_pca(feats, min(pca_dim, max(2, n - 1)), seed)
    floor_m = np.asarray(floors, dtype=np.int32)
    bay_u = [int(b) for b in bays]

    if n < 2:
        labels = np.zeros(n, dtype=np.int32)
    else:
        k = base.select_k(feats_pca, "spectral_rbf", min(k_max, n - 1), seed)
        k = max(2, min(k, n - 1))
        labels0 = base.cluster_features(feats_pca, "spectral_rbf", k, seed)
        W = reemb.spatial_affinity_units(
            boxes,
            bay_u,
            floor_m,
            iw=iw,
            ih=ih,
            col_tol=col_tol,
            row_tol=row_tol,
        )
        labels = reemb.neighbor_consistency_labels(
            labels0,
            W,
            feats_pca,
            strength=spatial_strength,
            iters=12,
            unary_weight=unary_weight,
        )
    return {"labels": labels, "feats": feats_pca}
