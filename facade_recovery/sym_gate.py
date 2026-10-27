"""Two-tier detection and symmetry-gated structure splits of box groups.

Mirror partners: each window box reflected about the facade's symmetry axis is paired with the other window it
overlaps most (IoU >= ``pair_iou``); windows reflecting onto themselves or onto nothing are unconstrained.
``gate_cluster`` walks each box group's average-linkage tree (cosine on a structure descriptor) top-down; a node
is split only if its merge height is above the cut and the split separates at most ``eps`` of the mirror pairs
inside the node.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist


def box_iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def overlap_small(a: list[float], b: list[float]) -> float:
    """Intersection over the smaller box's area."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih / max(min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])), 1e-9)


def two_tier(dets: list[dict[str, Any]], anchor: float, max_overlap: float = 0.2) -> list[dict[str, Any]]:
    """Keep detections scoring >= ``anchor``; add the lower-scoring ones (best first) that overlap no kept box by
    ``max_overlap`` of the smaller box. High-score boxes keep two-pane windows whole; the fill recovers
    windows a single high threshold drops."""
    keep = [d for d in dets if d["score"] >= anchor]
    for d in sorted((d for d in dets if d["score"] < anchor), key=lambda d: -d["score"]):
        if all(overlap_small(d["box_xyxy"], k["box_xyxy"]) < max_overlap for k in keep):
            keep.append(d)
    return keep


def mirror_partners(boxes: list[list[float]], cx: float, pair_iou: float = 0.3) -> list[int | None]:
    out: list[int | None] = []
    for i, b in enumerate(boxes):
        m = [2 * cx - b[2], b[1], 2 * cx - b[0], b[3]]
        if box_iou(m, b) >= 0.5:
            out.append(None)
            continue
        j = max((k for k in range(len(boxes)) if k != i), key=lambda k: box_iou(m, boxes[k]), default=None)
        out.append(j if j is not None and box_iou(m, boxes[j]) >= pair_iou else None)
    return out


def refine_axis(boxes: list[list[float]], cx: float, radius: float, pair_iou: float = 0.3) -> float:
    """Shift ``cx`` within +-radius px to the position pairing the most boxes (ties: nearest to ``cx``); narrow
    windows lose their partners at a few px of axis error."""
    if radius <= 0:
        return cx
    grid = np.arange(cx - radius, cx + radius + 0.5, 1.0)
    n = [sum(p is not None for p in mirror_partners(boxes, float(c), pair_iou)) for c in grid]
    best = max(n)
    return float(min((c for c, k in zip(grid, n) if k == best), key=lambda c: abs(c - cx)))


def gate_split(feat: np.ndarray, idx: np.ndarray, partner: list[int | None], thr: float, eps: float) -> np.ndarray:
    """Top-down average-linkage splits of units ``idx``, refusing splits that break > eps of mirror pairs."""
    if len(idx) == 1:
        return np.zeros(1, dtype=np.int64)
    Z = linkage(pdist(feat[idx], metric="cosine"), method="average")
    n = len(idx)
    leaves: dict[int, list[int]] = {i: [i] for i in range(n)}
    for k, (a, b, _, _) in enumerate(Z):
        leaves[n + k] = leaves[int(a)] + leaves[int(b)]
    lab = np.zeros(n, dtype=np.int64)
    nxt = [0]

    def visit(node: int) -> None:
        if node < n:
            lab[leaves[node]] = nxt[0]
            nxt[0] += 1
            return
        a, b, h, _ = Z[node - n]
        a, b = int(a), int(b)
        ok = h > thr
        if ok:
            side = {u: 0 for u in leaves[a]} | {u: 1 for u in leaves[b]}
            glob = {int(idx[u]): s for u, s in side.items()}
            pairs = [(i, partner[i]) for i in glob if partner[i] is not None and partner[i] in glob and i < partner[i]]
            broken = sum(glob[i] != glob[j] for i, j in pairs)
            ok = not pairs or broken / len(pairs) <= eps
        if ok:
            visit(a)
            visit(b)
        else:
            lab[leaves[node]] = nxt[0]
            nxt[0] += 1

    visit(2 * n - 2)
    return lab


def gate_cluster(
    feat: np.ndarray, groups: np.ndarray, partner: list[int | None], *, thr: float = 0.16, eps: float = 0.1
) -> np.ndarray:
    """Final type labels: ``gate_split`` inside every box group, numbered consecutively."""
    labels = np.zeros(len(groups), dtype=np.int64)
    nxt = 0
    for g in sorted(set(np.asarray(groups).tolist())):
        idx = np.where(groups == g)[0]
        _, sub = np.unique(gate_split(feat, idx, partner, thr, eps), return_inverse=True)
        labels[idx] = sub + nxt
        nxt += int(sub.max()) + 1
    return labels
