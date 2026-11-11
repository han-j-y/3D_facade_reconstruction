"""Joint vote after the fold gate: floor symmetry and column alignment pull windows into one type, occlusion-aware.

Run by ``run_pipeline.py --joint-vote`` between the fold gate and the local-symmetry vote (which still runs).

Occlusion is handled per axis instead of by editing boxes:

* a side hidden by an occluder (balcony, car, ...; the gate's hidden sides) is a lower bound on that extent;
* a shutter along a side (``--shutter-uncertain``: SAM3 'shutter' pixels in the strip just inside or just outside the
  left / right edge) makes the width unknown in both directions: folded shutters inside the box widen it, half-closed
  shutters over the glass narrow it. A box mostly covered by shutter, middle included, is a blind or closed shutter in
  the opening and keeps a certain width (:func:`shutter_width_uncertain`).

Weights (:func:`symmetry_weights`, :func:`column_weights`), with ``g(u) = max(0, (u - 0.5) / 0.5) ** 2`` so a match
below IoU 0.5 adds nothing:

* floor symmetry: the gate's per-floor fold axes; axes at the same x (within 0.15 median width) on different floors
  pool their strength (sum of g over their matched windows). Each window pairs with its best reflection about an axis;
  ``w_ij = sum over axes (strength / strongest) * g(IoU_ij)``;
* column: windows on adjacent floors with centres within a quarter of the narrower width, ``v_ij = g(overlap)``.

IoU / overlap = horizontal term x vertical term. Horizontal: 1-D IoU, or, when either width is uncertain, the centre
term ``max(0, 1 - |dx| / (0.5 * narrower width))``. Vertical: 1-D IoU with a hidden top / bottom extended to the
partner's (pairs), or the shorter / taller height ratio, 1 when the shorter one has a hidden top / bottom (columns).

Vote (:func:`joint_vote`): each window, lowest total weight first, takes
``argmax_T sum_j (w_ij + v_ij) [t(j) = T] - alpha * d(i, T)`` (d = mean structure-descriptor cosine distance to T's
other members), over its own type and the types of linked windows it is compatible with (same outline; heights within
``tol``, a hidden top / bottom a lower bound; widths within ``tol`` only when both are certain); repeated until stable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

Box = list[float]


def g(u: float) -> float:
    return max(0.0, (u - 0.5) / 0.5) ** 2


def _iv(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def shutter_width_uncertain(boxes: list[Box], mask: np.ndarray | None, *, strip: float = 0.15, cover: float = 0.4,
                            blind: float = 0.6) -> list[bool]:
    """Per box: is its width uncertain because of a shutter along its left or right edge? ``mask``: union of SAM3
    'shutter' instance masks (H, W). A side counts when shutter pixels cover >= ``cover`` of the strip just inside or
    just outside that edge (``strip`` of the box width, central 70 % of its height); none counts when the shutter covers
    >= ``blind`` of the box and of its middle third (blind or closed shutter inside the opening)."""
    if mask is None:
        return [False] * len(boxes)

    def cov(x0, x1, y0, y1):
        x0, x1 = max(0, int(x0)), min(mask.shape[1], int(x1))
        y0, y1 = max(0, int(y0)), min(mask.shape[0], int(y1))
        return float(mask[y0:y1, x0:x1].mean()) if x1 > x0 and y1 > y0 else 0.0

    out = []
    for x0, y0, x1, y1 in boxes:
        w = x1 - x0
        ya, yb = y0 + 0.15 * (y1 - y0), y1 - 0.15 * (y1 - y0)
        if cov(x0, x1, y0, y1) >= blind and cov(x0 + w / 3, x1 - w / 3, ya, yb) >= blind:
            out.append(False)
            continue
        side = False
        for a, c in ((x0 - strip * w, x0 + strip * w), (x1 - strip * w, x1 + strip * w)):
            m = (a + c) / 2
            side = side or max(cov(a, m, ya, yb), cov(m, c, ya, yb)) >= cover
        out.append(side)
    return out


def _pair_iou(bi: Box, bj: Box, c: float, unc_i: bool, unc_j: bool, oi: set[int], oj: set[int]) -> float:
    rx0, rx1 = 2 * c - bi[2], 2 * c - bi[0]
    if unc_i or unc_j:
        h = max(0.0, 1 - abs((rx0 + rx1) / 2 - (bj[0] + bj[2]) / 2) / (0.5 * min(bi[2] - bi[0], bj[2] - bj[0])))
    else:
        h = _iv(rx0, rx1, bj[0], bj[2])
    y0i, y1i, y0j, y1j = bi[1], bi[3], bj[1], bj[3]
    if 1 in oi:
        y0i = min(y0i, y0j)
    if 3 in oi:
        y1i = max(y1i, y1j)
    if 1 in oj:
        y0j = min(y0j, y0i)
    if 3 in oj:
        y1j = max(y1j, y1i)
    return h * _iv(y0i, y1i, y0j, y1j)


def symmetry_weights(boxes: list[Box], floor: dict[int, int], occ: list[set[int]], unc: list[bool],
                     axes: list[dict]) -> np.ndarray:
    """Floor-symmetry weights from the gate's fold axes (``fold_gate`` report ``fold_axes``)."""
    n = len(boxes)
    medw = float(np.median([b[2] - b[0] for b in boxes])) if boxes else 1.0
    inst = []
    for a in axes:
        mem = [i for i in range(n) if floor[i] == a["floor"]]
        pairs: dict[tuple[int, int], float] = {}
        for i in mem:
            best = max(((_pair_iou(boxes[i], boxes[j], a["x"], unc[i], unc[j], occ[i], occ[j]), j)
                        for j in mem if j != i), default=(0.0, None))
            if best[1] is not None and g(best[0]) > 0:
                key = (min(i, best[1]), max(i, best[1]))
                pairs[key] = max(pairs.get(key, 0.0), g(best[0]))
        inst.append((a["x"], pairs))
    groups: dict[float, list[float]] = {}
    for x in sorted({x for x, _ in inst}):
        k = next((k for k in groups if abs(k - x) <= 0.15 * medw), x)
        groups.setdefault(k, []).append(x)
    gid = {x: k for k, xs in groups.items() for x in xs}
    strength: dict[float, float] = {}
    for x, p in inst:
        strength[gid[x]] = strength.get(gid[x], 0.0) + sum(p.values())
    top = max(strength.values(), default=0.0) or 1.0
    W = np.zeros((n, n))
    for x, p in inst:
        for (i, j), v in p.items():
            W[i, j] += strength[gid[x]] / top * v
            W[j, i] = W[i, j]
    return W


def column_weights(boxes: list[Box], floor: dict[int, int], occ: list[set[int]], unc: list[bool]) -> np.ndarray:
    """Column weights between windows on adjacent floors whose centres are within a quarter of the narrower width."""
    n = len(boxes)
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(floor[i] - floor[j]) != 1:
                continue
            bi, bj = boxes[i], boxes[j]
            wi, wj = bi[2] - bi[0], bj[2] - bj[0]
            dc = abs((bi[0] + bi[2]) / 2 - (bj[0] + bj[2]) / 2)
            if dc > 0.25 * min(wi, wj):
                continue
            h = max(0.0, 1 - dc / (0.5 * min(wi, wj))) if (unc[i] or unc[j]) else _iv(bi[0], bi[2], bj[0], bj[2])
            hi, hj = bi[3] - bi[1], bj[3] - bj[1]
            short = i if hi < hj else j
            v = 1.0 if {1, 3} & occ[short] else min(hi, hj) / max(hi, hj)
            V[i, j] = V[j, i] = g(h * v)
    return V


def _compatible(i: int, j: int, boxes: list[Box], occ: list[set[int]], unc: list[bool], outline: dict[int, str],
                tol: float) -> bool:
    if outline.get(i) != outline.get(j):
        return False
    hi, hj = boxes[i][3] - boxes[i][1], boxes[j][3] - boxes[j][1]
    hid_i, hid_j = bool({1, 3} & occ[i]), bool({1, 3} & occ[j])
    if hid_i and not hid_j and hi > hj * tol:
        return False
    if hid_j and not hid_i and hj > hi * tol:
        return False
    if not hid_i and not hid_j and max(hi, hj) / max(1e-6, min(hi, hj)) > tol:
        return False
    if not (unc[i] or unc[j]):
        wi, wj = boxes[i][2] - boxes[i][0], boxes[j][2] - boxes[j][0]
        if max(wi, wj) / max(1e-6, min(wi, wj)) > tol:
            return False
    return True


def joint_vote(boxes: list[Box], labels: np.ndarray, floor: dict[int, int], occ: list[set[int]], outline: dict[int, str],
               feat: np.ndarray, axes: list[dict], unc: list[bool] | None = None, *, alpha: float = 4.0,
               tol: float = 1.3, iters: int = 20) -> tuple[np.ndarray, dict[str, Any]]:
    """New type labels and a JSON-able report (moves as [window, old type, new type])."""
    boxes = [list(map(float, b)) for b in boxes]
    n = len(boxes)
    occ = [set(o) for o in occ]
    unc = [bool(u) or bool({0, 2} & occ[i]) for i, u in enumerate(unc if unc is not None else [False] * n)]
    F = np.asarray(feat, dtype=np.float64)
    F = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
    W = symmetry_weights(boxes, floor, occ, unc, axes) + column_weights(boxes, floor, occ, unc)
    lab = np.asarray(labels).copy()
    moves = []
    for _ in range(iters):
        changed = False
        for i in np.argsort(W.sum(1)):
            cands = {int(lab[i])} | {int(lab[j]) for j in np.where(W[i] > 0)[0]
                                     if _compatible(i, j, boxes, occ, unc, outline, tol)}
            score = {}
            for t in cands:
                mem = [k for k in np.where(lab == t)[0] if k != i]
                d = float(1 - np.mean(F[mem] @ F[i])) if mem else 0.16
                score[t] = sum(W[i, j] for j in range(n) if j != i and lab[j] == t) - alpha * d
            new = max(score, key=score.get)
            if new != lab[i]:
                moves.append([int(i), int(lab[i]), int(new)])
                lab[i] = new
                changed = True
        if not changed:
            break
    return lab, {"moves": moves, "width_uncertain": [i for i in range(n) if unc[i]]}
