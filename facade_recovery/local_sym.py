"""Local symmetries + a weighted vote on window types (runs after gate clustering).

1. Candidate axes: midpoints of similar window pairs (same floor band, w/h within 1.5x) and window centres.
2. Each axis: every window's reflection is matched to its best window (IoU >= 0.3; itself counts). The axis's
   extent grows outward from the axis while >= 75% of the windows inside are matched; strength = summed match
   IoU inside the extent (a pair counts twice, a self-symmetric window once). A pair whose midpoint was folded
   into that group, but which falls below 0.3 about the chosen axis, is restored at its own midpoint. One pair
   is enough there: the same 0.3 test already accepted it.
3. Continuous weights on those axes: inside each axis's extent every window pairs with its best other member by
   reflection IoU, with no 0.3 cutoff; strength = Zhang et al.'s overlap curve at the axis over those members.
   Pair weight w_ij = sum over axes pairing i and j of strength * IoU (normalised by the strongest axis).
4. Weighted vote (ICM on a Potts energy): window i takes label L minimising
     alpha * D(i, L)  +  sum_j w_ij [L != L_j]
   with D = mean cosine distance of the pane-tree descriptor to label L's other members (``cut`` if none), over
   its own label and its mirror partners' labels (joins only; ``new_label`` also allows a type of its own).
   A window whose box size repeats (size-mates: w and h within ``mate_tol``) may only join a partner's type
   holding one of its size-mates; a one-off box size may join any partner's type, unless it is self-symmetric
   about a strong axis (a central feature, not a bad box).
Occlusion: a window side whose outside strip lies on an occluder mask (vehicles, fire escapes, trees, poles,
people, balconies) or on the image border is hidden; the pair IoU clips the partner at hidden sides. A hidden side makes
that axis a minimum for that window only: it may match a longer window, and the size tolerance rejects a
candidate only when this box is already the longer one. The longer window does not gain the shorter occluded
box as a size-mate. If no partner type meets the test, it stays. Windows lying on an occluder do not take part
in the vote.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from facade_recovery.sym_gate import box_iou

OCCLUDER_PROMPTS = ("car", "truck", "van", "fire escape", "tree", "street lamp", "traffic light", "person", "balcony")
MIRROR_SIDE = {0: 2, 1: 1, 2: 0, 3: 3}


def reflect(b: list[float], c: float) -> list[float]:
    return [2 * c - b[2], b[1], 2 * c - b[0], b[3]]


def similar(a: list[float], b: list[float], ratio: float = 1.5, yov: float = 0.5) -> bool:
    wa, ha, wb, hb = a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]
    ov = min(a[3], b[3]) - max(a[1], b[1])
    return max(wa, wb) / min(wa, wb) <= ratio and max(ha, hb) / min(ha, hb) <= ratio and ov >= yov * min(ha, hb)


def occluded_sides(
    boxes: list[list[float]], mask: np.ndarray | None, iw: int, ih: int, frac: float = 0.4, border: int = 3
) -> list[set[int]]:
    """Per box: hidden sides as xyxy indices (0 left, 1 top, 2 right, 3 bottom). A side is hidden when its outside
    strip (12% of the box, central 60% of the edge) is >= ``frac`` covered by ``mask``, or it touches the border."""
    out = []
    for x0, y0, x1, y1 in boxes:
        w, h = x1 - x0, y1 - y0
        tw, th = max(4, int(0.12 * w)), max(4, int(0.12 * h))
        mx0, mx1, my0, my1 = int(x0 + 0.2 * w), int(x1 - 0.2 * w), int(y0 + 0.2 * h), int(y1 - 0.2 * h)
        strips = {0: (my0, my1, int(x0) - tw, int(x0)), 2: (my0, my1, int(x1), int(x1) + tw),
                  1: (int(y0) - th, int(y0), mx0, mx1), 3: (int(y1), int(y1) + th, mx0, mx1)}
        s = set()
        for side, (a, b, c, d) in strips.items():
            a, b, c, d = max(0, a), min(ih, b), max(0, c), min(iw, d)
            if b > a and d > c and mask is not None and mask[a:b, c:d].mean() >= frac:
                s.add(side)
        if x0 <= border:
            s.add(0)
        if y0 <= border:
            s.add(1)
        if x1 >= iw - border:
            s.add(2)
        if y1 >= ih - border:
            s.add(3)
        out.append(s)
    return out


def pair_iou(boxes: list[list[float]], occ: list[set[int]], i: int, j: int, c: float) -> float:
    """IoU of window i and window j reflected about c; a side hidden on either window is clipped to the other
    box (a hidden part may extend further, so it must not count against the match)."""
    a, r = list(boxes[i]), reflect(boxes[j], c)
    for s in occ[i]:
        r[s] = max(r[s], a[s]) if s < 2 else min(r[s], a[s])
    for s in (MIRROR_SIDE[s] for s in occ[j]):
        a[s] = max(a[s], r[s]) if s < 2 else min(a[s], r[s])
    return box_iou(a, r)


def match(boxes: list[list[float]], occ: list[set[int]], c: float, min_iou: float = 0.3) -> list[tuple[int | None, float]]:
    """Per window: (partner index, IoU) of its reflection about c, or (None, 0)."""
    out = []
    for i in range(len(boxes)):
        ious = [pair_iou(boxes, occ, i, j, c) for j in range(len(boxes))]
        j = int(np.argmax(ious))
        out.append((j, ious[j]) if ious[j] >= min_iou else (None, 0.0))
    return out


def _one_pair_axis(boxes: list[list[float]], cx: np.ndarray, i: int, j: int, mid: float, iou: float) -> dict[str, Any]:
    """An axis that is only this pair, placed at the pair's own midpoint."""
    key = (i, j) if i < j else (j, i)
    ext = max(abs(cx[k] - mid) + 0.5 * (boxes[k][2] - boxes[k][0]) for k in key)
    return dict(cx=float(mid), strength=2.0 * float(iou), extent=float(ext), members=sorted(key),
                pairs={key: float(iou)}, selfs=[])


def local_symmetries(
    boxes: list[list[float]], occ: list[set[int]], min_pairs: int = 2, cover: float = 0.75
) -> list[dict[str, Any]]:
    cx = np.array([0.5 * (b[0] + b[2]) for b in boxes])
    medw = float(np.median([b[2] - b[0] for b in boxes]))
    tagged: list[tuple[float, tuple[int, int] | None]] = [
        (0.5 * (cx[i] + cx[j]), (i, j)) for i in range(len(boxes)) for j in range(i + 1, len(boxes))
        if similar(boxes[i], boxes[j])]
    tagged += [(float(c), None) for c in cx]
    tagged.sort(key=lambda t: t[0])
    tol = 0.15 * medw
    groups: list[list[tuple[float, tuple[int, int] | None]]] = []
    for item in tagged:
        if groups and item[0] - groups[-1][-1][0] <= tol:
            groups[-1].append(item)
        else:
            groups.append([item])
    syms = []
    lost: list[tuple[int, int, float, float]] = []
    for g in groups:
        xs = [t[0] for t in g]
        # position within the group's span (+-tol) with the largest summed match IoU
        grid = np.arange(min(xs) - tol, max(xs) + tol + 0.5, 1.0)
        c = float(max(grid, key=lambda x: sum(m[1] for m in match(boxes, occ, x))))
        m = match(boxes, occ, c)
        inside: list[int] = []
        best: list[int] = []
        for k in np.argsort(np.abs(cx - c)):
            inside.append(int(k))
            if np.mean([m[i][0] is not None for i in inside]) >= cover:
                best = list(inside)
        if not best:
            continue
        members = set(best)
        pairs = {tuple(sorted((i, m[i][0]))): m[i][1] for i in best
                 if m[i][0] is not None and m[i][0] != i and m[i][0] in members}
        if len(pairs) < min_pairs:
            continue
        strength = sum(m[i][1] for i in best if m[i][0] is not None and m[i][0] in members)
        ext = float(max(abs(cx[i] - c) + 0.5 * (boxes[i][2] - boxes[i][0]) for i in best))
        syms.append(dict(cx=c, strength=strength, extent=ext, members=sorted(best), pairs=pairs,
                         selfs=[i for i in best if m[i][0] == i]))
        seen: set[tuple[int, int]] = set()
        for mid, pair in g:
            if pair is None or pair in seen:
                continue
            seen.add(pair)
            iou_own = pair_iou(boxes, occ, pair[0], pair[1], mid)
            iou_here = pair_iou(boxes, occ, pair[0], pair[1], c)
            if iou_own >= 0.3 and iou_here < 0.3:
                lost.append((pair[0], pair[1], mid, iou_own))
    # neighbouring axes with the same windows: keep the strongest
    syms.sort(key=lambda s: -s["strength"])
    kept: list[dict[str, Any]] = []
    for s in syms:
        if not any(abs(s["cx"] - k["cx"]) <= tol for k in kept):
            kept.append(s)
    # a folded pair the chosen axis no longer matches comes back at its own midpoint
    restored: list[dict[str, Any]] = []
    lost.sort(key=lambda t: -t[3])
    for i, j, mid, iou in lost:
        if any(pair_iou(boxes, occ, i, j, s["cx"]) >= 0.3 for s in kept):
            continue
        if any(abs(mid - r["cx"]) <= tol for r in restored):
            continue
        restored.append(_one_pair_axis(boxes, cx, i, j, mid, iou))
    return sorted(kept + restored, key=lambda s: s["cx"])


def iou_matrix(boxes: np.ndarray, occ: np.ndarray, c: float) -> np.ndarray:
    """``pair_iou`` for all (i, j) at once; ``occ`` is (n, 4) bool hidden sides."""
    a = np.broadcast_to(boxes[:, None, :], (len(boxes), len(boxes), 4)).copy()
    ref = np.stack([2 * c - boxes[:, 2], boxes[:, 1], 2 * c - boxes[:, 0], boxes[:, 3]], 1)
    r = np.broadcast_to(ref[None, :, :], a.shape).copy()
    for s in range(4):
        hid = occ[:, None, s]
        r[..., s] = np.where(hid, np.maximum(r[..., s], a[..., s]) if s < 2 else np.minimum(r[..., s], a[..., s]),
                             r[..., s])
    for s in range(4):
        hid = occ[None, :, MIRROR_SIDE[s]]
        a[..., s] = np.where(hid, np.maximum(a[..., s], r[..., s]) if s < 2 else np.minimum(a[..., s], r[..., s]),
                             a[..., s])
    iw = np.clip(np.minimum(a[..., 2], r[..., 2]) - np.maximum(a[..., 0], r[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], r[..., 3]) - np.maximum(a[..., 1], r[..., 1]), 0, None)
    inter = iw * ih
    union = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1]) + (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1]) - inter
    return inter / np.maximum(union, 1e-9)


def continuous_weights(boxes: list[list[float]], occ: list[set[int]], syms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Same axes and extents, continuous weights: inside each axis's extent every window pairs with its best other
    member by reflection IoU (no 0.3 cutoff); strength = the overlap curve at the axis over those members."""
    b = np.asarray(boxes, dtype=np.float64)
    o = np.array([[s in hidden for s in range(4)] for hidden in occ], dtype=bool)
    out = []
    for s in syms:
        mem = np.array(sorted(set(s["members"]) | {i for key in s["pairs"] for i in key}))
        m = iou_matrix(b[mem], o[mem], s["cx"])
        np.fill_diagonal(m, 0.0)
        pairs: dict[tuple[int, int], float] = {}
        for a in range(len(mem)):
            c = int(np.argmax(m[a]))
            if m[a, c] > 0:
                key = (int(mem[a]), int(mem[c])) if mem[a] < mem[c] else (int(mem[c]), int(mem[a]))
                pairs[key] = max(pairs.get(key, 0.0), float(m[a, c]))
        out.append(dict(s, pairs=pairs, strength=float(m.max(1).sum()) if len(mem) > 1 else 0.0))
    return [s for s in out if s["pairs"]]


def pair_weights(boxes: list[list[float]], syms: list[dict[str, Any]]) -> np.ndarray:
    """w_ij = sum over axes pairing i and j of (axis strength / strongest) * match IoU."""
    n = len(boxes)
    w = np.zeros((n, n))
    top = max(s["strength"] for s in syms)
    for s in syms:
        for (i, j), iou in s["pairs"].items():
            w[i, j] += s["strength"] / top * iou
            w[j, i] = w[i, j]
    return w


def _axis_hidden(occ: set[int], axis: int) -> bool:
    """Width (0) is a lower bound when the left or right side is hidden; height (1) when the top or bottom is."""
    return bool(occ & ({0, 2} if axis == 0 else {1, 3}))


def _can_be(a: float, b: float, a_min: bool, tol: float) -> bool:
    """Can a window measured ``a`` be the same size as one measured ``b``? An exact window must agree within
    ``tol``. A hidden axis is a minimum: ``tol`` rejects ``b`` only when ``a`` is already the longer one."""
    if not a_min:
        return abs(a - b) / max(a, b, 1e-6) <= tol
    return a <= b * (1 + tol)


def size_mates(boxes: list[list[float]], occ: list[set[int]], tol: float = 0.12) -> list[set[int]]:
    """Per window: others it could still be, given hidden sides. Not symmetric: a short occluded box may list a
    longer window, and that longer window does not list it back."""
    wh = np.array([[b[2] - b[0], b[3] - b[1]] for b in boxes], dtype=np.float64)
    n = len(boxes)
    mates = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if all(
                _can_be(wh[i, axis], wh[j, axis], _axis_hidden(occ[i], axis), tol)
                for axis in (0, 1)
            ):
                mates[i].add(j)
    return mates


def vote(
    labels0: np.ndarray,
    w: np.ndarray,
    feat: np.ndarray,
    *,
    alpha: float = 4.0,
    cut: float = 0.16,
    iters: int = 20,
    mates: list[set[int]] | None = None,
    new_label: bool = False,
    centred: set[int] = frozenset(),
    bounded: set[int] = frozenset(),
    ids: list[int] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """``ids``: window ids used in the log (default: row index)."""
    labels = labels0.copy()
    fn = feat / np.linalg.norm(feat, axis=1, keepdims=True)
    log = []
    for _ in range(iters):
        changed = False
        for i in np.argsort(w.sum(1)):
            part = {int(labels[j]) for j in np.where(w[i] > 0)[0]}
            if mates is not None and mates[i]:
                part = {lab for lab in part if any(labels[k] == lab for k in mates[i])}
            elif i in bounded:
                part = set()
            elif i in centred:
                part = set()
            labs = sorted({int(labels[i])} | ({int(labels.max()) + 1} if new_label else set()) | part)
            cost = {}
            for lab in labs:
                mem = [k for k in np.where(labels == lab)[0] if k != i]
                d = float(1 - np.mean(fn[mem] @ fn[i])) if mem else cut
                cost[lab] = alpha * d + sum(w[i, j] for j in range(len(labels)) if j != i and labels[j] != lab)
            new = min(cost, key=cost.get)
            if new != labels[i]:
                log.append(f"unit {ids[i] if ids else i}: {labels[i]} -> {new}  costs " +
                           ", ".join(f"{k}:{v:.2f}" for k, v in sorted(cost.items())))
                labels[i] = new
                changed = True
        if not changed:
            break
    return labels, log


def local_sym_vote(
    boxes: list[list[float]],
    labels: np.ndarray,
    feat: np.ndarray,
    occluder_mask: np.ndarray | None,
    image_size: tuple[int, int],
    *,
    alpha: float = 4.0,
    cut: float = 0.16,
    mate_tol: float = 0.12,
    centred_min: float = 0.5,
    new_label: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """New type labels for all windows (windows on an occluder keep theirs) and a JSON-able report."""
    iw, ih = image_size
    labels = np.asarray(labels).copy()
    on = [occluder_mask is not None and float(occluder_mask[int(b[1]):int(b[3]), int(b[0]):int(b[2])].mean()) >= 0.5
          for b in boxes]
    idx = [i for i in range(len(boxes)) if not on[i]]
    report: dict[str, Any] = {"on_occluder": [i for i in range(len(boxes)) if on[i]], "axes": [], "moves": []}
    if len(idx) < 2:
        return labels, report
    sub = [list(map(float, boxes[i])) for i in idx]
    occ = occluded_sides(sub, occluder_mask, iw, ih)
    syms = continuous_weights(sub, occ, local_symmetries(sub, occ))
    if not syms:
        return labels, report
    top = max(s["strength"] for s in syms)
    w = pair_weights(sub, syms)
    mates = size_mates(sub, occ, mate_tol) if mate_tol > 0 else None
    centred = {i for s in syms if s["strength"] >= centred_min * top for i in s["selfs"] if not occ[i]} \
        if centred_min > 0 else set()
    bounded = {i for i, o in enumerate(occ) if o}
    new, log = vote(labels[idx], w, feat[idx], alpha=alpha, cut=cut, mates=mates, new_label=new_label,
                    centred=centred, bounded=bounded, ids=idx)
    report["axes"] = [{"cx": round(s["cx"], 1), "strength": round(s["strength"] / top, 3),
                       "extent": round(s["extent"], 1), "pairs": [[idx[i], idx[j]] for i, j in s["pairs"]]}
                      for s in syms]
    report["occluded_sides"] = {idx[i]: sorted(o) for i, o in enumerate(occ) if o}
    report["moves"] = [[idx[k], int(labels[idx[k]]), int(new[k])] for k in range(len(idx)) if new[k] != labels[idx[k]]]
    report["log"] = log
    labels[idx] = new
    return labels, report
