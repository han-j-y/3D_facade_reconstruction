"""Fold gate: window types from floor-wise fold symmetry, vertical alignment and the structure descriptor.

Replaces the global-axis "freeze" gate (sym_gate.gate_cluster). Steps, each switchable for ablations:

1. Fold pairs (``use_fold``). Per floor, the floor's own bays (merged x-ranges of its windows); every bay centre and
   every gap between neighbouring bays is a candidate axis. A candidate grows outward one bay per side; the extent kept
   maximises (sum IoU of matched windows - #unmatched windows). A window matches when its reflection overlaps another
   window of the same floor by IoU >= ``min_iou`` and the two can be the same size (``size_ok``: visible extents within
   ``size_ratio``, an extent with a hidden side is only a lower bound). Every axis with at least one real pair is kept
   (one floor is enough), all its pairs are used.
2. Vertical pairs (``use_vertical``): adjacent floors, centres within a quarter of the narrower width, widths within
   1.15x and heights within 1.25x for visible extents (hidden sides: lower bound).
3. Links across box groups (``use_cross``): allowed only when both windows' predicted outlines (rectangle / arched /
   other) match and the horizontal components they belong to share a majority outline; linked box groups merge.
4. Horizontal must-link: fold pairs tie windows together (within the possibly merged box group).
5. Vertical merges between components on adjacent floors: at least min(2, smaller size) aligned pairs, covering at
   least half of the smaller component, and the same majority predicted outline.
6. Structure split (``use_split``): average-linkage tree over the must-link components on the structure descriptor
   (component distance = mean pairwise cosine distance), cut at ``thr``. Off: every component is a type.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import pdist, squareform

from facade_recovery.local_sym import _axis_hidden, _can_be

Box = list[float]


def box_iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def make_size_ok(occ: list[set[int]], ratio_w: float, ratio_h: float | None = None) -> Callable[[int, int, list[Box]], bool]:
    """``size_ok(i, j, boxes)``: can windows i and j be the same size? Visible extents within the ratio, an extent with
    a hidden side (``occ``: hidden xyxy side indices per window) is a lower bound."""
    ratio_h = ratio_w if ratio_h is None else ratio_h

    def ok(i: int, j: int, boxes: list[Box]) -> bool:
        a, b = boxes[i], boxes[j]
        wh = ((a[2] - a[0], b[2] - b[0], 1 - 1 / ratio_w), (a[3] - a[1], b[3] - b[1], 1 - 1 / ratio_h))
        return all(_can_be(x, y, _axis_hidden(occ[i], k), tol) and _can_be(y, x, _axis_hidden(occ[j], k), tol)
                   for k, (x, y, tol) in enumerate(wh))
    return ok


def outline_class(ir: dict | None) -> str:
    """rect / arched / other / ? from a structure IR's boundary."""
    b = (ir or {}).get("boundary") or {}
    s = str(b.get("shape") or "")
    if not s:
        return "?"
    if s.startswith("rect"):
        return "rect"
    if "arch" in s or s == "head_body" or "semicircle" in s:
        return "arched"
    return s


# --- 1. fold pairs -------------------------------------------------------------------------------------------------

def _floor_bays(boxes: list[Box], idx: list[int]) -> list[list]:
    bays: list[list] = []
    for i in sorted(idx, key=lambda k: boxes[k][0]):
        if bays and boxes[i][0] < bays[-1][1]:
            bays[-1][1] = max(bays[-1][1], boxes[i][2]); bays[-1][2].append(i)
        else:
            bays.append([boxes[i][0], boxes[i][2], [i]])
    return bays


def _match(boxes, src, dst, c, size_ok, min_iou):
    out = []
    for i in src:
        b = boxes[i]; r = [2 * c - b[2], b[1], 2 * c - b[0], b[3]]
        best = max(((box_iou(r, boxes[j]), j) for j in dst if size_ok(i, j, boxes)), default=(0.0, None))
        out.append((i, best[1], best[0]) if best[0] >= min_iou else (i, None, 0.0))
    return out


def _region_value(recs, single):
    real = [q for q in recs if q[1] is not None and q[1] != q[0]]
    ok = [q for q in recs if q[1] is not None and (q[1] != q[0] or real or single)]
    return sum(q[2] for q in ok) - (len(recs) - len(ok)), ok


def _fold(boxes, bays, p, size_ok, min_iou):
    k = int(np.floor(p)); centre = p == k
    c0 = (bays[k][0] + bays[k][1]) / 2 if centre else (bays[k][1] + bays[k + 1][0]) / 2
    best = None
    for c in np.arange(c0 - 4, c0 + 4.5, 1.0):
        recs, d = [], 0
        while True:
            l, r = (k - d, k + d) if centre else (k - d, k + 1 + d)
            if l < 0 or r >= len(bays):
                break
            if l == r:
                recs = recs + _match(boxes, bays[l][2], bays[l][2], c, size_ok, min_iou)
            else:
                recs = recs + _match(boxes, bays[l][2], bays[r][2], c, size_ok, min_iou) \
                    + _match(boxes, bays[r][2], bays[l][2], c, size_ok, min_iou)
            v, ok = _region_value(recs, single=(l == r))
            if v > 0 and (best is None or (v, -abs(c - c0)) > (best[0], -abs(best[1][0] - c0))):
                best = (v, (float(c), ok))
            d += 1
    return None if best is None else best[1]


def fold_pairs(boxes: list[Box], floor: dict[int, int], size_ok, min_iou: float = 0.3) -> tuple[dict, list[dict]]:
    """Pairs (i < j) from every per-floor fold axis with >= 1 real pair -> {pair: best IoU}; plus the axes found."""
    pairs: dict[tuple[int, int], float] = {}
    axes = []
    for f in sorted(set(floor.values())):
        idx = [i for i in floor if floor[i] == f]
        if len(idx) < 2:
            continue
        bays = _floor_bays(boxes, idx)
        for p in np.arange(0, len(bays) - 1 + 1e-9, 0.5):
            r = _fold(boxes, bays, float(p), size_ok, min_iou)
            if r is None:
                continue
            c, ok = r
            real = [(i, j, s) for i, j, s in ok if i != j]
            if real:
                axes.append({"floor": f, "x": round(c, 1), "pairs": len(real) // 2 + len(real) % 2})
            for i, j, s in real:
                key = (min(i, j), max(i, j)); pairs[key] = max(pairs.get(key, 0.0), float(s))
    return pairs, axes


# --- 2. vertical pairs ---------------------------------------------------------------------------------------------

def vertical_pairs(boxes: list[Box], floor: dict[int, int], occ: list[set[int]]) -> list[tuple[int, int]]:
    ok = make_size_ok(occ, 1.15, 1.25)
    out = []
    for i in floor:
        for j in floor:
            if j <= i or abs(floor[i] - floor[j]) != 1:
                continue
            a, b = boxes[i], boxes[j]
            if abs((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2) <= 0.25 * min(a[2] - a[0], b[2] - b[0]) and ok(i, j, boxes):
                out.append((i, j))
    return out


# --- 3./4./5. links, box-group merging, components -----------------------------------------------------------------

class _UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _components(nodes, links, groups):
    uf = _UF(nodes)
    for i, j in links:
        if i in uf.p and j in uf.p and groups[i] == groups[j]:
            uf.union(i, j)
    comp: dict[int, list[int]] = {}
    for i in nodes:
        comp.setdefault(uf.find(i), []).append(i)
    return comp, {i: uf.find(i) for i in nodes}


def cross_group_filter(links, groups, outline: dict[int, str], hlinks) -> list[tuple[int, int]]:
    """Within-group links pass; a cross-group link needs matching outlines and matching component majority outlines."""
    comp, cid = _components(sorted(outline), hlinks, groups)
    maj = {c: Counter(outline[i] for i in m).most_common(1)[0][0] for c, m in comp.items()}
    keep = []
    for i, j in links:
        if groups[i] == groups[j]:
            keep.append((i, j))
        elif i in outline and j in outline and outline[i] == outline[j] and maj[cid[i]] == maj[cid[j]]:
            keep.append((i, j))
    return keep


def merge_groups(groups: np.ndarray, links) -> np.ndarray:
    uf = _UF([int(g) for g in set(groups.tolist())])
    for i, j in links:
        uf.union(int(groups[i]), int(groups[j]))
    return np.array([uf.find(int(g)) for g in groups])


def vertical_merges(hlinks, vlinks, nodes, groups, outline, min_pairs: int = 2, min_cover: float = 0.5):
    comp, cid = _components(nodes, hlinks, groups)
    maj = {c: Counter(outline.get(i, "?") for i in m).most_common(1)[0][0] for c, m in comp.items()}
    cnt: Counter = Counter()
    for i, j in vlinks:
        a, b = cid.get(i), cid.get(j)
        if a is not None and b is not None and a != b and groups[i] == groups[j]:
            cnt[(min(a, b), max(a, b))] += 1
    out = []
    for (a, b), n in cnt.items():
        small = min(len(comp[a]), len(comp[b]))
        if n >= min(min_pairs, small) and n >= min_cover * small and maj[a] == maj[b]:
            out.append((a, b))
    return out


# --- 6. structure split --------------------------------------------------------------------------------------------

def _split(D, idx, links, thr, use_split):
    uf = _UF(idx)
    for i, j in links:
        if i in uf.p and j in uf.p:
            uf.union(i, j)
    comps: dict[int, list[int]] = {}
    for i in idx:
        comps.setdefault(uf.find(i), []).append(i)
    comps_l = list(comps.values())
    if not use_split:
        return comps_l
    if len(comps_l) < 2:
        return [list(idx)]
    C = np.array([[D[np.ix_(a, b)].mean() if x != y else 0.0 for y, b in enumerate(comps_l)] for x, a in enumerate(comps_l)])
    root = to_tree(linkage(squareform(C, checks=False), method="average"))
    if root.dist <= thr:
        return [list(idx)]
    side = lambda n: [i for k in n.pre_order() for i in comps_l[k]]
    return _split(D, side(root.get_left()), links, thr, use_split) + _split(D, side(root.get_right()), links, thr, use_split)


def fold_gate(
    boxes: list[Box],
    groups: np.ndarray,
    floor: dict[int, int],
    feat: np.ndarray,
    occ: list[set[int]],
    outline: dict[int, str],
    *,
    thr: float = 0.16,
    size_ratio: float = 1.3,
    use_fold: bool = True,
    use_vertical: bool = True,
    use_cross: bool = True,
    use_split: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Type labels for every box (see module doc) and a JSON-able report."""
    boxes = [list(map(float, b)) for b in boxes]
    groups = np.asarray(groups)
    size_ok = make_size_ok(occ, size_ratio)
    H, axes = fold_pairs(boxes, floor, size_ok) if use_fold else ({}, [])
    hl = sorted(H)
    vl = vertical_pairs(boxes, floor, occ) if use_vertical else []
    if use_cross:
        hl = cross_group_filter(hl, groups, outline, hl)
        vl = cross_group_filter(vl, groups, outline, hl)
        sgroups = merge_groups(groups, hl + vl)
    else:
        hl = [(i, j) for i, j in hl if groups[i] == groups[j]]
        vl = [(i, j) for i, j in vl if groups[i] == groups[j]]
        sgroups = groups
    vm = vertical_merges(hl, vl, sorted(floor), sgroups, outline) if use_vertical else []
    D = squareform(pdist(np.asarray(feat), metric="cosine"))
    labels = np.zeros(len(boxes), dtype=np.int64); nxt = 0
    for g in sorted(set(sgroups.tolist())):
        idx = [int(i) for i in np.where(sgroups == g)[0]]
        for part in _split(D, idx, hl + vm, thr, use_split):
            labels[part] = nxt; nxt += 1
    report = {"fold_axes": axes, "fold_pairs": [[i, j] for i, j in hl], "vertical_pairs": [[i, j] for i, j in vl],
              "vertical_merges": [[int(a), int(b)] for a, b in vm],
              "box_groups": int(len(set(groups.tolist()))), "merged_groups": int(len(set(sgroups.tolist()))),
              "flags": {"fold": use_fold, "vertical": use_vertical, "cross": use_cross, "split": use_split,
                        "size_ratio": size_ratio}}
    return labels, report
