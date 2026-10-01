"""Keep window units from merging across floors."""

from __future__ import annotations

from typing import Any

import numpy as np


def _x_overlap_frac(a: list[int], b: list[int]) -> float:
    ov = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    w = min(max(1.0, float(a[2] - a[0])), max(1.0, float(b[2] - b[0])))
    return ov / w


def _box_area(b: list[int]) -> float:
    return max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))


def _intersection_area(a: list[int], b: list[int]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(x1 - x0) * float(y1 - y0)


def boxes_nested(
    a: list[int],
    b: list[int],
    *,
    contain_thr: float = 0.55,
) -> bool:
    """True if the smaller box is mostly inside the larger one."""
    aa, ba = _box_area(a), _box_area(b)
    if aa <= 0 or ba <= 0:
        return False
    ia = _intersection_area(a, b)
    return ia / min(aa, ba) >= contain_thr


def spanning_box_indices(
    boxes: list[list[int]],
    floors: np.ndarray,
    *,
    height_mult: float = 1.5,
    min_x_overlap: float = 0.40,
    contain_thr: float = 0.55,
) -> list[int]:
    """Tall boxes that cover a *non-nested* window center on another floor.

    Nested full-opening + inner-pane pairs are kept (merge will absorb the pane).
    """
    if not boxes:
        return []
    hs = [max(1.0, float(b[3] - b[1])) for b in boxes]
    med_h = float(np.median(hs))
    drop: list[int] = []
    for i, bi in enumerate(boxes):
        if hs[i] <= height_mult * med_h:
            continue
        for j, bj in enumerate(boxes):
            if i == j or int(floors[i]) == int(floors[j]):
                continue
            if boxes_nested(bi, bj, contain_thr=contain_thr):
                continue
            if _x_overlap_frac(bi, bj) < min_x_overlap:
                continue
            cj = 0.5 * (float(bj[1]) + float(bj[3]))
            if float(bi[1]) < cj < float(bi[3]):
                drop.append(i)
                break
    return drop


def filter_indices(items: list[Any], keep: list[int]) -> list[Any]:
    return [items[i] for i in keep]


def split_member_groups_by_floor(
    raw_boxes: list[list[int]],
    merged_boxes: list[list[int]],
    members: list[list[int]],
    floors: np.ndarray,
    *,
    contain_thr: float = 0.55,
) -> tuple[list[list[int]], list[list[int]]]:
    """Undo cross-floor unions unless held together by containment nesting.

    Adjacent/close-by merges must stay same-floor; nested full-opening + pane
    may span greedy floor bands and should remain one unit.
    """
    out_boxes: list[list[int]] = []
    out_members: list[list[int]] = []
    for box, mem in zip(merged_boxes, members):
        by_floor: dict[int, list[int]] = {}
        for i in mem:
            by_floor.setdefault(int(floors[i]), []).append(i)
        if len(by_floor) <= 1:
            out_boxes.append(box)
            out_members.append(mem)
            continue

        nested_ok = False
        floor_ids = list(by_floor.keys())
        for fi, f_a in enumerate(floor_ids):
            for f_b in floor_ids[fi + 1 :]:
                for i in by_floor[f_a]:
                    for j in by_floor[f_b]:
                        if boxes_nested(
                            raw_boxes[i], raw_boxes[j], contain_thr=contain_thr
                        ):
                            nested_ok = True
                            break
                    if nested_ok:
                        break
                if nested_ok:
                    break
            if nested_ok:
                break
        if nested_ok:
            out_boxes.append(box)
            out_members.append(mem)
            continue

        for idxs in by_floor.values():
            xs0 = min(int(raw_boxes[i][0]) for i in idxs)
            ys0 = min(int(raw_boxes[i][1]) for i in idxs)
            xs1 = max(int(raw_boxes[i][2]) for i in idxs)
            ys1 = max(int(raw_boxes[i][3]) for i in idxs)
            out_boxes.append([xs0, ys0, xs1, ys1])
            out_members.append(sorted(idxs))
    return out_boxes, out_members
