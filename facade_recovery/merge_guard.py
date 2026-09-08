"""Keep window units from merging across floors."""

from __future__ import annotations

from typing import Any

import numpy as np


def _x_overlap_frac(a: list[int], b: list[int]) -> float:
    ov = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    w = min(max(1.0, float(a[2] - a[0])), max(1.0, float(b[2] - b[0])))
    return ov / w


def spanning_box_indices(
    boxes: list[list[int]],
    floors: np.ndarray,
    *,
    height_mult: float = 1.5,
    min_x_overlap: float = 0.40,
) -> list[int]:
    """Boxes tall enough to cover another window's center on a different floor."""
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
) -> tuple[list[list[int]], list[list[int]]]:
    """Undo unions that mixed windows from different floor bands."""
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
        for idxs in by_floor.values():
            xs0 = min(int(raw_boxes[i][0]) for i in idxs)
            ys0 = min(int(raw_boxes[i][1]) for i in idxs)
            xs1 = max(int(raw_boxes[i][2]) for i in idxs)
            ys1 = max(int(raw_boxes[i][3]) for i in idxs)
            out_boxes.append([xs0, ys0, xs1, ys1])
            out_members.append(sorted(idxs))
    return out_boxes, out_members
