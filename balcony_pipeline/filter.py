"""Post-SAM3 filters for balcony boxes (wall-rail false positives).

A box below a window sill stays. A box with no window above it stays.

Among the boxes, drop the smaller one when two balcony boxes overlap
(intersection area > 0). The larger box stays. Equal areas both stay.

Juliet balconies stay, including a box that sits mostly inside a window.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from snap import _bands_from_window_instances


def _as_box(box: list[int] | list[float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = [float(v) for v in box]
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def _box_center_y(box: list[int] | list[float]) -> float:
    _, y0, _, y1 = _as_box(box)
    return 0.5 * (y0 + y1)


def _shrink_box_y(
    box: tuple[float, float, float, float],
    *,
    frac: float = 0.10,
) -> tuple[float, float, float, float]:
    """Inset top/bottom by 0.5*frac*height each (X unchanged)."""
    x0, y0, x1, y1 = box
    bh = max(1.0, y1 - y0)
    dy = 0.5 * float(frac) * bh
    sy0 = y0 + dy
    sy1 = y1 - dy
    if sy1 <= sy0:
        mid = 0.5 * (y0 + y1)
        return x0, mid, x1, mid
    return x0, sy0, x1, sy1


def _expand_box(
    box: tuple[float, float, float, float],
    *,
    frac: float = 0.10,
    iw: float,
    ih: float,
) -> tuple[float, float, float, float]:
    """Expand left/right and top/bottom by 0.5*frac*size each."""
    x0, y0, x1, y1 = box
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    dx, dy = 0.5 * float(frac) * bw, 0.5 * float(frac) * bh
    return (
        max(0.0, x0 - dx),
        max(0.0, y0 - dy),
        min(iw, x1 + dx),
        min(ih, y1 + dy),
    )


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _boxes_overlap_2d(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return (
        _overlap_1d(ax0, ax1, bx0, bx1) > 0
        and _overlap_1d(ay0, ay1, by0, by1) > 0
    )


def _box_area(box: list[int] | list[float] | tuple[float, ...]) -> float:
    x0, y0, x1, y1 = _as_box(list(box))
    return max(1.0, (x1 - x0) * (y1 - y0))


def _intersection_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    w = _overlap_1d(ax0, ax1, bx0, bx1)
    h = _overlap_1d(ay0, ay1, by0, by1)
    return w * h


def smaller_overlap_indices(boxes: list[list[int]]) -> set[int]:
    """Indices of boxes that overlap a strictly larger box.

    Overlap is a positive intersection area. Touching edges stay. When two
    boxes have the same area, both stay.
    """
    areas = [_box_area(b) for b in boxes]
    parsed = [_as_box(b) for b in boxes]
    drop: set[int] = set()
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i == j or areas[j] <= areas[i]:
                continue
            if _intersection_area(parsed[i], parsed[j]) > 0:
                drop.add(i)
                break
    return drop


def clearly_below_windows(
    box: list[int] | list[float],
    window_boxes: list[list[int]],
    *,
    iw: int,
    ih: int,
    shrink_frac: float = 0.10,
    expand_frac: float = 0.10,
    margin_frac: float = 0.0,
) -> bool:
    """True if balcony top is below the Y-shrunk overlapping window band.

    Overlap uses the window box expanded ~10% in X and Y; the sill threshold
    uses a ~10% shorter window in Y only.
    """
    b = _as_box(box)
    y0 = b[1]
    margin = float(margin_frac) * float(ih)

    band_bottoms: list[float] = []
    for wb in window_boxes:
        raw = _as_box(wb)
        expanded = _expand_box(raw, frac=expand_frac, iw=float(iw), ih=float(ih))
        if not _boxes_overlap_2d(b, expanded):
            continue
        _sx0, _sy0, _sx1, sy1 = _shrink_box_y(raw, frac=shrink_frac)
        band_bottoms.append(sy1)

    if not band_bottoms:
        return False

    return y0 > max(band_bottoms) + margin


def _balcony_bay_ids(
    box: list[int] | list[float],
    window_instances: list[dict[str, Any]],
    *,
    min_frac: float = 0.15,
) -> set[int]:
    """Bay ids whose X band overlaps the balcony width (same rule as ``snap_box``)."""
    if not window_instances:
        return set()
    _floor_y, bay_x = _bands_from_window_instances(window_instances)
    if not bay_x:
        return set()
    x0, _, x1, _ = _as_box(box)
    bw = max(1.0, x1 - x0)
    hits: list[int] = []
    for bid, (bx0, bx1) in bay_x.items():
        if _overlap_1d(x0, x1, bx0, bx1) / bw >= float(min_frac):
            hits.append(int(bid))
    if hits:
        return set(hits)
    cx = 0.5 * (x0 + x1)
    nearest = min(bay_x, key=lambda b: abs(cx - 0.5 * sum(bay_x[b])))
    return {int(nearest)}


def has_window_center_above(
    box: list[int] | list[float],
    window_instances: list[dict[str, Any]],
    *,
    ih: int,
    margin_frac: float = 0.0,
    bay_min_frac: float = 0.15,
) -> bool:
    """True if some window in the balcony bay span is above the balcony center.

    Image Y grows downward, so a higher window center has a smaller cy than
    the balcony. Side-bay or attic windows outside the balcony span are ignored.
    """
    if not window_instances:
        return False

    cy = _box_center_y(box)
    margin = float(margin_frac) * float(ih)
    bay_ids = _balcony_bay_ids(box, window_instances, min_frac=bay_min_frac)
    for u in window_instances:
        wb = u.get("box_xyxy")
        if not wb:
            continue
        if int(u.get("bay", -1)) not in bay_ids:
            continue
        if _box_center_y(wb) < cy - margin:
            return True
    return False


def no_window_above(
    box: list[int] | list[float],
    window_instances: list[dict[str, Any]],
    *,
    ih: int,
    margin_frac: float = 0.0,
    bay_min_frac: float = 0.15,
) -> bool:
    """True when no window in the balcony bay span lies above the balcony center."""
    return not has_window_center_above(
        box,
        window_instances,
        ih=ih,
        margin_frac=margin_frac,
        bay_min_frac=bay_min_frac,
    )


def filter_balcony_boxes(
    boxes: list[list[int]],
    facade: Image.Image,
    window_instances: list[dict[str, Any]],
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Return (kept_boxes, drop_log)."""
    del facade, window_instances

    preliminary: list[tuple[list[int], list[str], bool]] = []
    for box in boxes:
        b = [int(v) for v in box]
        preliminary.append((b, [], True))

    survivors = [i for i, (_b, _r, keep) in enumerate(preliminary) if keep]
    overlap_drop = {
        survivors[j]
        for j in smaller_overlap_indices([preliminary[i][0] for i in survivors])
    }

    kept: list[list[int]] = []
    log: list[dict[str, Any]] = []
    for i, (b, reasons, keep) in enumerate(preliminary):
        if i in overlap_drop:
            reasons = [*reasons, "smaller_overlap"]
            keep = False
        log.append({"box": b, "keep": keep, "reasons": reasons})
        if keep:
            kept.append(b)
    return kept, log
