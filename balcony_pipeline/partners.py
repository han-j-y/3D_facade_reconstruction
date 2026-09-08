"""Pair balcony units with window instances from the recovery DSL."""

from __future__ import annotations

from typing import Any

from filter import _as_box, _box_center_y, _boxes_overlap_2d, _expand_box, _overlap_1d


def _supporting_row(
    partners: list[dict[str, Any]],
    balcony_cy: float,
) -> list[dict[str, Any]]:
    """Keep partner windows on the row directly above the balcony (same Y band)."""
    above: list[tuple[float, dict[str, Any]]] = []
    for w in partners:
        box = w.get("box_xyxy")
        if not box:
            continue
        cy = _box_center_y(box)
        if cy < balcony_cy:
            above.append((balcony_cy - cy, w))
    if not above:
        return partners
    min_gap = min(g for g, _ in above)
    tol = max(8.0, min_gap * 0.20)
    return [w for g, w in above if g <= min_gap + tol]


def _overlap_partners(
    balcony: tuple[float, float, float, float],
    instances: list[dict[str, Any]],
    *,
    iw: float,
    ih: float,
    expand_frac: float = 0.10,
) -> list[dict[str, Any]]:
    """Windows whose expanded box 2D-overlaps the balcony (same rule as filter)."""
    hits: list[dict[str, Any]] = []
    for w in instances:
        box = w.get("box_xyxy")
        if not box:
            continue
        raw = _as_box(box)
        expanded = _expand_box(raw, frac=expand_frac, iw=iw, ih=ih)
        if _boxes_overlap_2d(balcony, expanded):
            hits.append(w)
    return hits


def _bay_span_partners(
    balcony: tuple[float, float, float, float],
    instances: list[dict[str, Any]],
    *,
    bay_ids: set[int],
    balcony_cy: float,
    floor_id: int | None = None,
) -> list[dict[str, Any]]:
    """Windows in bay span, horizontally aligned and above the balcony center."""
    hits: list[dict[str, Any]] = []
    bx0, _, bx1, _ = balcony
    for w in instances:
        if floor_id is not None and int(w.get("floor", -1)) != floor_id:
            continue
        if int(w.get("bay", -2)) not in bay_ids:
            continue
        box = w.get("box_xyxy")
        if not box:
            continue
        raw = _as_box(box)
        if _box_center_y(box) >= balcony_cy:
            continue
        if _overlap_1d(bx0, bx1, raw[0], raw[2]) <= 0.0:
            continue
        hits.append(w)
    return hits


def partner_windows_for_unit(
    unit: dict[str, Any],
    instances: list[dict[str, Any]],
    *,
    center_bays: list[int],
    span_bays: list[int],
    image_size: tuple[int, int] | list[int] | None = None,
) -> list[dict[str, Any]]:
    """Windows paired with a balcony unit for horizontal center placement.

    Priority (matches ``filter._partner_windows``):
      1. 2D overlap with 10% expanded window boxes; nearest row above balcony.
      2. Bay span + above balcony (ignore floor — snap can be off).
      3. Same floor + bay span + above balcony.
    """
    if not instances:
        return []

    box = unit.get("box_xyxy")
    if not box:
        return []

    iw = ih = 1.0
    if image_size and len(image_size) >= 2:
        iw = max(1.0, float(image_size[0]))
        ih = max(1.0, float(image_size[1]))
    elif image_size and len(image_size) >= 1:
        iw = max(1.0, float(image_size[0]))

    b = _as_box(box)
    cy_b = _box_center_y(box)
    span_set = {int(b) for b in span_bays} or {int(b) for b in center_bays}
    floor_i = int(unit.get("floor", 0))

    overlap_hits = _overlap_partners(b, instances, iw=iw, ih=ih)
    if overlap_hits:
        return _supporting_row(overlap_hits, cy_b)

    above_span = _bay_span_partners(
        b, instances, bay_ids=span_set, balcony_cy=cy_b, floor_id=None
    )
    if above_span:
        return _supporting_row(above_span, cy_b)

    above_floor = _bay_span_partners(
        b, instances, bay_ids=span_set, balcony_cy=cy_b, floor_id=floor_i
    )
    if above_floor:
        return _supporting_row(above_floor, cy_b)

    return []


def mean_window_center_norm(
    partners: list[dict[str, Any]],
    image_width: float,
) -> float | None:
    """Mean horizontal center of partner window boxes, normalized to image width."""
    iw = max(1.0, float(image_width))
    centers: list[float] = []
    for w in partners:
        box = w.get("box_xyxy")
        if not box:
            continue
        x0, _, x1, _ = _as_box(box)
        centers.append(0.5 * (x0 + x1))
    if not centers:
        return None
    return sum(centers) / len(centers) / iw
