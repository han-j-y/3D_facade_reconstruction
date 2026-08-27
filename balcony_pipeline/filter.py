"""Post-SAM3 filters for balcony boxes (wall-rail false positives).

Drop a candidate if:
  1. Railing-like: its top lies below the Y-shrunk overlapping window band
     (windows: ~10% expanded in X/Y for overlap; Y shrunk ~10% for sill).
     Skip this drop when the balcony is >=20% wider than an overlapping partner
     window (real multi-bay / wide slab cue).
  2. Juliet-like: width is close to a partner window (default 0.85–1.15×).
  3. Rooftop / top rail: drop when no window in the balcony's bay span has its
     center higher than the balcony center (image Y up = smaller cy).
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


def _box_width(box: list[int] | list[float] | tuple[float, ...]) -> float:
    x0, _, x1, _ = _as_box(list(box))
    return max(1.0, x1 - x0)


def _partner_windows(
    box: tuple[float, float, float, float],
    window_boxes: list[list[int]],
    *,
    iw: int,
    ih: int,
    expand_frac: float,
) -> list[tuple[float, float, float, float]]:
    """Raw window boxes that 2D-overlap the candidate vs an expanded window."""
    partners: list[tuple[float, float, float, float]] = []
    for wb in window_boxes:
        raw = _as_box(wb)
        expanded = _expand_box(raw, frac=expand_frac, iw=float(iw), ih=float(ih))
        if _boxes_overlap_2d(box, expanded):
            partners.append(raw)
    return partners


def wider_than_partner_window(
    box: list[int] | list[float],
    window_boxes: list[list[int]],
    *,
    iw: int,
    ih: int,
    expand_frac: float = 0.10,
    min_width_ratio: float = 1.20,
) -> bool:
    """True if balcony width is at least min_width_ratio times a partner window."""
    b = _as_box(box)
    bw = _box_width(b)
    for raw in _partner_windows(
        b, window_boxes, iw=iw, ih=ih, expand_frac=expand_frac
    ):
        if bw >= float(min_width_ratio) * _box_width(raw):
            return True
    return False


def similar_width_to_partner(
    box: list[int] | list[float],
    window_boxes: list[list[int]],
    *,
    iw: int,
    ih: int,
    expand_frac: float = 0.10,
    min_ratio: float = 0.85,
    max_ratio: float = 1.15,
) -> bool:
    """True if balcony width is close to a partner window (Juliet-like).

    Uses the same partner pairing as decoration filters (10% expanded window
    overlap). Drop cue when ratio is in ``[min_ratio, max_ratio]`` inclusive.
    """
    b = _as_box(box)
    bw = _box_width(b)
    lo, hi = float(min_ratio), float(max_ratio)
    for raw in _partner_windows(
        b, window_boxes, iw=iw, ih=ih, expand_frac=expand_frac
    ):
        ratio = bw / _box_width(raw)
        if lo <= ratio <= hi:
            return True
    return False


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
    *,
    shrink_frac: float = 0.10,
    expand_frac: float = 0.10,
    above_margin_frac: float = 0.0,
    min_width_ratio: float = 1.20,
    juliet_min_ratio: float = 0.85,
    juliet_max_ratio: float = 1.15,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Return (kept_boxes, drop_log)."""
    iw, ih = facade.size
    win_boxes = [
        [int(v) for v in (u.get("box_xyxy") or [])]
        for u in window_instances
        if u.get("box_xyxy")
    ]

    kept: list[list[int]] = []
    log: list[dict[str, Any]] = []
    for box in boxes:
        b = [int(v) for v in box]
        below_raw = clearly_below_windows(
            b,
            win_boxes,
            iw=iw,
            ih=ih,
            shrink_frac=shrink_frac,
            expand_frac=expand_frac,
        )
        wide_keep = below_raw and wider_than_partner_window(
            b,
            win_boxes,
            iw=iw,
            ih=ih,
            expand_frac=expand_frac,
            min_width_ratio=min_width_ratio,
        )
        below = below_raw and not wide_keep
        juliet = similar_width_to_partner(
            b,
            win_boxes,
            iw=iw,
            ih=ih,
            expand_frac=expand_frac,
            min_ratio=juliet_min_ratio,
            max_ratio=juliet_max_ratio,
        )
        rooftop = no_window_above(
            b, window_instances, ih=ih, margin_frac=above_margin_frac
        )
        reasons: list[str] = []
        if below:
            reasons.append("below_windows")
        if below_raw and wide_keep:
            reasons.append("wide_vs_window")
        if juliet:
            reasons.append("juliet_width")
        if rooftop:
            reasons.append("no_window_above")
        keep = (not below) and (not juliet) and (not rooftop)
        log.append({"box": b, "keep": keep, "reasons": reasons})
        if keep:
            kept.append(b)
    return kept, log
