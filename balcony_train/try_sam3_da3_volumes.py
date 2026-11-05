"""Detect the hand-drawn enclosed volumes with SAM3 seeds + DA3 local depth.

Does not replace the window pipeline. SAM3 finds candidate rectangles
(often one storey). Local wall-vs-box depth keeps projecting volumes, then
same-column boxes are merged across about one storey of gap. Stage 4 may
extend a box downward while that strip stays closer than the wall. The top
edge stays put.

The four photos and red-box targets are the attached depth-map markup:

    cmp_b0007  left oriel + right oriel + center gallery
    cmp_b0008  lower projecting bay
    cmp_b0010  three-storey glazed gallery
    cmp_b0223  left and right oriel columns

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/try_sam3_da3_volumes.py ^
      --device cuda --in-dir "data\\test for enclosure" ^
      --depth-dir runs\\da3_enclosure --out-dir runs\\sam3_da3_volumes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.extract_enclosed import (  # noqa: E402
    MIN_HEIGHT_FRAC,
    MAX_WIDTH_FRAC,
    foliage_mask,
    storey_width_reason,
)
from balcony_train.recrop_sam3 import (  # noqa: E402
    clamp_box_xyxy,
    detect_balcony_records,
    list_input_images,
)
from balcony_train.try_da3_enclosure import (  # noqa: E402
    DEFAULT_IN_DIR,
    is_closer,
    resize_depth,
    roi_median,
)
from balcony_train.try_sam3_prompt import merge_boxes  # noqa: E402

DEFAULT_DEPTH_DIR = ROOT / "runs" / "da3_enclosure"
DEFAULT_OUT_DIR = ROOT / "runs" / "sam3_da3_volumes"
DEFAULT_DSL_DIR = ROOT / "runs" / "batch"
STEMS = ("cmp_b0007", "cmp_b0008", "cmp_b0010", "cmp_b0223")

# Normalized xyxy matching the red markup on the DA3 color maps.
RED_GT: dict[str, tuple[dict[str, Any], ...]] = {
    "cmp_b0007": (
        {"name": "left_oriel", "box": (0.01, 0.26, 0.26, 0.56)},
        {"name": "right_oriel", "box": (0.74, 0.26, 0.99, 0.50)},
        {"name": "center_gallery", "box": (0.27, 0.50, 0.68, 0.78)},
    ),
    "cmp_b0008": (
        {"name": "lower_bay", "box": (0.28, 0.64, 0.66, 0.82)},
    ),
    "cmp_b0010": (
        {"name": "gallery_stack", "box": (0.26, 0.32, 0.72, 0.78)},
    ),
    "cmp_b0223": (
        {"name": "left_oriel", "box": (0.04, 0.10, 0.24, 0.38)},
        {"name": "right_oriel", "box": (0.68, 0.08, 0.90, 0.62)},
    ),
}

SAM3_PROMPTS = (
    "enclosed balcony",
    "closed balcony",
    "glazed balcony",
    "oriel window",
    "bay window",
)
EXCLUDE_PROMPTS = ("fire escape",)
EXCLUDE_COVER = 0.40

MIN_REL = 0.015
GROW_REL = 0.006
MIN_SEED_HEIGHT_FRAC = 0.055
# Pipeline floor height: drop Stage-1 crops at or below this fraction of one storey.
SEED_MIN_FLOOR_FRAC = 0.85
MIN_WIDTH_FRAC = 0.05
MAX_SEED_WIDTH_FRAC = 0.55
MAX_SEED_HEIGHT_FRAC = 0.62
MIN_COLUMN_FRAC = 0.10
# Clear space Stage 3 may bridge between same-column seeds, about one storey.
MERGE_Y_GAP_FRAC = 0.22
MATCH_IOU = 0.30


def box_iou(
    a: list[int] | tuple[int, int, int, int],
    b: list[int] | tuple[int, int, int, int],
) -> float:
    ax0, ay0, ax1, ay1 = (int(v) for v in a)
    bx0, by0, bx1, by1 = (int(v) for v in b)
    iw = min(ax1, bx1) - max(ax0, bx0)
    ih = min(ay1, by1) - max(ay0, by0)
    inter = max(0, iw) * max(0, ih)
    aa = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    ba = max(0, bx1 - bx0) * max(0, by1 - by0)
    denom = aa + ba - inter
    return float(inter) / float(denom) if denom > 0 else 0.0


def x_overlap_frac(
    a: list[int] | tuple[int, int, int, int],
    b: list[int] | tuple[int, int, int, int],
) -> float:
    aw = max(1, int(a[2]) - int(a[0]))
    bw = max(1, int(b[2]) - int(b[0]))
    overlap = min(int(a[2]), int(b[2])) - max(int(a[0]), int(b[0]))
    return max(0.0, float(overlap) / float(min(aw, bw)))


def same_column(
    a: list[int] | tuple[int, int, int, int],
    b: list[int] | tuple[int, int, int, int],
    *,
    min_each: float = 0.40,
) -> bool:
    """True only if the overlap is a large share of *both* widths (not a wide box nicking a column)."""
    aw = max(1, int(a[2]) - int(a[0]))
    bw = max(1, int(b[2]) - int(b[0]))
    overlap = min(int(a[2]), int(b[2])) - max(int(a[0]), int(b[0]))
    if overlap <= 0:
        return False
    return (overlap / aw) >= min_each and (overlap / bw) >= min_each


def y_gap(
    a: list[int] | tuple[int, int, int, int],
    b: list[int] | tuple[int, int, int, int],
) -> int:
    if int(a[3]) < int(b[1]):
        return int(b[1]) - int(a[3])
    if int(b[3]) < int(a[1]):
        return int(a[1]) - int(b[3])
    return 0


def union_box(
    a: list[int] | tuple[int, int, int, int],
    b: list[int] | tuple[int, int, int, int],
) -> list[int]:
    return [
        min(int(a[0]), int(b[0])),
        min(int(a[1]), int(b[1])),
        max(int(a[2]), int(b[2])),
        max(int(a[3]), int(b[3])),
    ]


def gt_px(stem: str, width: int, height: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in RED_GT.get(stem, ()):
        x0, y0, x1, y1 = item["box"]
        box = [
            int(round(x0 * width)),
            int(round(y0 * height)),
            int(round(x1 * width)),
            int(round(y1 * height)),
        ]
        out.append({"name": item["name"], "box_xyxy": box})
    return out


def robust_median(
    depth: np.ndarray,
    box: list[int] | tuple[int, int, int, int],
    ignore: np.ndarray | None = None,
) -> float | None:
    x0, y0, x1, y1 = (int(v) for v in box)
    patch = np.asarray(depth, dtype=np.float32)[y0:y1, x0:x1]
    valid = np.isfinite(patch) & (patch > 0)
    if ignore is not None:
        valid &= ~np.asarray(ignore, dtype=bool)[y0:y1, x0:x1]
    vals = patch[valid]
    if vals.size < 8:
        if ignore is not None:
            return None
        return roi_median(depth, (x0, y0, x1, y1))
    lo, hi = np.percentile(vals, (20.0, 85.0))
    mid = vals[(vals >= lo) & (vals <= hi)]
    if mid.size == 0:
        mid = vals
    return float(np.median(mid))


def local_wall_median(
    depth: np.ndarray,
    box: list[int] | tuple[int, int, int, int],
    *,
    veg: np.ndarray | None = None,
    farther_side: bool = False,
) -> float | None:
    """Median depth of the facade beside the box, ignoring trees and the box itself."""
    arr = np.asarray(depth, dtype=np.float32)
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in box)
    bw = max(8, x1 - x0)
    pad = max(12, int(round(1.1 * bw)))
    gap = max(6, int(round(0.18 * bw)))
    y0s = max(0, y0 + int(0.08 * (y1 - y0)))
    y1s = min(h, y1 - int(0.08 * (y1 - y0)))
    if y1s <= y0s + 2:
        y0s, y1s = max(0, y0), min(h, y1)
    strips: list[np.ndarray] = []
    side_meds: list[float] = []
    for xa, xb in (
        (max(0, x0 - gap - pad), max(0, x0 - gap)),
        (min(w, x1 + gap), min(w, x1 + gap + pad)),
    ):
        if xb <= xa + 2:
            continue
        patch = arr[y0s:y1s, xa:xb]
        if veg is not None:
            patch = patch[~veg[y0s:y1s, xa:xb]]
        else:
            patch = patch.ravel()
        vals = np.asarray(patch, dtype=np.float32).ravel()
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size < 8:
            continue
        strips.append(vals)
        lo, hi = np.percentile(vals, (25.0, 90.0))
        mid = vals[(vals >= lo) & (vals <= hi)]
        if mid.size == 0:
            mid = vals
        side_meds.append(float(np.median(mid)))
    if farther_side and side_meds:
        return max(side_meds)
    if not strips:
        return None
    vals = np.concatenate(strips)
    if vals.size < 12:
        return None
    lo, hi = np.percentile(vals, (25.0, 90.0))
    mid = vals[(vals >= lo) & (vals <= hi)]
    if mid.size == 0:
        mid = vals
    return float(np.median(mid))


def box_cover_frac(
    inner: list[int] | tuple[int, int, int, int],
    outer: list[int] | tuple[int, int, int, int],
) -> float:
    """Intersection area / area(inner)."""
    ax0, ay0, ax1, ay1 = (int(v) for v in inner)
    bx0, by0, bx1, by1 = (int(v) for v in outer)
    iw = min(ax1, bx1) - max(ax0, bx0)
    ih = min(ay1, by1) - max(ay0, by0)
    inter = max(0, iw) * max(0, ih)
    area = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    return float(inter) / float(area) if area > 0 else 0.0


def obstacle_ignore(
    depth: np.ndarray,
    box: list[int] | tuple[int, int, int, int],
    *,
    veg: np.ndarray | None,
    wall: float | None,
) -> np.ndarray | None:
    """Foliage plus pixels much closer than the local wall (poles, lamps)."""
    ignore = None
    if veg is not None:
        ignore = np.asarray(veg, dtype=bool)
    if wall is not None and wall > 0:
        close = clutter_mask(depth, wall)
        ignore = close if ignore is None else (ignore | close)
    return ignore


def box_rel_delta(
    depth: np.ndarray,
    box: list[int] | tuple[int, int, int, int],
    *,
    veg: np.ndarray | None = None,
) -> float | None:
    d_wall = local_wall_median(depth, box, veg=veg)
    if d_wall is None or d_wall <= 0:
        return None
    ignore = obstacle_ignore(depth, box, veg=veg, wall=d_wall)
    d_box = robust_median(depth, box, ignore=ignore)
    if d_box is None:
        return None
    return (d_wall - d_box) / d_wall


def front_fill_frac(
    depth: np.ndarray,
    box: list[int] | tuple[int, int, int, int],
    *,
    veg: np.ndarray | None = None,
    min_rel: float = MIN_REL,
) -> float | None:
    """Fraction of box pixels at least ``min_rel`` closer than the local wall."""
    d_wall = local_wall_median(depth, box, veg=veg)
    if d_wall is None or d_wall <= 0:
        return None
    x0, y0, x1, y1 = (int(v) for v in box)
    patch = np.asarray(depth, dtype=np.float32)[y0:y1, x0:x1]
    valid = np.isfinite(patch) & (patch > 0)
    if int(valid.sum()) < 8:
        return None
    rel = (d_wall - patch) / d_wall
    return float(np.mean(rel[valid] >= float(min_rel)))


def veg_frac_in_box(veg: np.ndarray | None, box: list[int] | tuple[int, int, int, int]) -> float:
    if veg is None:
        return 0.0
    x0, y0, x1, y1 = (int(v) for v in box)
    patch = veg[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(np.mean(patch))


def clutter_mask(depth: np.ndarray, wall: float, *, max_rel: float = 0.22) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(arr) & (arr > 0) & (wall > 0)
    return valid & (((wall - arr) / wall) > max_rel)


def clutter_frac(
    mask: np.ndarray | None, box: list[int] | tuple[int, int, int, int]
) -> float:
    if mask is None:
        return 0.0
    x0, y0, x1, y1 = (int(v) for v in box)
    patch = mask[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(np.mean(patch))


def x_gap(
    a: list[int] | tuple[int, int, int, int],
    b: list[int] | tuple[int, int, int, int],
) -> int:
    if int(a[2]) < int(b[0]):
        return int(b[0]) - int(a[2])
    if int(b[2]) < int(a[0]):
        return int(a[0]) - int(b[2])
    return 0


def y_overlap_frac(
    a: list[int] | tuple[int, int, int, int],
    b: list[int] | tuple[int, int, int, int],
) -> float:
    overlap = min(int(a[3]), int(b[3])) - max(int(a[1]), int(b[1]))
    if overlap <= 0:
        return 0.0
    ha = max(1, int(a[3]) - int(a[1]))
    hb = max(1, int(b[3]) - int(b[1]))
    return float(overlap) / float(min(ha, hb))


def starts_high(box: list[int] | tuple[int, int, int, int], ih: int) -> bool:
    return int(box[1]) < 0.38 * ih


def spans_ground(box: list[int] | tuple[int, int, int, int], ih: int) -> bool:
    x0, y0, x1, y1 = (int(v) for v in box)
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    return y1 > 0.72 * ih and bh > 1.6 * bw


def is_shop_floor(box: list[int] | tuple[int, int, int, int], ih: int) -> bool:
    return int(box[1]) > 0.62 * ih and int(box[3]) > 0.82 * ih


def clip_shop_span(box: list[int], ih: int) -> list[int]:
    """A mid-facade volume must not keep the ground-floor shop below it."""
    x0, y0, x1, y1 = [int(v) for v in box]
    if y0 < 0.55 * ih and y1 > 0.80 * ih:
        y1 = min(y1, int(round(0.78 * ih)))
    return [x0, y0, x1, y1]


def strip_projecting(
    depth: np.ndarray,
    probe: list[int] | tuple[int, int, int, int],
    *,
    wall0: float,
    seed_med: float | None,
    veg: np.ndarray | None,
    clutter: np.ndarray | None,
    min_rel: float,
    max_rel: float = 0.20,
    max_med_shift: float = 0.18,
    min_side: int = 4,
) -> bool:
    """True if a probe strip is still the same projecting volume as the seed."""
    x0, y0, x1, y1 = (int(v) for v in probe)
    if (x1 - x0) < min_side or (y1 - y0) < min_side:
        return False
    if veg_frac_in_box(veg, probe) > 0.35:
        return False
    if clutter_frac(clutter, probe) > 0.28:
        return False
    med = robust_median(depth, probe)
    if med is None or wall0 <= 0:
        return False
    rel = (wall0 - med) / wall0
    if rel < min_rel or rel > max_rel:
        return False
    if seed_med is not None and seed_med > 0:
        if abs(med - seed_med) / seed_med > max_med_shift:
            return False
    return True


def _axis_grow(
    depth: np.ndarray,
    box: list[int],
    *,
    wall0: float,
    seed_med: float | None,
    veg: np.ndarray | None,
    clutter: np.ndarray | None,
    min_rel: float,
    step: int,
    vertical: bool,
    max_extra: int,
    max_size: int,
    down_only: bool = False,
) -> list[int]:
    h, w = depth.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]

    def probe_ok(xa: int, ya: int, xb: int, yb: int) -> bool:
        return strip_projecting(
            depth,
            (xa, ya, xb, yb),
            wall0=wall0,
            seed_med=seed_med,
            veg=veg,
            clutter=clutter,
            min_rel=min_rel,
            min_side=max(4, step // 2),
        )

    def try_grow(cur: int, lo: int, hi: int, decreasing: bool) -> int | None:
        for k in range(1, 6):
            jump = max(lo, cur - k * step) if decreasing else min(hi, cur + k * step)
            jump = min(max(jump, lo), hi)
            if jump == cur:
                continue
            if vertical:
                xa, xb = x0, x1
                ya, yb = (jump, cur) if decreasing else (cur, jump)
                skipped = (x0, ya, x1, yb)
            else:
                ya, yb = y0, y1
                xa, xb = (jump, cur) if decreasing else (cur, jump)
                skipped = (xa, y0, xb, y1)
            # Cornice skip is allowed; jumping a tree/padding belt is not.
            if k > 1 and veg_frac_in_box(veg, skipped) > 0.30:
                continue
            if k > 1 and clutter_frac(clutter, skipped) > 0.22:
                continue
            far_a = jump if decreasing else max(lo, jump - step)
            far_b = min(hi, jump + step) if decreasing else jump
            if vertical:
                ok = probe_ok(x0, ya, x1, yb) or probe_ok(x0, far_a, x1, far_b)
            else:
                ok = probe_ok(xa, y0, xb, y1) or probe_ok(far_a, y0, far_b, y1)
            if ok:
                return jump
        return None

    if vertical:
        y0_orig, y1_orig = y0, y1
        down_cap = min(h, int(0.78 * h), y1_orig + max_extra)
        if y1_orig > int(0.60 * h):
            down_cap = y1_orig
        if not down_only:
            up_lo = max(0, y0_orig - max_extra)
            while y0 > up_lo and (y1 - y0) < max_size:
                nxt = try_grow(y0, up_lo, h, True)
                if nxt is None:
                    break
                y0 = nxt
        while y1 < down_cap and (y1 - y0) < max_size:
            nxt = try_grow(y1, 0, down_cap, False)
            if nxt is None:
                break
            y1 = nxt
    else:
        x0_orig, x1_orig = x0, x1
        while x0 > 0 and (x1 - x0) < max_size and (x0_orig - x0) < max_extra:
            nxt = try_grow(x0, 0, w, True)
            if nxt is None:
                break
            x0 = nxt
        while x1 < w and (x1 - x0) < max_size and (x1 - x1_orig) < max_extra:
            nxt = try_grow(x1, 0, w, False)
            if nxt is None:
                break
            x1 = nxt
    return list(clamp_box_xyxy((x0, y0, x1, y1), width=w, height=h))


def grow_vertical(
    depth: np.ndarray,
    box: list[int],
    *,
    veg: np.ndarray | None,
    clutter: np.ndarray | None = None,
    min_rel: float,
    step: int,
    max_height_frac: float = 0.85,
    seed_med: float | None = None,
    down_only: bool = False,
) -> list[int]:
    """Extend a seed while the new strip stays closer than the seed's wall.

    ``down_only`` keeps the top edge fixed. Stage 4 uses that.
    """
    h, w = depth.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0, x1, y1 = clamp_box_xyxy((x0, y0, x1, y1), width=w, height=h)
    wall0 = local_wall_median(depth, (x0, y0, x1, y1), veg=veg)
    if wall0 is None:
        return [x0, y0, x1, y1]
    if seed_med is None:
        seed_med = robust_median(depth, (x0, y0, x1, y1))
    return _axis_grow(
        depth,
        [x0, y0, x1, y1],
        wall0=wall0,
        seed_med=seed_med,
        veg=veg,
        clutter=clutter,
        min_rel=min_rel,
        step=step,
        vertical=True,
        max_extra=int(0.22 * h),
        max_size=int(max_height_frac * h),
        down_only=down_only,
    )


def grow_horizontal(
    depth: np.ndarray,
    box: list[int],
    *,
    veg: np.ndarray | None,
    clutter: np.ndarray | None = None,
    min_rel: float,
    step: int,
    seed_med: float | None = None,
) -> list[int]:
    """Widen a projecting volume while new side strips stay closer than the wall."""
    h, w = depth.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0, x1, y1 = clamp_box_xyxy((x0, y0, x1, y1), width=w, height=h)
    wall0 = local_wall_median(depth, (x0, y0, x1, y1), veg=veg, farther_side=True)
    if wall0 is None:
        return [x0, y0, x1, y1]
    if seed_med is None:
        seed_med = robust_median(depth, (x0, y0, x1, y1))
    bw = max(8, x1 - x0)
    return _axis_grow(
        depth,
        [x0, y0, x1, y1],
        wall0=wall0,
        seed_med=seed_med,
        veg=veg,
        clutter=clutter,
        min_rel=min_rel,
        step=step,
        vertical=False,
        max_extra=min(int(0.12 * w), int(0.90 * bw)),
        max_size=int(0.55 * w),
    )


def trim_to_projecting(
    depth: np.ndarray,
    box: list[int],
    *,
    veg: np.ndarray | None,
    clutter: np.ndarray | None,
    min_rel: float,
    step: int,
    seed_med: float | None = None,
) -> list[int]:
    """Shrink padding / tree / shop edges that failed the projecting test."""
    h, w = depth.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0, x1, y1 = clamp_box_xyxy((x0, y0, x1, y1), width=w, height=h)
    wall0 = local_wall_median(depth, (x0, y0, x1, y1), veg=veg)
    if wall0 is None:
        return [x0, y0, x1, y1]
    if seed_med is None:
        seed_med = robust_median(depth, (x0, y0, x1, y1))
    min_h = max(step * 3, int(0.12 * h))
    min_w = max(step * 2, int(0.06 * w))

    def edge_ok(probe: tuple[int, int, int, int]) -> bool:
        return strip_projecting(
            depth,
            probe,
            wall0=wall0,
            seed_med=seed_med,
            veg=veg,
            clutter=clutter,
            min_rel=min_rel,
            min_side=max(4, step // 2),
        )

    while y1 - y0 > min_h and not edge_ok((x0, y0, x1, min(h, y0 + step))):
        y0 += step
    if y1 > int(0.82 * h):
        while y1 - y0 > min_h and not edge_ok((x0, max(0, y1 - step), x1, y1)):
            y1 -= step
    while x1 - x0 > min_w and not edge_ok((x0, y0, min(w, x0 + step), y1)):
        x0 += step
    while x1 - x0 > min_w and not edge_ok((max(0, x1 - step), y0, x1, y1)):
        x1 -= step
    return list(clamp_box_xyxy((x0, y0, x1, y1), width=w, height=h))


def similar_box_width(
    a: list[int] | tuple[int, int, int, int],
    b: list[int] | tuple[int, int, int, int],
    *,
    min_ratio: float = 0.80,
) -> bool:
    """True when widths differ by at most ±20% (min/max >= 0.80)."""
    wa = max(1, int(a[2]) - int(a[0]))
    wb = max(1, int(b[2]) - int(b[0]))
    return float(min(wa, wb)) / float(max(wa, wb)) >= min_ratio


def drop_narrower_than_below(
    records: list[dict[str, Any]],
    *,
    min_width_ratio: float = 0.80,
    min_x_overlap: float = 0.40,
) -> list[dict[str, Any]]:
    """Drop a box more than 20% narrower than an enclosed volume below it.

    Same-column leftovers that failed the ±20% vertical merge stay as their
    own box; a skinny AC / fragment above a wider enclosed should not.
    """
    if len(records) < 2:
        return [dict(r) for r in records]
    drop: set[int] = set()
    for i, upper in enumerate(records):
        ua = upper["box_xyxy"]
        wa = max(1, int(ua[2]) - int(ua[0]))
        for j, lower in enumerate(records):
            if i == j:
                continue
            lb = lower["box_xyxy"]
            if int(ua[1]) >= int(lb[1]):
                continue
            if not same_column(ua, lb, min_each=min_x_overlap):
                continue
            wb = max(1, int(lb[2]) - int(lb[0]))
            if wa / float(wb) < min_width_ratio:
                drop.add(i)
                break
    return [dict(r) for k, r in enumerate(records) if k not in drop]


def merge_vertical_columns(
    records: list[dict[str, Any]],
    *,
    ih: int,
    min_x_overlap: float = 0.45,
    max_y_gap_frac: float = 0.16,
) -> list[dict[str, Any]]:
    """Union boxes that sit in the same facade column.

    Widths must stay within ±20%. A tall ground-floor portal and a shop floor
    still stay separate from the volume above them.
    """
    if not records:
        return []
    remaining = sorted(records, key=lambda r: (r["box_xyxy"][0], r["box_xyxy"][1]))
    merged: list[dict[str, Any]] = []
    max_gap = int(max_y_gap_frac * ih)
    while remaining:
        cur = dict(remaining.pop(0))
        changed = True
        while changed:
            changed = False
            nxt: list[dict[str, Any]] = []
            for rec in remaining:
                a, b = cur["box_xyxy"], rec["box_xyxy"]
                ground_mix = (spans_ground(a, ih) and starts_high(b, ih)) or (
                    spans_ground(b, ih) and starts_high(a, ih)
                )
                shop_mix = is_shop_floor(a, ih) != is_shop_floor(b, ih)
                if (
                    same_column(a, b, min_each=min_x_overlap)
                    and similar_box_width(a, b)
                    and y_gap(a, b) <= max_gap
                    and not ground_mix
                    and not shop_mix
                ):
                    cur["box_xyxy"] = union_box(cur["box_xyxy"], rec["box_xyxy"])
                    cur["score"] = max(float(cur.get("score") or 0), float(rec.get("score") or 0))
                    changed = True
                else:
                    nxt.append(rec)
            remaining = nxt
        merged.append(cur)
    return merged


def merge_adjacent_bays(
    records: list[dict[str, Any]],
    *,
    iw: int,
    max_x_gap_frac: float = 0.055,
    min_y_overlap: float = 0.28,
    max_union_width_frac: float = 0.42,
) -> list[dict[str, Any]]:
    """Join a corner turret with the in-plane window of the same oriel bay."""
    if not records:
        return []
    remaining = list(records)
    merged: list[dict[str, Any]] = []
    max_gap = int(max_x_gap_frac * iw)
    while remaining:
        cur = dict(remaining.pop(0))
        changed = True
        while changed:
            changed = False
            nxt: list[dict[str, Any]] = []
            for rec in remaining:
                a, b = cur["box_xyxy"], rec["box_xyxy"]
                aw = int(a[2]) - int(a[0])
                bw = int(b[2]) - int(b[0])
                uni = union_box(a, b)
                if (
                    max(aw, bw) < 0.28 * iw
                    and x_gap(a, b) <= max_gap
                    and y_overlap_frac(a, b) >= min_y_overlap
                    and (uni[2] - uni[0]) <= max_union_width_frac * iw
                ):
                    cur["box_xyxy"] = uni
                    cur["score"] = max(float(cur.get("score") or 0), float(rec.get("score") or 0))
                    changed = True
                else:
                    nxt.append(rec)
            remaining = nxt
        merged.append(cur)
    return merged


def nms_columns(
    records: list[dict[str, Any]],
    *,
    iw: int,
    ih: int,
    min_each: float = 0.40,
) -> list[dict[str, Any]]:
    """Drop a narrow ground portal under an oriel, not a wide gallery floor."""
    keep = [dict(r) for r in records]
    drop: set[int] = set()
    for i, a in enumerate(keep):
        for j in range(i + 1, len(keep)):
            if i in drop or j in drop:
                continue
            b = keep[j]
            if not same_column(a["box_xyxy"], b["box_xyxy"], min_each=min_each):
                continue
            ai, bi = a["box_xyxy"], b["box_xyxy"]
            upper, lower = (i, j) if ai[1] <= bi[1] else (j, i)
            low = keep[lower]["box_xyxy"]
            if (
                low[3] > 0.70 * ih
                and keep[upper]["box_xyxy"][1] < 0.18 * ih
            ):
                drop.add(lower)
            elif (
                low[3] > 0.70 * ih
                and keep[upper]["box_xyxy"][1] < 0.38 * ih
                and (low[2] - low[0]) < 0.28 * iw
            ):
                drop.add(lower)
    return [r for k, r in enumerate(keep) if k not in drop]


def column_storey_reason(
    box: list[int] | tuple[int, int, int, int],
    *,
    iw: int,
    ih: int,
) -> str | None:
    """Reject leftovers smaller than one facade column or one storey.

    Applied after grow / clip, not on SAM3 seeds, so one-floor pancakes can merge.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    bw, bh = max(0, x1 - x0), max(0, y1 - y0)
    if bw < MIN_COLUMN_FRAC * iw:
        return "too_narrow"
    if bh < MIN_HEIGHT_FRAC * ih:
        return "too_short"
    return None


def keep_column_storey(
    records: list[dict[str, Any]],
    *,
    iw: int,
    ih: int,
) -> list[dict[str, Any]]:
    return [
        rec
        for rec in records
        if column_storey_reason(rec["box_xyxy"], iw=iw, ih=ih) is None
    ]


def seed_ok(
    box: list[int],
    *,
    iw: int,
    ih: int,
    min_height_px: float | None = None,
) -> bool:
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    if min_height_px is not None:
        if bh <= float(min_height_px):
            return False
    elif bh < MIN_SEED_HEIGHT_FRAC * ih:
        return False
    if bw < MIN_WIDTH_FRAC * iw:
        return False
    if bw > MAX_SEED_WIDTH_FRAC * iw:
        return False
    if bh > MAX_SEED_HEIGHT_FRAC * ih:
        return False
    aspect = bh / max(1, bw)
    if aspect > 8.0 or bw / max(1, bh) > 6.5:
        return False
    return True


def match_gt(
    pred: list[dict[str, Any]],
    gt: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    used: set[int] = set()
    rows: list[dict[str, Any]] = []
    for g in gt:
        best_i = -1
        best = 0.0
        for i, p in enumerate(pred):
            if i in used:
                continue
            iou = box_iou(p["box_xyxy"], g["box_xyxy"])
            if iou > best:
                best = iou
                best_i = i
        hit = best >= MATCH_IOU
        if hit:
            used.add(best_i)
        rows.append({"name": g["name"], "iou": round(best, 3), "hit": hit})
    return rows


def draw_eval_overlay(
    photo: Image.Image,
    *,
    gt: list[dict[str, Any]],
    pred: list[dict[str, Any]],
) -> Image.Image:
    overlay = photo.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    line_w = max(3, photo.width // 220)
    for g in gt:
        draw.rectangle(g["box_xyxy"], outline=(255, 40, 40), width=line_w)
        draw.text((g["box_xyxy"][0] + 4, g["box_xyxy"][1] + 4), g["name"], fill=(255, 40, 40))
    for p in pred:
        box = p["box_xyxy"]
        rel = p.get("rel_delta")
        tag = f"{rel:+.2f}" if rel is not None else "pred"
        draw.rectangle(box, outline=(40, 255, 80), width=line_w)
        draw.text((box[0] + 4, box[3] - 16), tag, fill=(40, 255, 80))
    return overlay


def is_exclude_prompt(prompt: str | None) -> bool:
    return str(prompt or "") in EXCLUDE_PROMPTS


def _missing_exclude(records: list[dict[str, Any]]) -> bool:
    return not any(is_exclude_prompt(r.get("prompt")) for r in records)


def split_seed_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seeds: list[dict[str, Any]] = []
    exclude: list[dict[str, Any]] = []
    for rec in records:
        if is_exclude_prompt(rec.get("prompt")):
            exclude.append(rec)
        else:
            seeds.append(rec)
    return seeds, exclude


def covered_by_exclude(
    box: list[int],
    exclude_boxes: list[list[int]],
    *,
    min_frac: float = EXCLUDE_COVER,
) -> bool:
    return any(box_cover_frac(box, ex) >= min_frac for ex in exclude_boxes)


def filter_stems(paths: list[Path], stems: tuple[str, ...] | list[str]) -> list[Path]:
    want = set(stems)
    return [p for p in paths if p.stem in want]


def _load_facade_dsl(stem: str, dsl_dir: Path | None) -> dict[str, Any] | None:
    if dsl_dir is None:
        return None
    path = Path(dsl_dir) / stem / "facade_dsl.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _window_grid_api():
    pipe = str(ROOT / "balcony_pipeline")
    if pipe not in sys.path:
        sys.path.insert(0, pipe)
    from enclosed import MIN_COLUMN_COVER, MIN_FLOOR_COVER, median_floor_height
    from snap import _overlap_1d

    return median_floor_height, MIN_FLOOR_COVER, MIN_COLUMN_COVER, _overlap_1d


def resolve_floor_h(stem: str, dsl_dir: Path | None) -> float | None:
    """Median window-grid storey height from ``{dsl_dir}/{stem}/facade_dsl.json``."""
    dsl = _load_facade_dsl(stem, dsl_dir)
    if dsl is None:
        return None
    median_floor_height, _, _, _ = _window_grid_api()
    return median_floor_height(dsl)


def resolve_window_bands(
    stem: str, dsl_dir: Path | None
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]] | None:
    """Floor and bay pixel bands from the window facade DSL, or None."""
    dsl = _load_facade_dsl(stem, dsl_dir)
    if dsl is None:
        return None
    pipe = str(ROOT / "balcony_pipeline")
    if pipe not in sys.path:
        sys.path.insert(0, pipe)
    from snap import _bands_from_windows_dsl

    floor_y, bay_x = _bands_from_windows_dsl(dsl)
    if not floor_y or not bay_x:
        return None
    return floor_y, bay_x


def resolve_window_boxes(stem: str, dsl_dir: Path | None) -> list[list[int]]:
    """Window instance boxes from the window DSL (empty when missing)."""
    dsl = _load_facade_dsl(stem, dsl_dir)
    if dsl is None:
        return []
    from balcony_train.enclosed_symmetry import window_boxes_from_dsl

    return window_boxes_from_dsl(dsl)


def on_window_floor_bay(
    box: list[int] | tuple[int, int, int, int],
    floor_y: dict[int, tuple[int, int]] | None,
    bay_x: dict[int, tuple[int, int]] | None,
) -> bool:
    """True when the box covers a window storey and a window bay.

    Same floor rule as enclosed placement: 90% of an upper storey the box
    continues below, 50% of any other storey. A shop or edge sliver that
    misses the grid can still be dropped.
    """
    if not floor_y or not bay_x:
        return False
    _window_grid_api()
    from enclosed import overlapping_floors

    if not overlapping_floors(list(box), floor_y):
        return False
    _, _min_floor, min_bay, overlap_1d = _window_grid_api()
    x0, x1 = float(box[0]), float(box[2])
    for bx0, bx1 in bay_x.values():
        cover = overlap_1d(x0, x1, float(bx0), float(bx1)) / max(1.0, float(bx1) - float(bx0))
        if cover >= min_bay:
            return True
    return False


def _collect_prompt_boxes(
    image: Image.Image,
    *,
    processor,
    sam,
    device: torch.device,
    prompts: tuple[str, ...],
    threshold: float,
) -> list[dict[str, Any]]:
    pooled: list[dict[str, Any]] = []
    for prompt in prompts:
        recs = detect_balcony_records(
            image,
            processor=processor,
            sam=sam,
            device=device,
            prompt=prompt,
            threshold=threshold,
            min_side=12,
            max_side_frac=0.95,
        )
        for rec in recs:
            rec = dict(rec)
            rec["prompt"] = prompt
            rec["box_xyxy"] = list(
                clamp_box_xyxy(rec["box_xyxy"], width=image.width, height=image.height)
            )
            pooled.append(rec)
        print(f"  [{prompt}]  n={len(recs)}", flush=True)
    return pooled


def run_sam3(
    paths: list[Path],
    *,
    device: torch.device,
    cache_path: Path,
    threshold: float,
) -> dict[str, list[dict[str, Any]]]:
    cached: dict[str, list[dict[str, Any]]] = {}
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"SAM3 cache {cache_path}", flush=True)
    need_full = [p for p in paths if p.stem not in cached]
    need_ex = [
        p
        for p in paths
        if p.stem in cached and _missing_exclude(cached[p.stem])
    ]
    if need_full or need_ex:
        from transformers import Sam3Model, Sam3Processor

        print(
            f"loading SAM3…  full={len(need_full)} exclude={len(need_ex)}",
            flush=True,
        )
        processor = Sam3Processor.from_pretrained("facebook/sam3")
        sam = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
        for path in need_full:
            print(path.name, flush=True)
            image = Image.open(path).convert("RGB")
            cached[path.stem] = _collect_prompt_boxes(
                image,
                processor=processor,
                sam=sam,
                device=device,
                prompts=SAM3_PROMPTS + EXCLUDE_PROMPTS,
                threshold=threshold,
            )
        for path in need_ex:
            print(f"{path.name}  exclude only", flush=True)
            image = Image.open(path).convert("RGB")
            extra = _collect_prompt_boxes(
                image,
                processor=processor,
                sam=sam,
                device=device,
                prompts=EXCLUDE_PROMPTS,
                threshold=threshold,
            )
            cached[path.stem] = list(cached[path.stem]) + extra
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cached, indent=2) + "\n", encoding="utf-8")
        del sam, processor
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {p.stem: cached.get(p.stem, []) for p in paths}


STAGE_LABELS: tuple[tuple[str, str], ...] = (
    ("1_sam3", "1 SAM3 1-storey seeds"),
    ("2_depth", "2 depth vs wall >= 1.5% (no obstacles)"),
    ("3_merge", "3 merge same column / bay, drop narrow vs below"),
    ("4_grow", "4 grow down if 0.6% closer, drop <1-col"),
    ("5_clip", "5 clip shop, drop portal/small"),
)


def _stage_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "C:/Windows/Fonts/YuGothM.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_stage_overlay(
    photo: Image.Image,
    *,
    gt: list[dict[str, Any]],
    pred: list[dict[str, Any]],
    title: str,
) -> Image.Image:
    overlay = photo.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    line_w = max(3, photo.width // 220)
    for g in gt:
        draw.rectangle(g["box_xyxy"], outline=(255, 70, 70), width=max(2, line_w - 1))
    for p in pred:
        draw.rectangle(p["box_xyxy"], outline=(40, 255, 90), width=line_w)
    bar_h = max(28, photo.height // 28)
    draw.rectangle((0, 0, photo.width, bar_h), fill=(12, 16, 22))
    font = _stage_font(max(14, bar_h - 10))
    draw.text((8, 4), f"{title}  n={len(pred)}", fill=(240, 240, 240), font=font)
    return overlay


def compose_row(panels: list[Image.Image], *, height: int = 520) -> Image.Image:
    scaled: list[Image.Image] = []
    for panel in panels:
        w = max(1, int(round(panel.width * height / panel.height)))
        scaled.append(panel.resize((w, height), Image.Resampling.BILINEAR))
    gap = 8
    total_w = sum(p.width for p in scaled) + gap * (len(scaled) - 1)
    row = Image.new("RGB", (total_w, height), (30, 30, 30))
    x = 0
    for panel in scaled:
        row.paste(panel, (x, 0))
        x += panel.width + gap
    return row


def volume_stages(
    image: Image.Image,
    depth: np.ndarray,
    seeds: list[dict[str, Any]],
    *,
    min_rel: float = MIN_REL,
    grow_rel: float = GROW_REL,
    floor_h: float | None = None,
    floor_y: dict[int, tuple[int, int]] | None = None,
    bay_x: dict[int, tuple[int, int]] | None = None,
    windows: list[list[int]] | None = None,
    symmetry: bool = False,
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """Return the pipeline snapshots: (id, title, boxes).

    Five stages by default. ``symmetry`` appends Stage 6, the left-right
    repair in ``enclosed_symmetry`` (axis from the window grid).
    """
    iw, ih = image.size
    min_height_px = None
    stage1_title = STAGE_LABELS[0][1]
    if floor_h is not None and float(floor_h) > 0:
        min_height_px = SEED_MIN_FLOOR_FRAC * float(floor_h)
        stage1_title = (
            f"1 SAM3 seeds h>{min_height_px:.0f}px (0.85 floor)"
        )
    stage5_title = STAGE_LABELS[4][1]
    if floor_y and bay_x:
        stage5_title = "5 clip shop, keep floor×bay, drop portal/small"
    rgb = np.asarray(image, dtype=np.uint8)
    veg = foliage_mask(rgb)
    global_wall = local_wall_median(
        depth, [int(0.2 * iw), int(0.2 * ih), int(0.8 * iw), int(0.8 * ih)], veg=veg
    )
    clutter = clutter_mask(depth, global_wall) if global_wall else None

    seed_recs, exclude_recs = split_seed_records(seeds)
    exclude_boxes = [
        list(clamp_box_xyxy(r["box_xyxy"], width=iw, height=ih)) for r in exclude_recs
    ]

    sam3_seeds: list[dict[str, Any]] = []
    kept_seeds: list[dict[str, Any]] = []
    for rec in seed_recs:
        box = list(clamp_box_xyxy(rec["box_xyxy"], width=iw, height=ih))
        if not seed_ok(box, iw=iw, ih=ih, min_height_px=min_height_px):
            continue
        if covered_by_exclude(box, exclude_boxes):
            continue
        item = {
            "box_xyxy": box,
            "score": float(rec.get("score") or 0.0),
            "prompt": rec.get("prompt"),
        }
        sam3_seeds.append(item)
        if veg_frac_in_box(veg, box) > 0.45:
            continue
        if clutter_frac(clutter, box) > 0.40:
            continue
        rel = box_rel_delta(depth, box, veg=veg)
        if rel is None or rel < min_rel or rel > 0.22:
            continue
        kept = dict(item)
        kept["rel_delta"] = rel
        kept_seeds.append(kept)

    # Width is judged on the boxes Stage 2 draws. A seed hidden inside a
    # larger box must not bridge two outlines that differ by more than 20%.
    stage2 = merge_boxes(kept_seeds, 0.70)
    columns = merge_vertical_columns(
        stage2, ih=ih, min_x_overlap=0.40, max_y_gap_frac=MERGE_Y_GAP_FRAC
    )
    columns = merge_adjacent_bays(columns, iw=iw)
    columns = drop_narrower_than_below(columns)
    # Stage 4 extends downward only, where the strip is still closer than the
    # wall. It does not move the top, trim, or merge the boxes again.
    step = max(8, ih // 40)
    extended: list[dict[str, Any]] = []
    for rec in columns:
        item = dict(rec)
        item["box_xyxy"] = grow_vertical(
            depth,
            list(rec["box_xyxy"]),
            veg=veg,
            clutter=clutter,
            min_rel=grow_rel,
            step=step,
            down_only=True,
        )
        extended.append(item)
    grown = keep_column_storey(extended, iw=iw, ih=ih)

    clipped: list[dict[str, Any]] = []
    for rec in grown:
        box = clip_shop_span(list(rec["box_xyxy"]), ih)
        rec = dict(rec)
        rec["box_xyxy"] = box
        clipped.append(rec)
    on_grid: list[dict[str, Any]] = []
    loose: list[dict[str, Any]] = []
    for rec in clipped:
        box = list(clamp_box_xyxy(rec["box_xyxy"], width=iw, height=ih))
        keep_grid = on_window_floor_bay(box, floor_y, bay_x)
        if not keep_grid:
            if column_storey_reason(box, iw=iw, ih=ih) or storey_width_reason(box, iw=iw, ih=ih):
                continue
            if box[0] > 0.82 * iw:
                continue
            if (
                box[3] > 0.92 * ih
                and box[1] > 0.40 * ih
                and (box[2] - box[0]) < 0.28 * iw
            ):
                continue
            if clutter_frac(clutter, box) > 0.35:
                continue
        rel = box_rel_delta(depth, box, veg=veg)
        if not keep_grid and (rel is None or rel < min_rel or rel > 0.22):
            continue
        rec = dict(rec)
        rec["box_xyxy"] = box
        if rel is not None:
            rec["rel_delta"] = rel
        if keep_grid:
            on_grid.append(rec)
        else:
            loose.append(rec)
    final = merge_boxes(on_grid, 0.55) + merge_boxes(nms_columns(loose, iw=iw, ih=ih), 0.55)

    stages = [
        (STAGE_LABELS[0][0], stage1_title, merge_boxes(sam3_seeds, 0.70)),
        (STAGE_LABELS[1][0], STAGE_LABELS[1][1], stage2),
        (STAGE_LABELS[2][0], STAGE_LABELS[2][1], columns),
        (STAGE_LABELS[3][0], STAGE_LABELS[3][1], grown),
        (STAGE_LABELS[4][0], stage5_title, final),
    ]
    if symmetry:
        from balcony_train.enclosed_symmetry import (
            grid_axis_x,
            stage_title,
            symmetry_repair,
            window_symmetry,
        )

        cx = grid_axis_x(bay_x or {})
        sym_boxes, sym_log = symmetry_repair(
            final,
            cx=cx,
            window_sym=window_symmetry(windows or [], cx) if cx is not None else 0.0,
            iw=iw,
            ih=ih,
            rel_fn=lambda b: box_rel_delta(depth, b, veg=veg),
            on_grid=lambda b: on_window_floor_bay(b, floor_y, bay_x),
            min_rel=min_rel,
        )
        stages.append(("6_sym", stage_title(sym_log), sym_boxes))
    return stages


def detect_volumes(
    image: Image.Image,
    depth: np.ndarray,
    seeds: list[dict[str, Any]],
    *,
    min_rel: float = MIN_REL,
    grow_rel: float = GROW_REL,
    floor_h: float | None = None,
    floor_y: dict[int, tuple[int, int]] | None = None,
    bay_x: dict[int, tuple[int, int]] | None = None,
    windows: list[list[int]] | None = None,
    symmetry: bool = False,
) -> list[dict[str, Any]]:
    return volume_stages(
        image,
        depth,
        seeds,
        min_rel=min_rel,
        grow_rel=grow_rel,
        floor_h=floor_h,
        floor_y=floor_y,
        bay_x=bay_x,
        windows=windows,
        symmetry=symmetry,
    )[-1][2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    ap.add_argument("--depth-dir", type=Path, default=DEFAULT_DEPTH_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sam3-threshold", type=float, default=0.15)
    ap.add_argument("--min-rel", type=float, default=MIN_REL)
    ap.add_argument("--grow-rel", type=float, default=GROW_REL)
    ap.add_argument("--refresh-sam3", action="store_true")
    ap.add_argument("--stages", action="store_true", help="Save a 5-step overlay board")
    ap.add_argument(
        "--symmetry",
        action="store_true",
        help="Add Stage 6: mirror about the window-grid axis (add / align / drop)",
    )
    ap.add_argument(
        "--dsl-dir",
        type=Path,
        default=DEFAULT_DSL_DIR,
        help="Pipeline runs dir with {stem}/facade_dsl.json (same floor_h as run_balcony)",
    )
    ap.add_argument(
        "--stems",
        nargs="*",
        default=None,
        help="Optional subset of stems. Default: every image in --in-dir.",
    )
    args = ap.parse_args()

    paths = list_input_images(args.in_dir)
    if args.stems:
        paths = filter_stems(paths, tuple(args.stems))
    if not paths:
        raise SystemExit(f"no target images in {args.in_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    missing_depth = [
        p for p in paths if not (args.depth_dir / f"{p.stem}_depth.npy").is_file()
    ]
    if missing_depth:
        from balcony_train.try_da3_enclosure import DEFAULT_MODEL, run_da3

        print(f"DA3 depth missing for {len(missing_depth)} image(s)", flush=True)
        run_da3(
            missing_depth,
            out_dir=args.depth_dir,
            model_id=DEFAULT_MODEL,
            device=str(args.device),
            min_rel=0.02,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    cache_path = args.out_dir / "sam3_seeds.json"
    if args.refresh_sam3 and cache_path.exists():
        cache_path.unlink()
    device = torch.device(args.device)
    seeds = run_sam3(
        paths, device=device, cache_path=cache_path, threshold=args.sam3_threshold
    )

    summary: dict[str, Any] = {
        "min_rel": args.min_rel,
        "grow_rel": args.grow_rel,
        "prompts": list(SAM3_PROMPTS),
        "exclude_prompts": list(EXCLUDE_PROMPTS),
        "images": {},
    }
    hits = 0
    total_gt = 0
    stage_rows: list[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        depth_path = args.depth_dir / f"{path.stem}_depth.npy"
        if not depth_path.is_file():
            raise SystemExit(f"missing {depth_path}")
        depth = resize_depth(np.load(depth_path), image.width, image.height)
        floor_h = resolve_floor_h(path.stem, args.dsl_dir)
        bands = resolve_window_bands(path.stem, args.dsl_dir)
        floor_y, bay_x = bands if bands is not None else (None, None)
        if floor_h is not None:
            print(
                f"{path.name}  floor_h={floor_h:.0f}px  "
                f"stage1 drop <= {SEED_MIN_FLOOR_FRAC * floor_h:.0f}px",
                flush=True,
            )
        else:
            print(f"{path.name}  floor_h missing; stage1 uses {MIN_SEED_HEIGHT_FRAC:.1%} image", flush=True)
        if bands is None:
            print(f"{path.name}  window floor×bay missing; stage5 drops as usual", flush=True)
        stages = volume_stages(
            image,
            depth,
            seeds.get(path.stem, []),
            min_rel=args.min_rel,
            grow_rel=args.grow_rel,
            floor_h=floor_h,
            floor_y=floor_y,
            bay_x=bay_x,
            windows=resolve_window_boxes(path.stem, args.dsl_dir),
            symmetry=args.symmetry,
        )
        pred = stages[-1][2]
        gt = gt_px(path.stem, image.width, image.height)
        if args.stages:
            panels = [
                draw_stage_overlay(image, gt=gt, pred=boxes, title=title)
                for _sid, title, boxes in stages
            ]
            strip = compose_row(panels, height=560)
            strip.save(args.out_dir / f"{path.stem}_stages.jpg", quality=92)
            stage_rows.append(strip)
        rows = match_gt(pred, gt)
        total_gt += len(gt)
        hits += sum(1 for r in rows if r["hit"])
        overlay = draw_eval_overlay(image, gt=gt, pred=pred)
        overlay.save(args.out_dir / f"{path.stem}_eval.jpg", quality=92)
        scores = " ".join(f"{r['name']}={r['iou']:.2f}" for r in rows)
        extra = f"  {scores}" if scores else ""
        print(
            f"{path.name}  pred={len(pred)}  gt={len(gt)}{extra}",
            flush=True,
        )
        summary["images"][path.stem] = {
            "pred": pred,
            "gt": gt,
            "match": rows,
        }
    summary["hits"] = hits
    summary["n_gt"] = total_gt
    (args.out_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if args.stages and stage_rows:
        gap = 16
        board_w = max(r.width for r in stage_rows)
        board_h = sum(r.height for r in stage_rows) + gap * (len(stage_rows) - 1)
        board = Image.new("RGB", (board_w, board_h), (24, 24, 24))
        y = 0
        for row in stage_rows:
            board.paste(row, (0, y))
            y += row.height + gap
        board_path = args.out_dir / "stages_board.jpg"
        board.save(board_path, quality=90)
        print(f"stages -> {board_path}", flush=True)
    print(f"hits={hits}/{total_gt} -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
