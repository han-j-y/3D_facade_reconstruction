"""Windows seen on the side wall of an enclosed block.

A front window lies on a flat surface. A window on a block's side wall turns away from the camera, so its depth (wall
tilt removed) ramps steadily from one box edge to the other. Needs no camera position, so it also works on CMP photos.
"""

from __future__ import annotations

import statistics
from typing import Any

import numpy as np

from enclosed import _bay_set, enclosed_blocks

RAMP_MIN_M = 0.25  # depth change over one typical window width (validation, LSAA photos)
RAMP_FIT_MIN = 0.9  # share of the ramp a straight line explains
MIN_FLOORS = 2


def _residual_depth(depth: np.ndarray, boxes: list[list[int]]) -> np.ndarray:
    """Depth minus the wall plane, fitted robustly to pixels outside the window boxes."""
    h, w = depth.shape
    mask = np.ones((h, w), bool)
    mask[: int(h * 0.05)] = False
    for x0, y0, x1, y1 in boxes:
        mask[y0:y1, x0:x1] = False
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.c_[xx[mask], yy[mask], np.ones(int(mask.sum()))]
    z = depth[mask]
    keep = np.ones(len(z), bool)
    coef = np.zeros(3)
    for _ in range(4):
        coef = np.linalg.lstsq(a[keep], z[keep], rcond=None)[0]
        r = z - a @ coef
        keep = np.abs(r) < 2.0 * (1.4826 * np.median(np.abs(r[keep])) + 1e-6)
    return depth - (coef[0] * xx + coef[1] * yy + coef[2])


def window_ramp(residual: np.ndarray, box: list[int], typical_width: float) -> tuple[float, float]:
    """(depth change left to right per typical window width, straight-line fit share)."""
    x0, y0, x1, y1 = box
    col = np.median(residual[y0:y1, x0:x1], axis=0)
    xs = np.arange(len(col))
    slope, icpt = np.polyfit(xs, col, 1)
    fit = 1.0 - (col - (icpt + slope * xs)).var() / (col.var() + 1e-9)
    return float(slope * typical_width), float(fit)


def flag_side_windows(
    instances: list[dict[str, Any]],
    enclosed_units: list[dict[str, Any]],
    depth: np.ndarray,
) -> list[int]:
    """Set ``side_face`` on windows of enclosed blocks that sit on a side wall; returns their unit ids.

    A column counts when the ramp holds on at least ``MIN_FLOORS`` floors, depth grows towards the wall outside the
    block, no window of the block lies further out on that side, and the block keeps a front window.
    """
    wins = [u for u in instances if u.get("kind", "window") == "window"]
    if not wins or not enclosed_units:
        return []
    h, w = depth.shape
    def clip(b: list[float]) -> list[int]:
        return [max(0, int(b[0])), max(0, int(b[1])), min(w, int(b[2])), min(h, int(b[3]))]
    residual = _residual_depth(depth, [clip(u["box_xyxy"]) for u in wins])
    typical = statistics.median(u["box_xyxy"][2] - u["box_xyxy"][0] for u in wins)
    flagged: list[int] = []
    for block in enclosed_blocks(enclosed_units):
        cells = {(int(e["floor"]), b) for e in block for b in _bay_set(e)}
        members = [u for u in wins if (int(u["floor"]), int(u["bay"])) in cells]
        ramps: dict[int, list[tuple[dict[str, Any], float]]] = {}
        for u in members:
            box = clip(u["box_xyxy"])
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                continue
            ramp, fit = window_ramp(residual, box, typical)
            if abs(ramp) >= RAMP_MIN_M and fit >= RAMP_FIT_MIN:
                ramps.setdefault(int(u["bay"]), []).append((u, ramp))
        columns = sorted({int(u["bay"]) for u in members})
        for bay, hits in ramps.items():
            signs = {ramp > 0 for _, ramp in hits}
            if len(hits) < MIN_FLOORS or len(signs) != 1:
                continue
            # Depth grows to the right on a right side wall, so nothing of the block may lie further right.
            outer = bay == (columns[-1] if signs.pop() else columns[0])
            if not outer or len(columns) < 2:
                continue
            for u, _ in hits:
                u["side_face"] = True
                flagged.append(int(u["unit_id"]))
    return flagged
