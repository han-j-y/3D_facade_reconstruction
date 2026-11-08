"""Open-balcony structure from DA3 depth: does the balcony stick out of the wall right beside it?

Rule (user, 2026-11-07): projecting = sticks out of the outer wall beside it; a recess (inset / loggia) is
measured against that outer wall, so it is juliet. On the DA3 metric depth (smaller = closer):

* box depth = median depth inside the box;
* left / right wall = median depth of a strip beside the box at the same height (``strip`` x the box width,
  ``gap`` x the box width away). Above / below are not used: a recess is deeper there by design;
* protrusion = wall depth - box depth on the side where it is larger; projecting when > ``threshold`` metres.

Threshold 0.10 m was read off val30 (``depth_protrusion_val30.py``: acc 0.889, balanced 0.847 there).
Pipeline default since 2026-11-07 (user): ``run_balcony.py --structure-depth 0.10``.
"""

from __future__ import annotations

import numpy as np


def _strip_median(depth: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> float | None:
    h, w = depth.shape
    xa, xb = int(max(0, round(x0))), int(min(w, round(x1)))
    ya, yb = int(max(0, round(y0))), int(min(h, round(y1)))
    if xb - xa < 3 or yb - ya < 3:
        return None
    v = depth[ya:yb, xa:xb]
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.median(v)) if v.size else None


def protrusion(depth: np.ndarray, box, *, strip: float = 0.5, gap: float = 0.05) -> dict:
    """Box / left / right depths and the protrusion (metres and relative) on the larger side."""
    x0, y0, x1, y1 = (float(v) for v in box)
    bw = x1 - x0
    d_box = _strip_median(depth, x0, x1, y0, y1)
    d_l = _strip_median(depth, x0 - (gap + strip) * bw, x0 - gap * bw, y0, y1)
    d_r = _strip_median(depth, x1 + gap * bw, x1 + (gap + strip) * bw, y0, y1)
    sides = {k: d for k, d in (("left", d_l), ("right", d_r)) if d is not None}
    if d_box is None or not sides:
        return {"d_box": d_box, "d_left": d_l, "d_right": d_r, "side": None, "prot_m": None, "prot_rel": None}
    side = max(sides, key=lambda k: sides[k] - d_box)
    wall = sides[side]
    return {"d_box": d_box, "d_left": d_l, "d_right": d_r, "side": side,
            "prot_m": wall - d_box, "prot_rel": (wall - d_box) / wall}


def structure_from_depth(depth: np.ndarray | None, box, threshold: float = 0.10) -> str | None:
    """"projecting" / "juliet" from the protrusion; None when there is no depth or no wall strip."""
    if depth is None or threshold <= 0:
        return None
    m = protrusion(depth, box)
    if m["prot_m"] is None:
        return None
    return "projecting" if m["prot_m"] > threshold else "juliet"
