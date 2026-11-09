"""Undo the side-view parallax of a projecting balcony that straddles two windows (user, 2026-11-08).

The photo is rectified to the wall plane, so anything in front of the wall shifts away from the camera and shows its
side face on the camera side. A single projecting balcony can then straddle two windows and gets placed between them
(val30 cmp_b0360, Vienna way130743392).

Camera side of the facade, decided only when two cues agree:

* wall tilt: a robust plane fitted to the DA3 depth of the wall (facade region minus window / door / balcony boxes
  grown 10 % and the top 5 %). Depth growing to the right = right side farther = camera on the LEFT. Used only when
  the depth change across the facade is at least ``MIN_TILT_FRAC`` of the median wall depth;
* windows: per projecting open balcony, the box centre vs the centre of the nearest window right above it (a box
  left of its window = camera on the right); the facade's majority must equal the wall tilt.

A balcony is corrected when it is open and projecting, has exactly two partner windows in different bays and sticks
out at least ``MIN_PROTRUSION_M`` (DA3, wall beside the box minus the box). It keeps only the window on the camera
side (camera right: the balcony looks shifted left, so its window is the right one), is centred on it, and loses the
side face from its width: the depth-measured face width s when its sign agrees with the camera side, else the width
that keeps the camera-side box edge and centres on the window. ``parallax_fix`` keeps the box before.

With many windows (``REVEAL_MIN_WINDOWS`` kept, line fit R2 >= ``REVEAL_MIN_R2``) the window reveals also give the
camera's x (a jamb's visible width grows with the distance from the camera): a balcony whose predicted shift there is
below ``REVEAL_MIN_SHIFT`` is seen square and left alone, and one on the other side of the camera is left alone too.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from snap import _bands_from_windows_dsl, snap_box

MIN_TILT_FRAC = 0.03
MIN_PROTRUSION_M = 0.30
REVEAL_MIN_WINDOWS = 15  # windows kept in the reveal line fit before the camera's x is trusted
REVEAL_MIN_R2 = 0.5
REVEAL_MIN_SHIFT = 0.10  # predicted reveal asymmetry (fraction of a window width) at the balcony: below it, frontal


def wall_tilt(depth: np.ndarray, dsl: dict[str, Any]) -> dict[str, float] | None:
    """Depth change across the facade width (m, + = right side farther) and as a fraction of the median wall depth."""
    h, w = depth.shape
    reg = (dsl.get("meta") or {}).get("facade_region_xyxy") or [0, 0, w, h]
    x0, y0, x1, y1 = (int(round(float(v))) for v in reg)
    mask = np.zeros((h, w), bool)
    mask[max(y0, int(0.05 * h)):y1, max(0, x0):x1] = True
    boxes = [u["box_xyxy"] for u in dsl.get("instances") or [] if u.get("box_xyxy")]
    boxes += [b["box_xyxy"] for b in (dsl.get("layout") or {}).get("balconies") or [] if b.get("box_xyxy")]
    for bx0, by0, bx1, by1 in boxes:
        gx, gy = 0.1 * (bx1 - bx0), 0.1 * (by1 - by0)
        mask[max(0, int(by0 - gy)):int(by1 + gy), max(0, int(bx0 - gx)):int(bx1 + gx)] = False
    if mask.sum() < 100:
        return None
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.c_[xx[mask], yy[mask], np.ones(int(mask.sum()))]
    z = depth[mask]
    keep = np.isfinite(z)
    coef = np.zeros(3)
    for _ in range(4):
        coef = np.linalg.lstsq(a[keep], z[keep], rcond=None)[0]
        r = z - a @ coef
        keep = np.abs(r) < 2.0 * (1.4826 * np.median(np.abs(r[keep])) + 1e-6)
    change = float(coef[0] * (x1 - x0))
    return {"change_m": change, "frac": change / max(1e-6, float(np.median(z[keep])))}


def _reveal_asymmetry(depth: np.ndarray, box) -> float | None:
    """Reveal asymmetry of one window as a fraction of its width (+ = right jamb visible = window right of the camera).

    Depth across the window's middle band: wall = strips beside the box, glass = the box middle (deeper). On each side
    the depth crosses half-way from wall to glass; the box sits on the frame, so a visible jamb puts that crossing
    outside the box. a = (inset left - inset right) / width."""
    x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    w, h = x1 - x0, y1 - y0
    h_img, w_img = depth.shape
    ya, yb = max(0, int(y0 + 0.25 * h)), min(h_img, int(y1 - 0.25 * h))
    lo, hi = max(0, int(x0 - 0.5 * w)), min(w_img, int(x1 + 0.5 * w))
    if yb - ya < 3 or w < 6:
        return None
    prof = np.median(depth[ya:yb, lo:hi], axis=0)
    xs = np.arange(lo, hi)
    walls = [prof[xs < x0 - 0.1 * w], prof[xs > x1 + 0.1 * w]]
    walls = [float(np.median(v)) for v in walls if v.size >= 2]
    mid = prof[(xs > x0 + 0.35 * w) & (xs < x1 - 0.35 * w)]
    if len(walls) < 2 or mid.size == 0:
        return None
    wall, glass = float(np.median(walls)), float(np.median(mid))
    if glass - wall < 0.03:
        return None  # no recess seen
    inside = np.where((prof - wall) / (glass - wall) >= 0.5)[0]
    if inside.size == 0:
        return None
    return float((xs[inside[0]] - x0) - (x1 - xs[inside[-1]])) / w


def camera_x_from_reveals(dsl: dict[str, Any], depth: np.ndarray) -> dict[str, Any] | None:
    """The camera's x on the rectified photo from the window reveals (user, 2026-11-08): a jamb's visible width grows
    with the distance from the camera, so the reveal asymmetry of the windows lies on a line a = k (x - x_cam).
    Robust line fit; None when there are no windows to fit."""
    pts = []
    for u in dsl.get("instances") or []:
        if not u.get("box_xyxy") or u.get("side_face") or (u.get("kind") or "window") != "window":
            continue
        a = _reveal_asymmetry(depth, u["box_xyxy"])
        if a is not None:
            pts.append((0.5 * (float(u["box_xyxy"][0]) + float(u["box_xyxy"][2])), a))
    if len(pts) < 3:
        return None
    x = np.array([p[0] for p in pts])
    a = np.array([p[1] for p in pts])
    m = np.c_[x, np.ones_like(x)]
    keep = np.ones(len(x), bool)
    k = c = 0.0
    for _ in range(3):
        k, c = np.linalg.lstsq(m[keep], a[keep], rcond=None)[0]
        r = a - (k * x + c)
        keep = np.abs(r) < 2.5 * (1.4826 * np.median(np.abs(r[keep])) + 1e-6)
    var = float(np.sum((a[keep] - a[keep].mean()) ** 2))
    r2 = 1.0 - float(np.sum((a[keep] - (k * x[keep] + c)) ** 2)) / max(1e-9, var)
    reliable = int(keep.sum()) >= REVEAL_MIN_WINDOWS and r2 >= REVEAL_MIN_R2 and k > 0
    return {"k": float(k), "c": float(c), "x_cam": float(-c / k) if abs(k) > 1e-12 else None, "r2": r2,
            "n": int(keep.sum()), "reliable": reliable}


def protrusion_and_face(depth: np.ndarray, box) -> dict[str, float] | None:
    """Protrusion (m, wall beside the box minus the box middle) and the signed side-face width s (px, + = face on the
    right, i.e. camera on the right) from the depth profile across the box."""
    x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    w = x1 - x0
    h_img, w_img = depth.shape
    ya, yb = max(0, y0), min(h_img, y1)
    if yb - ya < 4 or w < 8:
        return None
    lo, hi = max(0, int(x0 - 0.4 * w)), min(w_img, int(x1 + 0.4 * w))
    prof = np.median(depth[ya:yb, lo:hi], axis=0)
    xs = np.arange(lo, hi)
    walls = [prof[xs < x0 - 0.05 * w], prof[xs > x1 + 0.05 * w]]
    walls = [float(np.median(v)) for v in walls if v.size >= 3]
    mid = prof[(xs > x0 + 0.3 * w) & (xs < x1 - 0.3 * w)]
    if not walls or mid.size == 0:
        return None
    wall, front = float(np.median(walls)), float(np.median(mid))
    if wall - front <= 1e-6:
        return {"protrusion_m": wall - front, "s": 0.0}
    inside = np.where((wall - prof) / (wall - front) >= 0.5)[0]
    if inside.size == 0:
        return {"protrusion_m": wall - front, "s": 0.0}
    in_l, in_r = float(xs[inside[0]] - x0), float(x1 - xs[inside[-1]])
    return {"protrusion_m": wall - front, "s": 2.0 * (in_r - in_l)}


def window_shift(box, windows: list[dict[str, Any]]) -> float | None:
    """+ when the box sits left of the nearest window right above it (camera on the right), - when right of it."""
    x0, y0, x1, y1 = (float(v) for v in box)
    cx = 0.5 * (x0 + x1)
    cand = [
        u for u in windows
        if y0 - 1.2 * (y1 - y0) - 40 <= float(u["box_xyxy"][3]) <= y1
        and min(x1, float(u["box_xyxy"][2])) - max(x0, float(u["box_xyxy"][0])) > 0
    ]
    if not cand:
        return None
    u = min(cand, key=lambda u: abs(0.5 * (float(u["box_xyxy"][0]) + float(u["box_xyxy"][2])) - cx))
    return -2.0 * (cx - 0.5 * (float(u["box_xyxy"][0]) + float(u["box_xyxy"][2])))


def _projecting_open(dsl: dict[str, Any]) -> list[dict[str, Any]]:
    types = {t["name"]: t.get("structure_ir") or {} for t in dsl.get("balcony_types") or []}
    out = []
    for rec in (dsl.get("layout") or {}).get("balconies") or []:
        ir = types.get(rec.get("type")) or {}
        if ir.get("enclosure") == "open" and (rec.get("structure") or ir.get("structure")) == "projecting":
            out.append(rec)
    return out


def camera_side(dsl: dict[str, Any], depth: np.ndarray) -> dict[str, Any]:
    """'right' / 'left' when the wall tilt (>= MIN_TILT_FRAC) and the window majority agree, else None."""
    windows = [u for u in dsl.get("instances") or [] if u.get("box_xyxy") and not u.get("side_face")
               and (u.get("kind") or "window") == "window"]
    cues = [c for c in (window_shift(r["box_xyxy"], windows) for r in _projecting_open(dsl)) if c]
    right, left = sum(c > 0 for c in cues), sum(c < 0 for c in cues)
    tilt = wall_tilt(depth, dsl)
    out = {"tilt": tilt, "votes_right": right, "votes_left": left, "side": None}
    if tilt is None or abs(tilt["frac"]) < MIN_TILT_FRAC or right == left:
        return out
    by_tilt = "left" if tilt["change_m"] > 0 else "right"
    by_windows = "right" if right > left else "left"
    if by_tilt == by_windows:
        out["side"] = by_tilt
    return out


def fix_straddling_balconies(dsl: dict[str, Any], depth: np.ndarray) -> dict[str, Any]:
    """Correct, in place, the straddling projecting balconies of a merged DSL. Returns the camera decision and a row
    per corrected balcony."""
    cam = camera_side(dsl, depth)
    cam["fixed"] = []
    side = cam["side"]
    if side is None:
        return cam
    reveals = camera_x_from_reveals(dsl, depth)
    cam["reveals"] = reveals
    iw = float(((dsl.get("meta") or {}).get("image_size") or [depth.shape[1]])[0])
    inst = {u["unit_id"]: u for u in dsl.get("instances") or [] if u.get("unit_id") is not None and u.get("box_xyxy")}
    floor_y, bay_x = _bands_from_windows_dsl(dsl)
    for rec in _projecting_open(dsl):
        ids = [i for i in rec.get("partner_window_unit_ids") or [] if not isinstance(i, str) and i in inst]
        if len(ids) != 2 or inst[ids[0]].get("bay") == inst[ids[1]].get("bay"):
            continue
        m = protrusion_and_face(depth, rec["box_xyxy"])
        if m is None or m["protrusion_m"] < MIN_PROTRUSION_M:
            continue
        if reveals and reveals["reliable"]:
            # Many windows: the camera's x is known. A balcony right in front of the camera is seen square; one beside
            # it must look shifted away from the camera, the same way as the facade's side.
            bx = 0.5 * (float(rec["box_xyxy"][0]) + float(rec["box_xyxy"][2]))
            pred = reveals["k"] * (bx - reveals["x_cam"])
            if abs(pred) < REVEAL_MIN_SHIFT or ("left" if pred > 0 else "right") != side:
                continue
        left_w, right_w = sorted((inst[i] for i in ids), key=lambda u: float(u["box_xyxy"][0]))
        win = right_w if side == "right" else left_w
        wcx = 0.5 * (float(win["box_xyxy"][0]) + float(win["box_xyxy"][2]))
        x0, y0, x1, y1 = (float(v) for v in rec["box_xyxy"])
        s = m["s"]
        if side == "right" and s > 0:
            new = [x0 + s, x1]  # camera-side (right) edge kept, the far edge loses the side face
        elif side == "left" and s < 0:
            new = [x0, x1 + s]
        elif side == "right":
            new = [2.0 * wcx - x1, x1]  # keep the right edge, centre on the window
        else:
            new = [x0, 2.0 * wcx - x0]
        if new[1] - new[0] < 0.25 * (x1 - x0):
            continue
        before = list(rec["box_xyxy"])
        box = [int(round(new[0])), int(round(y0)), int(round(new[1])), int(round(y1))]
        loc = snap_box(box, floor_y, bay_x)
        rec["parallax_fix"] = {"camera": side, "side_face_px": round(abs(s), 1), "box_before": before,
                               "windows_before": ids}
        rec["box_xyxy"] = box
        rec["width_norm"] = round((box[2] - box[0]) / iw, 6)
        rec.update({k: loc[k] for k in ("bay_start", "bay_end", "bays", "bays_center", "bay")})
        rec["partner_window_unit_ids"] = [win["unit_id"]]
        rec.pop("bay_cx_norm", None)
        rec.pop("cx_norm", None)
        rec["window_cx_norm"] = round(wcx / iw, 6)
        cam["fixed"].append({"type": rec["type"], "floor": rec["floor"], "box_before": before, "box_after": box,
                             "window": win["unit_id"], "protrusion_m": round(m["protrusion_m"], 2), "s": round(s, 1)})
    return cam
