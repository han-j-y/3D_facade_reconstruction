#!/usr/bin/env python3
"""Continuous vertical-axis search with crop-aware ghost partners.

Reads frozen ``e2e_sam3_cluster_*`` (read-only); writes only under ``--out-dir``.

For each candidate midline ``cx``:
  - If reflect(box) is still in-frame → score = best opposite-side partner IoU.
  - If reflect(box) leaves the image on a side flagged as *wall continued*
    (cropped façade) → treat as a ghost / potential partner (omit from mean).
  - If reflect leaves the frame on a non-continued side → score 0 (miss).

Note: the score curve is often roughly mirror-symmetric about bbox mid, so a
peak at ``mid - d`` and one at ``mid + d`` are dual. We mark both the raw
argmax and its mid-mirror (and prefer the side toward a continued wall when
disambiguating).

Example:
  python scripts/symmetry_axis_ghost_continuous.py --stems cmp_b0150,cmp_b0250
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.walkthrough_symmetry_repair import (  # noqa: E402
    box_iou,
    facade_center_x,
    reflect_box,
)
from window_ast.symmetry import bounding_box  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sam3-root",
        type=Path,
        default=ROOT / "runs",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "runs" / "symmetry_axis_sweep" / "ghost_continuous",
    )
    ap.add_argument(
        "--stems",
        default="",
        help="comma-separated stems; empty = all e2e_sam3_cluster_* under --sam3-root",
    )
    ap.add_argument("--n-samples", type=int, default=160)
    ap.add_argument(
        "--ghost-mode",
        choices=("exclude", "soft"),
        default="exclude",
        help="legacy flag; ghosts always contribute with --ghost-credit weight",
    )
    ap.add_argument(
        "--ghost-credit",
        type=float,
        default=0.25,
        help="constant weight on ghost align-IoU in the mean (1.0 collapses to edge)",
    )
    ap.add_argument(
        "--min-inframe",
        type=int,
        default=2,
        help="require at least this many in-frame partner evaluations (else score=0)",
    )
    ap.add_argument(
        "--edge-frac",
        type=float,
        default=0.08,
        help="strip / flush threshold as fraction of image width",
    )
    ap.add_argument("--color-dist-max", type=float, default=80.0)
    ap.add_argument("--strip-std-min", type=float, default=12.0)
    return ap.parse_args()


def _font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size
        )
    except Exception:
        return ImageFont.load_default()


def load_case(sam3_root: Path, stem: str) -> dict:
    run = sam3_root / f"e2e_sam3_cluster_{stem}"
    dsl = json.loads((run / "facade_dsl.json").read_text())
    idx = json.loads((run / "index_raw.json").read_text())
    fp = Path(idx.get("facade_path", ""))
    if not fp.is_file():
        fp = EXP / "data" / "facades" / "base" / f"{stem}.jpg"
    facade = Image.open(fp).convert("RGB")
    boxes = [
        list(map(float, u["box_xyxy"]))
        for u in dsl["instances"]
        if u.get("kind") != "door"
    ]
    labels = [
        int(u["type_id"])
        for u in dsl["instances"]
        if u.get("kind") != "door"
    ]
    cols = list(dsl.get("meta", {}).get("columns_xy") or [])
    return {
        "stem": stem,
        "run": run,
        "facade": facade,
        "boxes": boxes,
        "labels": labels,
        "columns_xy": cols,
        "bb": bounding_box(boxes),
    }


def edge_bays_complete(
    columns_xy: list | None,
    *,
    min_ratio: float = 0.70,
) -> dict:
    """Whether outermost column ranges look like full bays (not truncated)."""
    if not columns_xy or len(columns_xy) < 2:
        return {
            "left": False,
            "right": False,
            "both": False,
            "left_ratio": 0.0,
            "right_ratio": 0.0,
            "median_w": 0.0,
        }
    widths = [float(b - a) for a, b in columns_xy]
    med = float(np.median(widths))
    if med < 1.0:
        return {
            "left": False,
            "right": False,
            "both": False,
            "left_ratio": 0.0,
            "right_ratio": 0.0,
            "median_w": med,
        }
    lr = widths[0] / med
    rr = widths[-1] / med
    left_ok = lr >= min_ratio
    right_ok = rr >= min_ratio
    return {
        "left": left_ok,
        "right": right_ok,
        "both": bool(left_ok and right_ok),
        "left_ratio": float(lr),
        "right_ratio": float(rr),
        "median_w": med,
        "widths": widths,
    }


def select_focus_facade(
    boxes: list[list[float]],
    columns_xy: list | None,
    *,
    gap_pitch_ratio: float = 1.75,
) -> dict:
    """Pick one façade when detections span multiple buildings.

    Splits column centers on unusually large gaps; keeps the cluster with the
    most windows (tie → widest span). cmp_b0366-style street rows.
    """
    if not boxes:
        return {
            "boxes": boxes,
            "columns_xy": columns_xy or [],
            "changed": False,
            "n_groups": 1,
            "keep_cols": None,
        }
    cols = list(columns_xy or [])
    if len(cols) < 2:
        return {
            "boxes": boxes,
            "columns_xy": cols,
            "changed": False,
            "n_groups": 1,
            "keep_cols": None,
        }
    # Sort columns left→right (DSL order is not guaranteed).
    order = sorted(range(len(cols)), key=lambda i: 0.5 * (cols[i][0] + cols[i][1]))
    cols_sorted = [cols[i] for i in order]
    centers = [0.5 * (float(a) + float(b)) for a, b in cols_sorted]
    pitches = np.diff(np.asarray(centers, dtype=np.float64))
    small = [float(p) for p in pitches if p > 1.0]
    if not small:
        med_p = 0.0
    else:
        # Within-façade pitch ≈ the *smallest* regular step; large street
        # gaps between buildings should not dominate the median.
        med_p = float(min(small)) if len(small) == 1 else float(np.median(sorted(small)[: max(1, (len(small) + 1) // 2)]))
    if med_p < 1.0:
        return {
            "boxes": boxes,
            "columns_xy": cols,
            "changed": False,
            "n_groups": 1,
            "keep_cols": None,
        }
    box_ws = [b[2] - b[0] for b in boxes]
    med_bw = float(np.median(box_ws)) if box_ws else med_p
    gap_thr = max(gap_pitch_ratio * med_p, 2.5 * med_bw)

    groups: list[list[int]] = [[order[0]]]
    for i, p in enumerate(pitches):
        if float(p) >= gap_thr:
            groups.append([order[i + 1]])
        else:
            groups[-1].append(order[i + 1])
    if len(groups) <= 1:
        return {
            "boxes": boxes,
            "columns_xy": cols,
            "changed": False,
            "n_groups": 1,
            "keep_cols": None,
        }

    def bay_of(cx: float) -> int:
        for i, (a, b) in enumerate(cols):
            if a <= cx <= b:
                return i
        cds = [0.5 * (a + b) for a, b in cols]
        return int(np.argmin([abs(cx - c) for c in cds]))

    # Prefer the group nearest image/window mid when sizes are close; otherwise
    # the largest. Avoid locking onto a side building when the focus façade is
    # central (cmp_b0366).
    all_cx = [0.5 * (b[0] + b[2]) for b in boxes]
    scene_mid = float(np.median(all_cx)) if all_cx else 0.0
    scored = []
    for g in groups:
        gset = set(g)
        gboxes = [b for b in boxes if bay_of(0.5 * (b[0] + b[2])) in gset]
        if not gboxes:
            continue
        span = max(b[2] for b in gboxes) - min(b[0] for b in gboxes)
        gmid = float(np.mean([0.5 * (b[0] + b[2]) for b in gboxes]))
        scored.append((len(gboxes), span, -abs(gmid - scene_mid), g))
    if not scored:
        return {
            "boxes": boxes,
            "columns_xy": cols,
            "changed": False,
            "n_groups": len(groups),
            "keep_cols": None,
        }
    scored.sort(reverse=True)
    # If top two groups have similar counts, prefer nearer scene mid.
    best_g = scored[0][3]
    if len(scored) >= 2 and scored[0][0] <= scored[1][0] + 1:
        best_g = max(scored[:2], key=lambda t: t[2])[3]
    gset = set(best_g)
    fboxes = [b for b in boxes if bay_of(0.5 * (b[0] + b[2])) in gset]
    fcols = [
        cols[i]
        for i in sorted(best_g, key=lambda i: 0.5 * (cols[i][0] + cols[i][1]))
    ]
    return {
        "boxes": fboxes,
        "columns_xy": fcols,
        "changed": True,
        "n_groups": len(groups),
        "keep_cols": best_g,
        "group_sizes": [len(g) for g in groups],
        "gap_thr": gap_thr,
    }


def detect_wall_continuation(
    image: Image.Image,
    boxes: list[list[float]],
    *,
    columns_xy: list | None = None,
    edge_frac: float = 0.08,
    color_dist_max: float = 80.0,
    strip_std_min: float = 12.0,
    adjacent_color_min: float = 70.0,
    corner_vert_ratio: float = 2.5,
    black_thr: float = 20.0,
    mask_black_frac: float = 0.35,
    content_mask: np.ndarray | None = None,
) -> dict:
    """Heuristic L/R 'wall continues beyond frame' flags.

    Pipeline:
      0. **Fully-captured gate** — if both outermost bays look complete and
         there is no mask/rectification crop, refuse continuation (cmp_b0285).
      1. Raw flush+wall strip, with guards for full-bleed / adjacent / corner.
      2. **Mask crop** — black *or* pure-gray rectification pad with wall just
         inside (both are cut-off unseen; cmp_b0167 right, cmp_b0150 gray).
    """
    im = np.asarray(image, dtype=np.float32)
    H, W = im.shape[:2]
    bb = bounding_box(boxes)
    assert bb is not None
    strip_w = max(4, int(round(edge_frac * W)))
    gap_thr = edge_frac * W
    gray = im.mean(axis=2)
    # Unseen = pure black OR pure gray pad (same as ghost content_mask).
    if content_mask is None:
        content_mask = make_content_mask(
            image, boxes=boxes, black_thr=black_thr
        )
    content = np.asarray(content_mask, dtype=bool)
    if content.shape[:2] != (H, W):
        content = gray > black_thr

    ix0 = int(bb[0] + 0.15 * (bb[2] - bb[0]))
    ix1 = int(bb[2] - 0.15 * (bb[2] - bb[0]))
    iy0 = int(bb[1] + 0.20 * (bb[3] - bb[1]))
    iy1 = int(bb[3] - 0.20 * (bb[3] - bb[1]))
    ix0, ix1 = max(0, ix0), min(W, max(ix0 + 1, ix1))
    iy0, iy1 = max(0, iy0), min(H, max(iy0 + 1, iy1))
    interior = im[iy0:iy1, ix0:ix1].mean(axis=(0, 1))

    def vert_energy(x: float, band: int = 6) -> float:
        x0, x1 = max(0, int(x - band)), min(W, int(x + band))
        if x1 <= x0 + 1:
            return 0.0
        return float(np.abs(np.diff(gray[:, x0:x1], axis=1)).mean())

    def side_stats(side: str) -> dict:
        if side == "left":
            strip = im[:, :strip_w]
            cstrip = content[:, :strip_w]
            gap = float(bb[0])
            near = im[iy0:iy1, int(bb[0]) : int(min(bb[0] + 20, bb[2]))]
            x_bb = float(bb[0])
            x_in = float(bb[0] + 20.0)
            # just inside content for mask-crop wall check
            inner = im[:, strip_w : min(W, 2 * strip_w)]
            inner_m = content[:, strip_w : min(W, 2 * strip_w)]
        else:
            strip = im[:, -strip_w:]
            cstrip = content[:, -strip_w:]
            gap = float(W - bb[2])
            near = im[iy0:iy1, int(max(bb[2] - 20, bb[0])) : int(bb[2])]
            x_bb = float(bb[2])
            x_in = float(bb[2] - 20.0)
            inner = im[:, max(0, W - 2 * strip_w) : max(0, W - strip_w)]
            inner_m = content[:, max(0, W - 2 * strip_w) : max(0, W - strip_w)]
        # Fraction of edge strip that is cut-off unseen (black or pure gray).
        unseen_frac = float((~cstrip).mean()) if cstrip.size else 0.0
        if cstrip.any():
            pix = strip[cstrip]
            mean = pix.mean(axis=0)
            std = float(pix.std())
        else:
            mean = strip.mean(axis=(0, 1))
            std = float(strip.std())
        near_mean = near.mean(axis=(0, 1)) if near.size else interior
        color_dist = float(np.linalg.norm(mean - interior))
        color_vs_near = float(np.linalg.norm(mean - near_mean))
        flush = gap < gap_thr
        looks_wall = (color_dist <= color_dist_max) and (std >= strip_std_min)
        adjacent_other = color_vs_near >= adjacent_color_min
        ve_bb = vert_energy(x_bb)
        ve_in = vert_energy(x_in)
        corner_ratio = ve_bb / max(ve_in, 1e-6)
        has_corner = corner_ratio >= corner_vert_ratio
        continued_raw = bool(
            flush and looks_wall and not adjacent_other and not has_corner
        )
        # Mask / rectification crop: unseen pad + wall-like content inside.
        if inner_m.any():
            ipix = inner[inner_m]
            inner_dist = float(np.linalg.norm(ipix.mean(axis=0) - interior))
            inner_std = float(ipix.std())
            inner_wall = (inner_dist <= color_dist_max) and (inner_std >= strip_std_min)
        else:
            inner_dist, inner_std, inner_wall = 999.0, 0.0, False
        mask_continued = bool(
            unseen_frac >= mask_black_frac and inner_wall and flush
        )
        return {
            "gap_px": gap,
            "strip_std": std,
            "color_dist": color_dist,
            "color_vs_near": color_vs_near,
            "flush": flush,
            "looks_wall": looks_wall,
            "adjacent_other": adjacent_other,
            "corner_ratio": corner_ratio,
            "has_corner": has_corner,
            "continued_raw": continued_raw,
            "black_frac": unseen_frac,  # legacy key; includes pure-gray pad
            "unseen_frac": unseen_frac,
            "inner_wall": inner_wall,
            "inner_color_dist": inner_dist,
            "mask_continued": mask_continued,
        }

    left = side_stats("left")
    right = side_stats("right")
    bay_comp = edge_bays_complete(columns_xy)

    full_bleed = bool(left["flush"] and right["flush"])
    continued_left = bool(
        (left["continued_raw"] or left["mask_continued"]) and not full_bleed
    )
    continued_right = bool(
        (right["continued_raw"] or right["mask_continued"]) and not full_bleed
    )
    if continued_left and continued_right:
        continued_left = False
        continued_right = False
        full_bleed = True

    # Fully-captured façade: both edge bays look complete and neither side is a
    # mask crop → do not invent off-frame partners (cmp_b0285 hotel).
    mask_crop = bool(left["mask_continued"] or right["mask_continued"])
    fully_captured = bool(bay_comp["both"] and not mask_crop)
    if fully_captured:
        continued_left = False
        continued_right = False

    return {
        "image_size": [W, H],
        "window_bb": list(bb),
        "margin_L": float(bb[0]),
        "margin_R": float(W - bb[2]),
        "left": left,
        "right": right,
        "full_bleed": full_bleed,
        "fully_captured": fully_captured,
        "edge_bays_complete": bay_comp,
        "mask_crop": mask_crop,
        "continued_left": continued_left,
        "continued_right": continued_right,
        "params": {
            "edge_frac": edge_frac,
            "color_dist_max": color_dist_max,
            "strip_std_min": strip_std_min,
            "adjacent_color_min": adjacent_color_min,
            "corner_vert_ratio": corner_vert_ratio,
            "strip_w": strip_w,
            "mask_black_frac": mask_black_frac,
        },
    }


def _reflect(b: list[float], cx: float) -> list[float]:
    return reflect_box(b, axis="v", cx=cx, cy=0.0)


def _box_in_image_frac(b: list[float], W: float, H: float) -> float:
    x0, y0, x1, y1 = b
    ix0, iy0 = max(x0, 0.0), max(y0, 0.0)
    ix1, iy1 = min(x1, W), min(y1, H)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = max((x1 - x0) * (y1 - y0), 1e-9)
    return float(inter / area)


def bay_pitch_model(columns_xy: list | None) -> dict | None:
    """Median bay pitch + regularity from column ranges."""
    if not columns_xy or len(columns_xy) < 2:
        return None
    centers = [0.5 * (float(a) + float(b)) for a, b in columns_xy]
    pitches = np.diff(np.asarray(centers, dtype=np.float64))
    if len(pitches) == 0:
        return None
    med = float(np.median(pitches))
    if med < 1.0:
        return None
    rel_std = float(np.std(pitches) / med)
    return {
        "centers": centers,
        "pitches": pitches.tolist(),
        "median_pitch": med,
        "rel_std": rel_std,
        "regular": rel_std <= 0.35,
    }


def ghost_fits_bay_pattern(
    rb: list[float],
    *,
    side: str,
    columns_xy: list | None,
    pitch_tol: float = 0.35,
) -> bool:
    """True if reflected ghost center lands on an extrapolated bay.

    Requires a reasonably regular bay pitch; then the ghost's x-center must
    lie near ``edge_center ± k * pitch`` for small integer ``k >= 1``.
    """
    model = bay_pitch_model(columns_xy)
    if model is None or not model["regular"]:
        return False
    centers = model["centers"]
    p = model["median_pitch"]
    rc = 0.5 * (rb[0] + rb[2])
    tol = pitch_tol * p
    if side == "left":
        edge = centers[0]
        for k in range(1, 8):
            if abs(rc - (edge - k * p)) <= tol:
                return True
    else:
        edge = centers[-1]
        for k in range(1, 8):
            if abs(rc - (edge + k * p)) <= tol:
                return True
    return False


def _box_content_frac(
    content_mask: np.ndarray | None,
    b: list[float],
) -> float:
    """Fraction of reflected-box pixels on façade content (not black / pure-gray pad)."""
    if content_mask is None:
        return 1.0
    H, W = content_mask.shape[:2]
    x0 = max(0, int(np.floor(b[0])))
    y0 = max(0, int(np.floor(b[1])))
    x1 = min(W, int(np.ceil(b[2])))
    y1 = min(H, int(np.ceil(b[3])))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = content_mask[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(patch.mean())


def _local_std_map(gray: np.ndarray, k: int = 7) -> np.ndarray:
    """Box-filter local standard deviation (numpy only)."""
    k = int(max(3, k) | 1)  # odd
    g = gray.astype(np.float64)
    pad = k // 2
    gp = np.pad(g, pad, mode="edge")
    # integral images of g and g^2
    s = np.pad(gp, ((1, 0), (1, 0)), mode="constant")
    s = s.cumsum(0).cumsum(1)
    s2 = np.pad(gp * gp, ((1, 0), (1, 0)), mode="constant")
    s2 = s2.cumsum(0).cumsum(1)

    def rect(integral: np.ndarray) -> np.ndarray:
        return (
            integral[k:, k:]
            - integral[:-k, k:]
            - integral[k:, :-k]
            + integral[:-k, :-k]
        )

    area = float(k * k)
    mean = rect(s) / area
    mean2 = rect(s2) / area
    return np.sqrt(np.maximum(mean2 - mean * mean, 0.0))


def _border_connected(seed: np.ndarray) -> np.ndarray:
    """Keep True components of ``seed`` that touch the image border (4-conn)."""
    H, W = seed.shape
    out = np.zeros_like(seed, dtype=bool)
    if not seed.any():
        return out
    visited = np.zeros_like(seed, dtype=bool)
    stack: list[tuple[int, int]] = []
    for x in range(W):
        if seed[0, x]:
            stack.append((0, x))
        if seed[H - 1, x]:
            stack.append((H - 1, x))
    for y in range(H):
        if seed[y, 0]:
            stack.append((y, 0))
        if seed[y, W - 1]:
            stack.append((y, W - 1))
    while stack:
        y, x = stack.pop()
        if y < 0 or y >= H or x < 0 or x >= W:
            continue
        if visited[y, x] or not seed[y, x]:
            continue
        visited[y, x] = True
        out[y, x] = True
        stack.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
    return out


def make_content_mask(
    image: Image.Image,
    *,
    boxes: list[list[float]] | None = None,
    black_thr: float = 20.0,
    gray_facade_chroma_min: float = 22.0,
) -> np.ndarray:
    """True on façade content; False on cut-off unseen pad.

    Unseen = **pure black** or **pure gray** rectification pad (both are
    cropped / non-façade). Pure-gray pads are border-connected low-chroma
    smooth regions. If the façade itself is already low-chroma (grayscale
    building), fall back to black-only so the wall is not eaten.
    """
    im = np.asarray(image, dtype=np.float32)
    gray = im.mean(axis=2)
    black = gray <= black_thr
    chroma = im.max(axis=2) - im.min(axis=2)
    H, W = gray.shape

    facade_c: float | None = None
    if boxes:
        bb = bounding_box(boxes)
        if bb is not None:
            x0, y0 = max(0, int(bb[0])), max(0, int(bb[1]))
            x1, y1 = min(W, int(bb[2])), min(H, int(bb[3]))
            if x1 > x0 and y1 > y0:
                facade_c = float(np.median(chroma[y0:y1, x0:x1]))

    if facade_c is None or facade_c < gray_facade_chroma_min:
        return ~black

    chroma_max = float(np.clip(0.40 * facade_c, 12.0, 28.0))
    lstd = _local_std_map(gray, k=7)
    pure_gray = (chroma <= chroma_max) & (lstd <= 16.0) & (~black)
    pad = _border_connected(pure_gray | black)
    return ~pad


def _box_from_center(cx: float, cy: float, w: float, h: float) -> list[float]:
    return [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]


def best_symmetry_ghost_placement(
    rb: list[float],
    *,
    side: str,
    columns_xy: list | None,
) -> tuple[list[float], float]:
    """Place a ghost by horizontal symmetry, optionally snapped to bay lattice.

    Start from pure reflect ``rb``. If a lattice center ``edge ± k·pitch`` lies
    within ``0.35·pitch`` of the reflected center, snap to that center (same
    size / y). Return ``(placed, IoU(rb, placed))`` — equals 1 when unsnapped
    or when lattice agrees with symmetry.
    """
    pure = [float(x) for x in rb]
    model = bay_pitch_model(columns_xy)
    if model is None or not model["regular"]:
        return pure, 1.0
    w = float(rb[2] - rb[0])
    h = float(rb[3] - rb[1])
    cy = 0.5 * (rb[1] + rb[3])
    rc = 0.5 * (rb[0] + rb[2])
    p = model["median_pitch"]
    tol = 0.35 * p
    edge = model["centers"][0] if side == "left" else model["centers"][-1]
    best_c = None
    best_dist = float("inf")
    for k in range(1, 8):
        c = edge - k * p if side == "left" else edge + k * p
        dist = abs(c - rc)
        if dist < best_dist:
            best_dist = dist
            best_c = c
    if best_c is not None and best_dist <= tol:
        placed = _box_from_center(best_c, cy, w, h)
        return placed, float(box_iou(pure, placed))
    return pure, 1.0


def build_symmetry_ghosts(
    boxes: list[list[float]],
    cx: float,
    *,
    W: float,
    H: float,
    continued_left: bool,
    continued_right: bool,
    content_mask: np.ndarray | None = None,
    columns_xy: list | None = None,
    in_frac_min: float = 0.5,
    content_frac_min: float = 0.9,
) -> list[dict]:
    """Create virtual ghost windows at symmetry-reflected poses (cont. wall only).

    Unseen = OOB crop **or** reflection that is partly pure-gray
    (``content_frac < content_frac_min``, default 0.9 so ~10%+ gray → ghost).
    """
    ghosts: list[dict] = []
    for i, b in enumerate(boxes):
        rb = _reflect(b, cx)
        frac = _box_in_image_frac(rb, W, H)
        cfrac = _box_content_frac(content_mask, rb) if frac > 0 else 0.0
        out_or_gray = (frac < in_frac_min) or (cfrac < content_frac_min)
        if not out_or_gray:
            continue
        rc = 0.5 * (rb[0] + rb[2])
        overflow_L = max(0.0, -rb[0])
        overflow_R = max(0.0, rb[2] - W)
        if frac < in_frac_min:
            if rc < 0.0 or overflow_L > overflow_R:
                side = "left"
            else:
                side = "right"
        else:
            # in-bitmap gray zone: side toward nearer frame edge / overflow
            side = "left" if rc < cx else "right"
        cont = continued_left if side == "left" else continued_right
        if not cont:
            continue
        placed, align = best_symmetry_ghost_placement(
            rb, side=side, columns_xy=columns_xy
        )
        ghosts.append(
            {
                "source_i": i,
                "side": side,
                "pure_reflect": rb,
                "box": placed,
                "align_iou": align,
            }
        )
    return ghosts


def partner_score_for_box(
    i: int,
    boxes: list[list[float]],
    cx: float,
    *,
    W: float,
    H: float,
    continued_left: bool,
    continued_right: bool,
    ghost_credit: float,
    columns_xy: list | None = None,
    require_pattern_ghost: bool = True,
    in_frac_min: float = 0.5,
    content_mask: np.ndarray | None = None,
    content_frac_min: float = 0.9,
    ghosts: list[dict] | None = None,
    allow_self_iou: bool = True,
) -> tuple[float, str]:
    """Return (score, kind) with kind in {match, ghost, miss}.

    - **Self-IoU**: reflection may pair with the original box (on-axis columns).
    - **In-frame partners first**: even if the reflection overlaps gray/black
      pad, a real opposite-side window wins (cmp_b0010 right→left pair).
    - **Gray-zone / OOB**: only if no real partner — continued wall → ghost,
      else miss.
    """
    b = boxes[i]
    ci = 0.5 * (b[0] + b[2])
    rb = _reflect(b, cx)
    frac = _box_in_image_frac(rb, W, H)
    cfrac = _box_content_frac(content_mask, rb) if frac > 0 else 0.0
    out_or_gray = (frac < in_frac_min) or (cfrac < content_frac_min)

    def ghost_side() -> str:
        rc = 0.5 * (rb[0] + rb[2])
        overflow_L = max(0.0, -rb[0])
        overflow_R = max(0.0, rb[2] - W)
        if frac < in_frac_min:
            if rc < 0.0 or overflow_L > overflow_R:
                return "left"
            return "right"
        return "left" if rc < cx else "right"

    # Always try real partners (self / opposite boxes) first — not ghosts.
    # Matching a virtual ghost must stay kind="ghost" so ghost_credit applies.
    best_real = 0.0
    if allow_self_iou:
        best_real = max(best_real, float(box_iou(rb, b)))
    for j, c in enumerate(boxes):
        if j == i:
            continue
        cj = 0.5 * (c[0] + c[2])
        if (ci < cx) == (cj < cx):
            continue
        best_real = max(best_real, float(box_iou(rb, c)))

    # Real opposite/self overlap beats gray/ghost routing.
    if best_real > 0.0 and (not out_or_gray or best_real >= 0.15):
        return float(best_real), "match"

    if out_or_gray:
        side = ghost_side()
        cont = continued_left if side == "left" else continued_right
        if not cont:
            return 0.0, "miss"
        placed = None
        align = 0.0
        if ghosts is not None:
            for g in ghosts:
                if g["source_i"] == i:
                    placed = g["box"]
                    align = float(g["align_iou"])
                    break
        if placed is None:
            placed, align = best_symmetry_ghost_placement(
                rb, side=side, columns_xy=columns_xy
            )
        return float(align), "ghost"

    if best_real > 0.0:
        return float(best_real), "match"
    return 0.0, "miss"


def score_axis(
    boxes: list[list[float]],
    cx: float,
    *,
    W: float,
    H: float,
    continued_left: bool,
    continued_right: bool,
    ghost_mode: str = "exclude",
    ghost_credit: float = 0.25,
    min_inframe: int = 2,
    columns_xy: list | None = None,
    require_pattern_ghost: bool = True,
    content_mask: np.ndarray | None = None,
    allow_self_iou: bool = True,
) -> dict:
    ghosts = build_symmetry_ghosts(
        boxes,
        cx,
        W=W,
        H=H,
        continued_left=continued_left,
        continued_right=continued_right,
        content_mask=content_mask,
        columns_xy=columns_xy,
    )
    n_match = n_ghost = n_miss = 0
    match_ious: list[float] = []
    ghost_ious: list[float] = []
    for i in range(len(boxes)):
        s, kind = partner_score_for_box(
            i,
            boxes,
            cx,
            W=W,
            H=H,
            continued_left=continued_left,
            continued_right=continued_right,
            ghost_credit=1.0,
            columns_xy=columns_xy,
            require_pattern_ghost=require_pattern_ghost,
            content_mask=content_mask,
            ghosts=ghosts,
            allow_self_iou=allow_self_iou,
        )
        if kind == "ghost":
            n_ghost += 1
            ghost_ious.append(s)
        elif kind == "miss":
            n_miss += 1
        else:
            n_match += 1
            match_ious.append(s)

    n_eval = n_match + n_ghost
    g_w = 0.0
    if n_match < min_inframe:
        score = 0.0
    else:
        # Ghosts always share the same weight (default --ghost-credit 0.25),
        # whether or not they outnumber in-frame matches. Full ghost credit
        # (1.0) lets continued-edge axes dominate via gray/OOB reflections.
        g_w = float(ghost_credit)
        total = sum(match_ious) + g_w * sum(ghost_ious)
        denom = n_match + n_miss + n_ghost
        score = float(total / max(denom, 1))

    return {
        "cx": float(cx),
        "score": score,
        "n_match": n_match,
        "n_ghost": n_ghost,
        "n_miss": n_miss,
        "mean_match_iou": float(np.mean(match_ious)) if match_ious else 0.0,
        "mean_ghost_iou": float(np.mean(ghost_ious)) if ghost_ious else 0.0,
        "ghost_frac": n_ghost / max(len(boxes), 1),
        "ghost_mode": ghost_mode,
        "ghost_weight": float(g_w),
        "n_ghost_boxes": len(ghosts),
    }


def score_axis_no_ghost(
    boxes: list[list[float]],
    cx: float,
    *,
    W: float,
    H: float,
    min_inframe: int,
    columns_xy: list | None = None,
    content_mask: np.ndarray | None = None,
    allow_self_iou: bool = True,
) -> float:
    r = score_axis(
        boxes,
        cx,
        W=W,
        H=H,
        continued_left=False,
        continued_right=False,
        ghost_mode="exclude",
        ghost_credit=0.0,
        min_inframe=min_inframe,
        columns_xy=columns_xy,
        require_pattern_ghost=True,
        content_mask=content_mask,
        allow_self_iou=allow_self_iou,
    )
    return float(r["score"])


def sweep(
    boxes: list[list[float]],
    *,
    W: float,
    H: float,
    continued_left: bool,
    continued_right: bool,
    ghost_mode: str,
    ghost_credit: float,
    min_inframe: int,
    n_samples: int,
    columns_xy: list | None = None,
    require_pattern_ghost: bool = True,
    content_mask: np.ndarray | None = None,
    allow_self_iou: bool = True,
) -> tuple[np.ndarray, list[dict], list[float]]:
    bb = bounding_box(boxes)
    assert bb is not None
    xs = np.linspace(float(bb[0] + 1.0), float(bb[2] - 1.0), n_samples)
    rows = [
        score_axis(
            boxes,
            float(cx),
            W=W,
            H=H,
            continued_left=continued_left,
            continued_right=continued_right,
            ghost_mode=ghost_mode,
            ghost_credit=ghost_credit,
            min_inframe=min_inframe,
            columns_xy=columns_xy,
            require_pattern_ghost=require_pattern_ghost,
            content_mask=content_mask,
            allow_self_iou=allow_self_iou,
        )
        for cx in xs
    ]
    baseline = [
        score_axis_no_ghost(
            boxes,
            float(cx),
            W=W,
            H=H,
            min_inframe=min_inframe,
            columns_xy=columns_xy,
            content_mask=content_mask,
            allow_self_iou=allow_self_iou,
        )
        for cx in xs
    ]
    return xs, rows, baseline


def detect_edge_bay_contamination(
    boxes: list[list[float]],
    columns_xy: list,
    *,
    image_width: float,
    edge_frac: float = 0.08,
    width_ratio: float = 1.35,
    area_ratio_max: float = 0.55,
    aspect_ratio_min: float = 1.6,
) -> dict:
    """Flag outermost bay(s) that are not part of the main façade plane.

    Two failure modes:
      - **Adjacent façade** (cmp_b0220): flush + much *wider* than median bay.
      - **Side / corner return** (cmp_b0300): flush + much *smaller* / taller
        windows than the main bays (angled end wall).

    Both are excluded when computing ``clean_mid``.
    """
    if not columns_xy or len(columns_xy) < 3 or not boxes:
        return {
            "left": False,
            "right": False,
            "exclude_bay_idxs": [],
            "clean_mid": facade_center_x(boxes) if boxes else 0.0,
            "reasons": {},
        }

    def bay_of(cx: float) -> int:
        for i, (a, b) in enumerate(columns_xy):
            if a <= cx <= b:
                return i
        mids = [0.5 * (a + b) for a, b in columns_xy]
        return int(np.argmin([abs(cx - m) for m in mids]))

    by: dict[int, list[list[float]]] = {i: [] for i in range(len(columns_xy))}
    for b in boxes:
        by[bay_of(0.5 * (b[0] + b[2]))].append(b)

    widths = [float(b - a) for a, b in columns_xy]
    interior_w = widths[1:-1] if len(widths) > 2 else widths
    med_w = float(np.median(interior_w)) if interior_w else float(np.median(widths))

    def bay_stats(i: int) -> dict:
        group = by[i]
        if not group:
            return {"mean_area": 0.0, "mean_aspect": 0.0, "n": 0}
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in group]
        aspects = [(b[3] - b[1]) / max(b[2] - b[0], 1e-6) for b in group]
        return {
            "mean_area": float(np.mean(areas)),
            "mean_aspect": float(np.mean(aspects)),
            "n": len(group),
        }

    stats = [bay_stats(i) for i in range(len(columns_xy))]
    interior_areas = [
        stats[i]["mean_area"]
        for i in range(1, len(stats) - 1)
        if stats[i]["mean_area"] > 0
    ]
    med_area = float(np.median(interior_areas)) if interior_areas else float(
        np.median([s["mean_area"] for s in stats if s["mean_area"] > 0] or [1.0])
    )
    interior_asp = [
        stats[i]["mean_aspect"]
        for i in range(1, len(stats) - 1)
        if stats[i]["mean_aspect"] > 0
    ]
    med_asp = float(np.median(interior_asp)) if interior_asp else 1.0

    gap_thr = edge_frac * image_width

    def edge_flag(i: int, side: str) -> tuple[bool, str]:
        flush = (
            columns_xy[i][0] < gap_thr
            if side == "left"
            else (image_width - columns_xy[i][1]) < gap_thr
        )
        if not flush:
            return False, ""
        if widths[i] >= width_ratio * med_w:
            return True, "wide_adjacent"
        if med_area > 0 and stats[i]["mean_area"] <= area_ratio_max * med_area:
            return True, "small_side_return"
        if med_asp > 0 and stats[i]["mean_aspect"] >= aspect_ratio_min * med_asp:
            return True, "tall_side_return"
        return False, ""

    left, left_why = edge_flag(0, "left")
    right, right_why = edge_flag(len(columns_xy) - 1, "right")
    exclude: list[int] = []
    reasons: dict[str, str] = {}
    if left:
        exclude.append(0)
        reasons["left"] = left_why
    if right:
        exclude.append(len(columns_xy) - 1)
        reasons["right"] = right_why

    if exclude:
        clean = [
            b for b in boxes if bay_of(0.5 * (b[0] + b[2])) not in exclude
        ]
        if len(clean) < 2:
            clean = boxes
    else:
        clean = boxes

    return {
        "left": left,
        "right": right,
        "exclude_bay_idxs": exclude,
        "reasons": reasons,
        "median_bay_w": med_w,
        "median_bay_area": med_area,
        "bay_widths": widths,
        "bay_mean_areas": [s["mean_area"] for s in stats],
        "bay_mean_aspects": [s["mean_aspect"] for s in stats],
        "clean_mid": float(facade_center_x(clean)),
        "n_clean_boxes": len(clean),
    }


def _local_maxima(xs: np.ndarray, scores: list[float], min_score: float = 0.0) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for i in range(1, len(scores) - 1):
        if scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1] and scores[i] >= min_score:
            out.append((float(scores[i]), float(xs[i])))
    return out


def disambiguate_mirror_peak(
    raw_best_cx: float,
    mid: float,
    *,
    continued_left: bool,
    continued_right: bool,
    full_bleed: bool,
    xs: np.ndarray,
    scores: list[float],
    facade_width: float,
    clean_mid: float | None = None,
    fully_captured: bool = False,
    columns_xy: list | None = None,
    mid_score_ratio: float = 0.85,
    cont_flip_min_frac: float = 0.12,
    strong_peak_ratio: float = 0.85,
) -> dict:
    """Resolve dual peaks / false continuation / adjacent-bay contamination.

    Rules:
      - If ``clean_mid`` (mid after dropping wide flush edge bays) differs from
        bbox mid, pick the strong local peak nearest ``clean_mid`` (cmp_b0220).
      - One-sided crop → prefer mirror toward continued wall only if raw is
        clearly off-center.
      - Fully captured + odd bay count → prefer strong peak nearest the middle
        column center (cmp_b0147 five-bay, cmp_b0310 three-bay mirror side).
      - Fully captured (else) → among strong peaks, prefer nearest mid.
      - No crop / full-bleed → prefer mid only when its score is close to best
        (equal-bay folds like cmp_b0050). Do **not** snap a strong raw peak
        to mid when mid scores worse (cmp_b0300).
    """
    mirror_cx = float(2.0 * mid - raw_best_cx)
    j_m = int(np.argmin(np.abs(xs - mirror_cx)))
    j_r = int(np.argmin(np.abs(xs - raw_best_cx)))
    j_mid = int(np.argmin(np.abs(xs - mid)))
    mirror_score = float(scores[j_m])
    raw_score = float(scores[j_r])
    mid_score = float(scores[j_mid])
    best_score = max(scores) if scores else 0.0
    fw = max(float(facade_width), 1.0)
    dist_raw = abs(raw_best_cx - mid)
    target_mid = float(clean_mid) if clean_mid is not None else float(mid)

    # Adjacent-bay cleanup: prefer strong peak nearest the cleaned mid.
    if clean_mid is not None and abs(clean_mid - mid) > 0.03 * fw:
        peaks = _local_maxima(xs, scores, min_score=strong_peak_ratio * best_score)
        if peaks:
            _s, pick_cx = min(peaks, key=lambda t: abs(t[1] - clean_mid))
            return {
                "raw_best_cx": float(raw_best_cx),
                "raw_best_score": raw_score,
                "mirror_cx": mirror_cx,
                "mirror_score": mirror_score,
                "mid": float(mid),
                "mid_score": mid_score,
                "clean_mid": float(clean_mid),
                "chosen_cx": float(pick_cx),
                "reason": "strong_peak_nearest_clean_mid",
                "full_bleed": bool(full_bleed),
                "fully_captured": bool(fully_captured),
                "delta_raw": float(raw_best_cx - mid),
                "delta_mirror": float(mirror_cx - mid),
            }

    chosen = raw_best_cx
    reason = "raw_argmax"
    off_center = dist_raw >= cont_flip_min_frac * fw

    if continued_right and not continued_left and off_center:
        if mirror_cx > mid and raw_best_cx < mid:
            chosen = mirror_cx
            reason = "prefer_mirror_toward_continued_right"
        elif raw_best_cx > mid:
            chosen = raw_best_cx
            reason = "raw_already_toward_continued_right"
    elif continued_left and not continued_right and off_center:
        if mirror_cx < mid and raw_best_cx > mid:
            chosen = mirror_cx
            reason = "prefer_mirror_toward_continued_left"
        elif raw_best_cx < mid:
            chosen = raw_best_cx
            reason = "raw_already_toward_continued_left"
    elif continued_right and not continued_left and not off_center:
        chosen = raw_best_cx
        reason = "raw_near_mid_ignore_continued_right"
    elif continued_left and not continued_right and not off_center:
        chosen = raw_best_cx
        reason = "raw_near_mid_ignore_continued_left"
    elif fully_captured and not continued_left and not continued_right:
        # Prefer architectural center when the façade is complete in-frame.
        target = float(mid)
        soft = 0.55 * best_score
        if columns_xy and len(columns_xy) >= 3 and (len(columns_xy) % 2 == 1):
            # columns_xy may be unsorted in the DSL — always sort by center x
            # before taking the middle bay (cmp_b0168 had [898,…,410,499,…]
            # so unsorted[len//2] wrongly targeted bay 410 instead of 499).
            ordered = sorted(
                columns_xy, key=lambda ab: 0.5 * (float(ab[0]) + float(ab[1]))
            )
            mid_i = len(ordered) // 2
            a, b = ordered[mid_i]
            target = 0.5 * (float(a) + float(b))
            soft = 0.45 * best_score
        peaks = _local_maxima(xs, scores, min_score=soft)
        if peaks:
            _s, pick_cx = min(peaks, key=lambda t: abs(t[1] - target))
            chosen = float(pick_cx)
            reason = (
                "fully_captured_odd_bay_center"
                if columns_xy and len(columns_xy) % 2 == 1
                else "fully_captured_peak_nearest_mid"
            )
        elif mid_score >= mid_score_ratio * best_score:
            chosen = float(mid)
            reason = "prefer_mid_near_best_score"
        else:
            chosen = raw_best_cx
            reason = "raw_argmax"
    else:
        # Only prefer geometric mid when it is nearly as good as the best peak.
        if mid_score >= mid_score_ratio * best_score:
            chosen = float(mid)
            reason = "prefer_mid_near_best_score"
        else:
            chosen = raw_best_cx
            reason = "raw_argmax"

    return {
        "raw_best_cx": float(raw_best_cx),
        "raw_best_score": raw_score,
        "mirror_cx": mirror_cx,
        "mirror_score": mirror_score,
        "mid": float(mid),
        "mid_score": mid_score,
        "clean_mid": float(target_mid),
        "chosen_cx": float(chosen),
        "reason": reason,
        "full_bleed": bool(full_bleed),
        "fully_captured": bool(fully_captured),
        "delta_raw": float(raw_best_cx - mid),
        "delta_mirror": float(mirror_cx - mid),
    }


def draw_curve(
    xs: np.ndarray,
    ghost_scores: list[float],
    baseline: list[float],
    *,
    mid: float,
    raw_best_cx: float,
    mirror_cx: float,
    chosen_cx: float,
    cont: dict,
    stem: str,
    markers: list[tuple[float, str]] | None = None,
    width: int = 720,
    height: int = 300,
) -> Image.Image:
    img = Image.new("RGB", (width, height), (24, 24, 28))
    d = ImageDraw.Draw(img)
    font = _font(13)
    pad_l, pad_r, pad_t, pad_b = 48, 16, 40, 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    ymax = max(max(ghost_scores), max(baseline), 1e-6)

    def xpix(x: float) -> int:
        t = (x - float(xs[0])) / max(float(xs[-1] - xs[0]), 1e-9)
        return int(pad_l + max(0.0, min(1.0, t)) * plot_w)

    def ypix(v: float) -> int:
        return int(pad_t + (1.0 - v / ymax) * plot_h)

    d.rectangle(
        [pad_l, pad_t, pad_l + plot_w, pad_t + plot_h], outline=(70, 70, 80), width=1
    )
    for scores, color in (
        (baseline, (120, 120, 140)),
        (ghost_scores, (80, 220, 140)),
    ):
        pts = [(xpix(float(x)), ypix(v)) for x, v in zip(xs, scores)]
        if len(pts) >= 2:
            d.line(pts, fill=color, width=2)

    # Intentionally omit mid-reflection "mirror" marker — it confuses the chosen peak.
    _ = mirror_cx  # kept in signature for call-site compatibility
    marks = [
        ("mid", mid, (255, 180, 60)),
        ("chosen", chosen_cx, (80, 220, 140)),
    ]
    if abs(raw_best_cx - chosen_cx) > 1.0:
        marks.insert(1, ("raw", raw_best_cx, (255, 100, 100)))
    for x, label in markers or []:
        marks.append((label, x, (200, 200, 255)))
    for label, x, col in marks:
        if x < float(xs[0]) - 1 or x > float(xs[-1]) + 1:
            continue
        xp = xpix(x)
        d.line([(xp, pad_t), (xp, pad_t + plot_h)], fill=col, width=2)
        d.text((xp + 2, pad_t + 2), label, fill=col, font=font)

    d.text(
        (8, 6),
        f"{stem}  green=ghost-aware  gray=no-ghost  "
        f"L_cont={cont['continued_left']} R_cont={cont['continued_right']}  "
        f"full_bleed={cont.get('full_bleed', False)}  "
        f"fully_cap={cont.get('fully_captured', False)}",
        fill=(230, 230, 230),
        font=font,
    )
    d.text(
        (8, height - 28),
        f"score = (Σ match IoU + w·Σ ghost IoU) / (n_match+n_ghost+n_miss)   "
        f"w=ghost_credit   range 0..{ymax:.2f}",
        fill=(160, 160, 170),
        font=font,
    )
    return img


def draw_overlay(
    facade: Image.Image,
    boxes: list[list[float]],
    cx: float,
    *,
    continued_left: bool,
    continued_right: bool,
    ghost_credit: float,
    columns_xy: list,
    title: str,
    content_mask: np.ndarray | None = None,
    allow_self_iou: bool = True,
) -> Image.Image:
    """Draw obs / match / ghost overlays.

    Ghosts use a *light yellow* outline (not pink) so they read as partial-credit
    partners and do not clash with green matches / axis yellow. The canvas is
    padded so off-frame (cropped) ghost boxes remain fully visible.
    """
    W, H = facade.size
    ghosts = build_symmetry_ghosts(
        boxes,
        cx,
        W=float(W),
        H=float(H),
        continued_left=continued_left,
        continued_right=continued_right,
        content_mask=content_mask,
        columns_xy=columns_xy,
    )
    ghost_by_src = {g["source_i"]: g for g in ghosts}

    # Expand canvas so OOB / continued-wall ghosts are not clipped.
    margin = 24.0
    xs0, ys0, xs1, ys1 = 0.0, 0.0, float(W), float(H)
    for g in ghosts:
        gb = g["box"]
        xs0 = min(xs0, gb[0] - margin)
        ys0 = min(ys0, gb[1] - margin)
        xs1 = max(xs1, gb[2] + margin)
        ys1 = max(ys1, gb[3] + margin)
    for b in boxes:
        xs0 = min(xs0, b[0] - 4)
        ys0 = min(ys0, b[1] - 4)
        xs1 = max(xs1, b[2] + 4)
        ys1 = max(ys1, b[3] + 4)
    pad_l = int(max(0.0, -xs0))
    pad_t = int(max(0.0, -ys0))
    pad_r = int(max(0.0, xs1 - W))
    pad_b = int(max(0.0, ys1 - H))
    cw, ch = W + pad_l + pad_r, H + pad_t + pad_b

    def shift_box(b: list[float]) -> list[float]:
        return [b[0] + pad_l, b[1] + pad_t, b[2] + pad_l, b[3] + pad_t]

    base = Image.new("RGBA", (cw, ch), (28, 28, 32, 255))
    base.paste(facade.convert("RGBA"), (pad_l, pad_t))
    ov = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    font = _font(14)
    # Light yellow = ghost (weight ghost_credit); keep distinct from self-IoU gold fill.
    ghost_outline = (255, 236, 150, 230)
    ghost_x = (255, 228, 130, 140)
    ghost_txt = (255, 240, 180, 255)

    if continued_left:
        d.rectangle(
            [pad_l, pad_t, pad_l + 10, pad_t + H],
            fill=(255, 236, 150, 100),
        )
    if continued_right:
        d.rectangle(
            [pad_l + W - 10, pad_t, pad_l + W, pad_t + H],
            fill=(255, 236, 150, 100),
        )

    for a, b in columns_xy or []:
        x = a + pad_l
        d.line([(x, pad_t), (x, pad_t + H)], fill=(255, 255, 255, 35), width=1)
    cx_d = cx + pad_l
    d.line([(cx_d, pad_t), (cx_d, pad_t + H)], fill=(255, 200, 40, 255), width=3)

    # Always paint every ghost placement first (padded canvas keeps OOB ghosts visible).
    # Do this before obs/match so light-yellow is not overwritten by green "match"
    # reflections that only hit a ghost partner.
    labeled_ghost_src: set[int] = set()
    for g in ghosts:
        pb = shift_box(g["box"])
        d.rectangle(pb, outline=ghost_outline, width=3)
        d.line([pb[0], pb[1], pb[2], pb[3]], fill=ghost_x, width=1)
        src_i = int(g["source_i"])
        if src_i not in labeled_ghost_src:
            labeled_ghost_src.add(src_i)
            d.text(
                (pb[0], max(0, pb[1] - 12)),
                f"ghost×{ghost_credit:g} IoU={float(g.get('align_iou', 0.0)):.2f}",
                fill=ghost_txt,
                font=font,
            )

    for i, b in enumerate(boxes):
        bb = shift_box(b)
        d.rectangle(bb, outline=(60, 180, 255, 255), width=2)
        s, kind = partner_score_for_box(
            i,
            boxes,
            cx,
            W=float(W),
            H=float(H),
            continued_left=continued_left,
            continued_right=continued_right,
            ghost_credit=ghost_credit,
            columns_xy=columns_xy,
            require_pattern_ghost=False,
            content_mask=content_mask,
            ghosts=ghosts,
            allow_self_iou=allow_self_iou,
        )
        rb = _reflect(b, cx)
        frac = _box_in_image_frac(rb, float(W), float(H))
        cfrac = (
            _box_content_frac(content_mask, rb) if content_mask is not None and frac > 0 else 1.0
        )
        out_or_gray = (frac < 0.5) or (cfrac < 0.9)
        if kind == "ghost" or (kind == "match" and out_or_gray and i in ghost_by_src):
            # Ghost already drawn above; skip green reflection paint.
            continue
        if kind == "match" and s > 0.05 and not out_or_gray:
            rb_s = shift_box(rb)
            d.rectangle(rb_s, outline=(80, 255, 120, 200), width=2)
            if allow_self_iou and box_iou(rb, b) > 0.05:
                x0, y0 = max(rb_s[0], bb[0]), max(rb_s[1], bb[1])
                x1, y1 = min(rb_s[2], bb[2]), min(rb_s[3], bb[3])
                if x1 > x0 and y1 > y0:
                    d.rectangle(
                        [x0, y0, x1, y1],
                        fill=(255, 180, 40, 90),
                        outline=(255, 160, 20, 180),
                        width=1,
                    )
        elif kind == "miss":
            d.rectangle(shift_box(rb), outline=(255, 80, 80, 100), width=1)

    d.rectangle([4, 4, min(cw - 4, 820), 52], fill=(0, 0, 0, 210))
    d.text((8, 6), title, fill=(255, 255, 255, 255), font=font)
    d.text(
        (8, 28),
        (
            f"blue=obs  green=match(+selfIoU)  light-yellow=ghost(×{ghost_credit:g})  "
            f"L_cont={continued_left} R_cont={continued_right}"
        ),
        fill=(200, 200, 200, 255),
        font=font,
    )
    return Image.alpha_composite(base, ov).convert("RGB")


def run_stem(args: argparse.Namespace, stem: str) -> dict:
    case = load_case(args.sam3_root, stem)
    facade: Image.Image = case["facade"]
    boxes: list[list[float]] = case["boxes"]
    columns_xy = list(case["columns_xy"] or [])
    W, H = facade.size

    focus = select_focus_facade(boxes, columns_xy)
    if focus["changed"]:
        boxes = focus["boxes"]
        columns_xy = focus["columns_xy"]
    # After focus: pure-gray pad uses façade chroma from kept windows.
    content_mask = make_content_mask(facade, boxes=boxes)
    mid = facade_center_x(boxes)

    cont = detect_wall_continuation(
        facade,
        boxes,
        columns_xy=columns_xy,
        edge_frac=args.edge_frac,
        color_dist_max=args.color_dist_max,
        strip_std_min=args.strip_std_min,
        content_mask=content_mask,
    )
    cont["focus_facade"] = {
        "changed": focus["changed"],
        "n_groups": focus["n_groups"],
        "keep_cols": focus.get("keep_cols"),
        "group_sizes": focus.get("group_sizes"),
    }
    edge_bay = detect_edge_bay_contamination(
        boxes,
        columns_xy,
        image_width=float(W),
        edge_frac=args.edge_frac,
    )
    # Wide flush edge bay = adjacent façade, not our wall continuing.
    if edge_bay["left"]:
        cont["continued_left"] = False
        cont["left"]["adjacent_edge_bay"] = True
    if edge_bay["right"]:
        cont["continued_right"] = False
        cont["right"]["adjacent_edge_bay"] = True
    cont["edge_bay"] = edge_bay

    # Lattice snap is optional; irregular pitch does not kill continuation —
    # ghosts still follow pure horizontal symmetry.
    pitch = bay_pitch_model(columns_xy)
    cont["bay_pitch"] = pitch
    cont["pattern_blocks_ghost"] = bool(pitch is None or not pitch["regular"])

    xs, rows, baseline = sweep(
        boxes,
        W=float(W),
        H=float(H),
        continued_left=cont["continued_left"],
        continued_right=cont["continued_right"],
        ghost_mode=args.ghost_mode,
        ghost_credit=args.ghost_credit,
        min_inframe=args.min_inframe,
        n_samples=args.n_samples,
        columns_xy=columns_xy,
        require_pattern_ghost=False,
        content_mask=content_mask,
        allow_self_iou=True,
    )
    ghost_scores = [r["score"] for r in rows]
    best_i = int(np.argmax(ghost_scores))
    raw_best = rows[best_i]
    bb = bounding_box(boxes)
    fw = float(bb[2] - bb[0]) if bb is not None else float(W)
    dual = disambiguate_mirror_peak(
        raw_best["cx"],
        mid,
        continued_left=cont["continued_left"],
        continued_right=cont["continued_right"],
        full_bleed=cont.get("full_bleed", False),
        xs=xs,
        scores=ghost_scores,
        facade_width=fw,
        clean_mid=edge_bay["clean_mid"],
        fully_captured=cont.get("fully_captured", False),
        columns_xy=columns_xy,
    )
    chosen_cx = dual["chosen_cx"]
    # For overlays: only show continued-edge if it actually influenced the choice.
    used_cont = "continued" in dual["reason"] and "ignore" not in dual["reason"]
    draw_cont_L = bool(cont["continued_left"] and used_cont)
    draw_cont_R = bool(cont["continued_right"] and used_cont)

    chosen_row = score_axis(
        boxes,
        chosen_cx,
        W=float(W),
        H=float(H),
        continued_left=cont["continued_left"],
        continued_right=cont["continued_right"],
        ghost_mode=args.ghost_mode,
        ghost_credit=args.ghost_credit,
        min_inframe=args.min_inframe,
        columns_xy=columns_xy,
        require_pattern_ghost=False,
        content_mask=content_mask,
        allow_self_iou=True,
    )

    markers: list[tuple[float, str]] = []
    cols = columns_xy
    if stem == "cmp_b0150" and len(cols) > 4:
        markers.append((0.5 * (cols[4][0] + cols[4][1]), "B4"))

    out_dir = args.out_dir / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot uses detection flags; overlays use effective (axis-influencing) flags.
    cont_plot = dict(cont)
    cont_draw = dict(cont)
    cont_draw["continued_left"] = draw_cont_L
    cont_draw["continued_right"] = draw_cont_R

    curve = draw_curve(
        xs,
        ghost_scores,
        baseline,
        mid=mid,
        raw_best_cx=dual["raw_best_cx"],
        mirror_cx=dual["mirror_cx"],
        chosen_cx=chosen_cx,
        cont=cont_plot,
        stem=stem,
        markers=markers,
    )
    curve.save(out_dir / "score_curve.png")

    ov_chosen = draw_overlay(
        facade,
        boxes,
        chosen_cx,
        continued_left=cont["continued_left"],
        continued_right=cont["continued_right"],
        ghost_credit=args.ghost_credit,
        columns_xy=columns_xy,
        title=(
            f"{stem} CHOSEN cx={chosen_cx:.1f} ({dual['reason']})  "
            f"score={chosen_row['score']:.3f}"
        ),
        content_mask=content_mask,
        allow_self_iou=True,
    )
    ov_raw = draw_overlay(
        facade,
        boxes,
        dual["raw_best_cx"],
        continued_left=cont["continued_left"],
        continued_right=cont["continued_right"],
        ghost_credit=args.ghost_credit,
        columns_xy=columns_xy,
        title=f"{stem} RAW best cx={dual['raw_best_cx']:.1f}",
        content_mask=content_mask,
        allow_self_iou=True,
    )
    gap = 8
    canvas = Image.new(
        "RGB",
        (ov_raw.width + ov_chosen.width + gap, ov_raw.height),
        (20, 20, 20),
    )
    canvas.paste(ov_raw, (0, 0))
    canvas.paste(ov_chosen, (ov_raw.width + gap, 0))
    canvas.save(out_dir / "raw_vs_chosen.png")

    stack = Image.new(
        "RGB",
        (max(canvas.width, curve.width), canvas.height + curve.height + gap),
        (20, 20, 20),
    )
    stack.paste(canvas, (0, 0))
    stack.paste(curve, (0, canvas.height + gap))
    stack.save(out_dir / "overview.png")

    summary = {
        "stem": stem,
        "ghost_mode": args.ghost_mode,
        "mid_cx": mid,
        "dual": dual,
        "chosen_cx": chosen_cx,
        "chosen_score": chosen_row,
        "raw_best": raw_best,
        "continuation": cont,
        "edge_bay": edge_bay,
        "ghost_credit": args.ghost_credit,
        "min_inframe": args.min_inframe,
        "n_samples": args.n_samples,
        "markers": [{"label": lab, "cx": cx} for cx, lab in markers],
        "curve": [
            {
                "cx": float(x),
                "score": float(s),
                "baseline": float(b),
                "n_ghost": int(r["n_ghost"]),
                "n_match": int(r["n_match"]),
                "n_miss": int(r["n_miss"]),
            }
            for x, s, b, r in zip(xs, ghost_scores, baseline, rows)
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def discover_sam3_stems(sam3_root: Path) -> list[str]:
    stems = []
    for p in sorted(sam3_root.glob("e2e_sam3_cluster_cmp_*")):
        if p.is_dir() and (p / "facade_dsl.json").is_file():
            stems.append(p.name.replace("e2e_sam3_cluster_", "", 1))
    return stems


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stems = [s.strip() for s in args.stems.split(",") if s.strip()]
    if not stems:
        stems = discover_sam3_stems(args.sam3_root)
        print(f"auto-discovered {len(stems)} SAM3 e2e stems under {args.sam3_root}")
    all_rows = []
    for stem in stems:
        run = args.sam3_root / f"e2e_sam3_cluster_{stem}"
        if not (run / "facade_dsl.json").is_file():
            print(f"[skip] {stem}: no dsl")
            continue
        print(f"=== {stem} ===")
        s = run_stem(args, stem)
        d = s["dual"]
        all_rows.append(s)
        print(
            f"  cont L/R={s['continuation']['continued_left']}/"
            f"{s['continuation']['continued_right']}  "
            f"full_bleed={s['continuation'].get('full_bleed', False)}  "
            f"fully_cap={s['continuation'].get('fully_captured', False)}  "
            f"mid={s['mid_cx']:.1f}"
        )
        print(
            f"  raw_best={d['raw_best_cx']:.1f}  "
            f"mirror={d['mirror_cx']:.1f}  "
            f"chosen={d['chosen_cx']:.1f} ({d['reason']})"
        )
        print(f"  -> {args.out_dir / stem}")

    (args.out_dir / "summary_all.json").write_text(
        json.dumps(
            [
                {
                    "stem": r["stem"],
                    "mid_cx": r["mid_cx"],
                    "raw_best_cx": r["dual"]["raw_best_cx"],
                    "mirror_cx": r["dual"]["mirror_cx"],
                    "chosen_cx": r["dual"]["chosen_cx"],
                    "reason": r["dual"]["reason"],
                    "continued_left": r["continuation"]["continued_left"],
                    "continued_right": r["continuation"]["continued_right"],
                    "full_bleed": r["continuation"].get("full_bleed", False),
                    "fully_captured": r["continuation"].get("fully_captured", False),
                    "focus_changed": (r["continuation"].get("focus_facade") or {}).get(
                        "changed", False
                    ),
                }
                for r in all_rows
            ],
            indent=2,
        )
    )

    # contact sheet of per-stem overview.png
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18
        )
    except Exception:
        font = ImageFont.load_default()
    imgs = []
    tw = 420
    for r in all_rows:
        path = args.out_dir / r["stem"] / "overview.png"
        if not path.is_file():
            continue
        im = Image.open(path).convert("RGB")
        im = im.resize((tw, int(im.height * tw / im.width)))
        bar = Image.new("RGB", (im.width, 28), (30, 30, 30))
        ImageDraw.Draw(bar).text((6, 4), r["stem"], fill=(240, 240, 240), font=font)
        c = Image.new("RGB", (im.width, im.height + 28), (20, 20, 20))
        c.paste(bar, (0, 0))
        c.paste(im, (0, 28))
        imgs.append(c)
    if imgs:
        cols = 3
        rn = (len(imgs) + cols - 1) // cols
        cw = max(i.width for i in imgs)
        ch = max(i.height for i in imgs)
        out = Image.new(
            "RGB",
            (cols * cw + (cols - 1) * 8, rn * ch + (rn - 1) * 8),
            (15, 15, 15),
        )
        for i, im in enumerate(imgs):
            rr, cc = divmod(i, cols)
            out.paste(im, (cc * (cw + 8), rr * (ch + 8)))
        out.save(args.out_dir / "contact_all.png")
        print("wrote", args.out_dir / "contact_all.png", "n=", len(imgs))


if __name__ == "__main__":
    main()
