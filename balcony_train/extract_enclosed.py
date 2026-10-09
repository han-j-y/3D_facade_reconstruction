"""Extract enclosed volumes from DA3 depth: rectangles 2% in front of the wall.

1. Drop non-rectangular silhouettes (trees, cars, people).
2. Keep a rectangle if its median depth is at least 2% closer than the wall.

Uses existing ``*_depth.npy`` from ``try_da3_enclosure.py`` (does not reload DA3).

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/extract_enclosed.py ^
      --in-dir "data\\test for enclosure" --depth-dir runs\\da3_enclosure ^
      --out-dir runs\\da3_enclosed_extract
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.recrop_sam3 import list_input_images  # noqa: E402
from balcony_train.try_da3_enclosure import (  # noqa: E402
    DEFAULT_IN_DIR,
    DEFAULT_OUT_DIR,
    MIN_REL_CLOSER,
    colorize_depth,
    is_closer,
    resize_depth,
    roi_median,
)

DEFAULT_EXTRACT_DIR = ROOT / "runs" / "da3_enclosed_extract"

# Fill = silhouette area / bounding-box area. Rectangles sit near 1; trees/cars lower.
MIN_RECT_FILL = 0.72
MIN_AREA_FRAC = 0.004
MAX_AREA_FRAC = 0.48
MIN_SIDE_FRAC = 0.03
MAX_ASPECT = 6.0
# One storey ~10% of image height (same gate as the SAM3 one-floor test).
MIN_HEIGHT_FRAC = 0.10
MAX_WIDTH_FRAC = 0.75
SWEEP_RELS = (0.01, 0.02, 0.03, 0.04, 0.05)
# 1%..5% stacked on one photo (low threshold first so tighter boxes sit on top).
REL_STACK_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 60, 60),
    (255, 160, 0),
    (255, 220, 0),
    (80, 220, 80),
    (0, 200, 255),
)
CLOSE_KERNEL_FRAC = 0.025
VEG_FRAC = 0.25
BOTTOM_START_FRAC = 0.82
MAX_REL_CLOSER = 0.35


def estimate_wall_depth(
    depth: np.ndarray,
    *,
    top_frac: float = 0.08,
    bottom_frac: float = 0.18,
    side_frac: float = 0.06,
    trim: float = 0.15,
) -> float | None:
    """Median facade depth, ignoring sky, ground clutter, and the nearest blobs."""
    arr = np.asarray(depth, dtype=np.float32)
    h, w = arr.shape[:2]
    y0 = int(round(h * top_frac))
    y1 = int(round(h * (1.0 - bottom_frac)))
    x0 = int(round(w * side_frac))
    x1 = int(round(w * (1.0 - side_frac)))
    if y1 <= y0 + 2 or x1 <= x0 + 2:
        y0, y1, x0, x1 = 0, h, 0, w
    patch = arr[y0:y1, x0:x1]
    vals = patch[np.isfinite(patch) & (patch > 0)]
    if vals.size < 32:
        vals = arr[np.isfinite(arr) & (arr > 0)]
    if vals.size == 0:
        return None
    lo, hi = np.percentile(vals, (100.0 * trim, 100.0 * (1.0 - trim)))
    mid = vals[(vals >= lo) & (vals <= hi)]
    if mid.size == 0:
        mid = vals
    return float(np.median(mid))


def near_mask(
    depth: np.ndarray,
    wall: float,
    *,
    min_rel: float = MIN_REL_CLOSER,
) -> np.ndarray:
    """True where depth is at least ``min_rel`` closer than the wall."""
    arr = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(arr) & (arr > 0) & (wall > 0)
    return valid & (((wall - arr) / wall) >= min_rel)


def close_mask(mask: np.ndarray, *, kernel_frac: float = CLOSE_KERNEL_FRAC) -> np.ndarray:
    """Join nearby rectangular faces (stacked gallery floors) without filling trees."""
    h, w = mask.shape[:2]
    k = max(3, int(round(min(h, w) * kernel_frac)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(
        (mask.astype(np.uint8) * 255),
        cv2.MORPH_CLOSE,
        kernel,
    )
    return closed > 0


def silhouette_fill(area: int, box_w: int, box_h: int) -> float:
    denom = max(1, int(box_w) * int(box_h))
    return float(area) / float(denom)


def vegetation_fraction(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
) -> float:
    x0, y0, x1, y1 = box
    patch = rgb[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # OpenCV hue 0-179; foliage is yellow-green through green.
    veg = (h >= 25) & (h <= 95) & (s > 35) & (v > 35)
    return float(np.mean(veg))


def is_interference(
    box: tuple[int, int, int, int],
    *,
    height: int,
    rgb: np.ndarray | None = None,
    rel_delta: float | None = None,
    veg_frac: float = VEG_FRAC,
    bottom_start: float = BOTTOM_START_FRAC,
    max_rel: float = MAX_REL_CLOSER,
) -> str | None:
    """Tree / car / sky leftovers that can look rectangular after a crop."""
    _x0, y0, _x1, _y1 = box
    if y0 >= int(round(bottom_start * height)):
        return "bottom_clutter"
    if rel_delta is not None and rel_delta > max_rel:
        return "too_near"
    if rgb is not None and vegetation_fraction(rgb, box) >= veg_frac:
        return "vegetation"
    return None


def storey_width_reason(
    box: tuple[int, int, int, int] | list[int],
    *,
    iw: int,
    ih: int,
    min_height_frac: float = MIN_HEIGHT_FRAC,
    max_width_frac: float = MAX_WIDTH_FRAC,
) -> str | None:
    """Reject boxes shorter than one storey or wider than ``max_width_frac`` of the photo."""
    x0, y0, x1, y1 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    bw = max(0, x1 - x0)
    bh = max(0, y1 - y0)
    if bh < min_height_frac * ih:
        return "too_short"
    if bw > max_width_frac * iw:
        return "too_wide"
    return None


def is_rectangular_silhouette(
    area: int,
    box_w: int,
    box_h: int,
    *,
    min_fill: float = MIN_RECT_FILL,
) -> bool:
    """False for irregular blobs (trees, cars, people)."""
    if box_w < 1 or box_h < 1 or area < 1:
        return False
    return silhouette_fill(area, box_w, box_h) >= min_fill


def largest_ones_rectangle(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Largest axis-aligned rectangle of True pixels, or None if empty."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not np.any(m):
        return None
    h, w = m.shape
    heights = np.zeros(w, dtype=np.int32)
    best_area = 0
    best: tuple[int, int, int, int] | None = None
    for y in range(h):
        row = m[y]
        heights = np.where(row, heights + 1, 0)
        stack: list[int] = []
        for i in range(w + 1):
            cur = int(heights[i]) if i < w else 0
            while stack and cur < int(heights[stack[-1]]):
                hgt = int(heights[stack.pop()])
                left = stack[-1] + 1 if stack else 0
                width = i - left
                area = hgt * width
                if area > best_area:
                    best_area = area
                    y1 = y + 1
                    best = (left, y1 - hgt, left + width, y1)
            stack.append(i)
    return best


def foliage_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (
        (hsv[..., 0] >= 25)
        & (hsv[..., 0] <= 95)
        & (hsv[..., 1] > 35)
        & (hsv[..., 2] > 35)
    )


def extract_enclosures(
    depth: np.ndarray,
    *,
    rgb: np.ndarray | None = None,
    min_rel: float = MIN_REL_CLOSER,
    min_fill: float = MIN_RECT_FILL,
    min_area_frac: float = MIN_AREA_FRAC,
    max_area_frac: float = MAX_AREA_FRAC,
    min_side_frac: float = MIN_SIDE_FRAC,
    max_aspect: float = MAX_ASPECT,
    min_height_frac: float = MIN_HEIGHT_FRAC,
    max_width_frac: float = MAX_WIDTH_FRAC,
) -> dict[str, Any]:
    """Return kept enclosure boxes plus rejected interference blobs."""
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    h, w = arr.shape[:2]
    wall = estimate_wall_depth(arr)
    rejected: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    if wall is None:
        return {"wall_median": None, "kept": kept, "rejected": rejected}

    mask = close_mask(near_mask(arr, wall, min_rel=min_rel))
    if rgb is not None:
        if tuple(rgb.shape[:2]) != (h, w):
            raise ValueError("rgb shape must match depth")
        mask = mask & ~foliage_mask(np.asarray(rgb))
    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    img_area = float(h * w)
    min_side = max(4, int(round(min(h, w) * min_side_frac)))
    for lab in range(1, n_labels):
        x, y, bw, bh, area = (int(v) for v in stats[lab])
        box = (x, y, x + bw, y + bh)
        fill = silhouette_fill(area, bw, bh)
        med = roi_median(arr, box)
        rec: dict[str, Any] = {
            "box_xyxy": list(box),
            "area": area,
            "fill": round(fill, 4),
            "median": med,
            "rel_delta": None
            if med is None or wall <= 0
            else (wall - med) / wall,
        }
        if bw < min_side or bh < min_side:
            rec["reason"] = "too_small"
            rejected.append(rec)
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > max_aspect:
            rec["reason"] = "too_skinny"
            rejected.append(rec)
            continue
        frac = area / img_area
        if frac < min_area_frac or frac > max_area_frac:
            rec["reason"] = "area"
            rejected.append(rec)
            continue
        cand_box = box
        cand_fill = fill
        cand_area = area
        cand_med = med
        if not is_rectangular_silhouette(area, bw, bh, min_fill=min_fill):
            rec["reason"] = "not_rectangle"
            rejected.append(rec)
            comp = _labels[y : y + bh, x : x + bw] == lab
            inner = largest_ones_rectangle(comp)
            if inner is None:
                continue
            ix0, iy0, ix1, iy1 = inner
            cand_box = (x + ix0, y + iy0, x + ix1, y + iy1)
            cw = cand_box[2] - cand_box[0]
            ch = cand_box[3] - cand_box[1]
            cand_area = int(np.count_nonzero(comp[iy0:iy1, ix0:ix1]))
            cand_fill = silhouette_fill(cand_area, cw, ch)
            cand_med = roi_median(arr, cand_box)
            rec = {
                "box_xyxy": list(cand_box),
                "area": cand_area,
                "fill": round(cand_fill, 4),
                "median": cand_med,
                "rel_delta": None
                if cand_med is None or wall <= 0
                else (wall - cand_med) / wall,
            }
            if (
                cw < min_side
                or ch < min_side
                or cand_area / img_area < min_area_frac
                or max(cw, ch) / max(1, min(cw, ch)) > max_aspect
            ):
                rec["reason"] = "not_rectangle"
                rejected.append(rec)
                continue
        why_size = storey_width_reason(
            cand_box,
            iw=w,
            ih=h,
            min_height_frac=min_height_frac,
            max_width_frac=max_width_frac,
        )
        if why_size:
            rec["reason"] = why_size
            rec["box_xyxy"] = list(cand_box)
            rec["median"] = cand_med
            rec["fill"] = round(cand_fill, 4)
            rec["area"] = cand_area
            rec["rel_delta"] = (
                None
                if cand_med is None or wall <= 0
                else (wall - cand_med) / wall
            )
            rejected.append(rec)
            continue
        rec["rel_delta"] = (
            None
            if cand_med is None or wall <= 0
            else (wall - cand_med) / wall
        )
        rec["box_xyxy"] = list(cand_box)
        rec["median"] = cand_med
        rec["fill"] = round(cand_fill, 4)
        rec["area"] = cand_area
        if not is_closer(cand_med, wall, min_rel=min_rel):
            rec["reason"] = "not_front"
            rejected.append(rec)
            continue
        why = is_interference(
            cand_box, height=h, rgb=rgb, rel_delta=rec["rel_delta"]
        )
        if why:
            rec["reason"] = why
            rejected.append(rec)
            continue
        rec["reason"] = "enclosure"
        kept.append(rec)
    return {"wall_median": wall, "kept": kept, "rejected": rejected}


def draw_extract_overlay(
    photo: Image.Image,
    depth: np.ndarray,
    result: dict[str, Any],
) -> Image.Image:
    depth_rgb = colorize_depth(depth)
    photo_arr = np.asarray(photo.convert("RGB"), dtype=np.float32)
    blend = (0.55 * photo_arr + 0.45 * depth_rgb.astype(np.float32)).clip(0, 255)
    overlay = Image.fromarray(blend.astype(np.uint8))
    draw = ImageDraw.Draw(overlay)
    line_w = max(2, photo.width // 250)
    for rec in result.get("rejected") or []:
        if rec.get("reason") not in {
            "not_rectangle",
            "vegetation",
            "bottom_clutter",
            "too_near",
        }:
            continue
        box = rec["box_xyxy"]
        draw.rectangle(box, outline=(255, 60, 60), width=max(1, line_w - 1))
    for rec in result.get("kept") or []:
        box = rec["box_xyxy"]
        rel = rec.get("rel_delta")
        label = f"enc {rel:+.2f}" if rel is not None else "enc"
        draw.rectangle(box, outline=(255, 220, 0), width=line_w)
        draw.text((box[0] + 4, box[1] + 4), label, fill=(255, 220, 0))
    return overlay


def rel_subdir(min_rel: float) -> str:
    return f"rel_{int(round(min_rel * 100)):02d}pct"


def rel_stack_color(min_rel: float) -> tuple[int, int, int]:
    pct = int(round(min_rel * 100))
    idx = max(1, min(5, pct)) - 1
    return REL_STACK_COLORS[idx]


def draw_stacked_rel_overlay(
    photo: Image.Image,
    kept_by_rel: list[tuple[float, list[dict[str, Any]]]],
) -> Image.Image:
    """Draw kept boxes from several wall-delta thresholds on one photo."""
    overlay = photo.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    line_w = max(3, photo.width // 220)
    x = 8
    y = 8
    for min_rel, recs in kept_by_rel:
        color = rel_stack_color(min_rel)
        tag = f"{int(round(min_rel * 100))}%"
        draw.text((x, y), tag, fill=color)
        x += 48
        for rec in recs:
            box = rec["box_xyxy"]
            draw.rectangle(box, outline=color, width=line_w)
            draw.text((int(box[0]) + 4, int(box[1]) + 4), tag, fill=color)
    return overlay


def load_kept_by_rel(out_dir: Path, stem: str, rels: tuple[float, ...] | list[float]) -> list[tuple[float, list[dict[str, Any]]]]:
    out: list[tuple[float, list[dict[str, Any]]]] = []
    for min_rel in rels:
        summary_path = out_dir / rel_subdir(min_rel) / "extract_summary.json"
        if not summary_path.is_file():
            out.append((min_rel, []))
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        kept = list((data.get("images") or {}).get(stem, {}).get("kept") or [])
        out.append((min_rel, kept))
    return out


def write_stacked_rel_overlays(
    paths: list[Path],
    *,
    out_dir: Path,
    rels: tuple[float, ...] | list[float] = SWEEP_RELS,
) -> None:
    for path in paths:
        photo = Image.open(path).convert("RGB")
        kept = load_kept_by_rel(out_dir, path.stem, rels)
        stacked = draw_stacked_rel_overlay(photo, kept)
        dest = out_dir / f"{path.stem}_stacked.jpg"
        stacked.save(dest, quality=92)
        n = sum(len(recs) for _rel, recs in kept)
        print(f"{path.name}  stacked={n} -> {dest}", flush=True)


def compose_sweep_strip(overlays: list[Image.Image], labels: list[str]) -> Image.Image:
    """Side-by-side overlays, one column per wall-delta threshold."""
    if not overlays:
        raise ValueError("no overlays")
    gap = 8
    caption = 28
    w = max(im.width for im in overlays)
    h = max(im.height for im in overlays)
    strip = Image.new(
        "RGB",
        (len(overlays) * w + gap * (len(overlays) - 1), h + caption),
        (20, 20, 20),
    )
    draw = ImageDraw.Draw(strip)
    for i, (im, lab) in enumerate(zip(overlays, labels)):
        x = i * (w + gap)
        strip.paste(im.resize((w, h), Image.Resampling.BICUBIC), (x, caption))
        draw.text((x + 8, 6), lab, fill=(255, 220, 0))
    return strip


def run_extract(
    paths: list[Path],
    *,
    depth_dir: Path,
    out_dir: Path,
    min_rel: float,
    min_fill: float,
    min_height_frac: float = MIN_HEIGHT_FRAC,
    max_width_frac: float = MAX_WIDTH_FRAC,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "min_rel_closer": min_rel,
        "min_rect_fill": min_fill,
        "min_height_frac": min_height_frac,
        "max_width_frac": max_width_frac,
        "legend": "yellow=enclosure, red=non-rectangle (tree/car/person)",
        "images": {},
    }
    for path in paths:
        depth_path = depth_dir / f"{path.stem}_depth.npy"
        if not depth_path.is_file():
            raise SystemExit(
                f"missing {depth_path}; run balcony_train/try_da3_enclosure.py first"
            )
        image = Image.open(path).convert("RGB")
        depth = resize_depth(np.load(depth_path), image.width, image.height)
        rgb = np.asarray(image, dtype=np.uint8)
        result = extract_enclosures(
            depth,
            rgb=rgb,
            min_rel=min_rel,
            min_fill=min_fill,
            min_height_frac=min_height_frac,
            max_width_frac=max_width_frac,
        )
        overlay = draw_extract_overlay(image, depth, result)
        overlay.save(out_dir / f"{path.stem}_enclosed.jpg", quality=92)
        n_keep = len(result["kept"])
        n_irreg = sum(1 for r in result["rejected"] if r["reason"] == "not_rectangle")
        n_short = sum(1 for r in result["rejected"] if r["reason"] == "too_short")
        n_wide = sum(1 for r in result["rejected"] if r["reason"] == "too_wide")
        wall = result["wall_median"]
        wall_s = f"{wall:.3f}" if wall is not None else "nan"
        print(
            f"{path.name}  rel={min_rel:.2f}  wall={wall_s}  "
            f"enclosure={n_keep}  short={n_short}  wide={n_wide}  "
            f"dropped_irregular={n_irreg}",
            flush=True,
        )
        summary["images"][path.stem] = result
    (out_dir / "extract_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"done -> {out_dir}", flush=True)
    return summary


def run_rel_sweep(
    paths: list[Path],
    *,
    depth_dir: Path,
    out_dir: Path,
    rels: tuple[float, ...] | list[float],
    min_fill: float,
    min_height_frac: float,
    max_width_frac: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_stem: dict[str, list[Image.Image]] = {p.stem: [] for p in paths}
    labels: list[str] = []
    for min_rel in rels:
        labels.append(f"{int(round(min_rel * 100))}% vs wall")
        sub = out_dir / rel_subdir(min_rel)
        run_extract(
            paths,
            depth_dir=depth_dir,
            out_dir=sub,
            min_rel=min_rel,
            min_fill=min_fill,
            min_height_frac=min_height_frac,
            max_width_frac=max_width_frac,
        )
        for path in paths:
            im = Image.open(sub / f"{path.stem}_enclosed.jpg").convert("RGB")
            per_stem[path.stem].append(im)
    for stem, overlays in per_stem.items():
        compose_sweep_strip(overlays, labels).save(
            out_dir / f"{stem}_sweep.jpg", quality=90
        )
        print(f"strip -> {out_dir / f'{stem}_sweep.jpg'}", flush=True)
    write_stacked_rel_overlays(paths, out_dir=out_dir, rels=rels)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    ap.add_argument("--depth-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    ap.add_argument("--min-rel", type=float, default=MIN_REL_CLOSER)
    ap.add_argument("--min-fill", type=float, default=MIN_RECT_FILL)
    ap.add_argument(
        "--min-height-frac",
        type=float,
        default=MIN_HEIGHT_FRAC,
        help="keep boxes at least this fraction of image height (one storey)",
    )
    ap.add_argument(
        "--max-width-frac",
        type=float,
        default=MAX_WIDTH_FRAC,
        help="drop boxes wider than this fraction of image width",
    )
    ap.add_argument(
        "--sweep-rel",
        action="store_true",
        help="run 1%%, 2%%, 3%%, 4%%, 5%% vs wall and write a strip per image",
    )
    ap.add_argument(
        "--stack-only",
        action="store_true",
        help="overlay existing 1-5%% kept boxes on one photo per image (no re-extract)",
    )
    args = ap.parse_args(argv)
    paths = list_input_images(args.in_dir)
    if not paths:
        raise SystemExit(f"no images in {args.in_dir}")
    if args.stack_only:
        write_stacked_rel_overlays(paths, out_dir=args.out_dir, rels=SWEEP_RELS)
        return
    if args.sweep_rel:
        run_rel_sweep(
            paths,
            depth_dir=args.depth_dir,
            out_dir=args.out_dir,
            rels=SWEEP_RELS,
            min_fill=args.min_fill,
            min_height_frac=args.min_height_frac,
            max_width_frac=args.max_width_frac,
        )
        return
    run_extract(
        paths,
        depth_dir=args.depth_dir,
        out_dir=args.out_dir,
        min_rel=args.min_rel,
        min_fill=args.min_fill,
        min_height_frac=args.min_height_frac,
        max_width_frac=args.max_width_frac,
    )


if __name__ == "__main__":
    main()
