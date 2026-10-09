"""Depth Anything 3 as a side channel for enclosed-balcony detection.

Does not replace SAM3. Runs metric monocular depth on facade photos, then
compares median depth in enclosed / wall / open-rail ROIs. Enclosed should
be closer (smaller depth) than the wall if DA3 can help group the yellow
projecting blocks.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/try_da3_enclosure.py ^
      --device cuda --in-dir "data\\test for enclosure" --out-dir runs\\da3_enclosure

Writes per image ``<stem>_depth.jpg`` (color), ``<stem>_overlay.jpg`` (photo +
ROIs + depth blend), ``<stem>_depth.npy``, and ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.recrop_sam3 import list_input_images  # noqa: E402

DEFAULT_IN_DIR = ROOT / "data" / "test for enclosure"
DEFAULT_OUT_DIR = ROOT / "runs" / "da3_enclosure"
DEFAULT_MODEL = "depth-anything/DA3METRIC-LARGE"

# Normalized xyxy on the original photo. kind: enclosed | wall | open
ENCLOSURE_ROIS: dict[str, tuple[dict[str, Any], ...]] = {
    "cmp_b0007": (
        {"name": "left_oriel", "kind": "enclosed", "box": (0.00, 0.28, 0.28, 0.55)},
        {"name": "right_oriel", "kind": "enclosed", "box": (0.72, 0.28, 1.00, 0.55)},
        {"name": "center_gallery", "kind": "enclosed", "box": (0.30, 0.52, 0.70, 0.74)},
        {"name": "wall", "kind": "wall", "box": (0.38, 0.32, 0.48, 0.42)},
        {"name": "open_rail", "kind": "open", "box": (0.32, 0.16, 0.68, 0.25)},
    ),
    "cmp_b0008": (
        {"name": "left_colonnade", "kind": "enclosed", "box": (0.00, 0.38, 0.16, 0.70)},
        {"name": "lower_bay", "kind": "enclosed", "box": (0.32, 0.68, 0.68, 0.82)},
        {"name": "wall", "kind": "wall", "box": (0.38, 0.40, 0.60, 0.52)},
        {"name": "open_rail", "kind": "open", "box": (0.22, 0.10, 0.78, 0.18)},
    ),
    "cmp_b0010": (
        {"name": "gallery_stack", "kind": "enclosed", "box": (0.27, 0.33, 0.73, 0.76)},
        {"name": "wall", "kind": "wall", "box": (0.20, 0.46, 0.27, 0.58)},
        {"name": "open_rail", "kind": "open", "box": (0.00, 0.47, 0.22, 0.55)},
    ),
    "cmp_b0223": (
        {"name": "left_oriel", "kind": "enclosed", "box": (0.04, 0.10, 0.22, 0.32)},
        {"name": "right_oriel", "kind": "enclosed", "box": (0.70, 0.10, 0.90, 0.32)},
        {"name": "wall", "kind": "wall", "box": (0.40, 0.36, 0.52, 0.46)},
        {"name": "open_rail", "kind": "open", "box": (0.38, 0.48, 0.54, 0.56)},
    ),
    "cmp_b0307": (
        {"name": "left_bay", "kind": "enclosed", "box": (0.06, 0.52, 0.34, 0.80)},
        {"name": "wall", "kind": "wall", "box": (0.38, 0.48, 0.52, 0.68)},
        {"name": "open_rail", "kind": "open", "box": (0.58, 0.52, 0.95, 0.68)},
    ),
}

ROI_COLORS = {
    "enclosed": (255, 220, 0),
    "wall": (255, 255, 255),
    "open": (0, 220, 255),
}

# Enclosed is closer if it is this fraction shallower than the wall.
MIN_REL_CLOSER = 0.02


def box_from_norm(
    box: tuple[float, float, float, float] | list[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (float(v) for v in box)
    px0 = int(round(max(0.0, min(1.0, x0)) * width))
    py0 = int(round(max(0.0, min(1.0, y0)) * height))
    px1 = int(round(max(0.0, min(1.0, x1)) * width))
    py1 = int(round(max(0.0, min(1.0, y1)) * height))
    px1 = max(px0 + 1, min(width, px1))
    py1 = max(py0 + 1, min(height, py1))
    px0 = min(px0, px1 - 1)
    py0 = min(py0, py1 - 1)
    return px0, py0, px1, py1


def resize_depth(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape[0] == height and arr.shape[1] == width:
        return arr
    im = Image.fromarray(arr, mode="F")
    return np.asarray(im.resize((width, height), Image.BILINEAR), dtype=np.float32)


def roi_median(depth: np.ndarray, box_px: tuple[int, int, int, int]) -> float | None:
    x0, y0, x1, y1 = box_px
    patch = np.asarray(depth)[y0:y1, x0:x1]
    vals = patch[np.isfinite(patch) & (patch > 0)]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def is_closer(
    candidate: float | None,
    wall: float | None,
    *,
    min_rel: float = MIN_REL_CLOSER,
) -> bool:
    """True when candidate depth (meters / relative) is smaller than wall."""
    if candidate is None or wall is None or wall <= 0:
        return False
    return (wall - candidate) / wall >= min_rel


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Near (small) → warm, far → blue. uint8 HxWx3."""
    d = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(d) & (d > 0)
    out = np.zeros(d.shape + (3,), dtype=np.uint8)
    if not np.any(finite):
        return out
    lo, hi = np.percentile(d[finite], (2.0, 98.0))
    if hi <= lo:
        hi = lo + 1e-6
    t = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    t = 1.0 - t
    r = np.clip(1.5 * t, 0.0, 1.0)
    g = np.clip(1.5 * t - 0.25, 0.0, 1.0)
    b = np.clip(1.0 - t, 0.0, 1.0)
    out[..., 0] = (r * 255).astype(np.uint8)
    out[..., 1] = (g * 255).astype(np.uint8)
    out[..., 2] = (b * 255).astype(np.uint8)
    out[~finite] = 0
    return out


def score_image(
    depth: np.ndarray,
    *,
    width: int,
    height: int,
    rois: tuple[dict[str, Any], ...],
    min_rel: float = MIN_REL_CLOSER,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    wall_med: float | None = None
    for roi in rois:
        box_px = box_from_norm(roi["box"], width, height)
        med = roi_median(depth, box_px)
        rec = {
            "name": roi["name"],
            "kind": roi["kind"],
            "box_px": list(box_px),
            "median": med,
        }
        rows.append(rec)
        if roi["kind"] == "wall":
            wall_med = med
    for rec in rows:
        if rec["kind"] == "wall":
            rec["closer_than_wall"] = None
        else:
            rec["closer_than_wall"] = is_closer(rec["median"], wall_med, min_rel=min_rel)
            if rec["median"] is not None and wall_med is not None and wall_med > 0:
                rec["rel_delta"] = (wall_med - rec["median"]) / wall_med
            else:
                rec["rel_delta"] = None
    enclosed_ok = [
        rec["name"]
        for rec in rows
        if rec["kind"] == "enclosed" and rec.get("closer_than_wall")
    ]
    return {
        "wall_median": wall_med,
        "rois": rows,
        "enclosed_closer": enclosed_ok,
        "helps_enclosed": bool(enclosed_ok),
    }


def draw_overlay(
    photo: Image.Image,
    depth_rgb: np.ndarray,
    rois: tuple[dict[str, Any], ...],
) -> Image.Image:
    photo_arr = np.asarray(photo.convert("RGB"), dtype=np.float32)
    blend = (0.55 * photo_arr + 0.45 * depth_rgb.astype(np.float32)).clip(0, 255)
    overlay = Image.fromarray(blend.astype(np.uint8))
    draw = ImageDraw.Draw(overlay)
    line_w = max(2, photo.width // 250)
    for roi in rois:
        box_px = box_from_norm(roi["box"], photo.width, photo.height)
        color = ROI_COLORS.get(roi["kind"], (255, 0, 255))
        draw.rectangle(box_px, outline=color, width=line_w)
        draw.text((box_px[0] + 4, box_px[1] + 4), f"{roi['kind']}:{roi['name']}", fill=color)
    return overlay


def _extract_depth(prediction: Any) -> np.ndarray:
    depth = getattr(prediction, "depth", None)
    if depth is None and isinstance(prediction, dict):
        depth = prediction.get("depth")
    if depth is None:
        raise RuntimeError("DA3 prediction has no depth")
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    return arr


def _stub_module(name: str) -> None:
    import types

    if name in sys.modules:
        return
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        qual = ".".join(parts[:i])
        if qual not in sys.modules:
            sys.modules[qual] = types.ModuleType(qual)
        if i > 1:
            parent = sys.modules[".".join(parts[: i - 1])]
            setattr(parent, parts[i - 1], sys.modules[qual])


def import_depth_anything3():
    """Load DA3 without moviepy / trimesh / pycolmap / evo (export-only deps)."""
    import types

    _stub_module("evo.core.trajectory")
    traj = sys.modules["evo.core.trajectory"]
    if not hasattr(traj, "PosePath3D"):
        traj.PosePath3D = type("PosePath3D", (), {"__init__": lambda self, *a, **k: None})
    export_name = "depth_anything_3.utils.export"
    if export_name not in sys.modules:
        export_mod = types.ModuleType(export_name)
        export_mod.export = lambda *_a, **_k: None
        sys.modules[export_name] = export_mod
    from depth_anything_3.api import DepthAnything3

    return DepthAnything3


def run_da3(
    paths: list[Path],
    *,
    out_dir: Path,
    model_id: str,
    device: str,
    min_rel: float,
) -> dict[str, Any]:
    import torch

    DepthAnything3 = import_depth_anything3()

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading {model_id} on {device}…", flush=True)
    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device=torch.device(device)).eval()

    summary: dict[str, Any] = {
        "model": model_id,
        "min_rel_closer": min_rel,
        "roi_legend": "yellow=enclosed, white=wall, cyan=open",
        "images": {},
    }
    for path in paths:
        depth_npy = out_dir / f"{path.stem}_depth.npy"
        image = Image.open(path).convert("RGB")
        if depth_npy.is_file():
            print(f"{path.name}  reuse {depth_npy.name}", flush=True)
            depth = resize_depth(np.load(depth_npy), image.width, image.height)
        else:
            print(f"{path.name}  infer", flush=True)
            prediction = model.inference([str(path)])
            depth = resize_depth(np.asarray(_extract_depth(prediction)), image.width, image.height)
            np.save(depth_npy, depth)
        depth_rgb = colorize_depth(depth)
        Image.fromarray(depth_rgb).save(out_dir / f"{path.stem}_depth.jpg", quality=92)
        rois = ENCLOSURE_ROIS.get(path.stem, ())
        scored = score_image(
            depth,
            width=image.width,
            height=image.height,
            rois=rois,
            min_rel=min_rel,
        )
        overlay = draw_overlay(image, depth_rgb, rois)
        overlay.save(out_dir / f"{path.stem}_overlay.jpg", quality=92)
        summary["images"][path.stem] = scored
        closer = ",".join(scored["enclosed_closer"]) or "-"
        wall = scored["wall_median"]
        wall_s = f"{wall:.3f}" if wall is not None else "nan"
        print(
            f"{path.name}  wall={wall_s}  enclosed_closer={closer}  helps={scored['helps_enclosed']}",
            flush=True,
        )
        for rec in scored["rois"]:
            med = rec["median"]
            med_s = f"{med:.3f}" if med is not None else "nan"
            extra = ""
            if rec.get("rel_delta") is not None:
                extra = f"  rel={rec['rel_delta']:+.3f}"
            print(f"  {rec['kind']:9} {rec['name']:16}  median={med_s}{extra}", flush=True)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"done -> {out_dir}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    ap.add_argument(
        "--image",
        type=Path,
        nargs="*",
        default=None,
        help="Optional explicit photo(s). Overrides --in-dir.",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-rel", type=float, default=MIN_REL_CLOSER)
    args = ap.parse_args(argv)
    paths = [p.expanduser().resolve() for p in (args.image or [])]
    if not paths:
        paths = list_input_images(args.in_dir)
    if not paths:
        raise SystemExit(f"no images in {args.in_dir}")
    run_da3(
        paths,
        out_dir=args.out_dir,
        model_id=args.model,
        device=args.device,
        min_rel=args.min_rel,
    )


if __name__ == "__main__":
    main()
