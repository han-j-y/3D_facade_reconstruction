"""Score enclosed-volume detection against Label UI floor×bay cells.

Only photos saved from the grid UI are scored. An empty cell list means that
photo has no enclosure. A photo that was never saved is skipped.

Detection uses the same SAM3+DA3 volumes as the balcony pipeline, then snaps
each box onto the window grid. Rates are split by the label. Enclosure photos: detect means at least one
labeled cell was found, and bay/floor means the cell set matches. Photos
with no enclosure: both are correct only when the detector also finds none.

``--compare-dir`` paints the same window grid used in the Label UI:
blue is a labeled cell the detector missed, red is a predicted cell that
was not labeled, purple is both.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/evaluate_enclosed_grid.py --device cuda
    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/evaluate_enclosed_grid.py --device cuda --stems cmp_x0005 cmp_x0134
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_PIPE = ROOT / "balcony_pipeline"
if str(_PIPE) not in sys.path:
    sys.path.insert(0, str(_PIPE))

from enclosed import slice_volume  # noqa: E402

from balcony_train.enclosed_grid_store import (  # noqa: E402
    DEFAULT_DSL_DIR,
    DEFAULT_EVAL_LABELS_JSONL,
    DEFAULT_EVAL_PHOTO_DIR,
    Cell,
    find_photo,
    load_grid,
    load_labels,
)

DEFAULT_DEPTH_DIR = ROOT / "runs" / "da3_enclosure"
DEFAULT_SAM3_CACHE = ROOT / "runs" / "sam3_da3_volumes" / "sam3_seeds.json"
DEFAULT_COMPARE_DIR = ROOT / "runs" / "enclosed_grid" / "compare"


def cells_from_boxes(
    boxes: list[Any],
    floor_y: dict[int, tuple[int, int]],
    bay_x: dict[int, tuple[int, int]],
) -> set[Cell]:
    """Window-grid cells covered by detector boxes."""
    cells: set[Cell] = set()
    for box in boxes:
        xy = box.get("box_xyxy") if isinstance(box, dict) else box
        if not xy:
            continue
        for sl in slice_volume(list(xy), floor_y, bay_x):
            bays = sl.get("bays_center") or [sl.get("bay", 0)]
            for bay in bays:
                cells.add((int(sl["floor"]), int(bay)))
    return cells


def score_photo(gt_cells: list[Cell] | set[Cell], pred_cells: set[Cell]) -> dict[str, Any]:
    """Presence (detect) and exact floor×bay match for one labeled photo."""
    gt = set(gt_cells)
    pred = set(pred_cells)
    overlap = gt & pred
    if gt:
        detected = bool(overlap)
    else:
        detected = not pred
    return {
        "gt": sorted(gt),
        "pred": sorted(pred),
        "has_enclosure": bool(gt),
        "detected": detected,
        "exact": gt == pred,
        "tp": len(overlap),
        "fp": len(pred - gt),
        "fn": len(gt - overlap),
    }


def _group_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "n": len(rows),
        "detect": sum(1 for r in rows if r["detected"]),
        "bay_floor": sum(1 for r in rows if r["exact"]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    with_enc = [r for r in rows if r["has_enclosure"]]
    none = [r for r in rows if not r["has_enclosure"]]
    return {
        "n": len(rows),
        "enclosure": _group_counts(with_enc),
        "none": _group_counts(none),
        "tp": sum(int(r["tp"]) for r in rows),
        "fp": sum(int(r["fp"]) for r in rows),
        "fn": sum(int(r["fn"]) for r in rows),
    }


def _rate_line(name: str, group: dict[str, int]) -> str:
    n = int(group["n"])
    detect = int(group["detect"])
    bay = int(group["bay_floor"])
    denom = max(n, 1)
    return (
        f"{name} n={n}  "
        f"detect={detect}/{n}  detect_acc={detect / denom:.3f}  "
        f"bay_floor={bay}/{n}  bay_floor_acc={bay / denom:.3f}"
    )


def format_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    tp, fp, fn = int(summary["tp"]), int(summary["fp"]), int(summary["fn"])
    lines = [
        _rate_line("enclosure", summary["enclosure"]),
        _rate_line("none", summary["none"]),
        (
            f"cells tp={tp} fp={fp} fn={fn}  "
            f"recall={tp / max(tp + fn, 1):.3f}  "
            f"precision={tp / max(tp + fp, 1):.3f}"
        ),
    ]
    misses = [r for r in rows if not r["exact"]]
    if misses:
        lines.append("mismatch stem  gt  pred")
        for row in misses:
            lines.append(
                f"  {row['stem']}  gt={_fmt_cells(row['gt'])}  pred={_fmt_cells(row['pred'])}"
            )
    return "\n".join(lines)


def render_compare(
    photo: Any,
    floor_y: dict[int, tuple[int, int]],
    bay_x: dict[int, tuple[int, int]],
    gt_cells: list[Cell] | set[Cell],
    pred_cells: set[Cell],
) -> Any:
    """Grid overlay: blue = label only, red = prediction only, purple = both."""
    from PIL import Image, ImageDraw, ImageFont

    gt = set(gt_cells)
    pred = set(pred_cells)
    base = photo.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    line = max(2, base.width // 280)
    font = ImageFont.load_default()
    for path in ("C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/meiryo.ttc"):
        if Path(path).is_file():
            font = ImageFont.truetype(path, max(13, base.width // 52))
            break
    for fid, (y0, y1) in sorted(floor_y.items()):
        for bid, (x0, x1) in sorted(bay_x.items()):
            cell = (int(fid), int(bid))
            hit_gt = cell in gt
            hit_pred = cell in pred
            if hit_gt and hit_pred:
                fill = (170, 70, 220, 130)
                outline = (190, 90, 240, 255)
            elif hit_gt:
                fill = (40, 110, 255, 130)
                outline = (80, 150, 255, 255)
            elif hit_pred:
                fill = (255, 40, 40, 130)
                outline = (255, 80, 80, 255)
            else:
                fill = None
                outline = (40, 220, 120, 180)
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=outline, width=line)
            draw.text((x0 + 4, y0 + 2), f"f{fid} b{bid}", fill=(255, 255, 255, 235), font=font)
    return Image.alpha_composite(base, layer).convert("RGB")


def save_compare(
    path: Path,
    photo: Path,
    floor_y: dict[int, tuple[int, int]],
    bay_x: dict[int, tuple[int, int]],
    gt_cells: list[Cell] | set[Cell],
    pred_cells: set[Cell],
) -> Path:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(photo).convert("RGB")
    render_compare(image, floor_y, bay_x, gt_cells, pred_cells).save(path, quality=92)
    return path


def _fmt_cells(cells: list[Cell]) -> str:
    if not cells:
        return "none"
    return ",".join(f"f{f}b{b}" for f, b in cells)


def _floor_h(floor_y: dict[int, tuple[int, int]]) -> float | None:
    heights = [int(y1) - int(y0) for y0, y1 in floor_y.values() if int(y1) > int(y0)]
    if not heights:
        return None
    heights.sort()
    mid = len(heights) // 2
    if len(heights) % 2:
        return float(heights[mid])
    return 0.5 * (heights[mid - 1] + heights[mid])


def predict_cells(
    photo: Path,
    *,
    floor_y: dict[int, tuple[int, int]],
    bay_x: dict[int, tuple[int, int]],
    device: str,
    depth_dir: Path,
    sam3_cache: Path,
) -> set[Cell]:
    import torch
    from PIL import Image

    from balcony_pipeline.run_balcony import detect_enclosed_volume_boxes

    image = Image.open(photo).convert("RGB")
    boxes = detect_enclosed_volume_boxes(
        image,
        photo,
        device=torch.device(device),
        depth_dir=depth_dir,
        cache_path=sam3_cache,
        floor_h=_floor_h(floor_y),
        floor_y=floor_y,
        bay_x=bay_x,
    )
    return cells_from_boxes(boxes, floor_y, bay_x)


def select_labels(
    labels: dict[str, list[Cell]],
    stems: list[str] | None,
) -> tuple[dict[str, list[Cell]], list[str]]:
    """Keep only the named photos. Commas and spaces both split names.

    The second list is requested stems that are not in ``labels``.
    """
    if not stems:
        return labels, []
    want: list[str] = []
    seen: set[str] = set()
    for raw in stems:
        for part in str(raw).replace(",", " ").split():
            stem = part.strip()
            if stem and stem not in seen:
                seen.add(stem)
                want.append(stem)
    if not want:
        return labels, []
    missing = [stem for stem in want if stem not in labels]
    return {stem: labels[stem] for stem in want if stem in labels}, missing


def evaluate_labels(
    labels: dict[str, list[Cell]],
    *,
    photo_dir: Path,
    dsl_dir: Path,
    device: str,
    depth_dir: Path,
    sam3_cache: Path,
    compare_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for stem in sorted(labels):
        photo = find_photo(photo_dir, stem)
        dsl = Path(dsl_dir) / stem / "facade_dsl.json"
        if photo is None or not dsl.is_file():
            skipped.append(stem)
            continue
        floor_y, bay_x = load_grid(dsl)
        if not floor_y or not bay_x:
            skipped.append(stem)
            continue
        pred = predict_cells(
            photo,
            floor_y=floor_y,
            bay_x=bay_x,
            device=device,
            depth_dir=depth_dir,
            sam3_cache=sam3_cache,
        )
        row = score_photo(labels[stem], pred)
        row["stem"] = stem
        rows.append(row)
        if compare_dir is not None:
            out = save_compare(
                Path(compare_dir) / f"{stem}.jpg",
                photo,
                floor_y,
                bay_x,
                row["gt"],
                pred,
            )
            row["compare"] = str(out)
        print(f"{stem}  detect={int(row['detected'])}  bay_floor={int(row['exact'])}", flush=True)
    return rows, skipped


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=DEFAULT_EVAL_LABELS_JSONL)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_EVAL_PHOTO_DIR)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--depth-dir", type=Path, default=DEFAULT_DEPTH_DIR)
    ap.add_argument("--sam3-cache", type=Path, default=DEFAULT_SAM3_CACHE)
    ap.add_argument(
        "--device",
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    ap.add_argument(
        "--compare-dir",
        type=Path,
        default=DEFAULT_COMPARE_DIR,
        help="Blue=label only, red=prediction only, purple=both. Empty path skips images.",
    )
    ap.add_argument(
        "--stems",
        nargs="*",
        default=None,
        help="Only these labeled photos. Commas or spaces. Default: every saved label.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    if not labels:
        print(f"no labeled photos in {args.labels}")
        return
    labels, missing = select_labels(labels, args.stems)
    if missing:
        print("not in labels: " + ", ".join(missing))
    if not labels:
        print("no labeled photos matched --stems")
        return
    print(f"labels={args.labels}  n={len(labels)}")
    compare_dir = args.compare_dir
    if compare_dir is not None and str(compare_dir) == "":
        compare_dir = None
    rows, skipped = evaluate_labels(
        labels,
        photo_dir=args.in_dir,
        dsl_dir=args.dsl_dir,
        device=str(args.device),
        depth_dir=args.depth_dir,
        sam3_cache=args.sam3_cache,
        compare_dir=compare_dir,
    )
    if skipped:
        print("skipped (no photo or window grid): " + ", ".join(skipped))
    if not rows:
        print("nothing to score")
        return
    print(format_report(summarize(rows), rows))
    if compare_dir is not None:
        print(f"compare -> {Path(compare_dir).resolve()}")


if __name__ == "__main__":
    main()
