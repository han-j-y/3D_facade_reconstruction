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
    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/evaluate_enclosed_grid.py --device cuda --symmetry

``--symmetry`` scores Stage 5 and Stage 6 from one detection pass and writes
Stage 6 images to ``--compare-dir-sym``.
"""

from __future__ import annotations

import argparse
import json
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
    GRID_FAILURE,
    Cell,
    find_photo,
    load_grid,
    load_labels,
    load_records,
)
from balcony_train.labels import ENCLOSURE_SKINS  # noqa: E402

DEFAULT_DEPTH_DIR = ROOT / "runs" / "da3_enclosure"
DEFAULT_SAM3_CACHE = ROOT / "runs" / "sam3_da3_volumes" / "sam3_seeds.json"
DEFAULT_COMPARE_DIR = ROOT / "runs" / "enclosed_grid" / "compare"
DEFAULT_COMPARE_DIR_SYM = ROOT / "runs" / "enclosed_grid" / "compare_sym"


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


def scored_floor_shape(shape: str | None) -> str | None:
    """Circle stays circle. Triangle counts as rectangle. Anything else is unset."""
    raw = str(shape or "").strip().lower()
    if raw == "circle":
        return "circle"
    if raw in ("rectangle", "triangle"):
        return "rectangle"
    return None


# One box color per enclosed mesh: skin × floor. Triangle is already rectangle.
PLAN_TYPES: tuple[tuple[str, str], ...] = (
    ("punched_window", "rectangle"),
    ("punched_window", "circle"),
    ("curtain_wall", "rectangle"),
    ("curtain_wall", "circle"),
)
PLAN_TYPE_COLOR: dict[tuple[str, str], tuple[int, int, int]] = {
    ("punched_window", "rectangle"): (255, 150, 30),
    ("punched_window", "circle"): (230, 50, 170),
    ("curtain_wall", "rectangle"): (40, 130, 255),
    ("curtain_wall", "circle"): (30, 190, 110),
}
PLAN_TYPE_LABEL: dict[tuple[str, str], str] = {
    ("punched_window", "rectangle"): "punched rect",
    ("punched_window", "circle"): "punched circle",
    ("curtain_wall", "rectangle"): "curtain rect",
    ("curtain_wall", "circle"): "curtain circle",
}


def enclosed_plan_type(skin: str | None, floor: str | None) -> tuple[str, str]:
    """Map one cell prediction onto the four mesh types.

    An unknown or missing skin is punched. An unknown floor is rectangle.
    Triangle counts as rectangle, matching the photo-level vote.
    """
    shape = scored_floor_shape(floor) or "rectangle"
    skin_name = str(skin or "").strip().lower()
    if skin_name not in ENCLOSURE_SKINS:
        skin_name = "punched_window"
    return skin_name, shape


def vote_plan(
    image: Any,
    cells: list[Cell] | set[Cell],
    floor_y: dict[int, tuple[int, int]],
    bay_x: dict[int, tuple[int, int]],
    predict_full: Any,
) -> tuple[str | None, str | None]:
    """Majority floor shape and skin over the predicted cells.

    No predicted cell means there is no plan. That is different from the
    vote default (rectangle, punched window), which only applies once at
    least one crop was classified.
    """
    from balcony_train.depth_cells import cell_box
    from enclosed import vote_enclosed_floor_shape, vote_enclosed_skin

    chosen = sorted({(int(floor), int(bay)) for floor, bay in cells})
    shapes: list[str | None] = []
    skins: list[str | None] = []
    for cell in chosen:
        if cell[0] not in floor_y or cell[1] not in bay_x:
            continue
        crop = image.crop(tuple(cell_box(cell, floor_y, bay_x)))
        pred = predict_full(crop)
        shapes.append(pred.get("floor_shape") if isinstance(pred, dict) else None)
        skins.append(pred.get("skin") if isinstance(pred, dict) else None)
    if not shapes:
        return None, None
    shape, _counts = vote_enclosed_floor_shape(shapes)
    skin, _skin_counts = vote_enclosed_skin(skins)
    return shape, skin


def apply_plan_score(
    row: dict[str, Any],
    gt_shape: str | None,
    gt_skin: str | None,
    pred_shape: str | None,
    pred_skin: str | None,
) -> None:
    """Attach shape and skin correctness. Photos without both labels are not scored."""
    row["gt_shape"] = gt_shape
    row["gt_skin"] = gt_skin
    row["pred_shape"] = pred_shape
    row["pred_skin"] = pred_skin
    labeled = bool(row.get("has_enclosure") and gt_shape and gt_skin in ENCLOSURE_SKINS)
    row["plan_labeled"] = labeled
    if not labeled:
        row["shape_ok"] = None
        row["skin_ok"] = None
        return
    row["shape_ok"] = scored_floor_shape(pred_shape) == scored_floor_shape(gt_shape)
    row["skin_ok"] = pred_skin == gt_skin


def load_pred_cells(path: Path) -> dict[str, set[Cell]]:
    """Stem → predicted cells. Accepts a flat json or ``{"eval": {"preds": ...}}``."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("eval"), dict) and "preds" in raw["eval"]:
        raw = raw["eval"]["preds"]
    if not isinstance(raw, dict):
        raise ValueError(f"predicted cells must be an object, got {type(raw).__name__}")
    return {str(stem): {tuple(cell) for cell in cells} for stem, cells in raw.items()}


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
        "total": _group_counts(rows),
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
        _rate_line("total", summary["total"]),
        (
            f"cells tp={tp} fp={fp} fn={fn}  "
            f"recall={tp / max(tp + fn, 1):.3f}  "
            f"precision={tp / max(tp + fp, 1):.3f}"
        ),
    ]
    plans = [r for r in rows if r.get("plan_labeled")]
    # No predicted cell means floor and skin were never judged.
    judged = [r for r in plans if r.get("pred")]
    if judged:
        n = len(judged)
        shape_ok = sum(1 for r in judged if r["shape_ok"])
        skin_ok = sum(1 for r in judged if r["skin_ok"])
        held = len(plans) - n
        held_note = f"  no_cells={held}" if held else ""
        lines.append(
            f"shape {shape_ok}/{n}  shape_acc={shape_ok / n:.3f}  "
            f"skin {skin_ok}/{n}  skin_acc={skin_ok / n:.3f}{held_note}"
        )
    misses = [r for r in rows if not r["exact"]]
    if misses:
        lines.append("mismatch stem  gt  pred")
        for row in misses:
            lines.append(
                f"  {row['stem']}  gt={_fmt_cells(row['gt'])}  pred={_fmt_cells(row['pred'])}"
            )
    plan_misses = [
        r for r in judged if not r["shape_ok"] or not r["skin_ok"]
    ]
    if plan_misses:
        lines.append("plan stem  gt_shape  pred_shape  gt_skin  pred_skin")
        for row in plan_misses:
            lines.append(
                f"  {row['stem']}  {row['gt_shape']}  {row['pred_shape']}  "
                f"{row['gt_skin']}  {row['pred_skin']}"
            )
    return "\n".join(lines)


def render_plan_types(
    photo: Any,
    boxes: list[dict[str, Any]],
    *,
    title: str = "",
) -> Any:
    """Draw one bounding box per detected cell, colored by its plan type."""
    from PIL import Image, ImageDraw, ImageFont

    base = photo.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    line = max(3, base.width // 180)
    font = ImageFont.load_default()
    for path in ("C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/meiryo.ttc"):
        if Path(path).is_file():
            font = ImageFont.truetype(path, max(14, base.width // 48))
            break
    for item in boxes:
        skin, shape = item["plan"]
        color = PLAN_TYPE_COLOR[(skin, shape)]
        x0, y0, x1, y1 = item["box"]
        draw.rectangle((x0, y0, x1, y1), fill=(*color, 70), outline=(*color, 255), width=line)
        label = PLAN_TYPE_LABEL[(skin, shape)]
        tx, ty = x0 + 4, y0 + 2
        tw = max(8, font.getlength(label)) if hasattr(font, "getlength") else 8 * len(label)
        draw.rectangle((tx - 2, ty, tx + tw + 4, ty + 18), fill=(0, 0, 0, 180))
        draw.text((tx, ty), label, fill=(255, 255, 255, 255), font=font)
    painted = Image.alpha_composite(base, layer).convert("RGB")
    if not title:
        return painted
    bar_h = max(28, painted.height // 28)
    out = Image.new("RGB", (painted.width, painted.height + bar_h), (16, 18, 22))
    out.paste(painted, (0, bar_h))
    bar = ImageDraw.Draw(out)
    bar.text((8, 4), title, fill=(240, 240, 240), font=font)
    return out


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
    windows: list[list[int]] | None = None,
    symmetry: bool = False,
) -> dict[str, Any]:
    """Cells from Stage 5 (``base``) and, with ``symmetry``, Stage 6 (``sym``)."""
    import torch
    from PIL import Image

    from balcony_pipeline.run_balcony import enclosed_volume_stages

    image = Image.open(photo).convert("RGB")
    stages = enclosed_volume_stages(
        image,
        photo,
        device=torch.device(device),
        depth_dir=depth_dir,
        cache_path=sam3_cache,
        floor_h=_floor_h(floor_y),
        floor_y=floor_y,
        bay_x=bay_x,
        windows=windows,
        symmetry=symmetry,
    )
    by_id = {sid: (title, boxes) for sid, title, boxes in stages}
    base = by_id.get("5_clip", ("", []))[1]
    out: dict[str, Any] = {"base": cells_from_boxes(base, floor_y, bay_x)}
    if symmetry:
        title, boxes = by_id.get("6_sym", ("6 sym off (no boxes)", base))
        out["sym"] = cells_from_boxes(boxes, floor_y, bay_x)
        out["sym_title"] = title
    return out


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
    symmetry: bool = False,
    compare_dir_sym: Path | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Score each photo. With ``symmetry`` each row also holds ``row['sym']``."""
    import json

    from balcony_train.enclosed_symmetry import window_boxes_from_dsl

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
        windows = None
        if symmetry:
            windows = window_boxes_from_dsl(json.loads(dsl.read_text(encoding="utf-8")))
        pred = predict_cells(
            photo,
            floor_y=floor_y,
            bay_x=bay_x,
            device=device,
            depth_dir=depth_dir,
            sam3_cache=sam3_cache,
            windows=windows,
            symmetry=symmetry,
        )
        row = score_photo(labels[stem], pred["base"])
        row["stem"] = stem
        rows.append(row)
        if compare_dir is not None:
            out = save_compare(
                Path(compare_dir) / f"{stem}.jpg",
                photo,
                floor_y,
                bay_x,
                row["gt"],
                pred["base"],
            )
            row["compare"] = str(out)
        line = f"{stem}  detect={int(row['detected'])}  bay_floor={int(row['exact'])}"
        if symmetry:
            sym = score_photo(labels[stem], pred["sym"])
            sym["stem"] = stem
            sym["title"] = pred["sym_title"]
            row["sym"] = sym
            if compare_dir_sym is not None:
                sym["compare"] = str(
                    save_compare(
                        Path(compare_dir_sym) / f"{stem}.jpg",
                        photo,
                        floor_y,
                        bay_x,
                        sym["gt"],
                        pred["sym"],
                    )
                )
            line += (
                f"  | sym detect={int(sym['detected'])}  bay_floor={int(sym['exact'])}"
                f"  [{sym['title']}]"
            )
        print(line, flush=True)
    return rows, skipped


def format_symmetry_diff(rows: list[dict[str, Any]]) -> str:
    """Before/after rates and the photos whose cells changed."""
    sym_rows = [r["sym"] for r in rows if "sym" in r]
    before, after = summarize(rows), summarize(sym_rows)
    lines = ["=== symmetry: before (stage 5) -> after (stage 6) ==="]
    for group in ("enclosure", "none", "total"):
        b, a = before[group], after[group]
        lines.append(
            f"{group} n={b['n']}  detect {b['detect']} -> {a['detect']}  "
            f"bay_floor {b['bay_floor']} -> {a['bay_floor']}"
        )
    for key in ("tp", "fp", "fn"):
        lines.append(f"cells {key} {before[key]} -> {after[key]}")
    changed = [r for r in rows if "sym" in r and r["pred"] != r["sym"]["pred"]]
    if changed:
        lines.append("changed stem  exact before->after  +cells  -cells")
        for r in changed:
            b_cells, a_cells = set(r["pred"]), set(r["sym"]["pred"])
            lines.append(
                f"  {r['stem']}  {int(r['exact'])}->{int(r['sym']['exact'])}  "
                f"+{_fmt_cells(sorted(a_cells - b_cells))}  "
                f"-{_fmt_cells(sorted(b_cells - a_cells))}  [{r['sym']['title']}]"
            )
    else:
        lines.append("no photo changed")
    return "\n".join(lines)


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
    ap.add_argument(
        "--symmetry",
        action="store_true",
        help="Also score Stage 6 (mirror about the window-grid axis) and print before -> after.",
    )
    ap.add_argument(
        "--compare-dir-sym",
        type=Path,
        default=DEFAULT_COMPARE_DIR_SYM,
        help="Compare images for Stage 6 when --symmetry. Empty path skips images.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    failures = sorted(
        stem
        for stem, rec in load_records(args.labels).items()
        if rec.status == GRID_FAILURE
    )
    if failures:
        print(f"grid failure ({len(failures)}): " + ", ".join(failures))
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
    compare_dir_sym = args.compare_dir_sym
    if not args.symmetry or (compare_dir_sym is not None and str(compare_dir_sym) == ""):
        compare_dir_sym = None
    rows, skipped = evaluate_labels(
        labels,
        photo_dir=args.in_dir,
        dsl_dir=args.dsl_dir,
        device=str(args.device),
        depth_dir=args.depth_dir,
        sam3_cache=args.sam3_cache,
        compare_dir=compare_dir,
        symmetry=args.symmetry,
        compare_dir_sym=compare_dir_sym,
    )
    if skipped:
        print("skipped (no photo or window grid): " + ", ".join(skipped))
    if not rows:
        print("nothing to score")
        return
    print(format_report(summarize(rows), rows))
    if args.symmetry:
        sym_rows = [r["sym"] for r in rows]
        print("=== with symmetry (stage 6) ===")
        print(format_report(summarize(sym_rows), sym_rows))
        print(format_symmetry_diff(rows))
    if compare_dir is not None:
        print(f"compare -> {Path(compare_dir).resolve()}")
    if compare_dir_sym is not None:
        print(f"compare sym -> {Path(compare_dir_sym).resolve()}")


if __name__ == "__main__":
    main()
