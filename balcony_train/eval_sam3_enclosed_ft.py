"""Score the enclosed-facade SAM3 fine-tune on the cmp_x grid labels.

The checkpoint is ``proj-facade/sam3-enclosed-facade-fold0``. Inference follows
its model card: prompt ``enclosed balcony``, score threshold 0.45, and one
padding box so the geometry encoder stays on. Boxes are snapped to the window
floor x bay grid with the same overlap rule as ``evaluate_enclosed_grid``.

Cell TP / FP / FN / TN are pooled over every window cell of every labeled
photo. cmp_b is not read.

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/eval_sam3_enclosed_ft.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.enclosed_grid_store import (  # noqa: E402
    DEFAULT_DSL_DIR,
    DEFAULT_EVAL_LABELS_JSONL,
    DEFAULT_EVAL_PHOTO_DIR,
    Cell,
    find_photo,
    load_grid,
    load_labels,
)
from balcony_train.evaluate_enclosed_grid import cells_from_boxes, render_compare  # noqa: E402
from balcony_train.try_sam3_da3_volumes import (  # noqa: E402
    compose_row,
    draw_stage_overlay,
    resize_depth,
    resolve_floor_h,
    resolve_window_bands,
    resolve_window_boxes,
    volume_stages,
)

DEFAULT_DEPTH_DIR = ROOT / "runs" / "da3_enclosure"

MODEL_ID = "proj-facade/sam3-enclosed-facade-fold0"
PROMPT = "enclosed balcony"
THRESHOLD = 0.45
DEFAULT_OUT = ROOT / "runs" / "sam3_ft_fold0" / "cmp_x"


def detect_boxes(image: Image.Image, processor, model, device: torch.device) -> list[dict]:
    """One forward pass, matching the fold-0 model card."""
    inputs = processor(images=image, text=PROMPT, return_tensors="pt")
    moved = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    moved["input_boxes"] = torch.tensor([[[0.5, 0.5, 0.1, 0.1]]], device=device, dtype=torch.float32)
    moved["input_boxes_labels"] = torch.tensor([[-10]], device=device, dtype=torch.long)
    with torch.no_grad():
        outputs = model(**moved)
    result = processor.post_process_instance_segmentation(
        outputs,
        threshold=THRESHOLD,
        mask_threshold=0.5,
        target_sizes=moved["original_sizes"].tolist(),
    )[0]
    boxes = result["boxes"].detach().float().cpu().tolist()
    scores = result["scores"].detach().float().cpu().tolist()
    out = []
    for box, score in zip(boxes, scores):
        out.append(
            {
                "box_xyxy": [int(round(v)) for v in box],
                "score": round(float(score), 4),
                "prompt": PROMPT,
            }
        )
    return out


def photo_counts(gt: set[Cell], pred: set[Cell], n_cells: int) -> dict[str, int]:
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    tn = n_cells - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_cells": n_cells}


def rates(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def _font(size: int) -> ImageFont.ImageFont:
    for path in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = min(width / image.width, height / image.height)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def write_board(path: Path, summary: dict, examples: list[dict]) -> None:
    """Table on the left, two scored photos on the right."""
    board = Image.new("RGB", (1680, 920), (245, 245, 245))
    draw = ImageDraw.Draw(board)
    title = _font(28)
    body = _font(22)
    small = _font(18)
    panel = (28, 28, 32)
    draw.rounded_rectangle((36, 36, 760, 250), radius=16, fill=panel)
    draw.text((60, 56), f"{summary['n_photos']} images. {summary['n_cells']:,} cells in total", fill=(230, 230, 230), font=title)
    headers = ("TP", "FP", "FN", "TN")
    values = (summary["tp"], summary["fp"], summary["fn"], summary["tn"])
    for i, (name, value) in enumerate(zip(headers, values)):
        x = 70 + i * 170
        draw.text((x, 120), name, fill=(170, 170, 170), font=body)
        draw.text((x, 160), str(value), fill=(245, 245, 245), font=title)
    draw.rounded_rectangle((36, 280, 760, 860), radius=16, fill=panel)
    rows = [
        ("Precision", f"{summary['tp']} / ({summary['tp']} + {summary['fp']}) = {summary['precision']:.3f}"),
        ("Recall", f"{summary['tp']} / ({summary['tp']} + {summary['fn']}) = {summary['recall']:.3f}"),
        ("F1", f"2 x {summary['precision']:.3f} x {summary['recall']:.3f} / ({summary['precision']:.3f} + {summary['recall']:.3f}) = {summary['f1']:.3f}"),
        ("IoU", f"{summary['tp']} / ({summary['tp']} + {summary['fp']} + {summary['fn']}) = {summary['iou']:.3f}"),
    ]
    draw.text((60, 310), "Calculation", fill=(170, 170, 170), font=body)
    y = 370
    for name, formula in rows:
        draw.text((60, y), name, fill=(245, 245, 245), font=body)
        draw.text((230, y), formula, fill=(220, 220, 220), font=small)
        y += 110
    draw.text((820, 40), "Example", fill=(40, 40, 40), font=title)
    slots = ((820, 100), (1240, 100))
    for slot, example in zip(slots, examples):
        image = _fit(Image.open(example["path"]).convert("RGB"), 400, 680)
        board.paste(image, slot)
        caption = (
            f"{example['stem']}\n"
            f"Precision: {example['tp']}/{example['tp'] + example['fp']} = {example['precision']:.2f}\n"
            f"Recall: {example['tp']}/{example['tp'] + example['fn']} = {example['recall']:.2f}"
        )
        draw.multiline_text((slot[0], 800), caption, fill=(30, 30, 30), font=small, spacing=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    board.save(path, quality=92)


def choose_examples(rows: list[dict]) -> list[dict]:
    """Two enclosure photos: one with a false positive, one with a miss."""
    enclosed = [row for row in rows if row["gt"]]
    with_fp = sorted((row for row in enclosed if row["fp"]), key=lambda row: (-row["fp"], row["stem"]))
    with_fn = sorted((row for row in enclosed if row["fn"]), key=lambda row: (-row["fn"], row["stem"]))
    picked: list[dict] = []
    for group in (with_fp, with_fn, enclosed):
        for row in group:
            if row["stem"] not in {item["stem"] for item in picked}:
                picked.append(row)
            if len(picked) == 2:
                return picked
    return picked


def write_stage_strips(
    rows_by_stem: dict[str, list[dict]],
    *,
    photo_dir: Path,
    dsl_dir: Path,
    depth_dir: Path,
    out_dir: Path,
) -> None:
    """One six-stage strip per photo, plus a stacked board."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_rows: list[Image.Image] = []
    for stem, seeds in rows_by_stem.items():
        photo = find_photo(photo_dir, stem)
        depth_path = depth_dir / f"{stem}_depth.npy"
        if photo is None or not depth_path.is_file():
            print(f"skip stages {stem}", flush=True)
            continue
        image = Image.open(photo).convert("RGB")
        depth = resize_depth(np.load(depth_path), image.width, image.height)
        bands = resolve_window_bands(stem, dsl_dir)
        floor_y, bay_x = bands if bands is not None else (None, None)
        stages = volume_stages(
            image,
            depth,
            seeds,
            floor_h=resolve_floor_h(stem, dsl_dir),
            floor_y=floor_y,
            bay_x=bay_x,
            windows=resolve_window_boxes(stem, dsl_dir),
            symmetry=True,
        )
        panels = [
            draw_stage_overlay(image, gt=[], pred=boxes, title=title)
            for _sid, title, boxes in stages
        ]
        strip = compose_row(panels, height=420)
        strip.save(out_dir / f"{stem}_stages.jpg", quality=92)
        stage_rows.append(strip)
        print(f"stages {stem}  " + "  ".join(f"{title.split()[0]}={len(boxes)}" for _sid, title, boxes in stages), flush=True)
    if not stage_rows:
        return
    gap = 12
    board_w = max(row.width for row in stage_rows)
    board_h = sum(row.height for row in stage_rows) + gap * (len(stage_rows) - 1)
    board = Image.new("RGB", (board_w, board_h), (24, 24, 24))
    y = 0
    for row in stage_rows:
        board.paste(row, (0, y))
        y += row.height + gap
    board_path = out_dir / "stages_board.jpg"
    board.save(board_path, quality=85)
    print(f"stages -> {board_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=DEFAULT_EVAL_LABELS_JSONL)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_EVAL_PHOTO_DIR)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--depth-dir", type=Path, default=DEFAULT_DEPTH_DIR)
    ap.add_argument("--stages", action="store_true", help="Save the six-stage strip for each photo")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N labeled photos")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    stems = sorted(labels)
    if args.limit:
        stems = stems[: args.limit]
    out_dir = Path(args.out_dir)
    compare_dir = out_dir / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "boxes.json"
    cache: dict[str, list[dict]] = {}
    if cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    need = []
    for stem in stems:
        photo = find_photo(args.in_dir, stem)
        dsl = Path(args.dsl_dir) / stem / "facade_dsl.json"
        if photo is None or not dsl.is_file():
            print(f"skip {stem}", flush=True)
            continue
        if stem not in cache:
            need.append(stem)
    if need:
        from transformers import Sam3Model, Sam3Processor

        device = torch.device(args.device)
        print(f"loading {MODEL_ID} on {device}  n={len(need)}", flush=True)
        processor = Sam3Processor.from_pretrained(MODEL_ID)
        model = Sam3Model.from_pretrained(MODEL_ID).to(device).eval()
        for stem in need:
            photo = find_photo(args.in_dir, stem)
            image = Image.open(photo).convert("RGB")
            cache[stem] = detect_boxes(image, processor, model, device)
            cache_path.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
            print(f"{stem}  boxes={len(cache[stem])}", flush=True)
        del model, processor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = []
    for stem in stems:
        photo = find_photo(args.in_dir, stem)
        dsl = Path(args.dsl_dir) / stem / "facade_dsl.json"
        if photo is None or not dsl.is_file() or stem not in cache:
            continue
        floor_y, bay_x = load_grid(dsl)
        gt = {(int(f), int(b)) for f, b in labels[stem]}
        pred = cells_from_boxes(cache[stem], floor_y, bay_x)
        counts = photo_counts(gt, pred, len(floor_y) * len(bay_x))
        image_path = compare_dir / f"{stem}.jpg"
        render_compare(Image.open(photo).convert("RGB"), floor_y, bay_x, gt, pred).save(image_path, quality=92)
        row = {
            "stem": stem,
            "gt": sorted(gt),
            "pred": sorted(pred),
            "path": str(image_path),
            **counts,
            **rates(counts["tp"], counts["fp"], counts["fn"]),
        }
        rows.append(row)
        print(
            f"{stem}  tp={counts['tp']} fp={counts['fp']} fn={counts['fn']} tn={counts['tn']}",
            flush=True,
        )

    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    tn = sum(row["tn"] for row in rows)
    summary = {
        "model": MODEL_ID,
        "prompt": PROMPT,
        "threshold": THRESHOLD,
        "n_photos": len(rows),
        "n_cells": sum(row["n_cells"] for row in rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        **rates(tp, fp, fn),
        "exact": sum(1 for row in rows if set(map(tuple, row["gt"])) == set(map(tuple, row["pred"]))),
    }
    (out_dir / "summary.json").write_text(json.dumps({"summary": summary, "photos": rows}, indent=2) + "\n", encoding="utf-8")
    examples = choose_examples(rows)
    if examples:
        write_board(out_dir / "board.png", summary, examples)
    if args.stages:
        write_stage_strips(
            {stem: cache.get(stem, []) for stem in stems if stem in cache},
            photo_dir=args.in_dir,
            dsl_dir=args.dsl_dir,
            depth_dir=args.depth_dir,
            out_dir=out_dir / "stages",
        )
    text = (
        f"{summary['n_photos']} images, {summary['n_cells']} cells\n"
        f"TP {tp}  FP {fp}  FN {fn}  TN {tn}\n"
        f"Precision {summary['precision']:.3f}\n"
        f"Recall {summary['recall']:.3f}\n"
        f"F1 {summary['f1']:.3f}\n"
        f"IoU {summary['iou']:.3f}\n"
        f"exact {summary['exact']}/{summary['n_photos']}\n"
    )
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
