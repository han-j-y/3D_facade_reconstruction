"""IoU of SAM3 balcony boxes against CMP label-map balcony regions.

Ground truth is palette index 7 in ``data/extended/<stem>.png``. Prediction
is every detector box from the SAM3 ``balcony`` call at the score threshold
(the red panel of the comparison rows), before the size filter.

A SAM3 box is a true positive when at least 60% of its area lies inside a
PNG balcony box. A PNG box with no such SAM3 box is a false negative. A SAM3
box that is not 60% inside any PNG box is a false positive.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_pipeline/score_sam3_label_iou.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_pipeline as rp  # noqa: E402

from compare_sam3_filter import _row, segmentation_image  # noqa: E402
from draw import draw_boxes  # noqa: E402
from filter import filter_balcony_boxes  # noqa: E402
from balcony_train.crop_balconies_xml import (  # noqa: E402
    BALCONY_INDEX,
    balcony_pixel_boxes,
    scale_box,
)

DEFAULT_IMAGE_DIR = ROOT / "data" / "baseForBatch"
DEFAULT_LABEL_DIR = ROOT / "data" / "extended"
DEFAULT_OUT = ROOT / "runs" / "batch" / "_rows" / "baseForBatch_sam3_label_iou.json"
IOU_THRESHOLD = 0.5


def balcony_mask(label: Image.Image, *, width: int, height: int) -> np.ndarray:
    """Boolean balcony mask at the photo size."""
    if label.mode != "P":
        rgb = np.asarray(label.convert("RGB"))
        mask = (rgb[:, :, 0] == 170) & (rgb[:, :, 1] == 255) & (rgb[:, :, 2] == 85)
    else:
        mask = np.asarray(label) == BALCONY_INDEX
    if mask.shape == (height, width):
        return mask
    scaled = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (width, height), Image.Resampling.NEAREST
    )
    return np.asarray(scaled) > 0


def box_iou(a: list[int] | tuple[int, ...], b: list[int] | tuple[int, ...]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def rect_area(box: list[int] | tuple[int, ...]) -> float:
    x0, y0, x1, y1 = [float(v) for v in box]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def intersection_area(a: list[int] | tuple[int, ...], b: list[int] | tuple[int, ...]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    return iw * ih


def inside_fraction(pred: list[int] | tuple[int, ...], gt: list[int] | tuple[int, ...]) -> float:
    """Share of the SAM3 box that lies inside the PNG box."""
    area = rect_area(pred)
    if area <= 0:
        return 0.0
    return intersection_area(pred, gt) / area


def inside_counts(
    gt: list[list[int]],
    pred: list[list[int]],
    *,
    threshold: float,
) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, detected_gt).

    TP is a SAM3 box with at least ``threshold`` of its area inside some PNG
    box. FP is every other SAM3 box. FN is a PNG box that contains no such
    SAM3 box. ``detected_gt`` is the number of PNG boxes that are not FN.
    """
    tp = 0
    for p in pred:
        if any(inside_fraction(p, g) >= threshold for g in gt):
            tp += 1
    detected = 0
    for g in gt:
        if any(inside_fraction(p, g) >= threshold for p in pred):
            detected += 1
    fp = len(pred) - tp
    fn = len(gt) - detected
    return tp, fp, fn, detected


def cover_counts(
    gt: list[list[int]],
    pred: list[list[int]],
    *,
    threshold: float,
) -> tuple[int, int]:
    """Balconies covered at ``threshold``, and SAM3 boxes that cover one.

    A balcony is covered when some SAM3 box overlaps at least ``threshold``
    of that balcony rectangle. A SAM3 box counts when it does so for any balcony.
    """
    gt_tp = 0
    for g in gt:
        ga = rect_area(g)
        if ga <= 0:
            continue
        if any(intersection_area(g, p) / ga >= threshold for p in pred):
            gt_tp += 1
    box_tp = 0
    for p in pred:
        if any(
            rect_area(g) > 0 and intersection_area(g, p) / rect_area(g) >= threshold
            for g in gt
        ):
            box_tp += 1
    return gt_tp, box_tp


def match_iou(
    gt: list[list[int]],
    pred: list[list[int]],
    *,
    threshold: float,
) -> tuple[int, list[float]]:
    """Greedy one-to-one matches. Returns (tp, best IoU per ground-truth box)."""
    best = [0.0] * len(gt)
    pairs: list[tuple[float, int, int]] = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            iou = box_iou(g, p)
            if iou > best[i]:
                best[i] = iou
            if iou > 0:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_g: set[int] = set()
    used_p: set[int] = set()
    tp = 0
    for iou, i, j in pairs:
        if i in used_g or j in used_p:
            continue
        if iou < threshold:
            break
        used_g.add(i)
        used_p.add(j)
        tp += 1
    return tp, best


def area_iou(mask: np.ndarray, boxes: list[list[int]]) -> tuple[int, int]:
    """Return (intersection pixels, union pixels) of the balcony mask and boxes."""
    pred = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    for x0, y0, x1, y1 in boxes:
        x0 = max(0, min(width, int(x0)))
        x1 = max(0, min(width, int(x1)))
        y0 = max(0, min(height, int(y0)))
        y1 = max(0, min(height, int(y1)))
        if x1 > x0 and y1 > y0:
            pred[y0:y1, x0:x1] = True
    inter = int(np.logical_and(mask, pred).sum())
    union = int(np.logical_or(mask, pred).sum())
    return inter, union


def ratio(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return num / den


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    ap.add_argument(
        "--cover",
        type=float,
        default=0.6,
        help="SAM3 box is a true positive when at least this fraction of its area lies inside a PNG balcony box",
    )
    ap.add_argument("--prompt", default="balcony")
    ap.add_argument("--min-side", type=int, default=16)
    ap.add_argument("--max-side-frac", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--exclude",
        default="",
        help="comma-separated stems to leave out of the test set",
    )
    ap.add_argument(
        "--panels-dir",
        type=Path,
        default=None,
        help="write segmentation | SAM3 raw | SAM3 size | filtered for each scored image",
    )
    args = ap.parse_args()

    excluded = {s.strip() for s in args.exclude.replace(" ", ",").split(",") if s.strip()}
    images = sorted(p for p in Path(args.image_dir).glob("*.jpg") if p.is_file() and p.stem not in excluded)
    if not images:
        raise SystemExit(f"no jpg in {args.image_dir}")
    absent = sorted(excluded - {p.stem for p in Path(args.image_dir).glob("*.jpg")})
    if excluded:
        print(f"excluded {len(excluded - set(absent))}  absent from image dir: {absent}", flush=True)
    if args.panels_dir is not None:
        Path(args.panels_dir).mkdir(parents=True, exist_ok=True)

    from transformers import Sam3Model, Sam3Processor

    device = torch.device(args.device)
    print(f"{len(images)} images  device={device}  iou>={args.iou}", flush=True)
    print("loading SAM3…", flush=True)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    sam = Sam3Model.from_pretrained("facebook/sam3").to(device)
    sam.eval()

    rows: list[dict] = []
    for image_path in images:
        stem = image_path.stem
        label_path = Path(args.label_dir) / f"{stem}.png"
        if not label_path.is_file():
            print(f"skip {stem}: missing {label_path}", flush=True)
            continue
        photo = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)
        width, height = photo.size
        gt = [
            list(scale_box(box, src_w=label.size[0], src_h=label.size[1], dst_w=width, dst_h=height))
            for box in balcony_pixel_boxes(label)
        ]
        det_boxes: list[list[int]] = []
        sized = rp.detect_windows(
            photo,
            prompt=args.prompt,
            threshold=args.threshold,
            min_side=args.min_side,
            max_side_frac=args.max_side_frac,
            device=device,
            sam=sam,
            processor=processor,
            det_boxes_out=det_boxes,
        )
        if args.panels_dir is not None:
            sized_boxes = [r["box_xyxy"] for r in sized]
            kept, _log = filter_balcony_boxes(sized_boxes, photo, [])
            panels = [
                draw_boxes(photo, det_boxes, title=f"SAM3 raw  n={len(det_boxes)}", color=(220, 40, 40)),
                draw_boxes(photo, sized_boxes, title=f"SAM3 size  n={len(sized_boxes)}", color=(40, 120, 220)),
                draw_boxes(
                    photo,
                    kept,
                    title=f"Filtered  kept={len(kept)}/{len(sized_boxes)}",
                    color=(40, 180, 90),
                ),
            ]
            seg = segmentation_image(stem, Path(args.label_dir))
            if seg is not None:
                panels.insert(0, seg)
            _row(panels).save(Path(args.panels_dir) / f"{stem}.jpg", quality=90, optimize=True)
        tp, best = match_iou(gt, det_boxes, threshold=args.iou)
        cover_gt, cover_box = cover_counts(gt, det_boxes, threshold=args.cover)
        in_tp, in_fp, in_fn, in_detected = inside_counts(gt, det_boxes, threshold=args.cover)
        inter, union = area_iou(balcony_mask(label, width=width, height=height), det_boxes)
        row = {
            "stem": stem,
            "gt": len(gt),
            "sam3": len(det_boxes),
            "tp": tp,
            "precision": ratio(tp, len(det_boxes)),
            "recall": ratio(tp, len(gt)),
            "mean_iou": ratio(sum(best), len(gt)),
            "cover_gt_tp": cover_gt,
            "cover_box_tp": cover_box,
            "cover_precision": ratio(cover_box, len(det_boxes)),
            "cover_recall": ratio(cover_gt, len(gt)),
            "inside_tp": in_tp,
            "inside_fp": in_fp,
            "inside_fn": in_fn,
            "inside_detected": in_detected,
            "inside_precision": ratio(in_tp, len(det_boxes)),
            "inside_recall": ratio(in_detected, len(gt)),
            "area_inter": inter,
            "area_union": union,
            "area_iou": ratio(inter, union),
        }
        rows.append(row)
        print(
            f"{stem}  gt={row['gt']}  sam3={row['sam3']}  "
            f"TP={in_tp}  FP={in_fp}  FN={in_fn}  "
            f"P={_fmt(row['inside_precision'])}  R={_fmt(row['inside_recall'])}",
            flush=True,
        )

    del sam, processor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    gt_n = sum(r["gt"] for r in rows)
    sam_n = sum(r["sam3"] for r in rows)
    tp_n = sum(r["tp"] for r in rows)
    cover_gt_n = sum(r["cover_gt_tp"] for r in rows)
    cover_box_n = sum(r["cover_box_tp"] for r in rows)
    in_tp_n = sum(r["inside_tp"] for r in rows)
    in_fp_n = sum(r["inside_fp"] for r in rows)
    in_fn_n = sum(r["inside_fn"] for r in rows)
    in_det_n = sum(r["inside_detected"] for r in rows)
    inter_n = sum(r["area_inter"] for r in rows)
    union_n = sum(r["area_union"] for r in rows)
    mean_iou_num = sum((r["mean_iou"] or 0.0) * r["gt"] for r in rows)
    total = {
        "images": len(rows),
        "gt": gt_n,
        "sam3": sam_n,
        "tp": tp_n,
        "precision": ratio(tp_n, sam_n),
        "recall": ratio(tp_n, gt_n),
        "mean_iou": ratio(mean_iou_num, gt_n),
        "cover_gt_tp": cover_gt_n,
        "cover_box_tp": cover_box_n,
        "cover_precision": ratio(cover_box_n, sam_n),
        "cover_recall": ratio(cover_gt_n, gt_n),
        "cover_threshold": args.cover,
        "inside_tp": in_tp_n,
        "inside_fp": in_fp_n,
        "inside_fn": in_fn_n,
        "inside_detected": in_det_n,
        "inside_precision": ratio(in_tp_n, sam_n),
        "inside_recall": ratio(in_det_n, gt_n),
        "area_inter": inter_n,
        "area_union": union_n,
        "area_iou": ratio(inter_n, union_n),
        "iou_threshold": args.iou,
        "score_threshold": args.threshold,
        "prompt": args.prompt,
        "excluded": sorted(excluded),
    }
    payload = {"total": total, "images": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"TOTAL inside>={args.cover}  TP={in_tp_n}  FP={in_fp_n}  FN={in_fn_n}  "
        f"detected_png={in_det_n}  "
        f"P={_fmt(total['inside_precision'])}  R={_fmt(total['inside_recall'])}",
        flush=True,
    )
    print(
        f"TOTAL cover>={args.cover}  gt={gt_n}  sam3={sam_n}  "
        f"balcony_tp={cover_gt_n}  box_tp={cover_box_n}  "
        f"P={_fmt(total['cover_precision'])}  R={_fmt(total['cover_recall'])}",
        flush=True,
    )
    print(
        f"TOTAL iou>={args.iou}  tp={tp_n}  "
        f"P={_fmt(total['precision'])}  R={_fmt(total['recall'])}  "
        f"meanIoU={_fmt(total['mean_iou'])}  areaIoU={_fmt(total['area_iou'])}",
        flush=True,
    )
    print(f"wrote {args.out}", flush=True)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
