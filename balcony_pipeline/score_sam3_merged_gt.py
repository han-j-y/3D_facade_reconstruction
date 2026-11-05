"""SAM3 balcony boxes against CMP balcony boxes, with side-by-side GT boxes merged.

Ground truth is the 8-connected balcony regions of ``data/extended/<stem>.png``
(``balcony_pixel_boxes``), the same as ``score_sam3_label_iou.py``. Before
scoring, two GT boxes become one box when they sit side by side (their vertical
overlap is at least half the lower box's height) and the horizontal gap between
them is at most the lower box's height. Merging repeats until nothing changes.

Prediction is every raw SAM3 ``balcony`` box at the score threshold. Boxes are
cached in ``--cache`` so later runs skip SAM3.

Two criteria, pooled over the images:

* inside: a SAM3 box is TP when at least 60% of its area lies inside a GT box;
  a GT box is detected when it holds such a box (as ``score_sam3_label_iou.py``).
* IoU: one-to-one greedy matching by IoU, TP when IoU >= 0.5.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_pipeline/score_sam3_merged_gt.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from balcony_train.crop_balconies_xml import balcony_pixel_boxes, scale_box  # noqa: E402
from score_sam3_label_iou import inside_counts, match_iou  # noqa: E402

DEFAULT_IMAGE_DIR = ROOT / "data" / "extended_selected"
DEFAULT_LABEL_DIR = ROOT / "data" / "extended"
DEFAULT_CACHE = ROOT / "runs" / "batch" / "_rows" / "extended_selected_sam3_boxes.json"
DEFAULT_OUT = ROOT / "runs" / "batch" / "_rows" / "extended_selected_sam3_merged_gt.json"
# The 125-image set of extended_selected_sam3_*_125.xlsx.
EXCLUDED = (
    "cmp_x0002 cmp_x0003 cmp_x0006 cmp_x0008 cmp_x0011 cmp_x0012 cmp_x0014 cmp_x0016 cmp_x0021 cmp_x0036 "
    "cmp_x0043 cmp_x0044 cmp_x0067 cmp_x0107 cmp_x0116 cmp_x0127 cmp_x0162 cmp_x0173 cmp_x0176 cmp_x0227"
)


def side_by_side(a: list[int], b: list[int]) -> bool:
    """Same row, and the horizontal gap is at most the lower box's height."""
    low = min(a[3] - a[1], b[3] - b[1])
    if low <= 0:
        return False
    v_overlap = min(a[3], b[3]) - max(a[1], b[1])
    if v_overlap < 0.5 * low:
        return False
    gap = max(a[0], b[0]) - min(a[2], b[2])
    return gap <= low


def merge_side_by_side(boxes: list[list[int]]) -> list[list[int]]:
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if side_by_side(boxes[i], boxes[j]):
                    a, b = boxes[i], boxes[j]
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def sam3_boxes(images: list[Path], cache: Path, args: argparse.Namespace) -> dict[str, list[list[int]]]:
    done: dict[str, list[list[int]]] = {}
    if cache.is_file():
        done = json.loads(cache.read_text(encoding="utf-8"))
    todo = [p for p in images if p.stem not in done]
    if todo:
        import run_pipeline as rp
        from transformers import Sam3Model, Sam3Processor

        device = torch.device(args.device)
        processor = Sam3Processor.from_pretrained("facebook/sam3")
        sam = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
        for k, path in enumerate(todo, 1):
            det: list[list[int]] = []
            rp.detect_windows(
                Image.open(path).convert("RGB"), prompt="balcony", threshold=args.threshold,
                min_side=16, max_side_frac=1.0, device=device, sam=sam, processor=processor,
                det_boxes_out=det,
            )
            done[path.stem] = [list(map(int, b)) for b in det]
            print(f"[{k}/{len(todo)}] {path.stem} sam3={len(det)}", flush=True)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(done), encoding="utf-8")
    return done


def score(gt: list[list[int]], pred: list[list[int]], cover: float, iou: float) -> dict[str, int]:
    in_tp, in_fp, _in_fn, detected = inside_counts(gt, pred, threshold=cover)
    tp, _best = match_iou(gt, pred, threshold=iou)
    return {"gt": len(gt), "sam3": len(pred), "in_tp": in_tp, "in_fp": in_fp, "in_det": detected, "iou_tp": tp}


def rates(c: dict[str, int]) -> dict[str, float]:
    p_in = c["in_tp"] / max(c["sam3"], 1)
    r_in = c["in_det"] / max(c["gt"], 1)
    p_iou = c["iou_tp"] / max(c["sam3"], 1)
    r_iou = c["iou_tp"] / max(c["gt"], 1)
    f = lambda p, r: 2 * p * r / max(p + r, 1e-12)  # noqa: E731
    return {"in_P": p_in, "in_R": r_in, "in_F1": f(p_in, r_in), "iou_P": p_iou, "iou_R": r_iou, "iou_F1": f(p_iou, r_iou)}


def draw(photo: Image.Image, raw: list[list[int]], merged: list[list[int]], pred: list[list[int]]) -> Image.Image:
    im = photo.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    w = max(2, im.width // 350)
    for b in raw:
        d.rectangle(b, outline=(255, 255, 0), width=w)
    for b in merged:
        d.rectangle(b, outline=(0, 230, 0), width=w + 2)
    for b in pred:
        d.rectangle(b, outline=(230, 40, 40), width=w)
    return im


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--exclude", default=EXCLUDED)
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--cover", type=float, default=0.6)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--panels-dir", type=Path, default=None, help="photo with GT (yellow), merged GT (green), SAM3 (red)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    excluded = set(args.exclude.split())
    images = sorted(p for p in args.image_dir.glob("*.jpg") if p.stem not in excluded)
    preds = sam3_boxes(images, args.cache, args)

    rows = []
    tot = {"before": {}, "after": {}}
    for path in images:
        label_path = args.label_dir / f"{path.stem}.png"
        if not label_path.is_file():
            continue
        photo = Image.open(path)
        label = Image.open(label_path)
        gt = [list(scale_box(b, src_w=label.size[0], src_h=label.size[1], dst_w=photo.width, dst_h=photo.height))
              for b in balcony_pixel_boxes(label)]
        merged = merge_side_by_side(gt)
        pred = preds[path.stem]
        row = {"stem": path.stem, "before": score(gt, pred, args.cover, args.iou),
               "after": score(merged, pred, args.cover, args.iou)}
        rows.append(row)
        for k in ("before", "after"):
            for kk, v in row[k].items():
                tot[k][kk] = tot[k].get(kk, 0) + v
        if args.panels_dir is not None and len(merged) < len(gt):
            args.panels_dir.mkdir(parents=True, exist_ok=True)
            draw(photo, gt, merged, pred).save(args.panels_dir / f"{path.stem}.jpg", quality=88)

    lines = [f"images={len(rows)}  cover>={args.cover}  iou>={args.iou}  score>={args.threshold}", ""]
    lines.append("| GT boxes | GT | SAM3 | inside TP | inside FP | detected GT | P (inside) | R (inside) | F1 (inside) | IoU TP | P (IoU) | R (IoU) | F1 (IoU) |")
    lines.append("|---" * 13 + "|")
    for k, name in (("before", "as labeled"), ("after", "side-by-side merged")):
        c, r = tot[k], rates(tot[k])
        lines.append(
            f"| {name} | {c['gt']} | {c['sam3']} | {c['in_tp']} | {c['in_fp']} | {c['in_det']} | "
            f"{r['in_P']:.3f} | {r['in_R']:.3f} | {r['in_F1']:.3f} | {c['iou_tp']} | "
            f"{r['iou_P']:.3f} | {r['iou_R']:.3f} | {r['iou_F1']:.3f} |"
        )
    changed = [r["stem"] for r in rows if r["after"]["gt"] < r["before"]["gt"]]
    lines += ["", f"photos with merged GT: {len(changed)}/{len(rows)}"]
    text = "\n".join(lines)
    print(text)
    args.out.write_text(json.dumps({"total": tot, "rates": {k: rates(v) for k, v in tot.items()},
                                    "images": rows, "changed": changed}, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
