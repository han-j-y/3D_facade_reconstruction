"""Score SAM3 balcony boxes at each open-balcony pipeline stage against CMP labels.

Stages, as in ``run_balcony.py`` before the enclosed-balcony conflict step:

* raw: SAM3 detector boxes, prompt ``balcony``, score >= 0.45.
* sized: ``detect_windows`` output. The box is clamped to the mask, and boxes
  with a side under 16 px are dropped.
* filtered: ``filter_balcony_boxes`` (drop the smaller of two overlapping boxes).
* units: ``merge_adjacent_boxes(mode="balcony")`` with the ``run_balcony.py``
  defaults. Snapping to the window grid assigns floor and bay only and leaves
  the pixel box unchanged, so it is not run here.

``drop_open_under_enclosed`` is not applied; it needs the enclosed-cell
detections.

Each stage is scored with ``score_sam3_either_gt.score_image``: one-to-one
greedy IoU >= 0.5 against label fragments, against merged side-by-side groups,
and with each group counted whole or in fragments.

Boxes for every stage are cached in ``--cache``; a later run reuses them.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_pipeline/score_sam3_pipeline_stages.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from balcony_train.crop_balconies_xml import balcony_pixel_boxes, scale_box  # noqa: E402
from filter import filter_balcony_boxes  # noqa: E402
from score_sam3_either_gt import EXCLUDED, score_image  # noqa: E402

ROWS = ROOT / "runs" / "batch" / "_rows"
DEFAULT_DIR = ROOT / "data" / "extended_selected"
DEFAULT_CACHE = ROWS / "extended_selected_sam3_stage_boxes.json"
DEFAULT_OUT = ROWS / "extended_selected_sam3_pipeline_stages.json"
STAGES = ("raw", "sized", "filtered", "units")


def unit_boxes(boxes: list[list[int]], size: tuple[int, int], row_tol: float, col_tol: float) -> list[list[int]]:
    import run_pipeline as rp

    if not boxes:
        return []
    iw, ih = size
    cx = np.array([0.5 * (b[0] + b[2]) / iw for b in boxes], dtype=np.float64)
    cy = np.array([0.5 * (b[1] + b[3]) / ih for b in boxes], dtype=np.float64)
    merged, _members, _bay, _floor = rp.merge_mod.merge_adjacent_boxes(
        boxes,
        cx,
        cy,
        row_tol=row_tol,
        adj_gap=1.0,
        merge_bays=False,
        col_tol=col_tol,
        mode="balcony",
    )
    return [[int(v) for v in b] for b in merged]


def detect_stages(images: list[Path], cache: Path, args: argparse.Namespace) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if cache.is_file():
        done = json.loads(cache.read_text(encoding="utf-8"))
    todo = [p for p in images if p.stem not in done]
    if not todo:
        return done

    import torch
    from transformers import Sam3Model, Sam3Processor

    import run_pipeline as rp

    device = torch.device(args.device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    sam = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
    for k, path in enumerate(todo, 1):
        photo = Image.open(path).convert("RGB")
        raw: list[list[int]] = []
        records = rp.detect_windows(
            photo,
            prompt="balcony",
            threshold=args.threshold,
            min_side=16,
            max_side_frac=1.0,
            device=device,
            sam=sam,
            processor=processor,
            det_boxes_out=raw,
        )
        sized = [[int(v) for v in r["box_xyxy"]] for r in records]
        filtered, _log = filter_balcony_boxes(sized, photo, [])
        done[path.stem] = {
            "raw": [[int(v) for v in b] for b in raw],
            "sized": sized,
            "filtered": [[int(v) for v in b] for b in filtered],
            "size": list(photo.size),
        }
        print(
            f"[{k}/{len(todo)}] {path.stem}  raw={len(raw)}  sized={len(sized)}  filtered={len(filtered)}",
            flush=True,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(done), encoding="utf-8")
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--exclude", default=EXCLUDED + " cmp_x0009 cmp_x0047")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--row-tol", type=float, default=0.055)
    ap.add_argument("--col-tol", type=float, default=0.04)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    excluded = set(args.exclude.replace(",", " ").split())
    images = sorted(p for p in args.image_dir.glob("*.jpg") if p.stem not in excluded)
    boxes = detect_stages(images, args.cache, args)

    rows = []
    for path in images:
        label_path = args.label_dir / f"{path.stem}.png"
        if not label_path.is_file():
            print(f"skip {path.stem}: missing {label_path}", flush=True)
            continue
        entry = boxes[path.stem]
        width, height = entry["size"]
        label = Image.open(label_path)
        gt = [
            list(scale_box(b, src_w=label.size[0], src_h=label.size[1], dst_w=width, dst_h=height))
            for b in balcony_pixel_boxes(label)
        ]
        stage_boxes = {
            "raw": entry["raw"],
            "sized": entry["sized"],
            "filtered": entry["filtered"],
            "units": unit_boxes(entry["filtered"], (width, height), args.row_tol, args.col_tol),
        }
        row = {"stem": path.stem}
        for stage in STAGES:
            scored = score_image(gt, stage_boxes[stage], args.iou)
            row[stage] = {
                "sam3": scored["sam3"],
                "fragments": scored["fragments"],
                "merged": scored["merged"],
                "either": scored["either"],
                "groups_counted_whole": scored["groups_counted_whole"],
            }
        rows.append(row)

    total: dict = {"images": len(rows), "iou_threshold": args.iou, "excluded": sorted(excluded)}
    print(f"images={len(rows)}", flush=True)
    for stage in STAGES:
        sam_n = sum(r[stage]["sam3"] for r in rows)
        total[stage] = {"sam3": sam_n}
        for key in ("fragments", "merged", "either"):
            gt_n = sum(r[stage][key]["gt"] for r in rows)
            tp_n = sum(r[stage][key]["tp"] for r in rows)
            p = tp_n / sam_n if sam_n else 0.0
            r_ = tp_n / gt_n if gt_n else 0.0
            f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
            total[stage][key] = {
                "gt": gt_n,
                "tp": tp_n,
                "fp": sam_n - tp_n,
                "fn": gt_n - tp_n,
                "precision": p,
                "recall": r_,
                "f1": f1,
            }
            print(
                f"{stage:8s} {key:9s} gt={gt_n}  pred={sam_n}  tp={tp_n}  fp={sam_n - tp_n}  "
                f"fn={gt_n - tp_n}  P={p:.3f}  R={r_:.3f}  F1={f1:.3f}",
                flush=True,
            )
    args.out.write_text(json.dumps({"total": total, "images": rows}, indent=1), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
