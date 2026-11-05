#!/usr/bin/env python3
"""Stage overlays that call the existing pipeline without editing it.

Isolated venv (does not touch the env where the main code already runs)::

  python -m venv pipeline_preview/.venv
  pipeline_preview\\.venv\\Scripts\\pip install -r requirements.txt
  pipeline_preview\\.venv\\Scripts\\python pipeline_preview\\run_stages.py --image path.png

Outputs go to pipeline_preview/out/ (gitignored).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_pipeline as rp  # noqa: E402

OUT_DIR = HERE / "out"


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def detect_with_trace(
    image: Image.Image,
    *,
    prompt: str,
    threshold: float,
    min_side: int,
    max_side_frac: float,
    device,
) -> dict:
    """Same SAM3 + size filter as run_pipeline.detect_windows, plus raw boxes."""
    from transformers import Sam3Model, Sam3Processor

    iw, ih = image.size
    print("loading SAM3…")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    sam = Sam3Model.from_pretrained("facebook/sam3").to(device)
    sam.eval()
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    outputs = sam(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=0.5,
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]
    boxes = results["boxes"].detach().float().cpu().numpy()
    scores = results["scores"].detach().float().cpu().numpy()
    print(f"SAM3 raw detections: {len(boxes)}")
    del sam, outputs, inputs
    rp.torch.cuda.empty_cache()

    raw_all = []
    kept = []
    dropped = []
    for box, sc in zip(boxes, scores):
        x0, y0, x1, y1 = [float(v) for v in box]
        bw, bh = x1 - x0, y1 - y0
        rec = {
            "score": float(sc),
            "box_xyxy": [int(x0), int(y0), int(x1), int(y1)],
            "w": float(bw),
            "h": float(bh),
        }
        raw_all.append(rec)
        too_small = bw < min_side or bh < min_side
        too_big = bw > max_side_frac * iw or bh > max_side_frac * ih
        if too_small or too_big:
            rec = {
                **rec,
                "drop": "too_small" if too_small else "too_big",
            }
            dropped.append(rec)
            continue
        rec = {**rec, "idx": len(kept)}
        kept.append(rec)
    print(f"after size filter: {len(kept)} (dropped {len(dropped)})")
    return {"raw": raw_all, "kept": kept, "dropped": dropped}


def unitize_boxes(
    image: Image.Image,
    boxes: list[list[int]],
    *,
    row_tol: float,
    col_tol: float,
) -> dict:
    """Step 2 only: same merge as run_pipeline.cluster_units, no DINO."""
    iw, ih = image.size
    cx = np.array([0.5 * (b[0] + b[2]) / iw for b in boxes], dtype=np.float64)
    cy = np.array([0.5 * (b[1] + b[3]) / ih for b in boxes], dtype=np.float64)
    merged_boxes, members, bay_raw, floor_raw = rp.merge_mod.merge_adjacent_boxes(
        boxes,
        cx,
        cy,
        row_tol=row_tol,
        adj_gap=1.0,
        merge_bays=False,
        col_tol=col_tol,
    )
    bay_u = []
    for mem in members:
        labs = [int(bay_raw[i]) for i in mem]
        bay_u.append(max(set(labs), key=labs.count))
    mcy = np.array(
        [0.5 * (b[1] + b[3]) / ih for b in merged_boxes], dtype=np.float64
    )
    floor_m = rp.merge_mod.lay.assign_bays(mcy, row_tol)
    return {
        "merged_boxes": merged_boxes,
        "members": members,
        "bay": bay_u,
        "floor": [int(f) for f in floor_m.tolist()],
    }


def _banner(draw: ImageDraw.ImageDraw, text: str, fill: tuple[int, int, int]) -> None:
    draw.rectangle([8, 8, 8 + 11 * len(text), 40], fill=(20, 20, 20))
    draw.text((16, 12), text, fill=fill, font=_font(18))


def draw_boxes(
    image: Image.Image,
    records: list[dict],
    *,
    title: str,
    color: tuple[int, int, int],
    width: int = 3,
    scores: bool = False,
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font_s = _font(12)
    for rec in records:
        box = rec["box_xyxy"]
        draw.rectangle(box, outline=color, width=width)
        if scores and "score" in rec:
            draw.text((box[0] + 2, box[1] + 2), f"{rec['score']:.2f}", fill=color, font=font_s)
    _banner(draw, title, color)
    return canvas


def draw_detect(image: Image.Image, records: list[dict], n: int) -> Image.Image:
    return draw_boxes(
        image,
        records,
        title=f"Detect  prompt=window   n={n}",
        color=(220, 30, 30),
    )


def draw_unitize(
    image: Image.Image,
    raw_boxes: list[list[int]],
    clustered: dict,
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font_s = _font(13)
    for b in raw_boxes:
        draw.rectangle(b, outline=(220, 40, 40), width=1)
    units = clustered["merged_boxes"]
    for i, b in enumerate(units):
        draw.rectangle(b, outline=(0, 220, 255), width=4)
        nmem = len(clustered["members"][i])
        label = (
            f"U{i} F{clustered['floor'][i]} B{clustered['bay'][i]} n={nmem}"
        )
        tx, ty = b[0] + 3, max(0, b[1] - 18)
        draw.rectangle([tx, ty, tx + 8 * len(label), ty + 16], fill=(0, 0, 0))
        draw.text((tx + 2, ty), label, fill=(0, 220, 255), font=font_s)
    draw.rectangle([8, 8, 640, 40], fill=(20, 20, 20))
    draw.text(
        (16, 12),
        f"Unitize  {len(raw_boxes)} panes -> {len(units)} units  (red=detect, cyan=unit)",
        fill=(0, 220, 255),
        font=_font(16),
    )
    return canvas


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--from-index", type=Path, default=None)
    ap.add_argument("--prompt", type=str, default="window")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--min-side", type=int, default=24)
    ap.add_argument("--max-side-frac", type=float, default=0.55)
    ap.add_argument("--col-tol", type=float, default=0.04)
    ap.add_argument("--row-tol", type=float, default=0.055)
    ap.add_argument(
        "--device",
        default="cuda" if rp.torch.cuda.is_available() else "cpu",
    )
    ap.add_argument(
        "--until",
        choices=("detect", "unitize"),
        default="unitize",
        help="stop after this stage (detect = pipeline step 1 only)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem

    facade = Image.open(args.image).convert("RGB")
    input_path = out_dir / f"00_input_{stem}.png"
    facade.save(input_path)
    print(f"input -> {input_path}")

    if args.from_index is not None:
        facade, raw_windows, _ = rp.load_from_index(args.from_index)
        print(f"reusing index: {len(raw_windows)} windows")
        trace = {"raw": raw_windows, "kept": raw_windows, "dropped": []}
    else:
        trace = detect_with_trace(
            facade,
            prompt=args.prompt,
            threshold=args.threshold,
            min_side=args.min_side,
            max_side_frac=args.max_side_frac,
            device=rp.torch.device(args.device),
        )
        raw_windows = trace["kept"]

    raw_path = out_dir / f"01_sam3_raw_{stem}.png"
    draw_boxes(
        facade,
        trace["raw"],
        title=f"SAM3 raw  prompt={args.prompt!r}  thr={args.threshold}  n={len(trace['raw'])}",
        color=(255, 160, 0),
        scores=True,
    ).save(raw_path)
    print(f"sam3 raw n={len(trace['raw'])} -> {raw_path}")

    if trace["dropped"]:
        drop_path = out_dir / f"01b_size_dropped_{stem}.png"
        draw_boxes(
            facade,
            trace["dropped"],
            title=f"size-filter dropped  n={len(trace['dropped'])}",
            color=(180, 180, 180),
        ).save(drop_path)
        print(f"dropped n={len(trace['dropped'])} -> {drop_path}")

    raw_boxes = [w["box_xyxy"] for w in raw_windows]
    (out_dir / f"index_raw_{stem}.json").write_text(
        json.dumps(
            {
                "facade_path": str(Path(args.image).resolve()),
                "prompt": args.prompt,
                "threshold": args.threshold,
                "min_side": args.min_side,
                "max_side_frac": args.max_side_frac,
                "n_sam3_raw": len(trace["raw"]),
                "n_dropped": len(trace["dropped"]),
                "n_windows": len(raw_windows),
                "windows": raw_windows,
                "dropped": trace["dropped"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    detect_path = out_dir / f"02_detect_{stem}.png"
    draw_detect(facade, raw_windows, len(raw_windows)).save(detect_path)
    print(f"detect n={len(raw_windows)} -> {detect_path}")

    if args.until == "detect":
        print("stopped after detect")
        return

    clustered = unitize_boxes(
        facade, raw_boxes, row_tol=args.row_tol, col_tol=args.col_tol
    )
    unitize_path = out_dir / f"unitize_{stem}.png"
    draw_unitize(facade, raw_boxes, clustered).save(unitize_path)
    print(
        f"unitize {len(raw_boxes)} -> {len(clustered['merged_boxes'])} "
        f"-> {unitize_path}"
    )


if __name__ == "__main__":
    main()
