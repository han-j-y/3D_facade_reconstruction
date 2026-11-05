"""Four panels per photo: CMP label map, raw SAM3, size filter, balcony filter.

The label map is ``<stem>.png`` from ``--label-dir`` (CMP palette, shown in color).
Next is every detector box from the SAM3 ``balcony`` call (score >= threshold).
Then those boxes tightened to the mask, with short sides dropped. The right
panel drops only the smaller of two overlapping balcony boxes. A box below a
window sill stays, and a box with no window above it stays.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_pipeline/compare_sam3_filter.py --image-dir data/baseForBatch --dsl-dir runs/batch --out-dir runs/batch/_rows/baseForBatch_sam3_vs_filter_v3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_pipeline as rp  # noqa: E402

from draw import draw_boxes  # noqa: E402
from filter import filter_balcony_boxes  # noqa: E402

DEFAULT_IMAGE_DIR = ROOT / "data" / "baseForBatch"
DEFAULT_DSL_DIR = ROOT / "runs" / "batch"
DEFAULT_LABEL_DIR = ROOT / "data" / "extended"
DEFAULT_OUT = ROOT / "runs" / "batch" / "_rows" / "baseForBatch_sam3_vs_filter_v3"
DEFAULT_ROWS = ROOT / "runs" / "batch" / "_rows" / "baseForBatch_sam3_vs_filter_v3"


def _row(panels: list[Image.Image], *, height: int = 640, gap: int = 8) -> Image.Image:
    fitted = []
    for panel in panels:
        w = max(1, int(round(panel.width * height / panel.height)))
        fitted.append(panel.resize((w, height), Image.Resampling.BILINEAR))
    total_w = sum(p.width for p in fitted) + gap * (len(fitted) - 1)
    row = Image.new("RGB", (total_w, height), (30, 30, 30))
    x = 0
    for panel in fitted:
        row.paste(panel, (x, 0))
        x += panel.width + gap
    return row


def segmentation_image(stem: str, label_dir: Path) -> Image.Image | None:
    """CMP palette PNG for ``stem``, as an RGB panel with a title."""
    path = Path(label_dir) / f"{stem}.png"
    if not path.is_file():
        return None
    label = Image.open(path).convert("RGB")
    return draw_boxes(label, [], title="Segmentation", color=(255, 255, 255))


def attach_saved_rows(rows_dir: Path, out_dir: Path, label_dir: Path) -> None:
    """Paste the label map on the left of comparison JPGs already written."""
    rows_dir = Path(rows_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(p for p in rows_dir.glob("*.jpg") if p.is_file())
    if not sources:
        raise SystemExit(f"no jpg in {rows_dir}")
    missing: list[str] = []
    for src in sources:
        row = Image.open(src).convert("RGB")
        seg = segmentation_image(src.stem, label_dir)
        if seg is None:
            missing.append(src.stem)
            _row([row]).save(out_dir / src.name, quality=90, optimize=True)
            print(f"{src.stem}  no segmentation", flush=True)
            continue
        seg_w = max(1, int(round(seg.width * row.height / seg.height)))
        seg_r = seg.resize((seg_w, row.height), Image.Resampling.BILINEAR)
        gap = 8
        canvas = Image.new("RGB", (seg_r.width + gap + row.width, row.height), (30, 30, 30))
        canvas.paste(seg_r, (0, 0))
        canvas.paste(row, (seg_r.width + gap, 0))
        canvas.save(out_dir / src.name, quality=90, optimize=True)
        print(f"{src.stem}  + segmentation", flush=True)
    summary_src = rows_dir / "summary.json"
    if summary_src.is_file():
        (out_dir / "summary.json").write_text(summary_src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"wrote {len(sources)} rows -> {out_dir}", flush=True)
    if missing:
        print(f"missing segmentation: {', '.join(missing)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--label-dir",
        type=Path,
        default=DEFAULT_LABEL_DIR,
        help="CMP label PNGs named <stem>.png (palette). Empty file skips that panel.",
    )
    ap.add_argument(
        "--attach-rows",
        type=Path,
        default=None,
        help="skip SAM3; paste --label-dir maps onto comparison JPGs in this folder",
    )
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--min-side", type=int, default=16)
    ap.add_argument("--max-side-frac", type=float, default=1.0)
    ap.add_argument("--prompt", default="balcony")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.attach_rows is not None:
        attach_saved_rows(args.attach_rows, args.out_dir, args.label_dir)
        return

    images = sorted(p for p in Path(args.image_dir).glob("*.jpg") if p.is_file())
    if not images:
        raise SystemExit(f"no jpg in {args.image_dir}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"{len(images)} images  device={device}", flush=True)

    from transformers import Sam3Model, Sam3Processor

    print("loading SAM3…", flush=True)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    sam = Sam3Model.from_pretrained("facebook/sam3").to(device)
    sam.eval()

    summary: list[dict] = []
    for image_path in images:
        stem = image_path.stem
        dsl_path = Path(args.dsl_dir) / stem / "facade_dsl.json"
        if not dsl_path.is_file():
            print(f"skip {stem}: missing {dsl_path}", flush=True)
            summary.append({"stem": stem, "error": "missing dsl"})
            continue
        dsl = json.loads(dsl_path.read_text(encoding="utf-8"))
        facade = Image.open(image_path).convert("RGB")
        det_boxes: list[list[int]] = []
        sized = rp.detect_windows(
            facade,
            prompt=args.prompt,
            threshold=args.threshold,
            min_side=args.min_side,
            max_side_frac=args.max_side_frac,
            device=device,
            sam=sam,
            processor=processor,
            det_boxes_out=det_boxes,
        )
        sized_boxes = [r["box_xyxy"] for r in sized]
        kept, log = filter_balcony_boxes(
            sized_boxes, facade, dsl.get("instances") or []
        )
        dropped = [e for e in log if not e["keep"]]
        pure = draw_boxes(
            facade,
            det_boxes,
            title=f"SAM3 raw  n={len(det_boxes)}",
            color=(220, 40, 40),
        )
        middle = draw_boxes(
            facade,
            sized_boxes,
            title=f"SAM3 size  n={len(sized_boxes)}",
            color=(40, 120, 220),
        )
        right = draw_boxes(
            facade,
            kept,
            title=f"Filtered  kept={len(kept)}/{len(sized_boxes)}",
            color=(40, 180, 90),
        )
        dest = out_dir / f"{stem}.jpg"
        panels = [pure, middle, right]
        seg = segmentation_image(stem, args.label_dir)
        if seg is None:
            print(f"  no segmentation for {stem}", flush=True)
        else:
            panels.insert(0, seg)
        _row(panels).save(dest, quality=90, optimize=True)
        row = {
            "stem": stem,
            "raw": len(det_boxes),
            "size": len(sized_boxes),
            "kept": len(kept),
            "dropped": [
                {"box": e["box"], "reasons": e["reasons"]} for e in dropped
            ],
        }
        summary.append(row)
        print(
            f"{stem}  raw={len(det_boxes)}  size={len(sized_boxes)}  "
            f"kept={len(kept)}  drop={[e['reasons'] for e in dropped]}",
            flush=True,
        )

    del sam, processor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    report = out_dir / "summary.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {len(summary)} rows -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
