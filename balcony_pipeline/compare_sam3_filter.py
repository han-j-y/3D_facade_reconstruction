"""SAM3 balcony boxes beside the post-filter boxes, one row per photo.

Left panel is every SAM3 ``balcony`` detection. Right panel is what
``filter_balcony_boxes`` keeps (decoration, window-sized boxes, and rooftop
rails). Juliet balconies stay on the right.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_pipeline/compare_sam3_filter.py --image-dir data/baseForBatch --dsl-dir runs/batch --out-dir runs/batch/_rows/baseForBatch_sam3_vs_filter
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
DEFAULT_OUT = ROOT / "runs" / "batch" / "_rows" / "baseForBatch_sam3_vs_filter"


def _row(left: Image.Image, right: Image.Image, *, height: int = 640, gap: int = 8) -> Image.Image:
    panels = []
    for panel in (left, right):
        w = max(1, int(round(panel.width * height / panel.height)))
        panels.append(panel.resize((w, height), Image.Resampling.BILINEAR))
    total_w = panels[0].width + gap + panels[1].width
    row = Image.new("RGB", (total_w, height), (30, 30, 30))
    row.paste(panels[0], (0, 0))
    row.paste(panels[1], (panels[0].width + gap, 0))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--min-side", type=int, default=16)
    ap.add_argument("--max-side-frac", type=float, default=1.0)
    ap.add_argument("--prompt", default="balcony")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

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
        raw = rp.detect_windows(
            facade,
            prompt=args.prompt,
            threshold=args.threshold,
            min_side=args.min_side,
            max_side_frac=args.max_side_frac,
            device=device,
            sam=sam,
            processor=processor,
        )
        raw_boxes = [r["box_xyxy"] for r in raw]
        kept, log = filter_balcony_boxes(
            raw_boxes, facade, dsl.get("instances") or []
        )
        dropped = [e for e in log if not e["keep"]]
        left = draw_boxes(
            facade,
            raw_boxes,
            title=f"SAM3 balcony  n={len(raw_boxes)}",
            color=(220, 40, 40),
        )
        right = draw_boxes(
            facade,
            kept,
            title=f"Filtered  kept={len(kept)}/{len(raw_boxes)}",
            color=(40, 180, 90),
        )
        dest = out_dir / f"{stem}.jpg"
        _row(left, right).save(dest, quality=90, optimize=True)
        row = {
            "stem": stem,
            "raw": len(raw_boxes),
            "kept": len(kept),
            "dropped": [
                {"box": e["box"], "reasons": e["reasons"]} for e in dropped
            ],
        }
        summary.append(row)
        print(
            f"{stem}  raw={len(raw_boxes)}  kept={len(kept)}  "
            f"drop={[e['reasons'] for e in dropped]}",
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
