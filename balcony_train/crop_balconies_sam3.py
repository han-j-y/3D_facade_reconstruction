"""Crop every SAM3 ``balcony`` detection from facade images into unlabeled/.

Unlike ``recrop_sam3.py`` (one max-score box per image), this writes **all**
filtered boxes from each facade photo — intended for ``data/base`` bulk collect.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe ^
      balcony_train/crop_balconies_sam3.py --device cuda

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe ^
      balcony_train/crop_balconies_sam3.py ^
      --in-dir data/base ^
      --out-dir runs/balcony_clf/crops/unlabeled ^
      --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.labels import ensure_crop_dirs  # noqa: E402
from balcony_train.paths import DEFAULT_CROPS_DIR  # noqa: E402
from balcony_train.recrop_sam3 import (  # noqa: E402
    clamp_box_xyxy,
    detect_balcony_records,
    list_input_images,
)

DEFAULT_IN_DIR = ROOT / "data" / "base"
DEFAULT_OUT_DIR = DEFAULT_CROPS_DIR / "unlabeled"


def sort_records(records: list[dict]) -> list[dict]:
    """Stable order: higher score first, then top-to-bottom, left-to-right."""
    return sorted(
        records,
        key=lambda r: (
            -float(r.get("score", 0.0)),
            int(r["box_xyxy"][1]),
            int(r["box_xyxy"][0]),
        ),
    )


def crop_output_name(stem: str, index: int) -> str:
    return f"{stem}_bal_{index:03d}.png"


def crop_all_balconies(
    in_dir: Path,
    out_dir: Path,
    *,
    pattern: str = "",
    prompt: str = "balcony",
    threshold: float = 0.45,
    min_side: int = 16,
    max_side_frac: float = 1.0,
    device: str = "cuda",
) -> dict[str, int]:
    paths = list_input_images(in_dir, pattern)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not paths:
        return {
            "n_images": 0,
            "n_with_boxes": 0,
            "n_crops": 0,
            "n_skipped": 0,
        }

    from transformers import Sam3Model, Sam3Processor

    torch_device = torch.device(device)
    print(f"input={in_dir}  n={len(paths)}  pattern={pattern!r}", flush=True)
    print(f"out={out_dir}", flush=True)
    print("loading SAM3…", flush=True)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    sam = Sam3Model.from_pretrained("facebook/sam3").to(torch_device)
    sam.eval()

    n_with_boxes = 0
    n_crops = 0
    n_skipped = 0
    try:
        for i, path in enumerate(paths, start=1):
            image = Image.open(path).convert("RGB")
            records = detect_balcony_records(
                image,
                processor=processor,
                sam=sam,
                device=torch_device,
                prompt=prompt,
                threshold=threshold,
                min_side=min_side,
                max_side_frac=max_side_frac,
            )
            records = sort_records(records)
            if not records:
                print(f"[{i}/{len(paths)}] skip (no box): {path.name}", flush=True)
                n_skipped += 1
                continue

            n_with_boxes += 1
            stem = path.stem
            for j, rec in enumerate(records):
                box = clamp_box_xyxy(
                    rec["box_xyxy"], width=image.size[0], height=image.size[1]
                )
                crop = image.crop(box)
                out_path = out_dir / crop_output_name(stem, j)
                crop.save(out_path)
                n_crops += 1
            print(
                f"[{i}/{len(paths)}] {path.name}  boxes={len(records)}  "
                f"-> {stem}_bal_000..{len(records) - 1:03d}.png",
                flush=True,
            )
    finally:
        del sam
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "n_images": len(paths),
        "n_with_boxes": n_with_boxes,
        "n_crops": n_crops,
        "n_skipped": n_skipped,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-dir",
        type=Path,
        default=DEFAULT_IN_DIR,
        help=f"facade image folder (default: {DEFAULT_IN_DIR})",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"crop output folder (default: {DEFAULT_OUT_DIR})",
    )
    ap.add_argument(
        "--pattern",
        default="",
        help="glob under input dir; empty/all/* = every png|jpg|jpeg|webp",
    )
    ap.add_argument("--prompt", default="balcony", help="SAM3 text prompt")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="SAM3 score threshold (default 0.45; try 0.35 if few boxes)",
    )
    ap.add_argument("--min-side", type=int, default=16)
    ap.add_argument(
        "--max-side-frac",
        type=float,
        default=1.0,
        help="drop boxes larger than this fraction of image W/H",
    )
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    ensure_crop_dirs(DEFAULT_CROPS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.is_dir():
        print(f"error: input dir not found: {in_dir}", file=sys.stderr)
        return 1

    stats = crop_all_balconies(
        in_dir,
        out_dir,
        pattern=str(args.pattern),
        prompt=str(args.prompt),
        threshold=float(args.threshold),
        min_side=int(args.min_side),
        max_side_frac=float(args.max_side_frac),
        device=str(args.device),
    )
    print(
        f"done images={stats['n_images']} with_boxes={stats['n_with_boxes']} "
        f"crops={stats['n_crops']} skipped={stats['n_skipped']} -> {out_dir}"
    )
    print("Review / drop bad crops in Label UI (or label_from_prefix for flux2_*).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
