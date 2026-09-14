"""Re-crop images with SAM3 max-score ``balcony`` box (pipeline-aligned).

Loads ``facebook/sam3`` once, runs the same text prompt / threshold / size
filter pattern as ``run_pipeline.detect_windows``, then keeps the highest
score box and writes a PIL crop.

Typical use::

    python balcony_train/recrop_sam3.py --device cuda \\
      --in-dir "runs/balcony_clf/Before crops" \\
      --out-dir runs/balcony_clf/crops/unlabeled

    python balcony_train/recrop_sam3.py --device cuda
    # (defaults: unlabeled -> unlabeled, pattern = all png/jpg/jpeg/webp)

Originals can be backed up with ``--backup-dir`` (or crops-dir mode:
``unlabeled_precrop/``). Bad detections can be dropped later in Label UI.
"""

from __future__ import annotations

import argparse
import shutil
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

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def clamp_box_xyxy(
    box: list[int] | tuple[int, int, int, int],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(v) for v in box]
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return x0, y0, x1, y1


def pick_max_score_record(records: list[dict]) -> dict | None:
    if not records:
        return None
    return max(records, key=lambda r: float(r.get("score", 0.0)))


def filter_records(
    boxes,
    scores,
    *,
    iw: int,
    ih: int,
    min_side: int,
    max_side_frac: float,
) -> list[dict]:
    """Same size filter as ``run_pipeline.detect_windows``."""
    records: list[dict] = []
    for box, sc in zip(boxes, scores):
        x0, y0, x1, y1 = [float(v) for v in box]
        bw, bh = x1 - x0, y1 - y0
        if bw < min_side or bh < min_side:
            continue
        if bw > max_side_frac * iw or bh > max_side_frac * ih:
            continue
        records.append(
            {
                "score": float(sc),
                "box_xyxy": [int(x0), int(y0), int(x1), int(y1)],
            }
        )
    return records


@torch.no_grad()
def detect_balcony_records(
    image: Image.Image,
    *,
    processor,
    sam,
    device: torch.device,
    prompt: str,
    threshold: float,
    min_side: int,
    max_side_frac: float,
) -> list[dict]:
    iw, ih = image.size
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
    return filter_records(
        boxes,
        scores,
        iw=iw,
        ih=ih,
        min_side=min_side,
        max_side_frac=max_side_frac,
    )


def list_input_images(in_dir: Path, pattern: str = "") -> list[Path]:
    """List images under ``in_dir``.

    Empty / ``*`` / ``all`` → every ``png|jpg|jpeg|webp``.
    Otherwise ``Path.glob(pattern)``, then keep only those suffixes.
    """
    in_dir = Path(in_dir)
    if not in_dir.is_dir():
        return []
    pat = (pattern or "").strip()
    if not pat or pat in {"*", "all", "*.*"}:
        candidates = list(in_dir.iterdir())
    else:
        candidates = list(in_dir.glob(pat))
    uniq = {
        p.resolve(): p
        for p in candidates
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }
    return sorted(uniq.values(), key=lambda p: p.name.lower())


def recrop_images(
    in_dir: Path,
    out_dir: Path,
    *,
    pattern: str = "",
    backup_dir: Path | None = None,
    prompt: str = "balcony",
    threshold: float = 0.45,
    min_side: int = 16,
    max_side_frac: float = 1.0,
    device: str = "cuda",
) -> dict[str, int]:
    paths = list_input_images(in_dir, pattern)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if backup_dir is not None:
        Path(backup_dir).mkdir(parents=True, exist_ok=True)

    if not paths:
        return {"n": 0, "cropped": 0, "skipped": 0, "backed_up": 0}

    from transformers import Sam3Model, Sam3Processor

    torch_device = torch.device(device)
    print(f"input={in_dir}  n={len(paths)}  pattern={pattern!r}", flush=True)
    print("loading SAM3…", flush=True)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    sam = Sam3Model.from_pretrained("facebook/sam3").to(torch_device)
    sam.eval()

    cropped = 0
    skipped = 0
    backed_up = 0
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
            best = pick_max_score_record(records)
            if best is None:
                print(f"[{i}/{len(paths)}] skip (no box): {path.name}", flush=True)
                skipped += 1
                continue

            box = clamp_box_xyxy(best["box_xyxy"], width=image.size[0], height=image.size[1])
            if backup_dir is not None:
                dest_bak = Path(backup_dir) / path.name
                if not dest_bak.exists():
                    shutil.copy2(path, dest_bak)
                    backed_up += 1

            crop = image.crop(box)
            out_path = out_dir / path.name
            save_kw: dict = {}
            if out_path.suffix.lower() in {".jpg", ".jpeg"}:
                save_kw["quality"] = 95
            crop.save(out_path, **save_kw)
            cropped += 1
            print(
                f"[{i}/{len(paths)}] {path.name}  "
                f"score={best['score']:.3f}  box={list(box)}  "
                f"{image.size[0]}x{image.size[1]} -> {crop.size[0]}x{crop.size[1]}",
                flush=True,
            )
    finally:
        del sam
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "n": len(paths),
        "cropped": cropped,
        "skipped": skipped,
        "backed_up": backed_up,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-dir",
        type=Path,
        default=None,
        help="absolute/relative input folder (overrides --crops-dir/--in-subdir)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="absolute/relative output folder (overrides --crops-dir/--out-subdir)",
    )
    ap.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="absolute/relative backup folder for originals (optional)",
    )
    ap.add_argument(
        "--crops-dir",
        type=Path,
        default=DEFAULT_CROPS_DIR,
        help="label root when --in-dir/--out-dir omitted (default: runs/balcony_clf/crops)",
    )
    ap.add_argument(
        "--in-subdir",
        default="unlabeled",
        help="subdir under crops-dir to read (default: unlabeled)",
    )
    ap.add_argument(
        "--out-subdir",
        default="unlabeled",
        help="subdir under crops-dir to write crops (default: unlabeled)",
    )
    ap.add_argument(
        "--backup-subdir",
        default="unlabeled_precrop",
        help="backup originals under crops-dir when not using --backup-dir",
    )
    ap.add_argument(
        "--no-backup",
        action="store_true",
        help="do not copy originals before overwrite",
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
        help="SAM3 score threshold (pipeline window/balcony default 0.45)",
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
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.in_dir is not None:
        in_dir = Path(args.in_dir)
    else:
        ensure_crop_dirs(args.crops_dir)
        in_dir = Path(args.crops_dir) / args.in_subdir

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        ensure_crop_dirs(args.crops_dir)
        out_dir = Path(args.crops_dir) / args.out_subdir

    backup_dir: Path | None
    if args.no_backup:
        backup_dir = None
    elif args.backup_dir is not None:
        backup_dir = Path(args.backup_dir)
    elif args.in_dir is not None or args.out_dir is not None:
        # Absolute-dir mode: only backup if --backup-dir was set.
        backup_dir = None
    elif not str(args.backup_subdir).strip():
        backup_dir = None
    else:
        backup_dir = Path(args.crops_dir) / str(args.backup_subdir)

    stats = recrop_images(
        in_dir,
        out_dir,
        pattern=str(args.pattern),
        backup_dir=backup_dir,
        prompt=str(args.prompt),
        threshold=float(args.threshold),
        min_side=int(args.min_side),
        max_side_frac=float(args.max_side_frac),
        device=str(args.device),
    )
    print(
        f"done n={stats['n']} cropped={stats['cropped']} "
        f"skipped={stats['skipped']} backed_up={stats['backed_up']} "
        f"-> {out_dir}"
    )
    print("Review / drop bad crops in Label UI, then train.")


if __name__ == "__main__":
    main()
