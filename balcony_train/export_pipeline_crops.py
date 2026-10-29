"""Cut open-balcony crops exactly as the pipeline does, for hand labels on all of extended_selected.

Boxes are the pipeline's: SAM3 ``balcony`` at score >= 0.45, clamped to the
mask, sides under 16 px dropped, then ``filter_balcony_boxes``. The crop is
``photo.crop(box)`` with no margin, as ``run_balcony.py`` feeds the railing
classifier.

The 123-photo stage cache of ``score_sam3_pipeline_stages.py`` is copied into a
145-photo cache and SAM3 runs only on the missing photos. ``filtered`` is
recomputed for every photo from the cached ``sized`` boxes with the current
``filter.py``. The script stops if that differs from the cached boxes, which
would mean the filter changed since the cache was made.

Outputs under ``--out-dir``:

* ``unlabeled/<stem>_pbal_<NNN>.png``  the crops
* ``context/<crop stem>.jpg``           the photo with that box drawn, for labeling
* ``manifest.jsonl``                    crop, photo, box, photo size, cache, filter.py sha256
* ``prefill.jsonl``                     earlier hand labels from ``crops_ext_xml`` matched by
                                        box IoU (>= 0.3, best match). Starting values only.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/export_pipeline_crops.py --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "balcony_pipeline"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from filter import filter_balcony_boxes  # noqa: E402
from score_sam3_label_iou import box_iou  # noqa: E402
from score_sam3_pipeline_stages import detect_stages  # noqa: E402

ROWS = ROOT / "runs" / "batch" / "_rows"
DEFAULT_IMAGE_DIR = ROOT / "data" / "extended_selected"
DEFAULT_CACHE_123 = ROWS / "extended_selected_sam3_stage_boxes.json"
DEFAULT_CACHE = ROWS / "extended_selected_sam3_stage_boxes_145.json"
DEFAULT_OUT = ROOT / "runs" / "balcony_clf" / "crops_pipeline_ext"
DEFAULT_PREFILL_SRC = ROOT / "runs" / "balcony_clf" / "crops_ext_xml"
PREFILL_IOU = 0.3
CONTEXT_SIDE = 900
PREFILL_FIELDS = ("kind", "material", "floor_shape", "enclosure", "structure")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def context_image(photo: Image.Image, box: list[int]) -> Image.Image:
    scale = CONTEXT_SIDE / max(photo.size)
    im = photo.resize((round(photo.width * scale), round(photo.height * scale)), Image.Resampling.LANCZOS)
    b = [round(v * scale) for v in box]
    d = ImageDraw.Draw(im)
    d.rectangle(b, outline=(255, 255, 255), width=6)
    d.rectangle(b, outline=(230, 30, 30), width=3)
    return im


def load_prefill_source(src: Path) -> dict[str, list[tuple[list[int], dict]]]:
    """Photo stem -> [(box, labels)] from the segmentation-crop label set."""
    labels = {}
    for line in (src / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            labels[Path(rec["path"]).stem] = rec
    out: dict[str, list[tuple[list[int], dict]]] = {}
    for line in (src / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        rec = labels.get(Path(m["crop"]).stem)
        if rec is None:
            continue
        out.setdefault(Path(m["photo"]).stem, []).append((m["box_xyxy"], rec))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    ap.add_argument("--cache-123", type=Path, default=DEFAULT_CACHE_123)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--prefill-src", type=Path, default=DEFAULT_PREFILL_SRC)
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="overwrite an existing out-dir")
    args = ap.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise SystemExit(f"{args.out_dir} is not empty; labels may live there. Use --force to rebuild.")
    images = sorted(args.image_dir.glob("*.jpg"))
    if not args.cache.is_file():
        shutil.copyfile(args.cache_123, args.cache)
    stages = detect_stages(images, args.cache, args)

    old = json.loads(args.cache_123.read_text(encoding="utf-8"))
    mismatch = []
    filtered: dict[str, list[list[int]]] = {}
    for path in images:
        photo = Image.open(path).convert("RGB")
        kept, _log = filter_balcony_boxes(stages[path.stem]["sized"], photo, [])
        filtered[path.stem] = [[int(v) for v in b] for b in kept]
        if path.stem in old and filtered[path.stem] != old[path.stem]["filtered"]:
            mismatch.append(path.stem)
    if mismatch:
        raise SystemExit(f"filter.py output differs from the cached boxes on {len(mismatch)} photos: {mismatch[:10]}")

    filter_sha = sha256(ROOT / "balcony_pipeline" / "filter.py")
    prefill_src = load_prefill_source(args.prefill_src)
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    for sub in ("unlabeled", "context"):
        (args.out_dir / sub).mkdir(parents=True, exist_ok=True)

    manifest, prefill = [], []
    n_prefill = 0
    for path in images:
        photo = Image.open(path).convert("RGB")
        for k, box in enumerate(filtered[path.stem]):
            name = f"{path.stem}_pbal_{k:03d}"
            photo.crop(tuple(box)).save(args.out_dir / "unlabeled" / f"{name}.png")
            context_image(photo, box).save(args.out_dir / "context" / f"{name}.jpg", quality=88)
            manifest.append({
                "crop": f"{name}.png", "photo": path.name, "box_xyxy": box, "photo_size": list(photo.size),
                "cache": str(args.cache.relative_to(ROOT)), "filter_sha256": filter_sha,
            })
            cands = [(box_iou(box, b), rec) for b, rec in prefill_src.get(path.stem, [])]
            best = max(cands, key=lambda c: c[0], default=(0.0, None))
            if best[1] is not None and best[0] >= PREFILL_IOU:
                n_prefill += 1
                prefill.append({"path": f"{name}.png", **{f: best[1].get(f) for f in PREFILL_FIELDS},
                                "prefill_from": best[1]["path"], "prefill_iou": round(best[0], 3)})

    (args.out_dir / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest), encoding="utf-8")
    (args.out_dir / "prefill.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prefill), encoding="utf-8")
    n_old = sum(len(old[s]["filtered"]) for s in old)
    print(f"photos={len(images)}  crops={len(manifest)}  (cached 123 photos: {n_old})  prefilled={n_prefill}")
    print(f"filter.py sha256={filter_sha}")
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
