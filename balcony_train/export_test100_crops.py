"""Cut pipeline crops on the 100 test photos (``data/test100/photos``) for hand labels.

Boxes and crops are made exactly as in ``export_pipeline_crops.py``: SAM3
``balcony`` at score >= 0.45, mask clamp, sides under 16 px dropped,
``filter_balcony_boxes``, ``photo.crop(box)`` with no margin. The cmp_b photos
reuse the ``base_jpg`` stage cache (the boxes behind the ``*_pbal_*`` training
crops); SAM3 runs only on the LSAA facades. Crops are named
``<photo stem>_pbal_<NNN>``, so cmp_b crops keep their training names.

Starting values (``prefill.jsonl``) come only from earlier hand labels:

* cmp_b: the training label of the same crop (``runs/balcony_clf/labels.jsonl``),
  else of the old crop it duplicates (``crops_pipeline_train/manifest.jsonl``);
* LSAA: the FiftyOne tag export (``crops_lsaa``) whose box, moved into facade
  coordinates by the ``balconies_ctx.csv`` offset, has the best IoU >= 0.3.

Outputs under ``--out-dir`` (default ``runs/test100/crops``): ``unlabeled/``,
``context/``, ``manifest.jsonl``, ``prefill.jsonl``. Label them with::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/label_pipeline_crops_ui.py --dir runs/test100/crops
"""

from __future__ import annotations

import argparse
import csv
import json
import ntpath
import shutil
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "balcony_pipeline"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from filter import filter_balcony_boxes  # noqa: E402
from score_sam3_label_iou import box_iou  # noqa: E402
from score_sam3_pipeline_stages import detect_stages  # noqa: E402

from balcony_train.export_pipeline_crops import PREFILL_FIELDS, PREFILL_IOU, context_image, sha256  # noqa: E402
from balcony_train.label_store import load_labels  # noqa: E402
from balcony_train.paths import DEFAULT_LABELS_JSONL  # noqa: E402

ROWS = ROOT / "runs" / "batch" / "_rows"
DEFAULT_PHOTOS = ROOT / "data" / "test100" / "photos"
DEFAULT_OUT = ROOT / "runs" / "test100" / "crops"
DEFAULT_CACHE = ROWS / "test100_sam3_stage_boxes.json"
BASE_CACHE = ROWS / "base_jpg_sam3_stage_boxes.json"
TRAIN_MANIFEST = ROOT / "runs" / "balcony_clf" / "crops_pipeline_train" / "manifest.jsonl"
LSAA_CROPS = ROOT / "runs" / "balcony_clf" / "crops_lsaa"
LSAA_CTX_CSV = Path(r"C:\Users\hiroo\Desktop\temp\lsaa-dataset\data\Balconies_ctx\balconies_ctx.csv")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def base_prefill(labels: dict[str, dict]) -> dict[str, dict]:
    """cmp_b crop stem -> training label of that crop or of the old crop it duplicates."""
    out: dict[str, dict] = {}
    for m in read_jsonl(TRAIN_MANIFEST):
        stem = Path(m["crop"]).stem
        old = Path(ntpath.basename(m["old_crop"] or "")).stem
        for src in (stem, old):
            if src and src in labels and (labels[src].get("kind") or labels[src].get("enclosure")):
                out[stem] = {**labels[src], "prefill_from": labels[src]["path"], "prefill_iou": m["iou"] if src == old else 1.0}
                break
    return out


def lsaa_boxes(labels: dict[str, dict]) -> dict[str, list[tuple[list[float], dict]]]:
    """Facade image name -> [(box in facade pixels, label)] from the FiftyOne export."""
    ctx = {}
    with LSAA_CTX_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ctx[row["name"]] = (float(row["ctx_x0"]), float(row["ctx_y0"]))
    out: dict[str, list[tuple[list[float], dict]]] = {}
    for m in read_jsonl(LSAA_CROPS / "manifest.jsonl"):
        x0, y0 = ctx[ntpath.basename(m["photo"])]
        b = m["box_xyxy"]
        out.setdefault(m["facade_name"], []).append(
            ([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0], labels[Path(m["crop"]).stem]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="overwrite an existing out-dir")
    args = ap.parse_args()
    args.cache = args.cache.resolve()

    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise SystemExit(f"{args.out_dir} is not empty; labels may live there. Use --force to rebuild.")
    images = sorted(args.photos_dir.glob("*.jpg"))
    base = json.loads(BASE_CACHE.read_text(encoding="utf-8"))
    cache = json.loads(args.cache.read_text(encoding="utf-8")) if args.cache.is_file() else {}
    for p in images:
        if p.stem in base and p.stem not in cache:
            cache[p.stem] = base[p.stem]
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.cache.write_text(json.dumps(cache), encoding="utf-8")
    stages = detect_stages(images, args.cache, args)

    filtered: dict[str, list[list[int]]] = {}
    mismatch = []
    for path in images:
        photo = Image.open(path).convert("RGB")
        kept, _log = filter_balcony_boxes(stages[path.stem]["sized"], photo, [])
        filtered[path.stem] = [[int(v) for v in b] for b in kept]
        if filtered[path.stem] != stages[path.stem]["filtered"]:
            mismatch.append(path.stem)
    if mismatch:
        raise SystemExit(f"filter.py output differs from the cached boxes on {len(mismatch)} photos: {mismatch[:10]}")

    labels = load_labels(DEFAULT_LABELS_JSONL)
    pre_base = base_prefill(labels)
    pre_lsaa = lsaa_boxes(load_labels(LSAA_CROPS / "labels.jsonl"))
    filter_sha = sha256(ROOT / "balcony_pipeline" / "filter.py")
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    for sub in ("unlabeled", "context"):
        (args.out_dir / sub).mkdir(parents=True, exist_ok=True)

    manifest, prefill = [], []
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
            if path.stem.startswith("cmp_b"):
                rec = pre_base.get(name)
            else:
                cands = [(box_iou(box, b), r) for b, r in pre_lsaa.get(path.name, [])]
                iou, best = max(cands, key=lambda c: c[0], default=(0.0, None))
                rec = {**best, "prefill_from": best["path"], "prefill_iou": round(iou, 3)} \
                    if best is not None and iou >= PREFILL_IOU else None
            if rec is not None:
                prefill.append({"path": f"{name}.png", **{f: rec.get(f) for f in PREFILL_FIELDS},
                                "prefill_from": rec["prefill_from"], "prefill_iou": rec["prefill_iou"]})

    (args.out_dir / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest), encoding="utf-8")
    (args.out_dir / "prefill.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prefill), encoding="utf-8")
    n_base = sum(m["photo"].startswith("cmp_b") for m in manifest)
    print(f"photos={len(images)}  crops={len(manifest)} (cmp_b {n_base}, LSAA {len(manifest) - n_base})  "
          f"prefilled={len(prefill)}")
    print(f"filter.py sha256={filter_sha}")
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
