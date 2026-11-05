r"""Export the test100 boxes made by ``masonry_juliet_split`` as crops to label.

The final open-balcony pipeline on test100 (SAM3 ``filtered`` + ``side_wall`` +
``masonry_juliet_split`` with ``railing_test100_v2`` labels, as in
``balcony_pipeline/draw_test100_final.py``) splits some wide balustrade boxes into
one box per window. Those pieces have no crop in ``runs/test100/crops``. This
writes them to ``runs/test100/crops_split/`` in the layout
``label_pipeline_crops_ui.py`` reads:

* ``unlabeled/<parent crop>_s<k>.png``: ``photo.crop(box)``, no margin;
* ``context/<name>.jpg``: the photo with the piece's box drawn;
* ``manifest.jsonl``: box, photo, parent crop and parent box;
* ``prefill.jsonl``: the parent crop's hand label from ``runs/test100/crops/labels.jsonl``
  (never a model prediction). ``prefill_iou`` is the piece's IoU with the parent box.

Label with::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_train/label_pipeline_crops_ui.py --dir runs/test100/crops_split --open-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "balcony_pipeline"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from postfilter import apply_rules  # noqa: E402
from balcony_train.export_pipeline_crops import PREFILL_FIELDS, context_image  # noqa: E402

T100 = ROOT / "runs" / "test100"


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0.0


def inside(piece, parent) -> float:
    """Share of the piece's area inside the parent box."""
    ix = max(0, min(piece[2], parent[2]) - max(piece[0], parent[0]))
    iy = max(0, min(piece[3], parent[3]) - max(piece[1], parent[1]))
    a = (piece[2] - piece[0]) * (piece[3] - piece[1])
    return ix * iy / a if a else 0.0


def jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=ROOT / "runs" / "batch" / "_rows" / "test100_sam3_stage_boxes.json")
    ap.add_argument("--dsl-dir", type=Path, default=T100 / "batch_grid2")
    ap.add_argument("--depth-dir", type=Path, default=T100 / "da3_depth")
    ap.add_argument("--labels", type=Path, default=T100 / "railing_preds_v2.json", help="classifier labels for the split rule")
    ap.add_argument("--crops", type=Path, default=T100 / "crops", help="test crops: manifest + hand labels of the parents")
    ap.add_argument("--out-dir", type=Path, default=T100 / "crops_split")
    ap.add_argument("--force", action="store_true", help="overwrite --out-dir (it may hold labels)")
    args = ap.parse_args()

    if args.out_dir.exists():
        if not args.force:
            raise SystemExit(f"{args.out_dir} exists (may hold labels); pass --force to overwrite")
        shutil.rmtree(args.out_dir)
    for sub in ("unlabeled", "context"):
        (args.out_dir / sub).mkdir(parents=True, exist_ok=True)

    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    preds = json.loads(args.labels.read_text(encoding="utf-8"))
    hand = {Path(r["path"]).stem: r for r in jsonl(args.crops / "labels.jsonl")}
    crop_of = {}
    for m in jsonl(args.crops / "manifest.jsonl"):
        crop_of[(Path(m["photo"]).stem, tuple(m["box_xyxy"]))] = Path(m["crop"]).stem

    stems = sorted(p.stem for p in (T100 / "photos").glob("*.jpg"))
    manifest, prefill = [], []
    for s in stems:
        if s not in cache:
            continue
        dsl = json.loads((args.dsl_dir / s / "facade_dsl.json").read_text(encoding="utf-8"))
        dp = args.depth_dir / f"{s}_depth.npy"
        depth = np.load(dp) if dp.is_file() else None
        filt = cache[s]["filtered"]
        size = tuple(cache[s]["size"])
        sw, _ = apply_rules(filt, size, dsl, depth=depth, rules=("side_wall",))
        left = list(range(len(filt)))
        labs = []
        for b in sw:
            j = next(j for j in left if filt[j] == b)
            left.remove(j)
            labs.append(preds[s][j])
        pred, _ = apply_rules(sw, size, dsl, labels=labs, rules=("masonry_juliet_split",))
        pieces = [b for b in pred if b not in sw]
        if not pieces:
            continue
        parents = [b for b in sw if b not in pred]
        photo = Image.open(T100 / "photos" / f"{s}.jpg").convert("RGB")
        count: dict[str, int] = {}
        for b in sorted(pieces, key=lambda b: (b[1], b[0])):
            parent = max(parents, key=lambda p: inside(b, p))
            pname = crop_of.get((s, tuple(parent)), f"{s}_noparent")
            k = count.get(pname, 0)
            count[pname] = k + 1
            name = f"{pname}_s{k}"
            photo.crop(tuple(b)).save(args.out_dir / "unlabeled" / f"{name}.png")
            context_image(photo, b).save(args.out_dir / "context" / f"{name}.jpg", quality=88)
            manifest.append({"crop": f"{name}.png", "photo": f"{s}.jpg", "box_xyxy": list(b),
                             "photo_size": list(photo.size), "parent_crop": f"{pname}.png",
                             "parent_box_xyxy": list(parent), "rule": "masonry_juliet_split",
                             "split_labels": str(args.labels.relative_to(ROOT))})
            rec = hand.get(pname)
            if rec is not None:
                prefill.append({"path": f"{name}.png", **{f: rec.get(f) for f in PREFILL_FIELDS},
                                "prefill_from": f"{pname}.png", "prefill_iou": round(iou(b, parent), 3)})

    (args.out_dir / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest), encoding="utf-8")
    (args.out_dir / "prefill.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prefill), encoding="utf-8")
    photos = len({m["photo"] for m in manifest})
    print(f"pieces={len(manifest)} photos={photos} prefilled={len(prefill)} -> {args.out_dir}")


if __name__ == "__main__":
    main()
