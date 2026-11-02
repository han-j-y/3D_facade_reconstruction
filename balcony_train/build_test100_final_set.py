r"""Labeled crops of the final test100 pipeline boxes -> one dir for ``evaluate_test100.py``.

Final boxes = SAM3 ``filtered`` + ``side_wall`` + ``masonry_juliet_split`` (``railing_test100_v2``
labels), as ``balcony_pipeline/draw_test100_final.py``. Each box takes its hand label from
``runs/test100/crops`` (same box) or ``runs/test100/crops_split`` (split pieces). Boxes with no
label (never saved, or not_balcony) are left out.

``--gt-iou T`` (default 0.5, the detection match threshold) also leaves out a labeled crop whose box has
IoU < T with every box in ``runs/test100/balcony_gt_boxes.jsonl`` (user, 2026-11-01: a balcony that
is not in the box ground truth is not scored by the classifier). ``--gt-iou 0`` keeps them.

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_train/build_test100_final_set.py <out_dir>
    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_train/evaluate_test100.py checkpoints/railing_test100_juliet.pt --dir <out_dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "balcony_pipeline"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from postfilter import apply_rules  # noqa: E402
from balcony_train.label_store import find_image  # noqa: E402

T = ROOT / "runs" / "test100"


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.is_file() else []


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--gt-iou", type=float, default=0.5, help="min IoU with a GT box; 0 keeps every labeled crop")
    ap.add_argument("--cache", type=Path, default=ROOT / "runs" / "batch" / "_rows" / "test100_sam3_stage_boxes.json")
    ap.add_argument("--labels", type=Path, default=T / "railing_preds_v2.json", help="classifier labels for the split rule")
    args = ap.parse_args()

    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    preds = json.loads(args.labels.read_text(encoding="utf-8"))
    gt = {r["stem"]: r["boxes"] for r in jl(T / "balcony_gt_boxes.jsonl")}
    crop_of = {(Path(m["photo"]).stem, tuple(m["box_xyxy"])): Path(m["crop"]).stem for m in jl(T / "crops/manifest.jsonl")}
    split_of = {(Path(m["photo"]).stem, tuple(m["box_xyxy"])): Path(m["crop"]).stem for m in jl(T / "crops_split/manifest.jsonl")}
    lab_main = {Path(r["path"]).stem: r for r in jl(T / "crops/labels.jsonl")}
    lab_split = {Path(r["path"]).stem: r for r in jl(T / "crops_split/labels.jsonl")}

    if args.out.exists():
        shutil.rmtree(args.out)
    rows, dropped = [], []
    n_box = n_nolabel = 0
    for s in sorted(cache):
        if not (T / "photos" / f"{s}.jpg").is_file():
            continue
        dsl = json.loads((T / "batch_grid2" / s / "facade_dsl.json").read_text(encoding="utf-8"))
        dp = T / "da3_depth" / f"{s}_depth.npy"
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
        for b in pred:
            n_box += 1
            key = (s, tuple(b))
            if key in split_of:
                name, rec, src = split_of[key], lab_split.get(split_of[key]), T / "crops_split"
            else:
                name = crop_of.get(key)
                rec, src = (lab_main.get(name) if name else None), T / "crops"
            if rec is None:
                n_nolabel += 1
                continue
            best = max((iou(b, g) for g in gt.get(s, [])), default=0.0)
            if args.gt_iou > 0 and best < args.gt_iou:
                dropped.append({"crop": name, "best_gt_iou": round(best, 3)})
                continue
            img = find_image(src, name)
            dest = args.out / (rec.get("kind") or "unlabeled") / img.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dest)
            rows.append(rec)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (args.out / "dropped_not_in_gt.jsonl").write_text("".join(json.dumps(r) + "\n" for r in dropped), encoding="utf-8")
    n_split = sum(Path(r["path"]).stem in lab_split for r in rows)
    print(f"final boxes {n_box}: no label {n_nolabel}, not in GT (IoU < {args.gt_iou}) {len(dropped)}, "
          f"scored {len(rows)} (split pieces {n_split})")


if __name__ == "__main__":
    main()
