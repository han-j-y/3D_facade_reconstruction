r"""Railing classifier structure on val30 with context around the box (no retraining).

Each labeled balcony unit of ``data/val30/type_gt.jsonl`` is cropped from the photo with margins (fractions of
the box width for left / right, of the box height for up / down) and classified by the pipeline's railing
classifier. ``tight`` is the box itself (the pipeline's crop). Prints accuracy and balanced accuracy of the
structure head per variant, and the depth rule of ``depth_protrusion_val30.py`` (``protrusion.csv``) for
comparison. val30 only (not test100).

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_pipeline/crop_context_val30.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_balcony import juliet_prob  # noqa: E402

from balcony_train.model import try_load_predictor  # noqa: E402
from balcony_train.paths import DEFAULT_STRUCTURE_CKPT  # noqa: E402

# name: (left, right, up, down) margins; left / right x box width, up / down x box height
VARIANTS = {
    "tight": (0, 0, 0, 0),
    "down1_side0.5": (0.5, 0.5, 0.25, 1.0),
    "down0.5_side0.25": (0.25, 0.25, 0.1, 0.5),
    "all0.5": (0.5, 0.5, 0.5, 0.5),
    "all1.0": (1.0, 1.0, 1.0, 1.0),
}


def scores(pred: list[str], truth: list[str]) -> tuple[float, float]:
    acc = float(np.mean([p == t for p, t in zip(pred, truth)]))
    bal = float(np.mean([np.mean([p == c for p, t in zip(pred, truth) if t == c]) for c in ("projecting", "juliet")]))
    return acc, bal


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=ROOT / "data" / "val30" / "type_gt.jsonl")
    ap.add_argument("--photos", type=Path, default=ROOT / "data" / "val30")
    ap.add_argument("--protrusion", type=Path, default=ROOT / "runs" / "val30" / "_depth_protrusion" / "protrusion.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "val30" / "_depth_protrusion" / "crop_context.csv")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    pred = try_load_predictor(None, torch.device(args.device), structure_ckpt=DEFAULT_STRUCTURE_CKPT)
    rows = [json.loads(l) for l in args.gt.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs = []
    for r in rows:
        photo = Image.open(next(p for p in (args.photos / f"{r['stem']}.jpg", args.photos / f"{r['stem']}.png")
                                if p.is_file())).convert("RGB")
        for k, u in enumerate(r["units"]):
            if u["not_balcony"] or u["structure"] is None:
                continue
            x0, y0, x1, y1 = u["box"]
            w, h = x1 - x0, y1 - y0
            rec = {"id": r["id"], "unit": k, "truth": u["structure"]}
            for name, (ml, mr, mu, md) in VARIANTS.items():
                box = (max(0, int(x0 - ml * w)), max(0, int(y0 - mu * h)),
                       min(photo.width, int(x1 + mr * w)), min(photo.height, int(y1 + md * h)))
                rec[name] = round(juliet_prob(pred, photo.crop(box)), 4)
            recs.append(rec)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(recs[0]))
        wr.writeheader()
        wr.writerows(recs)

    truth = [x["truth"] for x in recs]
    print(f"{len(recs)} units (projecting {truth.count('projecting')}, juliet {truth.count('juliet')})")
    print(f"{'variant':18s} {'acc':>6s} {'balanced':>9s}  juliet called projecting / projecting called juliet")
    for name in VARIANTS:
        p = ["juliet" if x[name] > 0.5 else "projecting" for x in recs]
        a, b = scores(p, truth)
        jp = sum(pp == "projecting" and t == "juliet" for pp, t in zip(p, truth))
        pj = sum(pp == "juliet" and t == "projecting" for pp, t in zip(p, truth))
        print(f"{name:18s} {a:6.3f} {b:9.3f}  {jp:3d} / {pj:3d}")
    if args.protrusion.is_file():
        prot = {(r["id"], int(r["unit"])): r["prot_m"] for r in csv.DictReader(args.protrusion.open(encoding="utf-8"))}
        sub = [x for x in recs if prot.get((x["id"], x["unit"]))]
        p = ["projecting" if float(prot[(x["id"], x["unit"])]) > 0.10 else "juliet" for x in sub]
        a, b = scores(p, [x["truth"] for x in sub])
        print(f"{'depth > 0.10 m':18s} {a:6.3f} {b:9.3f}  (on {len(sub)} units with a side strip)")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
