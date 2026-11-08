r"""Does a balcony stick out of the wall beside it? DA3 depth test on the val30 structure ground truth.

For each labeled balcony unit (``data/val30/type_gt.jsonl``, not-a-balcony left out) on the DA3 metric depth
(``runs/da3_enclosure/<stem>_depth.npy``, resized to the photo; smaller = closer):

* box depth: median depth inside the box;
* left / right wall: median depth of a strip beside the box at the same height (``--strip`` x the box width,
  ``--gap`` x the box width away from it). Above / below are not used: a recess is deeper there by design;
* protrusion = wall depth - box depth on the side where it is larger (metres, and relative = / wall depth).

The user's rule (2026-11-07): projecting = sticks out of the outer wall right beside it. Prints the
protrusion per GT class, the accuracy of "projecting when protrusion > t" over a range of t (val30, not
test100), the railing classifier on the same units, and writes a CSV and a plot.

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_pipeline/depth_protrusion_val30.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from balcony_train.try_da3_enclosure import resize_depth  # noqa: E402
from structure_depth import protrusion as _protrusion  # noqa: E402


def sweep(vals: list[float], truth: list[str], ts) -> list[tuple[float, float, float]]:
    out = []
    for t in ts:
        pred = ["projecting" if v > t else "juliet" for v in vals]
        acc = float(np.mean([p == g for p, g in zip(pred, truth)]))
        recs = []
        for c in ("projecting", "juliet"):
            idx = [i for i, g in enumerate(truth) if g == c]
            recs.append(np.mean([pred[i] == c for i in idx]) if idx else 0.0)
        out.append((t, acc, float(np.mean(recs))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=ROOT / "data" / "val30" / "type_gt.jsonl")
    ap.add_argument("--photos", type=Path, default=ROOT / "data" / "val30")
    ap.add_argument("--depth-dir", type=Path, default=ROOT / "runs" / "da3_enclosure")
    ap.add_argument("--strip", type=float, default=0.5)
    ap.add_argument("--gap", type=float, default=0.05)
    ap.add_argument("--structure-json", type=Path,
                    default=ROOT / "runs" / "val30" / "_structure_errors" / "structure_val30.json",
                    help="classifier P(juliet) per unit (draw_val30_structure_errors.py), for comparison")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "val30" / "_depth_protrusion")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    clf = {}
    if args.structure_json.is_file():
        for r in json.loads(args.structure_json.read_text(encoding="utf-8")):
            clf[(r["id"], r["unit"])] = r["p_juliet"]
    rows = [json.loads(l) for l in args.gt.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs = []
    for r in rows:
        dp = args.depth_dir / f"{r['stem']}_depth.npy"
        img = next(p for p in (args.photos / f"{r['stem']}.jpg", args.photos / f"{r['stem']}.png") if p.is_file())
        W, H = Image.open(img).size
        depth = resize_depth(np.load(dp), W, H)
        for k, u in enumerate(r["units"]):
            if u["not_balcony"] or u["structure"] is None:
                continue
            m = _protrusion(depth, u["box"], strip=args.strip, gap=args.gap)
            recs.append({"id": r["id"], "unit": k, "group": u["group"], "truth": u["structure"],
                         "p_juliet": clf.get((r["id"], k)), **m})
    with (args.out_dir / "protrusion.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)

    ok = [x for x in recs if x["prot_m"] is not None]
    print(f"{len(recs)} units, {len(ok)} with depth on at least one side "
          f"(strip {args.strip} x width, gap {args.gap})")
    for key, unit in (("prot_m", "m"), ("prot_rel", "")):
        print(f"\nprotrusion ({key}) per GT class: median [p10, p90]")
        for c in ("projecting", "juliet"):
            v = np.array([x[key] for x in ok if x["truth"] == c])
            print(f"  {c:10s} n={len(v):3d}  {np.median(v):+.3f}{unit} [{np.percentile(v, 10):+.3f}, {np.percentile(v, 90):+.3f}]")
        vals = [x[key] for x in ok]
        truth = [x["truth"] for x in ok]
        lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)
        res = sweep(vals, truth, np.linspace(lo, hi, 41))
        best = max(res, key=lambda r: r[2])
        print(f"  threshold sweep (projecting if > t): best balanced acc {best[2]:.3f} at t={best[0]:+.3f}"
              f" (acc {best[1]:.3f})")
        for t, a, b in res[::5]:
            print(f"    t={t:+.3f}  acc {a:.3f}  balanced {b:.3f}")
    cl = [x for x in ok if x["p_juliet"] is not None]
    if cl:
        pred = ["juliet" if x["p_juliet"] > 0.5 else "projecting" for x in cl]
        acc = np.mean([p == x["truth"] for p, x in zip(pred, cl)])
        bal = np.mean([np.mean([p == c for p, x in zip(pred, cl) if x["truth"] == c]) for c in ("projecting", "juliet")])
        print(f"\nrailing classifier on the same {len(cl)} units: acc {acc:.3f}, balanced {bal:.3f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, lab in ((axes[0], "prot_m", "protrusion (m)"), (axes[1], "prot_rel", "protrusion / wall depth")):
        for c, col in (("projecting", "#d1495b"), ("juliet", "#2e86ab")):
            v = [x[key] for x in ok if x["truth"] == c]
            ax.hist(v, bins=30, alpha=0.6, color=col, label=f"GT {c} ({len(v)})")
        ax.set_xlabel(lab)
        ax.legend()
    fig.suptitle("val30: DA3 depth, wall beside the box minus box (larger side)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "protrusion_hist.png", dpi=110)
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
