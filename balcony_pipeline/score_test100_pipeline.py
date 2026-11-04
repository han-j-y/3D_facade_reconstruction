r"""Open-balcony boxes on test100, scored exactly as ``run_balcony.py`` produces them.

Calls the pipeline's own functions in the pipeline's order, reusing only the cached SAM3
boxes (``sized`` in ``runs/batch/_rows/test100_sam3_stage_boxes.json`` = what
``rp.detect_windows`` returns in ``run_balcony.py``):

1. ``snap_open_units``: ``filter_balcony_boxes`` (with the window instances) → ``side_wall``
   (DA3 depth resized to the photo) → ``masonry_juliet_split`` (live railing classifier,
   default ``railing_test100_v4.pt``) → ``merge_adjacent_boxes`` → ``snap_units``.
2. Oriel cells (``preds_official.json`` of the 457 model, = ``detect_enclosed_cells`` at the
   pipeline gate / t) → ``place_enclosed_cells`` → ``drop_open_under_enclosed``.

Scored stages (one-to-one greedy IoU, group rule, no ignore, IoU 0.5 and 0.3):
``split`` (before the merge), ``units`` (after merge + snap), ``final`` (after the oriel drop =
the boxes the pipeline types and writes to the DSL).

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_pipeline/score_test100_pipeline.py
"""

from __future__ import annotations

import argparse
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

from enclosed import drop_open_under_enclosed, place_enclosed_cells  # noqa: E402
from run_balcony import snap_open_units  # noqa: E402
from score_test100_gt_boxes import prf, score_photo  # noqa: E402
from balcony_train.model import try_load_predictor  # noqa: E402
from balcony_train.paths import DEFAULT_STRUCTURE_CKPT  # noqa: E402
from balcony_train.try_da3_enclosure import resize_depth  # noqa: E402

T100 = ROOT / "runs" / "test100"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=T100 / "balcony_gt_boxes.jsonl")
    ap.add_argument("--cache", type=Path, default=ROOT / "runs" / "batch" / "_rows" / "test100_sam3_stage_boxes.json")
    ap.add_argument("--oriel-preds", type=Path,
                    default=ROOT / "runs" / "oriel_grid_compare" / "final_457_test100" / "preds_official.json")
    ap.add_argument("--dsl-dir", type=Path, default=T100 / "batch_grid2")
    ap.add_argument("--depth-dir", type=Path, default=T100 / "da3_depth")
    ap.add_argument("--photos", type=Path, default=T100 / "photos")
    ap.add_argument("--railing-ckpt", type=Path, default=None, help="default: pipeline default (paths.py)")
    ap.add_argument("--row-tol", type=float, default=0.055)
    ap.add_argument("--col-tol", type=float, default=0.04)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=T100 / "balcony_pipeline_score.json")
    args = ap.parse_args()

    gt = {json.loads(l)["stem"]: json.loads(l) for l in args.gt.read_text(encoding="utf-8").splitlines() if l.strip()}
    stems = sorted(gt)
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    oriel = json.loads(args.oriel_preds.read_text(encoding="utf-8"))
    pred = try_load_predictor(args.railing_ckpt, torch.device(args.device), structure_ckpt=DEFAULT_STRUCTURE_CKPT)
    print(f"railing classifier {pred.ckpt_path}")

    stages: dict[str, dict[str, list]] = {"split": {}, "units": {}, "final": {}}
    dropped: dict[str, list] = {}
    for k, s in enumerate(stems, 1):
        facade = Image.open(args.photos / f"{s}.jpg").convert("RGB")
        dsl = json.loads((args.dsl_dir / s / "facade_dsl.json").read_text(encoding="utf-8"))
        dp = args.depth_dir / f"{s}_depth.npy"
        depth = resize_depth(np.load(dp), facade.width, facade.height) if dp.is_file() else None
        units, split_boxes, _ = snap_open_units(
            cache[s]["sized"], facade, dsl, row_tol=args.row_tol, col_tol=args.col_tol,
            side_wall=True, depth=depth, split_predictor=pred,
        )
        enc = place_enclosed_cells([tuple(c) for c in oriel.get(s, [])], dsl, unit_id0=len(units))
        kept, drop = drop_open_under_enclosed(units, enc) if units and enc else (units, [])
        stages["split"][s] = [list(b) for b in split_boxes]
        stages["units"][s] = [u["box_xyxy"] for u in units]
        stages["final"][s] = [u["box_xyxy"] for u in kept]
        dropped[s] = [u["box_xyxy"] for u in drop]
        print(f"[{k}/{len(stems)}] {s}  split {len(split_boxes)}  units {len(units)}  final {len(kept)}", flush=True)

    subsets = {"all": lambda s: True, "cmp": lambda s: s.startswith("cmp_"), "lsaa": lambda s: not s.startswith("cmp_")}
    results: dict = {}
    print(f"\n{'stage':6s} {'subset':5s} IoU   pred   TP   FP   FN     P      R     F1")
    for thr in (0.5, 0.3):
        for name, boxes in stages.items():
            for sub, f in subsets.items():
                t = prf([score_photo(gt[s]["boxes"], [], boxes[s], thr, True) for s in stems if f(s)])
                results[f"{name}|{sub}|{thr}"] = t
                print(f"{name:6s} {sub:5s} {thr:<4}  {t['pred']:4d} {t['tp']:4d} {t['fp']:4d} {t['fn']:4d}  "
                      f"{t['precision']:.3f}  {t['recall']:.3f}  {t['f1']:.3f}")
    args.out.write_text(json.dumps({"results": results, "boxes": stages, "oriel_dropped": dropped}, indent=1),
                        encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
