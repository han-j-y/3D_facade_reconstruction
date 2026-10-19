"""Score the depth-first cell detector step by step against Label UI cells.

Uses cached DA3 depth and SAM3 boxes only (no GPU). Every step's cell set is
scored as if it were the final answer, so the report shows what each step
adds or removes. Per photo, a strip shows all steps on the window grid
(blue = label only, red = prediction only, purple = both).

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/evaluate_depth_cells.py
    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/evaluate_depth_cells.py --stems cmp_x0097 cmp_x0137
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.depth_cells import (  # noqa: E402
    DEPTH_REL,
    STEP_TITLES,
    UPPER_MIN,
    cell_features,
    depth_cell_steps,
)
from balcony_train.enclosed_grid_store import (  # noqa: E402
    DEFAULT_DSL_DIR,
    DEFAULT_EVAL_LABELS_JSONL,
    DEFAULT_EVAL_PHOTO_DIR,
    find_photo,
    load_grid,
    load_labels,
)
from balcony_train.enclosed_symmetry import window_boxes_from_dsl  # noqa: E402
from balcony_train.evaluate_enclosed_grid import (  # noqa: E402
    DEFAULT_SAM3_CACHE,
    format_report,
    render_compare,
    score_photo,
    select_labels,
    summarize,
)
from balcony_train.extract_enclosed import foliage_mask  # noqa: E402
from balcony_train.try_da3_enclosure import resize_depth  # noqa: E402
from balcony_train.try_sam3_da3_volumes import (  # noqa: E402
    DEFAULT_DEPTH_DIR,
    _stage_font,
    compose_row,
)

DEFAULT_OUT_DIR = ROOT / "runs" / "depth_cells"


def step_strip(
    image: Image.Image,
    floor_y,
    bay_x,
    gt: set,
    steps: list[tuple[str, str, set]],
    rows: list[dict[str, Any]],
) -> Image.Image:
    panels = []
    font = _stage_font(max(14, image.width // 40))
    bar_h = max(28, image.height // 26)
    for (_sid, title, cells), row in zip(steps, rows):
        panel = render_compare(image, floor_y, bay_x, gt, cells)
        d = ImageDraw.Draw(panel)
        d.rectangle((0, 0, panel.width, bar_h), fill=(12, 16, 22))
        mark = "OK" if row["exact"] else f"tp{row['tp']} fp{row['fp']} fn{row['fn']}"
        d.text((6, 3), f"{title}  [{mark}]", fill=(240, 240, 240), font=font)
        panels.append(panel)
    return compose_row(panels, height=520)


def evaluate(
    labels: dict[str, list],
    *,
    photo_dir: Path,
    dsl_dir: Path,
    depth_dir: Path,
    sam3_cache: Path,
    out_dir: Path | None,
    depth_rel: float,
    upper_min: float,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    cache = json.loads(Path(sam3_cache).read_text(encoding="utf-8"))
    by_step: dict[str, list[dict[str, Any]]] = {sid: [] for sid, _ in STEP_TITLES}
    skipped: list[str] = []
    for stem in sorted(labels):
        photo = find_photo(photo_dir, stem)
        dsl_path = Path(dsl_dir) / stem / "facade_dsl.json"
        depth_path = Path(depth_dir) / f"{stem}_depth.npy"
        if photo is None or not dsl_path.is_file() or not depth_path.is_file():
            skipped.append(stem)
            continue
        floor_y, bay_x = load_grid(dsl_path)
        if not floor_y or not bay_x:
            skipped.append(stem)
            continue
        image = Image.open(photo).convert("RGB")
        depth = resize_depth(np.load(depth_path), image.width, image.height)
        veg = foliage_mask(np.asarray(image, dtype=np.uint8))
        windows = window_boxes_from_dsl(json.loads(dsl_path.read_text(encoding="utf-8")))
        feats = cell_features(depth, veg, floor_y, bay_x, cache.get(stem, []))
        steps, log = depth_cell_steps(
            feats,
            floor_y,
            bay_x,
            iw=image.width,
            windows=windows,
            depth_rel=depth_rel,
            upper_min=upper_min,
        )
        gt = {tuple(c) for c in labels[stem]}
        rows = []
        for sid, _title, cells in steps:
            row = score_photo(labels[stem], cells)
            row["stem"] = stem
            by_step[sid].append(row)
            rows.append(row)
        if out_dir is not None:
            (out_dir / "steps").mkdir(parents=True, exist_ok=True)
            step_strip(image, floor_y, bay_x, gt, steps, rows).save(
                out_dir / "steps" / f"{stem}.jpg", quality=90
            )
            (out_dir / "compare").mkdir(parents=True, exist_ok=True)
            render_compare(image, floor_y, bay_x, gt, steps[-1][2]).save(
                out_dir / "compare" / f"{stem}.jpg", quality=92
            )
            (out_dir / "features").mkdir(parents=True, exist_ok=True)
            (out_dir / "features" / f"{stem}.json").write_text(
                json.dumps(
                    {
                        "window_sym": log["window_sym"],
                        "label": sorted(gt),
                        "cells": {f"f{f}b{b}": v for (f, b), v in sorted(feats.items())},
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )
        flags = " ".join(f"{sid.split('_')[0]}:{int(r['exact'])}" for (sid, _), r in zip(STEP_TITLES, rows))
        print(f"{stem}  exact by step  {flags}", flush=True)
    return by_step, skipped


def step_table(by_step: dict[str, list[dict[str, Any]]]) -> str:
    lines = ["step        enc_detect  enc_exact  none_ok  total_exact  cells tp/fp/fn"]
    for sid, title in STEP_TITLES:
        s = summarize(by_step[sid])
        e, n, t = s["enclosure"], s["none"], s["total"]
        lines.append(
            f"{sid:<10}  {e['detect']:>3}/{e['n']:<3}     {e['bay_floor']:>3}/{e['n']:<3}    "
            f"{n['bay_floor']:>3}/{n['n']:<3}  {t['bay_floor']:>3}/{t['n']:<3} ({t['bay_floor'] / max(t['n'], 1):.3f})  "
            f"{s['tp']}/{s['fp']}/{s['fn']}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=DEFAULT_EVAL_LABELS_JSONL)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_EVAL_PHOTO_DIR)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--depth-dir", type=Path, default=DEFAULT_DEPTH_DIR)
    ap.add_argument("--sam3-cache", type=Path, default=DEFAULT_SAM3_CACHE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Empty path skips images.")
    ap.add_argument("--depth-rel", type=float, default=DEPTH_REL)
    ap.add_argument("--upper-min", type=float, default=UPPER_MIN)
    ap.add_argument("--stems", nargs="*", default=None)
    args = ap.parse_args()

    labels, missing = select_labels(load_labels(args.labels), args.stems)
    if missing:
        print("not in labels: " + ", ".join(missing))
    out_dir = None if str(args.out_dir) == "" else Path(args.out_dir)
    by_step, skipped = evaluate(
        labels,
        photo_dir=args.in_dir,
        dsl_dir=args.dsl_dir,
        depth_dir=args.depth_dir,
        sam3_cache=args.sam3_cache,
        out_dir=out_dir,
        depth_rel=args.depth_rel,
        upper_min=args.upper_min,
    )
    if skipped:
        print("skipped (no photo, grid or depth): " + ", ".join(skipped))
    for sid, title in STEP_TITLES:
        print(f"=== {title} ===")
        report = format_report(summarize(by_step[sid]), by_step[sid])
        print(report if sid == STEP_TITLES[-1][0] else "\n".join(report.splitlines()[:4]))
    print("=== by step ===")
    print(step_table(by_step))
    if out_dir is not None:
        final = by_step[STEP_TITLES[-1][0]]
        (out_dir / "results.json").write_text(
            json.dumps(
                {
                    "params": {"depth_rel": args.depth_rel, "upper_min": args.upper_min},
                    "eval": {
                        "summary": summarize(final),
                        "preds": {r["stem"]: [list(c) for c in r["pred"]] for r in final},
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"steps -> {(out_dir / 'steps').resolve()}")
        print(f"results -> {(out_dir / 'results.json').resolve()}")


if __name__ == "__main__":
    main()
