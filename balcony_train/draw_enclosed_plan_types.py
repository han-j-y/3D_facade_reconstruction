"""Draw the four enclosed plan types on eval photos that have predicted cells.

Uses the official gate cells (``compare_clf_rules_gate_ext_relabel/preds.json``). Each
predicted cell is cropped and labeled by the railing checkpoint. Triangle
counts as rectangle, and a missing skin is punched, so every box is one of:

    punched rect, punched circle, curtain rect, curtain circle

Photos with no predicted cell are skipped. One jpg per photo, plus a contact
sheet and ``assignments.json``.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/draw_enclosed_plan_types.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.depth_cells import cell_box  # noqa: E402
from balcony_train.draw_cell_clf_compare import board  # noqa: E402
from balcony_train.enclosed_grid_store import (  # noqa: E402
    DEFAULT_DSL_DIR,
    DEFAULT_EVAL_LABELS_JSONL,
    DEFAULT_EVAL_PHOTO_DIR,
    find_photo,
    load_grid,
    load_records,
)
from balcony_train.evaluate_enclosed_grid import (  # noqa: E402
    PLAN_TYPE_LABEL,
    PLAN_TYPES,
    enclosed_plan_type,
    load_pred_cells,
    render_plan_types,
    scored_floor_shape,
)
from balcony_train.paths import DEFAULT_RAILING_CKPT  # noqa: E402

DEFAULT_PREDS = ROOT / "runs" / "cell_clf" / "compare_clf_rules_gate_ext_relabel" / "preds.json"
DEFAULT_OUT = ROOT / "runs" / "enclosed_grid" / "plan_types"


def classify_cells(image: Image.Image, cells: set, floor_y: dict, bay_x: dict, predict_full) -> list[dict]:
    """One record per predicted cell that sits on the window grid."""
    rows: list[dict] = []
    for floor, bay in sorted({(int(f), int(b)) for f, b in cells}):
        if floor not in floor_y or bay not in bay_x:
            continue
        box = cell_box((floor, bay), floor_y, bay_x)
        pred = predict_full(image.crop(tuple(box)))
        raw_floor = pred.get("floor_shape") if isinstance(pred, dict) else None
        raw_skin = pred.get("skin") if isinstance(pred, dict) else None
        plan = enclosed_plan_type(raw_skin, raw_floor)
        rows.append(
            {
                "cell": [floor, bay],
                "box": box,
                "plan": plan,
                "raw_floor": raw_floor,
                "raw_skin": raw_skin,
            }
        )
    return rows


def _counts(rows: list[dict]) -> Counter:
    return Counter(PLAN_TYPE_LABEL[tuple(row["plan"])] for row in rows)


def _title(stem: str, rows: list[dict], record) -> str:
    counts = _counts(rows)
    parts = [f"{stem}  n={len(rows)}"]
    for _skin, shape in PLAN_TYPES:
        label = PLAN_TYPE_LABEL[(_skin, shape)]
        if counts[label]:
            parts.append(f"{label}={counts[label]}")
    if record is not None and record.cells:
        gt_shape = scored_floor_shape(record.floor_shape) or "-"
        gt_skin = record.skin or "-"
        parts.append(f"gt {gt_skin} {gt_shape}")
    elif record is not None:
        parts.append("gt none")
    return "  ".join(parts)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", type=Path, default=DEFAULT_PREDS)
    ap.add_argument("--labels", type=Path, default=DEFAULT_EVAL_LABELS_JSONL)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_EVAL_PHOTO_DIR)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_RAILING_CKPT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--device",
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.preds.is_file():
        raise SystemExit(f"no predicted cells at {args.preds}")
    if not args.ckpt.is_file():
        raise SystemExit(f"no railing checkpoint at {args.ckpt}")
    preds = load_pred_cells(args.preds)
    detected = {stem: cells for stem, cells in preds.items() if cells}
    if not detected:
        raise SystemExit("no photo has a predicted enclosed cell")
    records = load_records(args.labels)
    import torch

    from balcony_train.model import RailingPredictor

    predictor = RailingPredictor(args.ckpt, torch.device(args.device))
    if not getattr(predictor.model, "_skin_trained", False):
        print("warn: skin head is untrained; every box stays punched_window")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tiles: list[Image.Image] = []
    assignments: dict[str, dict] = {}
    skipped: list[str] = []
    for stem in sorted(detected):
        photo_path = find_photo(args.in_dir, stem)
        dsl = args.dsl_dir / stem / "facade_dsl.json"
        if photo_path is None or not dsl.is_file():
            skipped.append(stem)
            continue
        floor_y, bay_x = load_grid(dsl)
        image = Image.open(photo_path).convert("RGB")
        rows = classify_cells(image, detected[stem], floor_y, bay_x, predictor.predict_full)
        if not rows:
            skipped.append(stem)
            continue
        title = _title(stem, rows, records.get(stem))
        panel = render_plan_types(image, rows, title=title)
        panel.save(args.out_dir / f"{stem}.jpg", quality=90)
        tiles.append(panel)
        counts = _counts(rows)
        assignments[stem] = {
            "n": len(rows),
            "counts": {label: int(counts[label]) for label in counts},
            "cells": [
                {
                    "cell": row["cell"],
                    "box": row["box"],
                    "skin": row["plan"][0],
                    "floor": row["plan"][1],
                    "raw_floor": row["raw_floor"],
                    "raw_skin": row["raw_skin"],
                }
                for row in rows
            ],
        }
        print(title, flush=True)

    if tiles:
        board(tiles).save(args.out_dir / "_board.jpg", quality=85)
    payload = {"n_photos": len(assignments), "photos": assignments}
    (args.out_dir / "assignments.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    if skipped:
        print("skipped: " + ", ".join(skipped))
    print(f"photos={len(assignments)} -> {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
