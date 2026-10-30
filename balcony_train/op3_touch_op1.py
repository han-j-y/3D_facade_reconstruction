"""OP3 rectangle, then add OP1 cells that touch it.

By default a photo with gate max < 0.2 predicts nothing. ``--gate 0`` keeps
the pure classifier (no gate): every photo is decoded at t=0.3. An OP1
(Stage 6) cell is added only when all three hold:

  * it touches the rectangle (same bay, adjacent floor, or same floor, adjacent bay)
  * the bay is the rectangle's bay, its neighbor, or its mirror, and the floor
    is at most one storey outside that bay's rectangle range
  * weak evidence: depth L >= 1.5%, SAM3 enclosure cover >= 0.25, or p >= 0.10

The model is the one trained on cmp_b (``model_base.joblib``). The eval set
is scored once.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/op3_touch_op1.py
    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/op3_touch_op1.py --gate 0 --out-dir runs/cell_clf/compare_clf_op1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.cell_clf_experiments import DEFAULT_CELLS, Data, line, score_sets  # noqa: E402
from balcony_train.cell_clf_search import SPECS, Model  # noqa: E402
from balcony_train.cell_clf_train import DEFAULT_MODEL, DEFAULT_TRAIN_CELLS, DEFAULT_TRAIN_LABELS, apply_pair  # noqa: E402
from balcony_train.cell_clf_v2 import mirror_index  # noqa: E402
from balcony_train.cell_clf_v3 import rect_decode  # noqa: E402
from balcony_train.draw_cell_clf_compare import CATEGORIES, board, category  # noqa: E402
from balcony_train.draw_clf_compare import add_header, draw_probs, iou  # noqa: E402
from balcony_train.enclosed_grid_store import (  # noqa: E402
    DEFAULT_DSL_DIR,
    DEFAULT_EVAL_LABELS_JSONL,
    DEFAULT_EVAL_PHOTO_DIR,
    find_photo,
    load_floor_bays,
    load_grid,
)
from balcony_train.evaluate_enclosed_grid import format_report, render_compare, summarize  # noqa: E402

DEFAULT_OUT = ROOT / "runs" / "cell_clf" / "compare_touch"
GATE = 0.2
CELL_T = 0.3
MIN_L = 0.015
MIN_SAM3 = 0.25
MIN_P = 0.10


def _ranks(data: Data, stem: str) -> tuple[dict[int, int], dict[int, int]]:
    ii = data.idx[stem]
    floors = sorted({data.cells[i][0] for i in ii})
    bays = sorted({data.cells[i][1] for i in ii})
    return {f: k for k, f in enumerate(floors)}, {b: k for k, b in enumerate(bays)}


def _touches(cell: tuple[int, int], chosen: set, frank: dict[int, int], brank: dict[int, int]) -> bool:
    f, b = cell
    for cf, cb in chosen:
        if b == cb and abs(frank[f] - frank[cf]) == 1:
            return True
        if f == cf and abs(brank[b] - brank[cb]) == 1:
            return True
    return False


def _bay_ok(bay: int, op3: set, brank: dict[int, int], mirror_of: dict[int, int]) -> bool:
    chosen = {b for _f, b in op3}
    if bay in chosen:
        return True
    if any(abs(brank[bay] - brank[b]) == 1 for b in chosen):
        return True
    return mirror_of.get(bay) in chosen


def _floor_ok(floor: int, bay: int, op3: set, frank: dict[int, int]) -> bool:
    ranks = [frank[f] for f, b in op3 if b == bay]
    if not ranks:
        return True
    return min(ranks) - 1 <= frank[floor] <= max(ranks) + 1


def _evidence(i: int, data: Data, prob: np.ndarray) -> bool:
    ell = float(data.f("L")[i])
    cover = float(data.f("sam3_enc_cover")[i])
    return (np.isfinite(ell) and ell >= MIN_L) or cover >= MIN_SAM3 or float(prob[i]) >= MIN_P


def add_touching(data: Data, stem: str, op3: set, prob: np.ndarray, mir: np.ndarray) -> tuple[set, set]:
    """Return (op3 plus allowed OP1 cells, the cells that were added)."""
    frank, brank = _ranks(data, stem)
    ii = data.idx[stem]
    where = {data.cells[i]: i for i in ii}
    mirror_of: dict[int, int] = {}
    for i in ii:
        j = int(mir[i])
        if j >= 0:
            mirror_of[data.cells[i][1]] = data.cells[j][1]
    cur = data.f("in_cells_6")
    added = set()
    for i in ii:
        if cur[i] <= 0:
            continue
        cell = data.cells[i]
        if cell in op3 or cell not in where:
            continue
        if not _touches(cell, op3, frank, brank):
            continue
        if not _bay_ok(cell[1], op3, brank, mirror_of):
            continue
        if not _floor_ok(cell[0], cell[1], op3, frank):
            continue
        if not _evidence(i, data, prob):
            continue
        added.add(cell)
    return set(op3) | added, added


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--train-cells", type=Path, default=DEFAULT_TRAIN_CELLS)
    ap.add_argument("--train-labels", type=Path, default=DEFAULT_TRAIN_LABELS)
    ap.add_argument("--eval-cells", type=Path, default=DEFAULT_CELLS)
    ap.add_argument("--eval-labels", type=Path, default=DEFAULT_EVAL_LABELS_JSONL)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_EVAL_PHOTO_DIR)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--gate", type=float, default=GATE)
    ap.add_argument("--t", type=float, default=CELL_T)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    test = Data(args.eval_cells, args.eval_labels)
    if args.gate <= 0:
        train = Data(args.train_cells, args.train_labels)
        p_gate, p_cell = Model(train, train.stems, SPECS["base"], None).predict(test, test.stems)
    else:
        import joblib

        pair = joblib.load(args.model)
        p_gate, p_cell = apply_pair(pair, test, test.stems)
    mir = mirror_index(test, args.dsl_dir)
    cur = test.f("in_cells_6")

    preds: dict[str, set] = {}
    pure: dict[str, set] = {}
    rows = []
    tiles: dict[str, list[Image.Image]] = {c: [] for c in CATEGORIES}
    report: list[str] = []
    pure_mode = args.gate <= 0
    title = f"pure classifier + touching OP1  t={args.t}" if pure_mode else f"OP3+touching OP1  gate>={args.gate}  t={args.t}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for s in test.stems:
        ii = test.idx[s]
        gt = set(test.gt[s])
        gmax = float(p_gate[ii].max())
        op1 = {test.cells[i] for i in ii if cur[i] > 0}
        op3 = set() if (not pure_mode and gmax < args.gate) else rect_decode(test, s, p_cell, args.t, mir)
        pure[s] = op3
        pred, added = add_touching(test, s, op3, p_cell, mir)
        preds[s] = pred
        row = {
            "stem": s,
            "gt": sorted(gt),
            "pred": sorted(pred),
            "has_enclosure": bool(gt),
            "detected": bool(gt & pred) if gt else not pred,
            "exact": pred == gt,
            "tp": len(gt & pred),
            "fp": len(pred - gt),
            "fn": len(gt - pred),
        }
        rows.append(row)
        cat = category(gt, pred)
        added_txt = ",".join(f"f{f}b{b}" for f, b in sorted(added)) or "-"
        lines = [
            f"{s}  [{cat}]  label={len(gt)}  gate={gmax:.2f}",
            title,
            f"result exact={int(pred == gt)}  IoU={iou(pred, gt):.2f}  cells={len(pred)}  added={len(added)}",
            f"added: {added_txt}",
            f"classifier exact={int(op3 == gt)} IoU={iou(op3, gt):.2f}   OP1 exact={int(op1 == gt)} IoU={iou(op1, gt):.2f}",
        ]
        dsl = args.dsl_dir / s / "facade_dsl.json"
        floor_y, bay_x = load_grid(dsl)
        floor_bays = load_floor_bays(dsl, floor_y, bay_x)
        photo = Image.open(find_photo(args.in_dir, s)).convert("RGB")
        img = render_compare(photo, floor_y, bay_x, gt, pred, floor_bays)
        img = draw_probs(img, floor_y, bay_x, {test.cells[i]: float(p_cell[i]) for i in ii}, args.t, floor_bays)
        panel = add_header(img, lines)
        panel.save(args.out_dir / f"{cat}__{s}.jpg", quality=92)
        tiles[cat].append(panel)
        report.append(
            f"{s}  {cat:<9}  gate={gmax:.2f}  exact={int(pred == gt)}  iou={iou(pred, gt):.2f}  "
            f"added={len(added)} ({added_txt})  op3_exact={int(op3 == gt)}  op1_exact={int(op1 == gt)}"
        )

    for cat in CATEGORIES:
        if tiles[cat]:
            board(tiles[cat]).save(args.out_dir / f"_board_{cat}.jpg", quality=85)
    res = score_sets(test, preds)
    counts = "  ".join(f"{c}={len(tiles[c])}" for c in CATEGORIES)
    text = format_report(summarize(rows), rows)
    op1_preds = {s: {test.cells[i] for i in test.idx[s] if cur[i] > 0} for s in test.stems}
    base_name = "pure classifier (no gate)" if pure_mode else f"OP3 rectangle (gate>={args.gate})"
    touch_name = "pure + touching OP1" if pure_mode else f"OP3 + touching OP1 (gate>={args.gate})"
    header = "\n".join(
        [
            line(base_name, score_sets(test, pure)),
            line(touch_name, res),
            line("OP1 stage6", score_sets(test, op1_preds)),
            counts,
            "",
            text,
            "",
            "per photo",
            *sorted(report, key=lambda r: r.split()[1]),
        ]
    )
    (args.out_dir / "summary.txt").write_text(header + "\n", encoding="utf-8")
    (args.out_dir / "preds.json").write_text(
        json.dumps({s: sorted(v) for s, v in preds.items()}) + "\n", encoding="utf-8"
    )
    print(header)
    print(f"{len(test.stems)} images -> {args.out_dir}")


if __name__ == "__main__":
    main()
