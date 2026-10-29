"""Pure classifier, then four cell edits, scored once on the eval set.

The rectangle decoder (t=0.3, no gate) is kept. Then, in order:

1. Drop the top floor of that rectangle when its band is under 85% of the
   photo's median floor height.
2. Unless window symmetry is at least 0.75, drop a cell on the (new) top
   floor whose classifier probability is under 0.05. Other cells on that
   floor stay.
3. Add an OP1 (Stage 6) cell that is not already chosen when depth L is at
   least 1.5%, the upper front fraction is at least 0.55, SAM3 enclosure
   cover is at least 0.40, and the classifier probability is at least 0.10.
4. Add the cell one floor above a kept cell in the same bay when its depth L
   is within 0.02 of the cell directly below, upper is at least 0.75, and
   that floor band is at least 85% of the median floor height. Classifier
   probability, OP1 membership, and SAM3 cover are not required. Newly added
   cells do not grow another floor.

``--gate 0.10`` then blanks a photo whose gate-model max is below 0.10.
Photos at or above 0.10 keep the four-rule cells. 0.10 is below the four
no-enclosure photos left as false positives (gate 0.23 and above) and below
the lowest exact enclosure (0.14). ``--gate 0`` is the four rules alone.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/clf_three_rules.py
    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/clf_three_rules.py --gate 0.10
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
from balcony_train.cell_clf_train import DEFAULT_TRAIN_CELLS, DEFAULT_TRAIN_LABELS  # noqa: E402
from balcony_train.cell_clf_v2 import mirror_index  # noqa: E402
from balcony_train.cell_clf_v3 import rect_decode  # noqa: E402
from balcony_train.draw_cell_clf_compare import CATEGORIES, board, category  # noqa: E402
from balcony_train.draw_clf_compare import add_header, draw_probs, iou  # noqa: E402
from balcony_train.enclosed_grid_store import (  # noqa: E402
    DEFAULT_DSL_DIR,
    DEFAULT_EVAL_LABELS_JSONL,
    DEFAULT_EVAL_PHOTO_DIR,
    find_photo,
    load_grid,
)
from balcony_train.enclosed_symmetry import SYM_GATE  # noqa: E402
from balcony_train.evaluate_enclosed_grid import format_report, render_compare, summarize  # noqa: E402

DEFAULT_OUT = ROOT / "runs" / "cell_clf" / "compare_clf_rules"
DEFAULT_GATE_OUT = ROOT / "runs" / "cell_clf" / "compare_clf_rules_gate_ext_relabel"
MIN_L = 0.015
MIN_UPPER = 0.55
MIN_COVER = 0.40
MIN_P = 0.10
MIN_TOP_FRAC = 0.85
MIN_TOP_P = 0.05
MAX_L_DELTA = 0.02
MIN_UP_UPPER = 0.75


def _top_floor(cells: set, floor_y: dict[int, tuple[int, int]]) -> int | None:
    if not cells:
        return None
    return min((f for f, _b in cells), key=lambda f: floor_y[f][0])


def _median_height(floor_y: dict[int, tuple[int, int]]) -> float:
    heights = [fy1 - fy0 for fy0, fy1 in floor_y.values() if fy1 > fy0]
    return float(np.median(heights)) if heights else 0.0


def _floor_above(floor: int, floor_y: dict[int, tuple[int, int]]) -> int | None:
    order = sorted(floor_y, key=lambda f: floor_y[f][0])
    if floor not in order:
        return None
    k = order.index(floor)
    return order[k - 1] if k > 0 else None


def apply_rules(
    data: Data,
    stem: str,
    clf: set,
    prob: np.ndarray,
    floor_y: dict[int, tuple[int, int]],
) -> tuple[set, set, set, set, set]:
    """Return final cells, short-top drops, low-p drops, OP1 adds, one-up adds."""
    ii = data.idx[stem]
    where = {data.cells[i]: i for i in ii}
    sym = float(data.f("window_sym")[ii[0]]) if len(ii) else 0.0
    med = _median_height(floor_y)

    dropped_top: set = set()
    top = _top_floor(clf, floor_y)
    if top is not None and med > 0:
        height = floor_y[top][1] - floor_y[top][0]
        if height < MIN_TOP_FRAC * med:
            dropped_top = {cell for cell in clf if cell[0] == top}
    kept = set(clf) - dropped_top

    dropped_low: set = set()
    top = _top_floor(kept, floor_y)
    if top is not None and sym < SYM_GATE:
        for cell in list(kept):
            if cell[0] != top or cell not in where:
                continue
            if float(prob[where[cell]]) < MIN_TOP_P:
                dropped_low.add(cell)
    kept -= dropped_low

    ell = data.f("L")
    upper = data.f("upper")
    cover = data.f("sam3_enc_cover")
    op1 = data.f("in_cells_6")
    added: set = set()
    for i in ii:
        cell = data.cells[i]
        if op1[i] <= 0 or cell in kept:
            continue
        if not (np.isfinite(ell[i]) and ell[i] >= MIN_L):
            continue
        if not (np.isfinite(upper[i]) and upper[i] >= MIN_UPPER):
            continue
        if not (np.isfinite(cover[i]) and cover[i] >= MIN_COVER):
            continue
        if float(prob[i]) < MIN_P:
            continue
        added.add(cell)

    base = kept | added
    added_up: set = set()
    for cell in base:
        if cell not in where:
            continue
        above = _floor_above(cell[0], floor_y)
        if above is None or med <= 0:
            continue
        cand = (above, cell[1])
        if cand in base or cand in added_up or cand not in where:
            continue
        height = floor_y[above][1] - floor_y[above][0]
        if height < MIN_TOP_FRAC * med:
            continue
        i = where[cand]
        j = where[cell]
        if not (
            np.isfinite(ell[i])
            and np.isfinite(ell[j])
            and abs(float(ell[i]) - float(ell[j])) <= MAX_L_DELTA
        ):
            continue
        if not (np.isfinite(upper[i]) and upper[i] >= MIN_UP_UPPER):
            continue
        added_up.add(cand)
    return kept | added | added_up, dropped_top, dropped_low, added, added_up


def _cells(cells: set) -> str:
    return ",".join(f"f{f}b{b}" for f, b in sorted(cells)) or "-"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-cells", type=Path, default=DEFAULT_TRAIN_CELLS)
    ap.add_argument("--train-labels", type=Path, default=DEFAULT_TRAIN_LABELS)
    ap.add_argument("--eval-cells", type=Path, default=DEFAULT_CELLS)
    ap.add_argument("--eval-labels", type=Path, default=DEFAULT_EVAL_LABELS_JSONL)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_EVAL_PHOTO_DIR)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--t", type=float, default=0.3)
    ap.add_argument(
        "--gate",
        type=float,
        default=0.0,
        help="blank a photo when its gate max is below this; 0 keeps every photo",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = DEFAULT_GATE_OUT if args.gate > 0 else DEFAULT_OUT

    train = Data(args.train_cells, args.train_labels)
    test = Data(args.eval_cells, args.eval_labels)
    p_gate, p_cell = Model(train, train.stems, SPECS["base"], None).predict(test, test.stems)
    mir = mirror_index(test, args.dsl_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    preds: dict[str, set] = {}
    rules_preds: dict[str, set] = {}
    pure: dict[str, set] = {}
    rows = []
    tiles: dict[str, list[Image.Image]] = {c: [] for c in CATEGORIES}
    report: list[str] = []
    title = (
        f"4 rules + gate>={args.gate:g}  t={args.t}"
        if args.gate > 0
        else f"pure classifier + 4 rules  t={args.t}"
    )
    for s in test.stems:
        ii = test.idx[s]
        gt = set(test.gt[s])
        clf = rect_decode(test, s, p_cell, args.t, mir)
        pure[s] = clf
        floor_y, bay_x = load_grid(args.dsl_dir / s / "facade_dsl.json")
        rules, dropped_top, dropped_low, added, added_up = apply_rules(
            test, s, clf, p_cell, floor_y
        )
        gmax = float(p_gate[ii].max())
        gated_off = args.gate > 0 and gmax < args.gate
        pred = set() if gated_off else rules
        rules_preds[s] = rules
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
        lines = [
            f"{s}  [{cat}]  label={len(gt)}  gate={gmax:.2f}" + ("  GATE OFF" if gated_off else ""),
            title,
            f"result exact={int(pred == gt)}  IoU={iou(pred, gt):.2f}  cells={len(pred)}",
            f"drop top {_cells(dropped_top)}   drop p<0.05 {_cells(dropped_low)}",
            f"add OP1 {_cells(added)}   add up {_cells(added_up)}"
            + (f"   blanked {_cells(rules)}" if gated_off else ""),
        ]
        photo = Image.open(find_photo(args.in_dir, s)).convert("RGB")
        img = render_compare(photo, floor_y, bay_x, gt, pred)
        img = draw_probs(img, floor_y, bay_x, {test.cells[i]: float(p_cell[i]) for i in ii}, args.t)
        panel = add_header(img, lines)
        panel.save(args.out_dir / f"{cat}__{s}.jpg", quality=92)
        tiles[cat].append(panel)
        report.append(
            f"{s}  {cat:<9}  gate={gmax:.2f}  gated={int(gated_off)}  "
            f"exact={int(pred == gt)}  iou={iou(pred, gt):.2f}  "
            f"clf_exact={int(clf == gt)}  drop_top={len(dropped_top)}  "
            f"drop_low={len(dropped_low)}  added={len(added)} ({_cells(added)})  "
            f"up={len(added_up)} ({_cells(added_up)})"
        )

    for cat in CATEGORIES:
        if tiles[cat]:
            board(tiles[cat]).save(args.out_dir / f"_board_{cat}.jpg", quality=85)
    counts = "  ".join(f"{c}={len(tiles[c])}" for c in CATEGORIES)
    text = format_report(summarize(rows), rows)
    score_lines = [
        line("pure classifier (no gate)", score_sets(test, pure)),
        line("pure + 4 rules", score_sets(test, rules_preds)),
    ]
    if args.gate > 0:
        score_lines.append(line(f"4 rules + gate>={args.gate:g}", score_sets(test, preds)))
    header = "\n".join(
        [
            *score_lines,
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
        json.dumps({s: [list(c) for c in sorted(v)] for s, v in preds.items()}) + "\n",
        encoding="utf-8",
    )
    print(header)
    print(f"{len(test.stems)} images -> {args.out_dir}")


if __name__ == "__main__":
    main()
