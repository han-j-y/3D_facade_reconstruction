"""OP3 gate -> OP1: the gate drops photos without an enclosure, OP1 places the cells.

A photo is gated on when the max cell probability of the OP3 gate model is
>= the threshold; gated-on photos keep the OP1 (Stage 6) cells, others get none.
The threshold is chosen on cmp_b out-of-fold gate scores (3 seeds x 5 folds);
the eval set is scored once with the gate trained on all of cmp_b.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/gate_op1.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.cell_clf_experiments import DEFAULT_CELLS, Data, line, score_sets  # noqa: E402
from balcony_train.cell_clf_search import SEEDS, SPECS, Model, balanced, oof  # noqa: E402
from balcony_train.cell_clf_train import DEFAULT_TRAIN_CELLS, DEFAULT_TRAIN_LABELS  # noqa: E402
from balcony_train.enclosed_grid_store import DEFAULT_EVAL_LABELS_JSONL  # noqa: E402

DEFAULT_DUMP = ROOT / "runs" / "cell_clf" / "gate_op1.json"
THRESHOLDS = [round(x, 2) for x in np.arange(0.0, 0.91, 0.05)]


def op1_cells(data: Data) -> dict[str, set]:
    cur = data.f("in_cells_6")
    return {s: {data.cells[i] for i in data.idx[s] if cur[i] > 0} for s in data.stems}


def gated(data: Data, op1: dict[str, set], gmax: dict[str, float], thr: float) -> dict[str, set]:
    return {s: (op1[s] if gmax[s] >= thr else set()) for s in data.stems}


def hits(data: Data, preds: dict[str, set]) -> dict[str, int]:
    return {s: int(set(data.gt[s]) == preds[s]) for s in data.stems}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-cells", type=Path, default=DEFAULT_TRAIN_CELLS)
    ap.add_argument("--train-labels", type=Path, default=DEFAULT_TRAIN_LABELS)
    ap.add_argument("--eval-cells", type=Path, default=DEFAULT_CELLS)
    ap.add_argument("--eval-labels", type=Path, default=DEFAULT_EVAL_LABELS_JSONL)
    ap.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    args = ap.parse_args()

    train = Data(args.train_cells, args.train_labels)
    test = Data(args.eval_cells, args.eval_labels)
    if set(train.stems) & set(test.stems):
        raise SystemExit("train and eval share photos")
    op1_tr, op1_te = op1_cells(train), op1_cells(test)
    print(line("train OP1 only", score_sets(train, op1_tr)))

    per_seed = []
    for seed in SEEDS:
        p_gate, _p_cell, _fold = oof(train, SPECS["base"], None, seed)
        gmax = {s: float(p_gate[train.idx[s]].max()) for s in train.stems}
        per_seed.append({t: hits(train, gated(train, op1_tr, gmax, t)) for t in THRESHOLDS})

    print("\ntrain CV (mean of 3 seeds)")
    print("  thr   balanced  total    enc     none")
    table = {}
    for t in THRESHOLDS:
        bal = float(np.mean([balanced(train, h[t], train.stems) for h in per_seed]))
        tot = float(np.mean([sum(h[t].values()) for h in per_seed]))
        enc = float(np.mean([sum(h[t][s] for s in train.stems if train.gt[s]) for h in per_seed]))
        table[t] = {"balanced": bal, "total": tot, "enc": enc, "none": tot - enc}
        print(f"  {t:.2f}  {bal:.3f}     {tot:6.1f}  {enc:5.1f}  {tot - enc:6.1f}")
    best = max(THRESHOLDS, key=lambda t: (table[t]["balanced"], -t))
    print(f"\nchosen on train (balanced): gate >= {best}\n")

    model = Model(train, train.stems, SPECS["base"], None)
    p_gate, _ = model.predict(test, test.stems)
    gmax_te = {s: float(p_gate[test.idx[s]].max()) for s in test.stems}
    print(line("eval OP1 only", score_sets(test, op1_te)))
    res = score_sets(test, gated(test, op1_te, gmax_te, best))
    print(line(f"eval gate>={best} -> OP1", res))

    args.dump.parent.mkdir(parents=True, exist_ok=True)
    args.dump.write_text(
        json.dumps({"threshold": best, "train_cv": table, "eval": res["summary"], "gate_max": gmax_te}, default=list)
        + "\n",
        encoding="utf-8",
    )
    print(f"results -> {args.dump}")


if __name__ == "__main__":
    main()
