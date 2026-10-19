"""Round 2: photo gate + structured cell decoding (leave-one-photo-out).

Ideas tested on top of ``cell_clf_experiments``:
  gate      photo-level "has enclosure" from the classifier's max cell score;
            a gated-off photo predicts no cells.
  current   inside gated-on photos, keep the current method's cells.
  blend     cell score = a * classifier + (1 - a) * current cell, then
            threshold, optional mirror averaging and column runs.
All parameters are chosen on the other 60 photos for each held-out photo.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.cell_clf_experiments import (  # noqa: E402
    CURRENT,
    DEFAULT_CELLS,
    Data,
    column_runs,
    hgb,
    line,
    logreg,
    oof_probs,
    score_sets,
)
from balcony_train.depth_cells import mirror_bay  # noqa: E402
from balcony_train.enclosed_grid_store import DEFAULT_DSL_DIR, DEFAULT_EVAL_LABELS_JSONL, load_grid  # noqa: E402
from balcony_train.enclosed_symmetry import SYM_GATE, grid_axis_x  # noqa: E402


def mirror_index(data: Data, dsl_dir: Path) -> np.ndarray:
    """Row index of each cell's mirror cell (or -1), for photos with symmetric windows."""
    out = np.full(len(data.y), -1, dtype=int)
    wsym = data.f("window_sym")
    for s in data.stems:
        ii = data.idx[s]
        if wsym[ii[0]] < SYM_GATE:
            continue
        _fy, bay_x = load_grid(dsl_dir / s / "facade_dsl.json")
        cx = grid_axis_x(bay_x)
        where = {data.cells[i]: i for i in ii}
        for i in ii:
            f, b = data.cells[i]
            mb = mirror_bay(b, bay_x, cx) if cx is not None else None
            if mb is not None and mb != b and (f, mb) in where:
                out[i] = where[(f, mb)]
    return out


def choose_nested(data: Data, table: dict, grid: list) -> tuple[dict, list]:
    hits = {p: {s: int(set(data.gt[s]) == table[p][s]) for s in data.stems} for p in grid}
    tot = {p: sum(h.values()) for p, h in hits.items()}
    preds, chosen = {}, []
    for s in data.stems:
        best = max(grid, key=lambda p: (tot[p] - hits[p][s], str(p)))
        preds[s] = table[best][s]
        chosen.append(best)
    return preds, chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    ap.add_argument("--labels", type=Path, default=DEFAULT_EVAL_LABELS_JSONL)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--dump", type=Path, default=ROOT / "runs" / "cell_clf" / "results_v2.json")
    args = ap.parse_args()

    data = Data(args.cells, args.labels)
    mir = mirror_index(data, args.dsl_dir)
    cur = data.f("in_cells_6")
    hand = np.array([i for i, n in enumerate(data.names) if n not in CURRENT])
    allc = np.arange(len(data.names))
    models = {
        "hgb hand+dino16": oof_probs(data, hgb(), hand, dino_pca=16),
        "hgb hand+current+dino16": oof_probs(data, hgb(), allc, dino_pca=16),
        "lr hand+current+dino16": oof_probs(data, logreg(0.1), allc, dino_pca=16),
    }
    models["avg3"] = np.mean(list(models.values()), axis=0)
    results = {}

    def report(name, preds, extra=""):
        res = score_sets(data, preds)
        results[name] = {"preds": {s: sorted(v) for s, v in preds.items()}, "summary": res["summary"]}
        print(line(name, res) + (f"  {extra}" if extra else ""), flush=True)

    base = {s: {data.cells[i] for i in data.idx[s] if cur[i] > 0} for s in data.stems}
    report("current stage6 (sym)", base)

    for mname, prob in models.items():
        # A. gate only: classifier decides "any enclosure", cells come from the current method.
        gates = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        table = {
            (g,): {s: (base[s] if prob[data.idx[s]].max() >= g else set()) for s in data.stems} for g in gates
        }
        preds, chosen = choose_nested(data, table, list(table))
        report(f"gate({mname}) + current", preds, f"usual gate={max(set(chosen), key=chosen.count)}")

        # B. blend + mirror + runs + gate.
        grid = list(
            itertools.product(
                (0.3, 0.5, 0.7),  # a: classifier weight
                (0.3, 0.4, 0.5, 0.6),  # t: cell threshold
                (0.0, 0.4, 0.6),  # gate
                (False, True),  # mirror averaging
                (1, 2),  # min run
            )
        )
        table = {}
        for a, t, g, m, r in grid:
            score = a * prob + (1 - a) * cur
            if m:
                ms = np.where(mir >= 0, score[np.maximum(mir, 0)], score)
                score = np.maximum(score, 0.5 * (score + ms))
            out = {}
            for s in data.stems:
                ii = data.idx[s]
                if prob[ii].max() < g:
                    out[s] = set()
                    continue
                cells = {data.cells[i] for i in ii if score[i] >= t}
                out[s] = column_runs(cells, data.floor_order(s), r)
            table[(a, t, g, m, r)] = out
        preds, chosen = choose_nested(data, table, grid)
        report(f"blend({mname})", preds, f"usual (a,t,gate,mirror,run)={max(set(chosen), key=chosen.count)}")

    args.dump.write_text(json.dumps(results) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
