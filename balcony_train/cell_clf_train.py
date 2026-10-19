"""Train the cell classifier on cmp_b only, then score the held-out eval set once.

Same method as the round 5 best (``cell_clf_v5`` "dino16"):
  gate   HGB on hand features + DINOv2 PCA16, every train photo; a photo is
         gated on when its max cell probability >= gate.
  cell   HGB (min_leaf 10) on all features + DINOv2 PCA16, enclosure photos only.
  score  a * cell + (1 - a) * in_cells_6, rectangle + mirror decoding at t.

Eval photos are never used for training or for choosing (a, t, gate): the
parameters are picked on grouped K-fold out-of-fold scores inside the train
set. ``--objective balanced`` (default) maximizes the mean of the enclosure
and none exact rates, since train has far fewer enclosure photos than eval.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/cell_clf_train.py
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

from balcony_train.cell_clf_experiments import CURRENT, DEFAULT_CELLS, Data, hgb, line, score_sets  # noqa: E402
from balcony_train.cell_clf_v2 import mirror_index  # noqa: E402
from balcony_train.cell_clf_v3 import rect_decode  # noqa: E402
from balcony_train.enclosed_grid_store import DEFAULT_DSL_DIR, DEFAULT_EVAL_LABELS_JSONL  # noqa: E402

DEFAULT_TRAIN_CELLS = ROOT / "runs" / "cell_clf" / "cells_train.npz"
DEFAULT_TRAIN_LABELS = ROOT / "runs" / "enclosed_grid" / "labels_train.jsonl"
DEFAULT_MODEL = ROOT / "runs" / "cell_clf" / "model_base.joblib"
DEFAULT_DUMP = ROOT / "runs" / "cell_clf" / "results_base.json"
DINO_PCA = 16
GRID = list(
    itertools.product(
        (0.0, 0.3, 0.5, 0.7, 1.0),  # a: classifier weight vs current cells
        (0.3, 0.4, 0.5, 0.6),  # t: cell threshold
        (0.3, 0.4, 0.5, 0.6, 0.7, 0.8),  # gate
    )
)


def hand_cols(data: Data) -> np.ndarray:
    return np.array([i for i, n in enumerate(data.names) if n not in CURRENT])


def fit_model(data: Data, rows: np.ndarray, cols: np.ndarray, make):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=DINO_PCA, random_state=0).fit(data.dino[rows])
    X = np.hstack([data.X[rows][:, cols], pca.transform(data.dino[rows])])
    return make().fit(X, data.y[rows]), pca


def predict(model, pca, data: Data, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    X = np.hstack([data.X[rows][:, cols], pca.transform(data.dino[rows])])
    return model.predict_proba(X)[:, 1]


def fit_pair(data: Data, stems: list[str]) -> dict:
    hand = hand_cols(data)
    allc = np.arange(len(data.names))
    rows = np.concatenate([data.idx[s] for s in stems])
    enc_rows = np.concatenate([data.idx[s] for s in stems if data.gt[s]])
    gate_model, gate_pca = fit_model(data, rows, hand, hgb())
    cell_model, cell_pca = fit_model(data, enc_rows, allc, hgb(min_leaf=10))
    return {
        "gate": (gate_model, gate_pca, hand),
        "cell": (cell_model, cell_pca, allc),
        "names": list(data.names),
    }


def apply_pair(pair: dict, data: Data, stems: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if list(data.names) != pair["names"]:
        raise SystemExit("feature names differ between train and this cells file")
    p_gate = np.zeros(len(data.y))
    p_cell = np.zeros(len(data.y))
    rows = np.concatenate([data.idx[s] for s in stems])
    m, p, c = pair["gate"]
    p_gate[rows] = predict(m, p, data, rows, c)
    m, p, c = pair["cell"]
    p_cell[rows] = predict(m, p, data, rows, c)
    return p_gate, p_cell


def oof_train(data: Data, folds: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import StratifiedGroupKFold

    stems = np.asarray(data.stems)
    has = np.array([int(bool(data.gt[s])) for s in stems])
    p_gate = np.zeros(len(data.y))
    p_cell = np.zeros(len(data.y))
    split = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=0)
    for k, (tr, te) in enumerate(split.split(stems, has, groups=stems)):
        pair = fit_pair(data, list(stems[tr]))
        g, c = apply_pair(pair, data, list(stems[te]))
        rows = np.concatenate([data.idx[s] for s in stems[te]])
        p_gate[rows] = g[rows]
        p_cell[rows] = c[rows]
        print(f"fold {k + 1}/{folds}  train={len(tr)}  held-out={len(te)}", flush=True)
    return p_gate, p_cell


def decode_all(data: Data, p_gate, p_cell, mir, params) -> dict[str, set]:
    a, t, g = params
    score = a * p_cell + (1 - a) * data.f("in_cells_6")
    out = {}
    for s in data.stems:
        if float(p_gate[data.idx[s]].max()) < g:
            out[s] = set()
        else:
            out[s] = rect_decode(data, s, score, t, mir)
    return out


def objective(data: Data, preds: dict[str, set], kind: str) -> float:
    enc = [s for s in data.stems if data.gt[s]]
    none = [s for s in data.stems if not data.gt[s]]
    hit = {s: int(set(data.gt[s]) == preds[s]) for s in data.stems}
    if kind == "total":
        return float(sum(hit.values()))
    return 0.5 * (sum(hit[s] for s in enc) / max(1, len(enc)) + sum(hit[s] for s in none) / max(1, len(none)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-cells", type=Path, default=DEFAULT_TRAIN_CELLS)
    ap.add_argument("--train-labels", type=Path, default=DEFAULT_TRAIN_LABELS)
    ap.add_argument("--eval-cells", type=Path, default=DEFAULT_CELLS)
    ap.add_argument("--eval-labels", type=Path, default=DEFAULT_EVAL_LABELS_JSONL)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--objective", choices=("balanced", "total"), default="balanced")
    ap.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    args = ap.parse_args()

    import joblib

    train = Data(args.train_cells, args.train_labels)
    test = Data(args.eval_cells, args.eval_labels)
    overlap = set(train.stems) & set(test.stems)
    if overlap:
        raise SystemExit(f"train and eval share photos: {sorted(overlap)[:5]}")
    n_enc = sum(bool(train.gt[s]) for s in train.stems)
    print(f"train photos={len(train.stems)} (enclosure {n_enc})  cells={len(train.y)} pos={int(train.y.sum())}")
    print(f"eval  photos={len(test.stems)}  cells={len(test.y)} pos={int(test.y.sum())}")

    mir_tr = mirror_index(train, args.dsl_dir)
    p_gate, p_cell = oof_train(train, args.folds)
    tables = {p: decode_all(train, p_gate, p_cell, mir_tr, p) for p in GRID}
    best = max(GRID, key=lambda p: (objective(train, tables[p], args.objective), str(p)))
    a, t, g = best
    print(f"chosen on train ({args.objective}): a={a} t={t} gate={g}")
    print(line("train OOF (chosen params)", score_sets(train, tables[best])))

    pair = fit_pair(train, train.stems)
    eg, ec = apply_pair(pair, test, test.stems)
    preds = decode_all(test, eg, ec, mirror_index(test, args.dsl_dir), best)
    base = {s: {test.cells[i] for i in test.idx[s] if test.f("in_cells_6")[i] > 0} for s in test.stems}
    res_base = score_sets(test, base)
    res = score_sets(test, preds)
    print(line("eval current stage6 (sym)", res_base))
    print(line("eval cell_clf (train=cmp_b)", res))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({**pair, "params": {"a": a, "t": t, "gate": g}, "dino_pca": DINO_PCA}, args.model_out)
    args.dump.write_text(
        json.dumps(
            {
                "params": {"a": a, "t": t, "gate": g, "objective": args.objective},
                "eval": {"summary": res["summary"], "preds": {s: sorted(v) for s, v in preds.items()}},
                "eval_current": res_base["summary"],
                "gate_max": {s: float(eg[test.idx[s]].max()) for s in test.stems},
            },
            default=list,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"model -> {args.model_out}\nresults -> {args.dump}")


if __name__ == "__main__":
    main()
