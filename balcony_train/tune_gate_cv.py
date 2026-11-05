"""Pick the photo gate and cell threshold by cross-validation on the training photos.

The pipeline cell model (``Model(..., SPECS["base"])``, as in
``enclosed_detect.load_model``) is refit on 4/5 of the training photos and
predicts the held-out fifth (folds by photo, stratified on oriel / no oriel).
Every held-out photo is then decoded exactly like ``decode_cells`` (rectangle
decode, four cell rules, photo gate) for each (gate, t) on the grid. The
choice maximises pooled per-column cell F1 over all training photos. No test
or dev photo is read.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/tune_gate_cv.py
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
_PIPE = ROOT / "balcony_pipeline"
if str(_PIPE) not in sys.path:
    sys.path.insert(0, str(_PIPE))

from balcony_train.cell_clf_experiments import Data  # noqa: E402
from balcony_train.cell_clf_search import SPECS, Model  # noqa: E402
from balcony_train.cell_clf_train import DEFAULT_TRAIN_CELLS, DEFAULT_TRAIN_LABELS  # noqa: E402
from balcony_train.enclosed_detect import CELL_T, PHOTO_GATE, decode_cells  # noqa: E402
from balcony_train.enclosed_grid_store import DEFAULT_DSL_DIR  # noqa: E402
from snap import _bands_from_windows_dsl  # noqa: E402

GATES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
TS = (0.2, 0.3, 0.4, 0.5, 0.6)


def oof(data: Data, folds: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import StratifiedGroupKFold

    stems = np.asarray(data.stems)
    has = np.array([int(bool(data.gt[s])) for s in stems])
    p_gate = np.zeros(len(data.y))
    p_cell = np.zeros(len(data.y))
    split = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=0)
    for k, (tr, te) in enumerate(split.split(stems, has, groups=stems)):
        g, c = Model(data, list(stems[tr]), SPECS["base"], None).predict(data, list(stems[te]))
        rows = np.concatenate([data.idx[s] for s in stems[te]])
        p_gate[rows], p_cell[rows] = g[rows], c[rows]
        print(f"fold {k + 1}/{folds}  train={len(tr)}  held-out={len(te)}", flush=True)
    return p_gate, p_cell


def pooled(data: Data, preds: dict[str, set], stems: list[str]) -> dict:
    tp = fp = fn = exact = none_ok = n_none = 0
    for s in stems:
        g, p = set(data.gt[s]), preds[s]
        tp, fp, fn = tp + len(g & p), fp + len(p - g), fn + len(g - p)
        exact += g == p
        if not g:
            n_none += 1
            none_ok += not p
    pr = tp / max(1, tp + fp)
    rc = tp / max(1, tp + fn)
    return {
        "tp": tp, "fp": fp, "fn": fn, "precision": pr, "recall": rc,
        "f1": 2 * pr * rc / max(1e-9, pr + rc), "exact": exact, "none_ok": none_ok, "none": n_none, "photos": len(stems),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-cells", type=Path, default=DEFAULT_TRAIN_CELLS)
    ap.add_argument("--train-labels", type=Path, default=DEFAULT_TRAIN_LABELS)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "cell_clf" / "tune_gate_cv.json")
    args = ap.parse_args()

    data = Data(args.train_cells, args.train_labels)
    n_enc = sum(bool(data.gt[s]) for s in data.stems)
    print(f"train photos {len(data.stems)} (oriel {n_enc})  cells {len(data.y)}  pos {int(data.y.sum())}")
    grids = {
        s: _bands_from_windows_dsl(json.loads((args.dsl_dir / s / "facade_dsl.json").read_text(encoding="utf-8")))
        for s in data.stems
    }
    p_gate, p_cell = oof(data, args.folds)
    cmpb = [s for s in data.stems if s.startswith("cmp_b")]
    rows = []
    for gate in GATES:
        for t in TS:
            preds = {s: decode_cells(data, s, p_gate, p_cell, *grids[s], gate=gate, t=t)[0] for s in data.stems}
            rows.append({"gate": gate, "t": t, "all": pooled(data, preds, data.stems), "cmp_b": pooled(data, preds, cmpb)})
    best = max(rows, key=lambda r: (round(r["all"]["f1"], 6), -abs(r["gate"] - PHOTO_GATE), -abs(r["t"] - CELL_T)))
    cur = next(r for r in rows if r["gate"] == PHOTO_GATE and r["t"] == CELL_T)
    print("| gate | t | P | R | F1 | exact | no-oriel ok | cmp_b F1 | cmp_b no-oriel ok |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        a, b = r["all"], r["cmp_b"]
        mark = " (chosen)" if r is best else " (current)" if r is cur else ""
        print(
            f"| {r['gate']:.2f} | {r['t']:.1f}{mark} | {a['precision']:.3f} | {a['recall']:.3f} | {a['f1']:.3f} | "
            f"{a['exact']}/{a['photos']} | {a['none_ok']}/{a['none']} | {b['f1']:.3f} | {b['none_ok']}/{b['none']} |"
        )
    print(f"chosen gate={best['gate']} t={best['t']}  (current gate={PHOTO_GATE} t={CELL_T})")
    args.out.write_text(json.dumps({"chosen": {"gate": best["gate"], "t": best["t"]}, "rows": rows}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
