"""Round 4: photo-relative cell scores + learned photo gate (leave-one-photo-out).

Cell scores differ in level between photos, but the ranking inside a photo
is good. Decoders here threshold ``p / max(p in photo)``. The photo gate is
a small logistic model on photo summaries (max/top-3 cell score, current
method output, SAM3 enclosure strength), itself trained leave-one-out.
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
from balcony_train.cell_clf_v2 import choose_nested, mirror_index  # noqa: E402
from balcony_train.cell_clf_v3 import oof_enc_only, rect_decode  # noqa: E402
from balcony_train.enclosed_grid_store import DEFAULT_DSL_DIR, DEFAULT_EVAL_LABELS_JSONL  # noqa: E402


def photo_summary(data: Data, probs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    cur = data.f("in_cells_6")
    enc_sam3 = np.nanmax(np.vstack([data.f(f"sam3_{k}") for k in range(3)]), axis=0)
    L = np.nan_to_num(data.f("L"), nan=0.0)
    rows, ys = [], []
    for s in data.stems:
        ii = data.idx[s]
        feats = []
        for p in probs:
            top = np.sort(p[ii])[::-1]
            feats += [top[0], top[: min(3, len(top))].mean(), float((p[ii] >= 0.5).sum())]
        feats += [
            float(cur[ii].sum() > 0),
            float(cur[ii].sum()),
            float(enc_sam3[ii].max()),
            float(np.sort(enc_sam3[ii])[::-1][:3].mean()),
            float(np.mean(L[ii] >= 0.015)),
            float(data.f("window_sym")[ii[0]]),
        ]
        rows.append(feats)
        ys.append(int(bool(data.gt[s])))
    return np.asarray(rows), np.asarray(ys)


def loo_gate(Xp: np.ndarray, yp: np.ndarray, C: float) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = np.zeros(len(yp))
    for k in range(len(yp)):
        tr = np.arange(len(yp)) != k
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(Xp[tr], yp[tr])
        out[k] = m.predict_proba(Xp[k : k + 1])[0, 1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    ap.add_argument("--labels", type=Path, default=DEFAULT_EVAL_LABELS_JSONL)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--dump", type=Path, default=ROOT / "runs" / "cell_clf" / "results_v4.json")
    args = ap.parse_args()

    data = Data(args.cells, args.labels)
    mir = mirror_index(data, args.dsl_dir)
    cur = data.f("in_cells_6")
    hand = np.array([i for i, n in enumerate(data.names) if n not in CURRENT])
    allc = np.arange(len(data.names))
    p_gate = oof_probs(data, hgb(), hand, dino_pca=16)
    p_all = oof_probs(data, hgb(), allc, dino_pca=16)
    p_enc = oof_enc_only(data, hgb(min_leaf=10), allc)
    p_enc_lr = oof_enc_only(data, logreg(0.1), allc)

    Xp, yp = photo_summary(data, [p_gate, p_all, p_enc])
    gates = {"max": {s: float(p_gate[data.idx[s]].max()) for s in data.stems}}
    for C in (0.05, 0.2, 1.0):
        g = loo_gate(Xp, yp, C)
        gates[f"lr C={C}"] = {s: float(v) for s, v in zip(data.stems, g)}
        acc = np.mean((g >= 0.5) == yp)
        print(f"photo gate lr C={C}: LOO acc@0.5={acc:.3f}")
    results = {}

    def report(name, preds, extra=""):
        res = score_sets(data, preds)
        results[name] = {"preds": {s: sorted(v) for s, v in preds.items()}, "summary": res["summary"]}
        print(line(name, res) + (f"  {extra}" if extra else ""), flush=True)

    cell_scores = {
        "enc hgb": p_enc,
        "enc avg": 0.5 * (p_enc + p_enc_lr),
        "all hgb": p_all,
    }
    for gname, gate in gates.items():
        for mname, prob in cell_scores.items():
            for decoder in ("rel runs", "rel rect+mirror", "abs rect+mirror"):
                grid = list(
                    itertools.product(
                        (0.5, 0.7, 1.0),  # a
                        (0.2, 0.3, 0.4, 0.5, 0.6) if decoder.startswith("rel") else (0.3, 0.4, 0.5),
                        (0.3, 0.4, 0.5, 0.6, 0.7, 0.8),  # gate threshold
                    )
                )
                table = {}
                for a, t, gthr in grid:
                    score = a * prob + (1 - a) * cur
                    out = {}
                    for s in data.stems:
                        ii = data.idx[s]
                        if gate[s] < gthr:
                            out[s] = set()
                            continue
                        sc = score.copy()
                        if decoder.startswith("rel"):
                            sc[ii] = sc[ii] / max(1e-6, sc[ii].max())
                        if decoder == "rel runs":
                            out[s] = column_runs({data.cells[i] for i in ii if sc[i] >= t}, data.floor_order(s), 1)
                        else:
                            out[s] = rect_decode(data, s, sc, t, mir)
                    table[(a, t, gthr)] = out
                preds, chosen = choose_nested(data, table, grid)
                report(f"[{gname}] {mname} / {decoder}", preds, f"usual={max(set(chosen), key=chosen.count)}")
    args.dump.write_text(json.dumps(results) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
