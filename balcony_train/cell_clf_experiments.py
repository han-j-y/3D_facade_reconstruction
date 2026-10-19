"""Cell-classifier and hybrid experiments with leave-one-photo-out CV.

Every photo is predicted by a model that never saw it. Thresholds are picked
per held-out photo on the other photos' out-of-fold scores, then applied to
the held-out photo. Reports the same rates as ``evaluate_enclosed_grid``.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/cell_clf_experiments.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.enclosed_grid_store import DEFAULT_EVAL_LABELS_JSONL, load_labels  # noqa: E402
from balcony_train.evaluate_enclosed_grid import score_photo, summarize  # noqa: E402

DEFAULT_CELLS = ROOT / "runs" / "cell_clf" / "cells.npz"
CURRENT = ("in_cells_5", "in_cells_6", "cover_2_depth", "cover_3_merge", "cover_4_grow", "cover_5_clip", "mirror_in_6")


class Data:
    def __init__(self, path: Path, labels_path: Path) -> None:
        z = np.load(path, allow_pickle=False)
        self.X = z["X"].astype(np.float64)
        self.y = z["y"].astype(int)
        self.groups = z["groups"].astype(str)
        self.names = [str(n) for n in z["names"]]
        self.cells = [tuple(map(int, c)) for c in z["cells"]]
        self.dino = z["dino"].astype(np.float64)
        self.stems = sorted(set(self.groups))
        labels = load_labels(labels_path)
        self.gt = {s: [tuple(c) for c in labels.get(s, [])] for s in self.stems}
        self.idx = {s: np.where(self.groups == s)[0] for s in self.stems}
        self.col = {n: i for i, n in enumerate(self.names)}

    def f(self, name: str) -> np.ndarray:
        return self.X[:, self.col[name]]

    def floor_order(self, stem: str) -> dict[int, int]:
        floors = sorted({self.cells[i][0] for i in self.idx[stem]})
        return {f: k for k, f in enumerate(floors)}


def column_runs(cells: set, rank: dict[int, int], min_run: int) -> set:
    if min_run <= 1:
        return set(cells)
    inv = {k: f for f, k in rank.items()}
    keep = set()
    for b in {c[1] for c in cells}:
        ks = sorted(rank[f] for f, bb in cells if bb == b)
        run: list[int] = []
        for k in ks + [None]:  # type: ignore[list-item]
            if k is not None and run and k == run[-1] + 1:
                run.append(k)
                continue
            if len(run) >= min_run:
                keep.update((inv[r], b) for r in run)
            run = [k] if k is not None else []
    return keep


def score_sets(data: Data, preds: dict[str, set]) -> dict[str, Any]:
    rows = []
    for s in data.stems:
        row = score_photo(data.gt[s], preds.get(s, set()))
        row["stem"] = s
        rows.append(row)
    return {"summary": summarize(rows), "rows": rows}


def exact_count(data: Data, preds: dict[str, set], stems: list[str]) -> int:
    return sum(int(set(data.gt[s]) == preds.get(s, set())) for s in stems)


def decode(data: Data, stem: str, prob: np.ndarray, t: float, gate: float, min_run: int) -> set:
    ii = data.idx[stem]
    p = prob[ii]
    if p.max() < gate:
        return set()
    cells = {data.cells[i] for i, v in zip(ii, p) if v >= t}
    return column_runs(cells, data.floor_order(stem), min_run)


def nested_decode(
    data: Data,
    prob: np.ndarray,
    grid: list[tuple[float, float, int]],
) -> tuple[dict[str, set], list[tuple[float, float, int]]]:
    """Pick (t, gate, min_run) on the other photos, apply to the held-out one."""
    table = {
        params: {s: decode(data, s, prob, *params) for s in data.stems} for params in grid
    }
    hits = {params: {s: int(set(data.gt[s]) == table[params][s]) for s in data.stems} for params in grid}
    totals = {params: sum(h.values()) for params, h in hits.items()}
    preds, chosen = {}, []
    for s in data.stems:
        best = max(grid, key=lambda p: (totals[p] - hits[p][s], -abs(p[0] - 0.5)))
        preds[s] = table[best][s]
        chosen.append(best)
    return preds, chosen


def oof_probs(data: Data, make_model: Callable[[], Any], cols: np.ndarray, *, dino_pca: int = 0) -> np.ndarray:
    from sklearn.decomposition import PCA

    prob = np.zeros(len(data.y))
    for s in data.stems:
        te = data.idx[s]
        tr = np.where(data.groups != s)[0]
        Xtr, Xte = data.X[tr][:, cols], data.X[te][:, cols]
        if dino_pca:
            pca = PCA(n_components=dino_pca, random_state=0).fit(data.dino[tr])
            Xtr = np.hstack([Xtr, pca.transform(data.dino[tr])])
            Xte = np.hstack([Xte, pca.transform(data.dino[te])])
        model = make_model().fit(Xtr, data.y[tr])
        prob[te] = model.predict_proba(Xte)[:, 1]
    return prob


def hgb(**kw):
    from sklearn.ensemble import HistGradientBoostingClassifier

    def make():
        return HistGradientBoostingClassifier(
            max_iter=kw.get("iters", 200),
            learning_rate=kw.get("lr", 0.05),
            max_leaf_nodes=kw.get("leaves", 15),
            min_samples_leaf=kw.get("min_leaf", 20),
            l2_regularization=kw.get("l2", 1.0),
            class_weight=kw.get("cw", None),
            random_state=0,
        )

    return make


def logreg(C: float = 0.3):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def make():
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=C, max_iter=2000, class_weight="balanced"),
        )

    return make


def line(name: str, res: dict[str, Any]) -> str:
    s = res["summary"]
    e, n, t = s["enclosure"], s["none"], s["total"]
    return (
        f"{name:<34} total exact {t['bay_floor']:>2}/{t['n']} ({t['bay_floor'] / t['n']:.3f})  "
        f"detect {t['detect']:>2}/{t['n']} ({t['detect'] / t['n']:.3f})  "
        f"enc {e['bay_floor']:>2}/{e['n']} det {e['detect']:>2}  none {n['bay_floor']:>2}/{n['n']}  "
        f"cells {s['tp']}/{s['fp']}/{s['fn']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    ap.add_argument("--labels", type=Path, default=DEFAULT_EVAL_LABELS_JSONL)
    ap.add_argument("--only", default="", help="comma list of experiment names")
    ap.add_argument("--dump", type=Path, default=ROOT / "runs" / "cell_clf" / "results.json")
    args = ap.parse_args()

    data = Data(args.cells, args.labels)
    print(f"photos={len(data.stems)} cells={len(data.y)} pos={int(data.y.sum())}")
    only = {x for x in args.only.split(",") if x}
    results: dict[str, Any] = {}

    def report(name: str, preds: dict[str, set], extra: str = "") -> None:
        res = score_sets(data, preds)
        results[name] = {"preds": {s: sorted(v) for s, v in preds.items()}, "summary": res["summary"]}
        print(line(name, res) + (f"  {extra}" if extra else ""), flush=True)

    base6 = {s: {data.cells[i] for i in data.idx[s] if data.f("in_cells_6")[i] > 0} for s in data.stems}
    base5 = {s: {data.cells[i] for i in data.idx[s] if data.f("in_cells_5")[i] > 0} for s in data.stems}
    report("current stage5", base5)
    report("current stage6 (sym)", base6)

    grid = [
        (t, g, r)
        for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
        for g in (0.0, 0.5, 0.6, 0.7, 0.8)
        for r in (1, 2)
        if g == 0.0 or g >= t
    ]
    hand = np.array([i for i, n in enumerate(data.names) if n not in CURRENT])
    allc = np.arange(len(data.names))
    experiments = {
        "lr hand": (logreg(0.3), hand, 0),
        "lr hand+current": (logreg(0.3), allc, 0),
        "hgb hand": (hgb(), hand, 0),
        "hgb hand+current": (hgb(), allc, 0),
        "hgb hand+current+dino16": (hgb(), allc, 16),
        "hgb hand+dino16": (hgb(), hand, 16),
        "lr hand+current+dino16": (logreg(0.1), allc, 16),
    }
    probs: dict[str, np.ndarray] = {}
    for name, (make, cols, pca) in experiments.items():
        if only and name not in only:
            continue
        prob = oof_probs(data, make, cols, dino_pca=pca)
        probs[name] = prob
        preds, chosen = nested_decode(data, prob, grid)
        common = max(set(chosen), key=chosen.count)
        report(name, preds, f"usual (t,gate,run)={common}")

    # Hybrid: current output plus classifier cells (union), and veto (drop low-prob current cells).
    for name, prob in probs.items():
        grid_h = [(t, v, r) for t in (0.5, 0.6, 0.7, 0.8, 0.9) for v in (0.0, 0.1, 0.2, 0.3) for r in (1, 2)]

        def hyb(s, t, v, r):
            ii = data.idx[s]
            add = {data.cells[i] for i in ii if prob[i] >= t}
            keep = {data.cells[i] for i in ii if data.f("in_cells_6")[i] > 0 and prob[i] >= v}
            return column_runs(keep | add, data.floor_order(s), r) if (add - keep) else keep

        table = {p: {s: hyb(s, *p) for s in data.stems} for p in grid_h}
        hits = {p: {s: int(set(data.gt[s]) == table[p][s]) for s in data.stems} for p in grid_h}
        tot = {p: sum(h.values()) for p, h in hits.items()}
        preds, chosen = {}, []
        for s in data.stems:
            best = max(grid_h, key=lambda p: (tot[p] - hits[p][s], p[0]))
            preds[s] = table[best][s]
            chosen.append(best)
        report(f"hybrid current+{name}", preds, f"usual (add_t,veto,run)={max(set(chosen), key=chosen.count)}")

    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        args.dump.write_text(json.dumps(results, default=list) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
