"""Search cell-classifier variants on cmp_b only; score the eval set once per variant.

Variants (model x decoder):
  model    base | weighted (balanced class weights) | flip (mirrored train
           photos) | weighted+flip
  decoder  clf          classifier cells only
           clf+current  when the classifier gives nothing, fall back to the
                        current method's cells if gate >= g2
Selection never looks at the eval set:
  * probabilities are out-of-fold (StratifiedGroupKFold over train photos),
    repeated over 3 seeds;
  * (a, t, gate, g2) for each fold are chosen on the other folds (nested);
  * the variant is the one with the best mean nested balanced exact rate
    (mean of enclosure and none exact rates).
A learning curve (50 / 75 / 100 % of train photos) for the chosen model shows
whether more labels are likely to help.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/cell_clf_search.py
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
from balcony_train.cell_clf_train import DEFAULT_TRAIN_CELLS, DEFAULT_TRAIN_LABELS, DINO_PCA  # noqa: E402
from balcony_train.cell_clf_v2 import mirror_index  # noqa: E402
from balcony_train.cell_clf_v3 import rect_decode  # noqa: E402
from balcony_train.enclosed_grid_store import DEFAULT_DSL_DIR, DEFAULT_EVAL_LABELS_JSONL  # noqa: E402

DEFAULT_FLIP = ROOT / "runs" / "cell_clf" / "cells_train_dino_flip.npy"
DEFAULT_DUMP = ROOT / "runs" / "cell_clf" / "search_base.json"
SEEDS = (0, 1, 2)
FOLDS = 5
SPECS = {
    "base": {"cw": None, "flip": False},
    "weighted": {"cw": "balanced", "flip": False},
    "flip": {"cw": None, "flip": True},
    "weighted+flip": {"cw": "balanced", "flip": True},
}
A_T = list(itertools.product((0.5, 0.7, 1.0), (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)))
GATES = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
FALLBACK = (0.0, 0.1, 0.2, 0.3)
DECODERS = {
    "clf": [(a, t, g, None) for (a, t), g in itertools.product(A_T, GATES)],
    "clf+current": [(a, t, g, g2) for (a, t), g, g2 in itertools.product(A_T, GATES, (None,) + FALLBACK)],
}


class Model:
    """Gate (all photos, hand features) + cell model (enclosure photos, all features)."""

    def __init__(self, data: Data, stems: list[str], spec: dict, dflip: np.ndarray | None) -> None:
        from sklearn.decomposition import PCA

        self.hand = np.array([i for i, n in enumerate(data.names) if n not in CURRENT])
        self.allc = np.arange(len(data.names))
        rows = np.concatenate([data.idx[s] for s in stems])
        enc = np.concatenate([data.idx[s] for s in stems if data.gt[s]])
        self.parts = []
        for rr, cols, leaf in ((rows, self.hand, 20), (enc, self.allc, 10)):
            X, D, y = data.X[rr], data.dino[rr], data.y[rr]
            if spec["flip"]:
                Xf = X.copy()
                k = data.col["bay_rank"]
                Xf[:, k] = 1.0 - Xf[:, k]
                X, D, y = np.vstack([X, Xf]), np.vstack([D, dflip[rr]]), np.concatenate([y, y])
            pca = PCA(n_components=DINO_PCA, random_state=0).fit(D)
            model = hgb(min_leaf=leaf, cw=spec["cw"])().fit(np.hstack([X[:, cols], pca.transform(D)]), y)
            self.parts.append((model, pca, cols))

    def predict(self, data: Data, stems: list[str]) -> tuple[np.ndarray, np.ndarray]:
        rows = np.concatenate([data.idx[s] for s in stems])
        out = []
        for model, pca, cols in self.parts:
            p = np.zeros(len(data.y))
            p[rows] = model.predict_proba(np.hstack([data.X[rows][:, cols], pca.transform(data.dino[rows])]))[:, 1]
            out.append(p)
        return out[0], out[1]


def subsample(data: Data, stems: list[str], frac: float, rng: np.random.Generator) -> list[str]:
    if frac >= 1.0:
        return list(stems)
    keep = []
    for has in (True, False):
        pool = [s for s in stems if bool(data.gt[s]) == has]
        keep += list(rng.choice(pool, size=max(1, round(frac * len(pool))), replace=False))
    return keep


def oof(data: Data, spec: dict, dflip, seed: int, frac: float = 1.0):
    from sklearn.model_selection import StratifiedGroupKFold

    stems = np.asarray(data.stems)
    has = np.array([int(bool(data.gt[s])) for s in stems])
    p_gate, p_cell = np.zeros(len(data.y)), np.zeros(len(data.y))
    fold_of = {}
    rng = np.random.default_rng(seed)
    split = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(split.split(stems, has, groups=stems)):
        m = Model(data, subsample(data, list(stems[tr]), frac, rng), spec, dflip)
        g, c = m.predict(data, list(stems[te]))
        rows = np.concatenate([data.idx[s] for s in stems[te]])
        p_gate[rows], p_cell[rows] = g[rows], c[rows]
        fold_of.update({s: k for s in stems[te]})
    return p_gate, p_cell, fold_of


class Decoded:
    """All decoder outputs for one set of probabilities."""

    def __init__(self, data: Data, p_gate, p_cell, mir) -> None:
        cur = data.f("in_cells_6")
        self.data = data
        self.gmax = {s: float(p_gate[data.idx[s]].max()) for s in data.stems}
        self.cur = {s: {data.cells[i] for i in data.idx[s] if cur[i] > 0} for s in data.stems}
        self.clf = {}
        for a, t in A_T:
            score = a * p_cell + (1 - a) * cur
            self.clf[(a, t)] = {s: rect_decode(data, s, score, t, mir) for s in data.stems}

    def pred(self, s: str, p) -> set:
        a, t, g, g2 = p
        gm = self.gmax[s]
        if gm >= g and self.clf[(a, t)][s]:
            return self.clf[(a, t)][s]
        if g2 is not None and gm >= g2 and self.cur[s]:
            return self.cur[s]
        return set()

    def preds(self, p) -> dict[str, set]:
        return {s: self.pred(s, p) for s in self.data.stems}

    def hits(self, grid, metric: str = "exact") -> dict:
        """Per-photo score: exact match, or cell IoU (an empty prediction on a none photo is 1)."""
        gt = self.data.gt

        def one(s, p):
            g, q = set(gt[s]), self.pred(s, p)
            if metric == "exact":
                return int(g == q)
            return 1.0 if not (g | q) else len(g & q) / len(g | q)

        return {p: {s: one(s, p) for s in self.data.stems} for p in grid}


def balanced(data: Data, hit: dict[str, int], stems) -> float:
    enc = [s for s in stems if data.gt[s]]
    none = [s for s in stems if not data.gt[s]]
    return 0.5 * (sum(hit[s] for s in enc) / max(1, len(enc)) + sum(hit[s] for s in none) / max(1, len(none)))


def nested(data: Data, dec: Decoded, hits: dict, grid, fold_of) -> dict[str, set]:
    preds = {}
    for k in sorted(set(fold_of.values())):
        inner = [s for s in data.stems if fold_of[s] != k]
        best = max(grid, key=lambda p: (balanced(data, hits[p], inner), str(p)))
        for s in data.stems:
            if fold_of[s] == k:
                preds[s] = dec.pred(s, best)
    return preds


def photo_scores(data: Data, preds: dict[str, set], metric: str) -> dict[str, float]:
    out = {}
    for s in data.stems:
        g, q = set(data.gt[s]), preds[s]
        out[s] = float(g == q) if metric == "exact" else (1.0 if not (g | q) else len(g & q) / len(g | q))
    return out


def rates(data: Data, preds: dict[str, set]) -> tuple[float, float, int, int]:
    hit = {s: int(set(data.gt[s]) == preds[s]) for s in data.stems}
    enc = sum(hit[s] for s in data.stems if data.gt[s])
    return balanced(data, hit, data.stems), sum(hit.values()) / len(hit), enc, sum(hit.values()) - enc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-cells", type=Path, default=DEFAULT_TRAIN_CELLS)
    ap.add_argument("--train-labels", type=Path, default=DEFAULT_TRAIN_LABELS)
    ap.add_argument("--train-flip", type=Path, default=DEFAULT_FLIP)
    ap.add_argument("--eval-cells", type=Path, default=DEFAULT_CELLS)
    ap.add_argument("--eval-labels", type=Path, default=DEFAULT_EVAL_LABELS_JSONL)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    ap.add_argument(
        "--metric",
        choices=("exact", "iou"),
        default="exact",
        help="per-photo score used to choose parameters and the variant",
    )
    args = ap.parse_args()

    train = Data(args.train_cells, args.train_labels)
    test = Data(args.eval_cells, args.eval_labels)
    if set(train.stems) & set(test.stems):
        raise SystemExit("train and eval share photos")
    dflip = np.load(args.train_flip) if args.train_flip.is_file() else None
    mir_tr = mirror_index(train, args.dsl_dir)
    mir_te = mirror_index(test, args.dsl_dir)
    n_enc = sum(bool(train.gt[s]) for s in train.stems)
    print(f"train {len(train.stems)} photos ({n_enc} enclosure), eval {len(test.stems)} photos", flush=True)

    cv: dict[str, dict] = {}
    for sname, spec in SPECS.items():
        if spec["flip"] and dflip is None:
            print(f"skip {sname}: no {args.train_flip}")
            continue
        runs = []
        for seed in SEEDS:
            pg, pc, fold_of = oof(train, spec, dflip, seed)
            dec = Decoded(train, pg, pc, mir_tr)
            runs.append((dec, fold_of, {d: dec.hits(grid, args.metric) for d, grid in DECODERS.items()}))
        for dname, grid in DECODERS.items():
            nested_preds = [nested(train, dec, h[dname], grid, fo) for dec, fo, h in runs]
            scores = [rates(train, p) for p in nested_preds]
            sel = [balanced(train, photo_scores(train, p, args.metric), train.stems) for p in nested_preds]
            mean_hits = {p: np.mean([balanced(train, h[dname][p], train.stems) for _d, _f, h in runs]) for p in grid}
            full = max(grid, key=lambda p: (mean_hits[p], str(p)))
            key = f"{sname} / {dname}"
            cv[key] = {
                "spec": sname,
                "decoder": dname,
                "nested_select": float(np.mean(sel)),
                "nested_select_sd": float(np.std(sel)),
                "nested_balanced": float(np.mean([x[0] for x in scores])),
                "nested_balanced_sd": float(np.std([x[0] for x in scores])),
                "nested_total": float(np.mean([x[1] for x in scores])),
                "nested_enc": float(np.mean([x[2] for x in scores])),
                "nested_none": float(np.mean([x[3] for x in scores])),
                "params": full,
            }
            r = cv[key]
            print(
                f"CV {key:<28} {args.metric} {r['nested_select']:.3f}±{r['nested_select_sd']:.3f}  "
                f"exact balanced {r['nested_balanced']:.3f}  "
                f"total {r['nested_total']:.3f}  enc {r['nested_enc']:.1f}/{n_enc}  "
                f"none {r['nested_none']:.1f}/{len(train.stems) - n_enc}  params(a,t,gate,g2)={full}",
                flush=True,
            )

    chosen = max(cv, key=lambda k: cv[k]["nested_select"])
    print(f"\nchosen by train CV: {chosen}\n", flush=True)

    base = {s: {test.cells[i] for i in test.idx[s] if test.f("in_cells_6")[i] > 0} for s in test.stems}
    print(line("eval current stage6 (sym)", score_sets(test, base)))
    results = {"cv": cv, "chosen": chosen, "eval": {}}
    models = {}
    for key, r in cv.items():
        if r["spec"] not in models:
            m = Model(train, train.stems, SPECS[r["spec"]], dflip)
            models[r["spec"]] = Decoded(test, *m.predict(test, test.stems), mir_te)
        preds = models[r["spec"]].preds(tuple(r["params"]))
        res = score_sets(test, preds)
        results["eval"][key] = {"summary": res["summary"], "preds": {s: sorted(v) for s, v in preds.items()}}
        print(line(("* " if key == chosen else "  ") + f"eval {key}", res), flush=True)

    spec = cv[chosen]["spec"]
    dname = cv[chosen]["decoder"]
    print(f"\nlearning curve ({spec} / {dname}, nested train CV)", flush=True)
    results["learning_curve"] = {}
    for frac in (0.5, 0.75, 1.0):
        vals = []
        for seed in SEEDS:
            pg, pc, fo = oof(train, SPECS[spec], dflip, seed, frac)
            dec = Decoded(train, pg, pc, mir_tr)
            grid = DECODERS[dname]
            p = nested(train, dec, dec.hits(grid, args.metric), grid, fo)
            vals.append((*rates(train, p), balanced(train, photo_scores(train, p, args.metric), train.stems)))
        mb = float(np.mean([v[0] for v in vals]))
        results["learning_curve"][frac] = vals
        print(
            f"  train frac {frac:.2f}  {args.metric} {np.mean([v[4] for v in vals]):.3f}  "
            f"exact balanced {mb:.3f}±{np.std([v[0] for v in vals]):.3f}  "
            f"enc {np.mean([v[2] for v in vals]):.1f}/{n_enc}",
            flush=True,
        )

    args.dump.parent.mkdir(parents=True, exist_ok=True)
    args.dump.write_text(json.dumps(results, default=list) + "\n", encoding="utf-8")
    print(f"results -> {args.dump}")


if __name__ == "__main__":
    main()
