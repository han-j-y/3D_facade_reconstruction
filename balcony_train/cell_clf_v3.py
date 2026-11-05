"""Round 3: gate + enclosure-conditioned cell model + rectangle decoding.

Most labeled photos share one floor range across all enclosed bays (oriel
pairs, galleries). ``rect`` picks the floor range and bay set that maximize
sum(score - t); ``rect+mirror`` adds a bay together with its mirror bay.

Cell models:
  all   trained on every photo (same as rounds 1-2)
  enc   trained on enclosure photos only, so it answers "which cells, given
        that this photo has an enclosure"
Parameters are chosen on the other photos for each held-out photo.
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
from balcony_train.enclosed_grid_store import DEFAULT_DSL_DIR, DEFAULT_EVAL_LABELS_JSONL  # noqa: E402


def oof_enc_only(data: Data, make, cols, *, dino_pca: int = 16) -> np.ndarray:
    """Model trained on enclosure photos only; held-out photo never in training."""
    from sklearn.decomposition import PCA

    enc = [s for s in data.stems if data.gt[s]]
    prob = np.zeros(len(data.y))
    cache = {}
    for s in data.stems:
        train_stems = tuple(x for x in enc if x != s)
        if train_stems not in cache:
            tr = np.concatenate([data.idx[x] for x in train_stems])
            Xtr = data.X[tr][:, cols]
            pca = None
            if dino_pca:
                pca = PCA(n_components=dino_pca, random_state=0).fit(data.dino[tr])
                Xtr = np.hstack([Xtr, pca.transform(data.dino[tr])])
            cache[train_stems] = (make().fit(Xtr, data.y[tr]), pca)
        model, pca = cache[train_stems]
        te = data.idx[s]
        Xte = data.X[te][:, cols]
        if pca is not None:
            Xte = np.hstack([Xte, pca.transform(data.dino[te])])
        prob[te] = model.predict_proba(Xte)[:, 1]
    return prob


def rect_decode(
    data: Data,
    stem: str,
    score: np.ndarray,
    t: float,
    mir: np.ndarray | None,
    *,
    min_len: int = 1,
) -> set:
    ii = data.idx[stem]
    rank = data.floor_order(stem)
    nf = len(rank)
    bays = sorted({data.cells[i][1] for i in ii})
    grid = {data.cells[i]: score[i] - t for i in ii}
    inv = {k: f for f, k in rank.items()}
    mirror_of = {}
    if mir is not None:
        for i in ii:
            j = mir[i]
            if j >= 0:
                mirror_of[data.cells[i][1]] = data.cells[j][1]
    best_val, best = 0.0, set()
    for a in range(nf):
        for b in range(a + min_len - 1, nf):
            floors = [inv[k] for k in range(a, b + 1)]
            col = {bay: sum(grid.get((f, bay), -1.0) for f in floors) for bay in bays}
            chosen = set()
            done = set()
            for bay in bays:
                if bay in done:
                    continue
                mb = mirror_of.get(bay)
                if mb is not None and mb in col and mb not in done:
                    pair = col[bay] + col[mb]
                    if pair > 0:
                        chosen |= {bay, mb}
                    elif col[bay] > 0 and col[bay] > -col[mb]:
                        chosen.add(bay)
                    elif col[mb] > 0 and col[mb] > -col[bay]:
                        chosen.add(mb)
                    done |= {bay, mb}
                    continue
                if col[bay] > 0:
                    chosen.add(bay)
                done.add(bay)
            val = sum(col[x] for x in chosen)
            if val > best_val:
                best_val, best = val, {(f, x) for f in floors for x in chosen}
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    ap.add_argument("--labels", type=Path, default=DEFAULT_EVAL_LABELS_JSONL)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--dump", type=Path, default=ROOT / "runs" / "cell_clf" / "results_v3.json")
    args = ap.parse_args()

    data = Data(args.cells, args.labels)
    mir = mirror_index(data, args.dsl_dir)
    cur = data.f("in_cells_6")
    hand = np.array([i for i, n in enumerate(data.names) if n not in CURRENT])
    allc = np.arange(len(data.names))
    gate_prob = oof_probs(data, hgb(), hand, dino_pca=16)
    gate_max = {s: float(gate_prob[data.idx[s]].max()) for s in data.stems}
    cell_models = {
        "all hgb": oof_probs(data, hgb(), allc, dino_pca=16),
        "enc hgb": oof_enc_only(data, hgb(min_leaf=10), allc),
        "enc lr": oof_enc_only(data, logreg(0.1), allc),
    }
    cell_models["enc avg"] = 0.5 * (cell_models["enc hgb"] + cell_models["enc lr"])
    results = {}

    def report(name, preds, extra=""):
        res = score_sets(data, preds)
        results[name] = {"preds": {s: sorted(v) for s, v in preds.items()}, "summary": res["summary"]}
        print(line(name, res) + (f"  {extra}" if extra else ""), flush=True)

    for mname, prob in cell_models.items():
        for decoder in ("runs", "rect", "rect+mirror"):
            grid = list(
                itertools.product(
                    (0.0, 0.3, 0.5, 0.7, 1.0),  # a: classifier weight vs current cells
                    (0.3, 0.4, 0.5, 0.6),  # t
                    (0.5, 0.6, 0.7, 0.8),  # gate
                )
            )
            table = {}
            for a, t, g in grid:
                score = a * prob + (1 - a) * cur
                out = {}
                for s in data.stems:
                    if gate_max[s] < g:
                        out[s] = set()
                    elif decoder == "runs":
                        ii = data.idx[s]
                        out[s] = column_runs({data.cells[i] for i in ii if score[i] >= t}, data.floor_order(s), 1)
                    else:
                        out[s] = rect_decode(data, s, score, t, mir if decoder == "rect+mirror" else None)
                table[(a, t, g)] = out
            preds, chosen = choose_nested(data, table, grid)
            report(f"{mname} / {decoder}", preds, f"usual (a,t,gate)={max(set(chosen), key=chosen.count)}")
    args.dump.write_text(json.dumps(results) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
