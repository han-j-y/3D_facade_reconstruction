"""Round 5: DINOv2 contrast features (cell vs its storey row / photo mean).

Same pipeline as the round 3-4 best (enclosure-conditioned HGB, max-score
gate, rectangle+mirror decoding), only the visual features change.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.cell_clf_experiments import CURRENT, DEFAULT_CELLS, Data, hgb, line, score_sets  # noqa: E402
from balcony_train.cell_clf_v2 import choose_nested, mirror_index  # noqa: E402
from balcony_train.cell_clf_v3 import rect_decode  # noqa: E402
from balcony_train.enclosed_grid_store import DEFAULT_DSL_DIR, DEFAULT_EVAL_LABELS_JSONL  # noqa: E402


def contrast_blocks(data: Data) -> dict[str, np.ndarray]:
    d = data.dino / (np.linalg.norm(data.dino, axis=1, keepdims=True) + 1e-8)
    row = np.zeros_like(d)
    col = np.zeros_like(d)
    pho = np.zeros_like(d)
    for s in data.stems:
        ii = data.idx[s]
        pho[ii] = d[ii].mean(axis=0)
        for f in {data.cells[i][0] for i in ii}:
            jj = [i for i in ii if data.cells[i][0] == f]
            row[jj] = d[jj].mean(axis=0)
        for b in {data.cells[i][1] for i in ii}:
            jj = [i for i in ii if data.cells[i][1] == b]
            col[jj] = d[jj].mean(axis=0)
    cos = lambda a, b: np.sum(a * b, axis=1, keepdims=True) / (  # noqa: E731
        np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1, keepdims=True) + 1e-8
    )
    scal = np.hstack([cos(d, row), cos(d, pho), cos(d, col)])
    return {"cell": d, "row_diff": d - row, "photo_diff": d - pho, "scal": scal}


def oof(data: Data, cols: np.ndarray, blocks: list[tuple[np.ndarray, int]], *, enc_only: bool, make) -> np.ndarray:
    from sklearn.decomposition import PCA

    enc = [s for s in data.stems if data.gt[s]]
    prob = np.zeros(len(data.y))
    cache = {}
    for s in data.stems:
        pool = enc if enc_only else data.stems
        key = tuple(x for x in pool if x != s)
        if key not in cache:
            tr = np.concatenate([data.idx[x] for x in key])
            parts = [data.X[tr][:, cols]]
            fitted = []
            for M, k in blocks:
                if k:
                    p = PCA(n_components=k, random_state=0).fit(M[tr])
                    parts.append(p.transform(M[tr]))
                else:
                    p = None
                    parts.append(M[tr])
                fitted.append(p)
            cache[key] = (make().fit(np.hstack(parts), data.y[tr]), fitted)
        model, fitted = cache[key]
        te = data.idx[s]
        parts = [data.X[te][:, cols]]
        for (M, _k), p in zip(blocks, fitted):
            parts.append(p.transform(M[te]) if p is not None else M[te])
        prob[te] = model.predict_proba(np.hstack(parts))[:, 1]
    return prob


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    ap.add_argument("--labels", type=Path, default=DEFAULT_EVAL_LABELS_JSONL)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    args = ap.parse_args()

    data = Data(args.cells, args.labels)
    mir = mirror_index(data, args.dsl_dir)
    cur = data.f("in_cells_6")
    hand = np.array([i for i, n in enumerate(data.names) if n not in CURRENT])
    allc = np.arange(len(data.names))
    B = contrast_blocks(data)
    variants = {
        "dino16 (round 3)": [(B["cell"], 16)],
        "dino16 + row_diff8": [(B["cell"], 16), (B["row_diff"], 8), (B["scal"], 0)],
        "dino16 + row/photo diff8": [(B["cell"], 16), (B["row_diff"], 8), (B["photo_diff"], 8), (B["scal"], 0)],
        "dino32 + diffs16": [(B["cell"], 32), (B["row_diff"], 16), (B["photo_diff"], 16), (B["scal"], 0)],
        "scal only": [(B["scal"], 0)],
    }
    for vname, blocks in variants.items():
        p_gate = oof(data, hand, blocks, enc_only=False, make=hgb())
        p_cell = oof(data, allc, blocks, enc_only=True, make=hgb(min_leaf=10))
        gate = {s: float(p_gate[data.idx[s]].max()) for s in data.stems}
        grid = list(itertools.product((0.7, 1.0), (0.3, 0.4, 0.5), (0.5, 0.6, 0.7, 0.8)))
        table = {}
        for a, t, g in grid:
            score = a * p_cell + (1 - a) * cur
            table[(a, t, g)] = {
                s: (rect_decode(data, s, score, t, mir) if gate[s] >= g else set()) for s in data.stems
            }
        preds, chosen = choose_nested(data, table, grid)
        print(line(vname, score_sets(data, preds)) + f"  usual={max(set(chosen), key=chosen.count)}", flush=True)


if __name__ == "__main__":
    main()
