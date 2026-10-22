"""Threshold search for the enclosed detector, scored on pooled window cells.

TP / FP / FN / TN are summed over every window cell of every photo in the set;
precision, recall, F1 and IoU come from those totals.

Selection uses cmp_b only. Probabilities are out-of-fold (the same
StratifiedGroupKFold as ``cell_clf_search.oof``, 3 seeds). ``--objective``:

* ``plain``    pooled F1 on cmp_b as it is (8% enclosure photos);
* ``balanced`` FP cells on none photos are weighted by n_enclosure / n_none,
  so enclosure and none photos count 1:1 (cmp_x is 26 / 35).

Every method is scored by nested CV (parameters chosen on the other folds).
The eval set (cmp_x) is scored once, after the choice.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/threshold_f1_search.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.cell_clf_experiments import DEFAULT_CELLS, Data  # noqa: E402
from balcony_train.cell_clf_search import SEEDS, SPECS, Model, oof  # noqa: E402
from balcony_train.cell_clf_train import DEFAULT_TRAIN_CELLS, DEFAULT_TRAIN_LABELS  # noqa: E402
from balcony_train.cell_clf_v2 import mirror_index  # noqa: E402
from balcony_train.cell_clf_v3 import rect_decode  # noqa: E402
from balcony_train.enclosed_grid_store import (  # noqa: E402
    DEFAULT_DSL_DIR,
    DEFAULT_EVAL_LABELS_JSONL,
    load_grid,
)
from balcony_train.enclosed_symmetry import SYM_GATE  # noqa: E402

DEFAULT_OUT = ROOT / "runs" / "cell_clf" / "threshold_f1"
T_GRID = tuple(round(x, 2) for x in np.arange(0.05, 0.851, 0.05))


# ---------------------------------------------------------------- metric


# FP weight on none photos for the selection objective; set from --objective.
NONE_WEIGHT = 1.0


def counts(gts: dict[str, set], preds: dict[str, set], ncells: dict[str, int], stems) -> dict[str, float]:
    """Pooled cell counts and rates, plus photo-level exact / none-correct."""
    tp = fp = fn = tn = 0
    fp_none = 0
    exact = enc_exact = none_ok = n_enc = 0
    for s in stems:
        gt, pr = gts[s], preds.get(s, set())
        a, b, c = len(gt & pr), len(pr - gt), len(gt - pr)
        tp, fp, fn = tp + a, fp + b, fn + c
        tn += ncells[s] - a - b - c
        exact += int(gt == pr)
        if gt:
            n_enc += 1
            enc_exact += int(gt == pr)
        else:
            fp_none += b
            none_ok += int(not pr)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "fp_none": fp_none,
        "P": p, "R": r, "F1": 2 * p * r / max(p + r, 1e-12), "IoU": tp / max(tp + fp + fn, 1),
        "exact": exact, "enc_exact": enc_exact, "none_ok": none_ok,
        "n": len(stems), "n_enc": n_enc,
    }


def weighted_f1(c: dict[str, float], w_none: float) -> float:
    fp = (c["fp"] - c["fp_none"]) + w_none * c["fp_none"]
    return 2 * c["tp"] / max(2 * c["tp"] + fp + c["fn"], 1e-12)


# ---------------------------------------------------------------- decoder


@dataclass(frozen=True)
class Params:
    """Current pipeline values are the defaults (``enclosed_detect`` + ``clf_three_rules``)."""

    t: float = 0.30
    gate: float = 0.10
    top_frac: float = 0.85  # rule 1 / rule 4 floor-height ratio
    top_p: float = 0.05  # rule 2
    min_l: float = 0.015  # rule 3
    min_upper: float = 0.55  # rule 3
    min_cover: float = 0.40  # rule 3
    min_p: float = 0.10  # rule 3
    l_delta: float = 0.02  # rule 4
    up_upper: float = 0.75  # rule 4
    rules: bool = True
    # Per-photo relative cell threshold: t_eff = max(t, rel * max p_cell), on the T step grid.
    rel: float = 0.0
    # Photo score the gate threshold applies to: "gmax" (current) or "gmax*cmax".
    score: str = "gmax"
    # Rectangle on the joint score gmax * p_cell instead of p_cell.
    joint: bool = False


class Photo:
    """Per-photo arrays used by the rules, cached once."""

    def __init__(self, data: Data, stem: str, floor_y: dict[int, tuple[int, int]]) -> None:
        ii = data.idx[stem]
        self.ii = ii
        self.cells = [data.cells[i] for i in ii]
        self.where = {c: k for k, c in enumerate(self.cells)}
        self.sym = float(data.f("window_sym")[ii[0]]) if len(ii) else 0.0
        self.L = data.f("L")[ii]
        self.upper = data.f("upper")[ii]
        self.cover = data.f("sam3_enc_cover")[ii]
        self.op1 = data.f("in_cells_6")[ii]
        self.floor_y = floor_y
        heights = [b - a for a, b in floor_y.values() if b > a]
        self.med = float(np.median(heights)) if heights else 0.0
        order = sorted(floor_y, key=lambda f: floor_y[f][0])
        self.above = {f: (order[k - 1] if k > 0 else None) for k, f in enumerate(order)}

    def height(self, f: int) -> float:
        a, b = self.floor_y[f]
        return float(b - a)

    def top(self, cells: set) -> int | None:
        if not cells:
            return None
        return min((f for f, _b in cells), key=lambda f: self.floor_y[f][0])


def apply_rules_p(ph: Photo, clf: set, prob: np.ndarray, p: Params) -> set:
    """``clf_three_rules.apply_rules`` with every threshold as a parameter."""
    kept = set(clf)
    top = ph.top(kept)
    if top is not None and ph.med > 0 and ph.height(top) < p.top_frac * ph.med:
        kept = {c for c in kept if c[0] != top}

    top = ph.top(kept)
    if top is not None and ph.sym < SYM_GATE:
        kept = {
            c for c in kept
            if not (c[0] == top and c in ph.where and float(prob[ph.where[c]]) < p.top_p)
        }

    added = set()
    for k, c in enumerate(ph.cells):
        if ph.op1[k] <= 0 or c in kept:
            continue
        if not (np.isfinite(ph.L[k]) and ph.L[k] >= p.min_l):
            continue
        if not (np.isfinite(ph.upper[k]) and ph.upper[k] >= p.min_upper):
            continue
        if not (np.isfinite(ph.cover[k]) and ph.cover[k] >= p.min_cover):
            continue
        if float(prob[k]) < p.min_p:
            continue
        added.add(c)

    base = kept | added
    up = set()
    for c in base:
        if c not in ph.where or ph.med <= 0:
            continue
        above = ph.above.get(c[0])
        if above is None:
            continue
        cand = (above, c[1])
        if cand in base or cand in up or cand not in ph.where:
            continue
        if ph.height(above) < p.top_frac * ph.med:
            continue
        i, j = ph.where[cand], ph.where[c]
        if not (np.isfinite(ph.L[i]) and np.isfinite(ph.L[j]) and abs(ph.L[i] - ph.L[j]) <= p.l_delta):
            continue
        if not (np.isfinite(ph.upper[i]) and ph.upper[i] >= p.up_upper):
            continue
        up.add(cand)
    return base | up


class Set_:
    """One probability set (OOF seed or full-model eval) with cached rectangles."""

    def __init__(self, data: Data, p_gate: np.ndarray, p_cell: np.ndarray, mir: np.ndarray, dsl_dir: Path) -> None:
        self.data = data
        self.stems = list(data.stems)
        self.gts = {s: set(data.gt[s]) for s in self.stems}
        self.p_gate = p_gate
        self.p_cell = p_cell
        self.mir = mir
        self.gmax = {s: float(p_gate[data.idx[s]].max()) for s in self.stems}
        self.scores = {
            "gmax": self.gmax,
            "gmax*cmax": {s: self.gmax[s] * float(p_cell[data.idx[s]].max()) for s in self.stems},
        }
        self.photos = {s: Photo(data, s, load_grid(dsl_dir / s / "facade_dsl.json")[0]) for s in self.stems}
        self.prob = {s: p_cell[data.idx[s]] for s in self.stems}
        self.ncells = {s: len(data.idx[s]) for s in self.stems}
        self.joint_score = p_cell.copy()
        for s in self.stems:
            self.joint_score[data.idx[s]] *= self.gmax[s]
        self._rect: dict[tuple[str, float, bool], set] = {}

    def rect(self, s: str, t: float, joint: bool = False) -> set:
        key = (s, round(t, 4), joint)
        if key not in self._rect:
            score = self.joint_score if joint else self.p_cell
            self._rect[key] = rect_decode(self.data, s, score, t, self.mir)
        return self._rect[key]

    def predict(self, s: str, p: Params) -> set:
        if self.scores[p.score][s] < p.gate:
            return set()
        t = p.t
        if p.rel > 0:
            t = max(t, p.rel * float(self.prob[s].max()))
            t = min(T_GRID, key=lambda x: abs(x - t))
        clf = self.rect(s, t, p.joint)
        return apply_rules_p(self.photos[s], clf, self.prob[s], p) if p.rules else clf

    def preds(self, p: Params, stems=None) -> dict[str, set]:
        return {s: self.predict(s, p) for s in (stems or self.stems)}

    def counts(self, preds: dict[str, set], stems=None) -> dict[str, float]:
        return counts(self.gts, preds, self.ncells, stems or self.stems)


# ---------------------------------------------------------------- probabilities


def load_or_build(out_dir: Path, dsl_dir: Path) -> tuple[list[Set_], list[dict[str, int]], Set_]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "probs.npz"
    train = Data(DEFAULT_TRAIN_CELLS, DEFAULT_TRAIN_LABELS)
    test = Data(DEFAULT_CELLS, DEFAULT_EVAL_LABELS_JSONL)
    if cache.is_file():
        z = np.load(cache, allow_pickle=True)
        oof_g, oof_c = z["oof_g"], z["oof_c"]
        folds = json.loads(str(z["folds"]))
        ev_g, ev_c = z["ev_g"], z["ev_c"]
    else:
        t0 = time.time()
        oof_g, oof_c, folds = [], [], []
        for seed in SEEDS:
            g, c, fold_of = oof(train, SPECS["base"], None, seed)
            oof_g.append(g)
            oof_c.append(c)
            folds.append(fold_of)
            print(f"oof seed {seed} done {time.time() - t0:.0f}s", flush=True)
        oof_g, oof_c = np.stack(oof_g), np.stack(oof_c)
        ev_g, ev_c = Model(train, train.stems, SPECS["base"], None).predict(test, test.stems)
        np.savez(cache, oof_g=oof_g, oof_c=oof_c, folds=json.dumps(folds), ev_g=ev_g, ev_c=ev_c)
    mir_tr = mirror_index(train, dsl_dir)
    mir_ev = mirror_index(test, dsl_dir)
    sets = [Set_(train, oof_g[k], oof_c[k], mir_tr, dsl_dir) for k in range(len(SEEDS))]
    ev = Set_(test, ev_g, ev_c, mir_ev, dsl_dir)
    return sets, folds, ev


def fmt(name: str, c: dict, digits: int = 0) -> str:
    n = lambda x: f"{x:.{digits}f}"  # noqa: E731
    return (
        f"{name:<40} P={c['P']:.3f} R={c['R']:.3f} F1={c['F1']:.3f} IoU={c['IoU']:.3f} | "
        f"TP={n(c['tp'])} FP={n(c['fp'])} (none {n(c['fp_none'])}) FN={n(c['fn'])} TN={n(c['tn'])} | "
        f"exact {n(c['exact'])}/{c['n']:.0f} enc {n(c['enc_exact'])}/{c['n_enc']:.0f} "
        f"none ok {n(c['none_ok'])}/{c['n'] - c['n_enc']:.0f}"
    )


# ---------------------------------------------------------------- search

GRIDS: dict[str, tuple] = {
    "t": T_GRID,
    "gate": (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5),
    "top_frac": (0.0, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
    "top_p": (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3),
    "min_l": (0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05),
    "min_upper": (0.0, 0.3, 0.45, 0.55, 0.65, 0.75, 0.9),
    "min_cover": (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 9.0),
    "min_p": (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5),
    "l_delta": (0.0, 0.01, 0.02, 0.03, 0.04, 0.06),
    "up_upper": (0.5, 0.6, 0.75, 0.85, 0.95, 9.0),
}


def objective(sets: list[Set_], p: Params, stems) -> float:
    return float(np.mean([weighted_f1(st.counts(st.preds(p, stems), stems), NONE_WEIGHT) for st in sets]))


def fit_tg(sets, stems, start: Params) -> Params:
    best, best_v = start, -1.0
    for t, g in itertools.product(GRIDS["t"], GRIDS["gate"]):
        p = replace(start, t=t, gate=g)
        v = objective(sets, p, stems)
        if v > best_v + 1e-9:
            best, best_v = p, v
    return best


def fit_coord(sets, stems, start: Params, keys=tuple(GRIDS), passes: int = 3) -> Params:
    best = fit_tg(sets, stems, start)
    best_v = objective(sets, best, stems)
    for _ in range(passes):
        moved = False
        for key in keys:
            for val in GRIDS[key]:
                p = replace(best, **{key: val})
                v = objective(sets, p, stems)
                if v > best_v + 1e-9:
                    best, best_v, moved = p, v, True
        if not moved:
            break
    return best


def fit_rel(sets, stems, start: Params) -> Params:
    best, best_v = start, -1.0
    for rel, t, g in itertools.product((0.3, 0.4, 0.5, 0.6, 0.7, 0.8), (0.1, 0.15, 0.2, 0.25, 0.3, 0.4), GRIDS["gate"]):
        p = replace(start, rel=rel, t=t, gate=g)
        v = objective(sets, p, stems)
        if v > best_v + 1e-9:
            best, best_v = p, v
    return best


def fit_joint(sets, stems, start: Params) -> Params:
    """One threshold on gmax * p_cell; no separate photo gate."""
    best, best_v = start, -1.0
    for t in (0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3):
        p = replace(start, joint=True, gate=0.0, t=t)
        v = objective(sets, p, stems)
        if v > best_v + 1e-9:
            best, best_v = p, v
    return best


CELL_KEYS = tuple(k for k in GRIDS if k != "gate")


def fit_t_fixed_gate(sets, stems, start: Params) -> Params:
    best, best_v = start, -1.0
    for t in GRIDS["t"]:
        p = replace(start, t=t)
        v = objective(sets, p, stems)
        if v > best_v + 1e-9:
            best, best_v = p, v
    return best


def fit_coord_fixed_gate(sets, stems, start: Params, passes: int = 3) -> Params:
    best = fit_t_fixed_gate(sets, stems, start)
    best_v = objective(sets, best, stems)
    for _ in range(passes):
        moved = False
        for key in CELL_KEYS:
            for val in GRIDS[key]:
                p = replace(best, **{key: val})
                v = objective(sets, p, stems)
                if v > best_v + 1e-9:
                    best, best_v, moved = p, v, True
        if not moved:
            break
    return best


def fit_rel_fixed_gate(sets, stems, start: Params) -> Params:
    best, best_v = start, -1.0
    for rel, t in itertools.product((0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)):
        p = replace(start, rel=rel, t=t)
        v = objective(sets, p, stems)
        if v > best_v + 1e-9:
            best, best_v = p, v
    return best


def matched_gate(sets, stems, score: str) -> float:
    """Threshold on ``score`` that blanks the same share of cmp_b none photos as gmax >= 0.10."""
    vals = []
    for st in sets:
        none = [s for s in stems if not st.gts[s]]
        kept = np.mean([st.gmax[s] >= Params().gate for s in none])
        sc = np.sort([st.scores[score][s] for s in none])
        k = int(round((1.0 - kept) * len(sc)))
        k = min(max(k, 1), len(sc) - 1)
        vals.append(0.5 * (sc[k - 1] + sc[k]))
    return float(np.mean(vals))


def fit_product_matched(sets, stems, start: Params) -> Params:
    return replace(start, score="gmax*cmax", gate=matched_gate(sets, stems, "gmax*cmax"))


def fit_product_grid(sets, stems, start: Params) -> Params:
    best, best_v = start, -1.0
    for g in (0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3):
        p = replace(start, score="gmax*cmax", gate=g)
        v = objective(sets, p, stems)
        if v > best_v + 1e-9:
            best, best_v = p, v
    return best


METHODS = {
    "A baseline (current, hand-set)": None,
    "B grid t x gate": fit_tg,
    "C coordinate, all 10 thresholds": fit_coord,
    "D relative t = max(t, r*max p)": fit_rel,
    "E joint score gmax*p_cell, one threshold": fit_joint,
    "F gate 0.10 fixed, t only": fit_t_fixed_gate,
    "G gate 0.10 fixed, coordinate 9 cell thr": fit_coord_fixed_gate,
    "H gate 0.10 fixed, relative t": fit_rel_fixed_gate,
    "J gmax*cmax, train-FPR matched": fit_product_matched,
    "L gmax*cmax, threshold by grid": fit_product_grid,
}


def nested(sets: list[Set_], folds: list[dict[str, int]], fit) -> dict:
    """Per seed: fit on the other folds, predict the held-out fold; pool all held-out cells."""
    per_seed = []
    for k, st in enumerate(sets):
        fold_of = folds[k]
        preds: dict[str, set] = {}
        for f in sorted(set(fold_of.values())):
            sel = [s for s in st.stems if fold_of[s] != f]
            held = [s for s in st.stems if fold_of[s] == f]
            p = fit([st], sel, Params()) if fit else Params()
            preds.update(st.preds(p, held))
        per_seed.append(st.counts(preds))
    out = {k: float(np.mean([c[k] for c in per_seed])) for k in per_seed[0]}
    out["wF1"] = float(np.mean([weighted_f1(c, NONE_WEIGHT) for c in per_seed]))
    return out


def show(p: Params) -> str:
    d = {k: v for k, v in p.__dict__.items() if v != getattr(Params(), k) or k in ("t", "gate")}
    return json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()})


def main() -> None:
    global NONE_WEIGHT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--methods", default="".join(name[0] for name in METHODS))
    ap.add_argument("--objective", choices=("plain", "balanced"), default="balanced")
    args = ap.parse_args()
    sets, folds, ev = load_or_build(args.out_dir, args.dsl_dir)
    if args.objective == "balanced":
        n_enc = sum(bool(g) for g in sets[0].gts.values())
        NONE_WEIGHT = n_enc / (len(sets[0].stems) - n_enc)
    log: list[str] = []

    def say(msg: str) -> None:
        print(msg, flush=True)
        log.append(msg)

    say(f"objective: pooled F1, {args.objective} (FP weight on none photos {NONE_WEIGHT:.3f})")
    say("== nested CV on cmp_b (OOF probs, 3 seeds x 5 folds; thresholds chosen on the other 4 folds)")
    finals: dict[str, Params] = {}
    for name, fit in METHODS.items():
        if name[0] not in args.methods:
            continue
        sc = nested(sets, folds, fit)
        say(fmt(name, sc, digits=1) + f" | objective F1={sc['wF1']:.3f}")
        finals[name] = fit(sets, list(sets[0].stems), Params()) if fit else Params()
        say(f"    params on all cmp_b: {show(finals[name])}")

    say("")
    say("== eval (cmp_x), scored once with the params above")
    rows = {}
    for name, p in finals.items():
        preds = ev.preds(p)
        rows[name] = {s: sorted(v) for s, v in preds.items()}
        say(fmt(name, ev.counts(preds)))
    tag = args.objective
    (args.out_dir / f"summary_{tag}.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    (args.out_dir / f"eval_preds_{tag}.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
