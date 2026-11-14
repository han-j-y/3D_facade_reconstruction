#!/usr/bin/env python3
"""Soft structure score: F1 of the drawn division lines of a window type against its label.

Each structure is reduced to the canonical pane tree that TED and exact use (eval_testset.pane_tree: split positions
ignored, through-cuts flattened, a head / body seam = a horizontal cut), and drawn as line segments in the unit
window square, every node cutting its box evenly (the outline is not used, shape is scored separately). Equal trees
draw equal lines, so exact implies F1 1. Both line sets are sampled every 0.005; precision = share of predicted line length within
``--tol`` of a label line, recall the reverse, F1 per window. No lines on both sides (a single pane) = 1, lines on one
side only = 0. A T (mullion below the transom) against a cross (mullion through it) scores 0.85 instead of
exact 0 / TED 0.43.

Matching and averaging as eval_testset.py: units matched one-to-one to labelled windows by greedy IoU >= --min-iou,
the matched unit's type structure is scored, unmatched labelled windows score 0, mean over labelled windows.

  python soft_structure.py --run RUN_DIR [--only-tids FILE.json] [--min-iou 0.1]
  (Astra: datasets/Astra/eval_astra_protocol.py --soft)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import eval_testset as ev

STEP = 0.005


def tree_lines(tree: tuple, box=(0.0, 0.0, 1.0, 1.0)) -> list[tuple[float, float, float, float]]:
    """Division lines of a canonical pane tree (eval_testset.pane_tree): each V / H node cuts its box evenly."""
    axis, kids = tree
    if axis == "P" or not kids:
        return []
    x0, y0, x1, y1 = box
    k, out = len(kids), []
    for i, kid in enumerate(kids):
        if axis == "V":
            a, b = x0 + (x1 - x0) * i / k, x0 + (x1 - x0) * (i + 1) / k
            sub = (a, y0, b, y1)
            if i:
                out.append((a, y0, a, y1))
        else:
            a, b = y0 + (y1 - y0) * i / k, y0 + (y1 - y0) * (i + 1) / k
            sub = (x0, a, x1, b)
            if i:
                out.append((x0, a, x1, a))
        out += tree_lines(kid, sub)
    return out


def segments(tokens: list[str] | None) -> list[tuple[float, float, float, float]] | None:
    """Lines of the structure's canonical pane tree, the tree TED and exact use (split positions, head / body seam
    vs horizontal split and through-cut nesting are normalised the same way); None if it does not parse."""
    if not tokens:
        return None
    tree = ev.tokens_tree(tokens)
    return None if tree is None else tree_lines(tree)


def _samples(segs) -> np.ndarray:
    pts = []
    for x0, y0, x1, y1 in segs:
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / STEP) + 1)
        t = np.linspace(0, 1, n)
        pts.append(np.c_[x0 + (x1 - x0) * t, y0 + (y1 - y0) * t])
    return np.vstack(pts)


def _dist(p: np.ndarray, segs) -> np.ndarray:
    best = np.full(len(p), np.inf)
    for x0, y0, x1, y1 in segs:
        a, b = np.array([x0, y0]), np.array([x1, y1])
        ab = b - a
        t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        best = np.minimum(best, np.linalg.norm(p - (a + t[:, None] * ab), axis=1))
    return best


def line_f1(pred: list | None, gold: list | None, tol: float) -> float:
    """pred / gold: segment lists (None = unparseable / missing)."""
    if gold is None or pred is None:
        return 0.0
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    ps, gs = _samples(pred), _samples(gold)
    p = float((_dist(ps, gold) <= tol).mean())
    r = float((_dist(gs, pred) <= tol).mean())
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def score_facade(dsl: dict, gt: dict, min_iou: float, tol: float) -> list[float]:
    units = [u for u in dsl["instances"] if u.get("kind", "window") == "window"]
    tsegs = {int(t["type_id"]): segments(ev.canon(t["structure_tokens"])) for t in dsl["window_types"]}
    wins = gt["windows"]
    m = ev.match(units, wins, min_iou)
    out = []
    for j, w in enumerate(wins):
        if w["unusable"]:
            continue
        gold = segments(w["tokens"]) if w["tokens"] else None
        out.append(line_f1(tsegs.get(int(units[m[j]]["type_id"])), gold, tol) if j in m else 0.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="dir of <tid>/facade_dsl.json")
    ap.add_argument("--labels", type=Path, default=ev.TESTSET / "labels")
    ap.add_argument("--min-iou", type=float, default=0.5)
    ap.add_argument("--tol", type=float, default=0.03, help="line distance tolerance, fraction of the window")
    ap.add_argument("--only-tids", type=Path, help="json with a 'tids' list (e.g. eval_astra_protocol.json)")
    a = ap.parse_args()
    gt = ev.load_gt(a.labels)
    only = set(json.loads(a.only_tids.read_text())["tids"]) if a.only_tids else None
    vals = []
    for tid in sorted(gt):
        if only is not None and tid not in only:
            continue
        p = a.run / tid / "facade_dsl.json"
        dsl = json.loads(p.read_text()) if p.is_file() else {"instances": [], "window_types": []}
        vals += score_facade(dsl, gt[tid], a.min_iou, a.tol)
    print(f"soft line F1 {np.mean(vals):.3f} over {len(vals)} labelled windows (tol {a.tol}, IoU >= {a.min_iou})")


if __name__ == "__main__":
    main()
