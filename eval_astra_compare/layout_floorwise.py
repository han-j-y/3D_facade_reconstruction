#!/usr/bin/env python3
"""Floor-by-floor layout agreement between a rendering and its photo, from window boxes only (no pixel IoU, no
alignment to the photo).

Photo: the labelled test windows (SAM3 proposals the labeller kept; deleted ones dropped; unusable ones = don't care).
Rendering: SAM3 ``window`` detections on the method's own render (prepare_detections.py, same settings as the photo
labels), so every method is read through SAM3 on its rendering.

Per façade, the render's windows are placed on the photo with one scale + shift per axis, fitted to the usable photo
windows by eval_astra.init_alignment / refine_alignment (the same placement for every method; ``--norm extent`` puts
each set in its own unit square instead, which breaks when the outermost windows differ, e.g. an extra side window),
then:
1. floors = window rows (centre bands, gap 0.5 median height, eval_astra_protocol._bands); photo and render floors
   are matched in order by dynamic programming on their relative heights (a match needs a gap below half the photo's
   median floor pitch); floors left over are unmatched;
2. on a matched floor pair, windows are relative x-intervals: a photo window is found when the render windows on that
   floor cover >= 50 % of its interval (one wide render box may cover several photo windows, e.g. an oriel), a render
   window is correct when the photo windows (usable or not) cover >= 50 % of its interval; render windows lying mostly
   on unusable photo windows are not counted;
3. windows on unmatched floors are misses (photo) or extras (render).
Reported: pooled precision (render windows correct), recall (photo windows found), F1, and floor-count exact share.

  python layout_floorwise.py --render-dets DIR [DIR ...] --names ours astra
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent)); import paths  # noqa: E402
HERE = paths.ASTRA_DIR
TESTSET = paths.TESTSET_DIR
sys.path.insert(0, str(TESTSET / "tools"))
sys.path.insert(0, str(paths.HERE))
import eval_testset as ev  # noqa: E402
from eval_astra import apply, init_alignment, refine_alignment  # noqa: E402
from eval_astra_protocol import _bands, grid_map  # noqa: E402


def normalise(boxes: list, ref: list | None = None) -> list:
    """Boxes in the unit square of the extent of ``ref`` (default: the boxes themselves)."""
    ref = ref or boxes
    x0, y0 = min(b[0] for b in ref), min(b[1] for b in ref)
    x1, y1 = max(b[2] for b in ref), max(b[3] for b in ref)
    sx, sy = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    return [[(b[0] - x0) / sx, (b[1] - y0) / sy, (b[2] - x0) / sx, (b[3] - y0) / sy] for b in boxes]


def floors(boxes: list) -> list[list[int]]:
    mh = float(np.median([b[3] - b[1] for b in boxes]))
    return _bands(boxes, 1, 0.5 * mh)


def ycentre(boxes, idx):
    return float(np.mean([(boxes[i][1] + boxes[i][3]) / 2 for i in idx]))


def match_floors(pc: list[float], rc: list[float], tol: float) -> list[tuple[int, int]]:
    """Monotone matching maximising matches, then minimising total |dy| (DP over prefixes)."""
    n, m = len(pc), len(rc)
    best = [[(0, 0.0)] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            cands = []
            if i > 0:
                cands.append((best[i - 1][j], (i - 1, j, False)))
            if j > 0:
                cands.append((best[i][j - 1], (i, j - 1, False)))
            if i > 0 and j > 0 and abs(pc[i - 1] - rc[j - 1]) <= tol:
                k, c = best[i - 1][j - 1]
                cands.append(((k + 1, c - abs(pc[i - 1] - rc[j - 1])), (i - 1, j - 1, True)))
            v, b = max(cands, key=lambda x: x[0])
            best[i][j], back[i][j] = v, b
    out, i, j = [], n, m
    while back[i][j] is not None:
        pi, pj, mt = back[i][j]
        if mt:
            out.append((pi, pj))
        i, j = pi, pj
    return out[::-1]


def cover_1d(target: tuple[float, float], intervals: list[tuple[float, float]]) -> float:
    a, b = target
    if b <= a:
        return 0.0
    segs = sorted((max(a, x), min(b, y)) for x, y in intervals if min(b, y) > max(a, x))
    tot, cur = 0.0, None
    for x, y in segs:
        if cur is None or x > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [x, y]
        else:
            cur[1] = max(cur[1], y)
    if cur:
        tot += cur[1] - cur[0]
    return tot / (b - a)


def score_facade(photo: list[dict], render: list, norm: str = "align") -> dict:
    use = [not w["unusable"] for w in photo]
    if norm == "extent":
        pb = normalise([w["box"] for w in photo], [w["box"] for w in photo if not w["unusable"]])
        rb = normalise(render)
    else:  # one scale + shift per axis, fitted to the usable photo windows (eval_astra.py), the same for every method
        pb = [w["box"] for w in photo]
        gb = [w["box"] for w in photo if not w["unusable"]]
        sc, sh = init_alignment(render, gb)
        sc, sh = refine_alignment(render, gb, sc, sh)
        if norm == "grid":  # then floor / bay bands (eval_astra_protocol.grid_map), the same for every method
            fm = grid_map(render, gb, sc, sh)
            rb = [fm(b) for b in render]
        else:
            rb = [apply(b, sc, sh) for b in render]
    pf, rf = floors(pb), floors(rb)
    pc, rc = [ycentre(pb, f) for f in pf], [ycentre(rb, f) for f in rf]
    pitch = float(np.median(np.diff(pc))) if len(pc) > 1 else 1.0
    pairs = match_floors(pc, rc, 0.5 * pitch)
    found = correct = n_render = 0
    matched_p, matched_r = {i for i, _ in pairs}, {j for _, j in pairs}
    for i, j in pairs:
        rint = [(rb[k][0], rb[k][2]) for k in rf[j]]
        pall = [(pb[k][0], pb[k][2]) for k in pf[i]]
        punu = [(pb[k][0], pb[k][2]) for k in pf[i] if not use[k]]
        for k in pf[i]:
            if use[k] and cover_1d((pb[k][0], pb[k][2]), rint) >= 0.5:
                found += 1
        for k in rf[j]:
            iv = (rb[k][0], rb[k][2])
            if punu and cover_1d(iv, punu) >= 0.5 and cover_1d(iv, [(pb[q][0], pb[q][2]) for q in pf[i] if use[q]]) < 0.5:
                continue  # on an unusable photo window: don't care
            n_render += 1
            correct += int(cover_1d(iv, pall) >= 0.5)
    n_render += sum(len(rf[j]) for j in range(len(rf)) if j not in matched_r)
    n_photo = sum(use)
    return {"found": found, "n_photo": n_photo, "correct": correct, "n_render": n_render,
            "floors_photo": len(pf), "floors_render": len(rf), "floors_matched": len(pairs),
            "unmatched_photo_floors": len(pf) - len(matched_p)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render-dets", type=Path, nargs="+", required=True, help="dirs of <stem>.json detections")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--tids", type=Path, default=HERE / "eval_protocol_grid_iou01" / "eval_astra_protocol.json")
    ap.add_argument("--norm", choices=["align", "grid", "extent"], default="align",
                    help="align: global scale + shift fitted to the photo windows (default); grid: then per floor / bay band; extent: own unit squares")
    ap.add_argument("--out", type=Path, default=HERE / "eval_layout_floorwise.json")
    a = ap.parse_args()
    gt = ev.load_gt(TESTSET / "labels")
    tids = json.loads(a.tids.read_text())["tids"]
    res = {}
    for name, d in zip(a.names, a.render_dets):
        per = {}
        for t in tids:
            p = d / f"{gt[t]['stem']}.json"
            render = [w["box_xyxy"] for w in json.loads(p.read_text())["windows"]]
            per[t] = score_facade(gt[t]["windows"], render, a.norm)
        F, N, C, R = (sum(v[k] for v in per.values()) for k in ("found", "n_photo", "correct", "n_render"))
        P, Rc = C / max(1, R), F / max(1, N)
        fl = sum(v["floors_photo"] == v["floors_render"] for v in per.values()) / len(per)
        res[name] = {"precision": P, "recall": Rc, "f1": 2 * P * Rc / max(1e-9, P + Rc), "floor_count_exact": fl, "per_facade": per}
        print(f"{name:8s} P {P:.3f} R {Rc:.3f} F1 {res[name]['f1']:.3f} | floor count exact {fl:.0%} | windows photo {N} render {R}")
    a.out.write_text(json.dumps(res, indent=1, default=int) + "\n")


if __name__ == "__main__":
    main()
