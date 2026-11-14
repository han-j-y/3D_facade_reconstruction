#!/usr/bin/env python3
"""Score Astra's façades (manually labeled renders, Astra/light_labels/labels) against the test-set labels.

Astra windows are labeled in Astra's front-render frame. Per façade an axis-aligned map render -> photo
(x' = sx x + tx, y' = sy y + ty) is fitted: initialised from the window-set extents, refined by mutual-nearest-centre
least squares, then by maximising the summed IoU of a one-to-one (Hungarian) assignment (the placement most favourable
to Astra). Two matchings of test windows (not unusable) to aligned Astra windows, both one-to-one:
  * iou   - IoU >= 0.5, as tools/eval_testset.py (measures placement too);
  * centre - Hungarian on centre distance, accepted when the distance is below half the smaller window width
             (correspondence without penalising placement; used to compare structure and grouping).
Per matched window, as eval_testset.py: pane exact / TED of the Astra window's cluster structure label vs the test
window's cluster structure label; outline shape; pairwise same/different cluster. Astra windows marked unusable carry
no structure (scored as a missing prediction, no cluster).

Usage: python eval_astra.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment, minimize

sys.path.insert(0, str(Path(__file__).resolve().parent)); import paths  # noqa: E402
HERE = paths.ASTRA_DIR
TESTSET = paths.TESTSET_DIR
sys.path.insert(0, str(TESTSET / "tools"))
import eval_testset as ev  # noqa: E402


def centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def apply(b, s, t):
    return [b[0] * s[0] + t[0], b[1] * s[1] + t[1], b[2] * s[0] + t[0], b[3] * s[1] + t[1]]


def init_alignment(a_boxes, g_boxes, iters=30):
    A = np.array([centre(b) for b in a_boxes]); G = np.array([centre(b) for b in g_boxes])
    amin, amax, gmin, gmax = A.min(0), A.max(0), G.min(0), G.max(0)
    s = (gmax - gmin) / np.maximum(amax - amin, 1e-6); t = gmin - s * amin
    med = float(np.median([b[2] - b[0] for b in g_boxes]))
    for _ in range(iters):
        P = A * s + t
        d = np.linalg.norm(P[:, None] - G[None], axis=2)
        a2g, g2a = d.argmin(1), d.argmin(0)
        pairs = [(i, a2g[i]) for i in range(len(A)) if g2a[a2g[i]] == i and d[i, a2g[i]] < 1.5 * med]
        if len(pairs) < 3:
            break
        ia, ig = map(list, zip(*pairs))
        ns, nt = s.copy(), t.copy()
        for k in (0, 1):
            ns[k], nt[k] = np.linalg.lstsq(np.c_[A[ia, k], np.ones(len(ia))], G[ig, k], rcond=None)[0]
        if np.allclose(ns, s) and np.allclose(nt, t):
            break
        s, t = ns, nt
    return s, t


def iou_matrix(a_boxes, g_boxes):
    return np.array([[ev.iou(a, g) for g in g_boxes] for a in a_boxes])


def refine_alignment(a_boxes, g_boxes, s, t):
    """Maximise the summed IoU of the Hungarian assignment over (sx, sy, tx, ty)."""
    def neg(p):
        M = iou_matrix([apply(b, p[:2], p[2:]) for b in a_boxes], g_boxes)
        r, c = linear_sum_assignment(-M)
        return -float(M[r, c].sum())
    best = minimize(neg, np.r_[s, t], method="Nelder-Mead",
                    options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 2000, "initial_simplex": None})
    p = best.x if best.fun <= neg(np.r_[s, t]) else np.r_[s, t]
    return p[:2], p[2:]


def match_centre(a_boxes, g_boxes):
    A = np.array([centre(b) for b in a_boxes]); G = np.array([centre(b) for b in g_boxes])
    d = np.linalg.norm(A[:, None] - G[None], axis=2)
    r, c = linear_sum_assignment(d)
    out = {}
    for i, j in zip(r, c):
        lim = 0.5 * min(a_boxes[i][2] - a_boxes[i][0], g_boxes[j][2] - g_boxes[j][0])
        if d[i, j] <= lim:
            out[i] = j
    return out


def score(m, awins, gwins):
    recs = []
    for ui, gi in m.items():
        a, g = awins[ui], gwins[gi]
        exact, tedv, _ = ev.pane_score(None if a["unusable"] else a["pane"], g["pane"])
        sh = (not a["unusable"]) and a["tokens"] is not None and g["tokens"] is not None \
            and ev.shape_of(a["tokens"]) == ev.shape_of(g["tokens"])
        recs.append(dict(gid=g["id"], aid=a["id"], gcl=g["cluster"], acl=None if a["unusable"] else a["cluster"],
                         exact=bool(exact), ted=float(tedv), shape=bool(sh)))
    tp = fp = fn = 0
    for x, y in combinations(recs, 2):
        sa = x["acl"] is not None and x["acl"] == y["acl"]; sg = x["gcl"] == y["gcl"]
        tp += sa and sg; fp += sa and not sg; fn += sg and not sa
    return recs, (tp, fp, fn)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "light_labels" / "eval")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    gt_all = ev.load_gt(TESTSET / "labels")
    as_all = ev.load_gt(HERE / "light_labels" / "labels")
    per, panels = {}, []
    tot = {k: dict(gt=0, pred=0, n=0, ex=0, ted=0.0, sh=0, tp=0, fp=0, fn=0) for k in ("iou", "centre")}
    for tid in sorted(as_all):
        G, A = gt_all[tid], as_all[tid]
        gwins = [w for w in G["windows"] if not w["unusable"]]
        awins = A["windows"]
        gb = [w["box"] for w in gwins]
        s, t = init_alignment([w["box"] for w in awins], gb)
        s, t = refine_alignment([w["box"] for w in awins], gb, s, t)
        ab = [apply(w["box"], s, t) for w in awins]
        units = [{"box_xyxy": b} for b in ab]
        matches = {"iou": {ui: gi for gi, ui in ev.match(units, gwins, 0.5).items()}, "centre": match_centre(ab, gb)}
        f = {"n_gt": len(gwins), "n_astra": len(awins), "align": {"sx": float(s[0]), "sy": float(s[1]), "tx": float(t[0]), "ty": float(t[1])},
             "n_clusters_astra": len({w["cluster"] for w in awins if not w["unusable"]}), "n_clusters_gt": len({w["cluster"] for w in gwins})}
        for mode, m in matches.items():
            recs, (tp, fp, fn) = score(m, awins, gwins); n = len(recs)
            f[mode] = dict(matched=n, recall=n / max(1, len(gwins)), precision=n / max(1, len(awins)),
                           pane_exact=sum(r["exact"] for r in recs) / max(1, n), pane_ted=sum(r["ted"] for r in recs) / max(1, n),
                           shape=sum(r["shape"] for r in recs) / max(1, n), pair=(tp, fp, fn), recs=recs)
            T = tot[mode]
            for k, v in (("gt", len(gwins)), ("pred", len(awins)), ("n", n), ("ex", sum(r["exact"] for r in recs)),
                         ("ted", sum(r["ted"] for r in recs)), ("sh", sum(r["shape"] for r in recs)), ("tp", tp), ("fp", fp), ("fn", fn)):
                T[k] += v
        per[tid] = f
        im = Image.open(TESTSET / "photos" / f"{G['stem']}.jpg").convert("RGB"); dr = ImageDraw.Draw(im)
        for w in gwins:
            dr.rectangle(w["box"], outline=(0, 230, 90), width=3)
        for ui, b in enumerate(ab):
            dr.rectangle(b, outline=(255, 0, 200), width=3 if ui in matches["iou"] else 1)
        im = im.resize((int(im.width * 360 / im.height), 360))
        c = Image.new("RGB", (im.width, 384), (20, 20, 20)); c.paste(im, (0, 24))
        ImageDraw.Draw(c).text((4, 5), f"{tid}: IoU-matched {f['iou']['matched']}/{len(gwins)}, centre-matched {f['centre']['matched']}/{len(gwins)}",
                               fill=(255, 255, 255))
        panels.append(c)
    summary = {}
    for mode, T in tot.items():
        P = T["tp"] / max(1, T["tp"] + T["fp"]); R = T["tp"] / max(1, T["tp"] + T["fn"])
        summary[mode] = dict(recall=T["n"] / T["gt"], precision=T["n"] / T["pred"], pane_exact=T["ex"] / max(1, T["n"]),
                             pane_ted=T["ted"] / max(1, T["n"]), shape=T["sh"] / max(1, T["n"]), pair_P=P, pair_R=R,
                             pair_F1=2 * P * R / max(1e-9, P + R), matched=T["n"], gt_windows=T["gt"], astra_windows=T["pred"])
    (args.out / "eval_astra.json").write_text(json.dumps({"summary": summary, "per_facade": per}, indent=1) + "\n")
    W = sum(p.width + 6 for p in panels[:5])
    sheet = Image.new("RGB", (W, 2 * 390), (20, 20, 20)); x = y = 0
    for k, p in enumerate(panels):
        if k == 5:
            x, y = 0, 390
        sheet.paste(p, (x, y)); x += p.width + 6
    sheet.save(args.out / "alignment_check.jpg", quality=88)
    print(json.dumps(summary, indent=1))
    for tid, f in per.items():
        i, c = f["iou"], f["centre"]
        print(f"{tid}: test {f['n_gt']} astra {f['n_astra']} | IoU-matched {i['matched']} centre-matched {c['matched']} | centre: exact {c['pane_exact']:.2f} "
              f"TED {c['pane_ted']:.2f} shape {c['shape']:.2f} | clusters astra {f['n_clusters_astra']} test {f['n_clusters_gt']}")


if __name__ == "__main__":
    main()
