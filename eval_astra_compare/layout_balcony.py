#!/usr/bin/env python3
"""Balcony recall / precision of renderings against the test balcony boxes, read through SAM3 on each rendering.

Render balconies: SAM3 ``balcony`` instances >= 0.45 (the pipeline's balcony threshold) on the method's full render
(cached in eval_layout_dets/<method>_balcony/<stem>.json). They are placed on the photo with the same per-method
placement as layout_floorwise.py --norm grid (global scale + shift, then floor / bay bands, fitted from the render's
SAM3 windows to the usable test windows). Photo: runs/test100/balcony_gt_boxes.jsonl (``boxes``; ``ignore`` regions =
don't care). A photo balcony is found when the union of placed render balconies covers >= --cover of its area (one
render box may cover several, several may cover one); a render balcony is correct when the union of photo balconies
covers >= --cover of its area; render balconies lying mostly on ignore regions are not counted. Default ``--mode floor``
instead: each balcony (photo and placed render) belongs to the window floor whose median window bottom is nearest its
centre, and matching needs the same floor and x-interval coverage >= 0.5 (thin balcony boxes a little higher or
lower than the photo's fail the area rule though they are in the right place).
CMP façades are reported separately: the LSAA boxes were prefilled from our pipeline's boxes.

  python layout_balcony.py [--cover 0.5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent)); import paths  # noqa: E402
HERE = paths.ASTRA_DIR
import eval_testset as ev  # noqa: E402
from eval_astra import init_alignment, refine_alignment  # noqa: E402
from eval_astra_protocol import grid_map  # noqa: E402
from layout_floorwise_viz import RENDERS  # noqa: E402
import layout_floorwise as lf  # noqa: E402

GT = paths.LABELS_DIR / "balcony_gt_boxes.jsonl"


def coverage(target, boxes, shape_hw):
    x0, y0, x1, y1 = (int(round(v)) for v in target)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(shape_hw[1], x1), min(shape_hw[0], y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    m = np.zeros((y1 - y0, x1 - x0), bool)
    for b in boxes:
        a0, b0, a1, b1 = (int(round(v)) for v in b)
        a0, a1, b0, b1 = max(a0, x0) - x0, min(a1, x1) - x0, max(b0, y0) - y0, min(b1, y1) - y0
        if a1 > a0 and b1 > b0:
            m[b0:b1, a0:a1] = True
    return float(m.mean())


def floor_of(box, row_bottoms) -> int:
    """Window floor a balcony belongs to: the floor whose windows' median bottom is nearest the balcony's centre."""
    cy = (box[1] + box[3]) / 2
    return int(np.argmin([abs(cy - b) for b in row_bottoms]))


def floor_match(gboxes, placed, ign, gb):
    """--mode floor: same floor (nearest window-row bottom) and x-interval coverage >= 0.5 (layout_floorwise.cover_1d)."""
    rows = lf.floors(gb)
    bottoms = [float(np.median([gb[i][3] for i in r])) for r in rows]
    gf = [floor_of(g, bottoms) for g in gboxes]
    pf = [floor_of(p, bottoms) for p in placed]
    found = sum(lf.cover_1d((g[0], g[2]), [(p[0], p[2]) for p, f in zip(placed, pf) if f == k]) >= 0.5
                for g, k in zip(gboxes, gf))
    igf = [floor_of(g, bottoms) for g in ign]
    counted = ok = 0
    for p, k in zip(placed, pf):
        on_gt = lf.cover_1d((p[0], p[2]), [(g[0], g[2]) for g, f in zip(gboxes, gf) if f == k])
        if ign and lf.cover_1d((p[0], p[2]), [(g[0], g[2]) for g, f in zip(ign, igf) if f == k]) >= 0.5 and on_gt < 0.5:
            continue
        counted += 1
        ok += on_gt >= 0.5
    return found, counted, ok


def cell_confusion(gboxes, placed, ign, gb):
    """Cell-by-cell: every usable photo window is a cell; it has a balcony when the balconies on its floor (nearest
    window-row bottom) cover >= 50 % of the window's x-interval. Photo vs render: TP / FN / FP / TN; cells covered by
    an ignore region are skipped."""
    rows = lf.floors(gb)
    bottoms = [float(np.median([gb[i][3] for i in r])) for r in rows]
    fl = {i: k for k, r in enumerate(rows) for i in r}
    gf, pf, igf = ([floor_of(b, bottoms) for b in bs] for bs in (gboxes, placed, ign))
    c = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for i, w in enumerate(gb):
        iv, k = (w[0], w[2]), fl[i]
        if ign and lf.cover_1d(iv, [(g[0], g[2]) for g, f in zip(ign, igf) if f == k]) >= 0.5:
            continue
        has_g = lf.cover_1d(iv, [(g[0], g[2]) for g, f in zip(gboxes, gf) if f == k]) >= 0.5
        has_r = lf.cover_1d(iv, [(p[0], p[2]) for p, f in zip(placed, pf) if f == k]) >= 0.5
        c[("tp" if has_r else "fn") if has_g else ("fp" if has_r else "tn")] += 1
    return c


def row_match(gboxes, placed, ign, gb):
    """--mode rows, as layout_floorwise.py for windows: each image's balconies grouped into their own rows (centre bands,
    gap 0.5 median height), rows matched in order by DP on centre height (tolerance half the photo's window-floor
    pitch), then x-interval coverage >= 0.5 within matched rows (one render box may cover several). Render balconies
    mostly on ignore regions are not counted; balconies in unmatched rows are misses / extras."""
    if not gboxes and not placed:
        return 0, 0, 0
    rows_w = lf.floors(gb)
    wc = [lf.ycentre(gb, r) for r in rows_w]
    tol = 0.5 * (float(np.median(np.diff(wc))) if len(wc) > 1 else float(np.median([b[3] - b[1] for b in gb])) * 2)
    gr = lf.floors(gboxes) if gboxes else []
    pr = lf.floors(placed) if placed else []
    pairs = lf.match_floors([lf.ycentre(gboxes, r) for r in gr], [lf.ycentre(placed, r) for r in pr], tol)
    found = counted = ok = 0
    mr = {j for _, j in pairs}
    for i, j in pairs:
        rint = [(placed[k][0], placed[k][2]) for k in pr[j]]
        gint = [(gboxes[k][0], gboxes[k][2]) for k in gr[i]]
        found += sum(lf.cover_1d(g, rint) >= 0.5 for g in gint)
        for k in pr[j]:
            p = placed[k]
            on = lf.cover_1d((p[0], p[2]), gint)
            if ign and on < 0.5 and coverage(p, ign, (10 ** 6, 10 ** 6)) >= 0.5:
                continue
            counted += 1
            ok += on >= 0.5
    for j, r in enumerate(pr):
        if j not in mr:
            counted += sum(1 for k in r if not (ign and coverage(placed[k], ign, (10 ** 6, 10 ** 6)) >= 0.5))
    return found, counted, ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cover", type=float, default=0.5)
    ap.add_argument("--mode", choices=["rows", "floor", "area"], default="rows",
                    help="rows: own balcony rows matched in order, as the windows (default); floor: same window floor + x coverage; area: 2-D area coverage >= --cover")
    ap.add_argument("--methods", nargs="+", default=["ours_full", "astra"])
    ap.add_argument("--out", type=Path, default=HERE / "eval_layout_balcony.json")
    ap.add_argument("--tids", type=Path, default=HERE / "eval_protocol_grid_iou01" / "eval_astra_protocol.json")
    a = ap.parse_args()
    gt = ev.load_gt(ev.TESTSET / "labels")
    tids = json.loads(a.tids.read_text())["tids"]
    bgt = {json.loads(l)["stem"]: json.loads(l) for l in GT.open()}
    sam = None
    res = {}
    for m in a.methods:
        cache = HERE / "eval_layout_dets" / f"{m}_balcony"
        cache.mkdir(parents=True, exist_ok=True)
        per = {}
        for t in tids:
            stem = gt[t]["stem"]
            cp = cache / f"{stem}.json"
            if not cp.is_file():
                if sam is None:
                    import sam_iou_testset as si
                    sam = si.Sam(torch.device("cuda"))
                img = Image.open(RENDERS[m](stem)).convert("RGB")
                dets = [d["box_f"] for d in sam.instances(img, "balcony", 0.45)]
                cp.write_text(json.dumps({"boxes": dets}) + "\n")
            rbal = json.loads(cp.read_text())["boxes"]
            rwin = [x["box_xyxy"] for x in json.loads((HERE / "eval_layout_dets" / m / f"{stem}.json").read_text())["windows"]]
            gb = [w["box"] for w in gt[t]["windows"] if not w["unusable"]]
            sc, sh = init_alignment(rwin, gb)
            sc, sh = refine_alignment(rwin, gb, sc, sh)
            fm = grid_map(rwin, gb, sc, sh)
            placed = [fm(b) for b in rbal]
            B = bgt[stem]
            hw = (B["height"], B["width"])
            gboxes, ign = B["boxes"], B.get("ignore") or []
            if a.mode == "rows":
                found, counted, ok = row_match(gboxes, placed, ign, gb)
            elif a.mode == "floor":
                found, counted, ok = floor_match(gboxes, placed, ign, gb)
            else:
                found = sum(coverage(g, placed, hw) >= a.cover for g in gboxes)
                counted = ok = 0
                for p in placed:
                    if ign and coverage(p, ign, hw) >= 0.5 and coverage(p, gboxes, hw) < a.cover:
                        continue
                    counted += 1
                    ok += coverage(p, gboxes, hw) >= a.cover
            per[t] = {"stem": stem, "n_gt": len(gboxes), "found": found, "n_render": counted, "correct": ok,
                      "cmp": stem.startswith("cmp_"), **cell_confusion(gboxes, placed, ign, gb)}
        out = {}
        for name, sel in (("all", lambda v: True), ("cmp", lambda v: v["cmp"]), ("lsaa", lambda v: not v["cmp"])):
            vs = [v for v in per.values() if sel(v)]
            N, Fd, R, C = (sum(v[k] for v in vs) for k in ("n_gt", "found", "n_render", "correct"))
            P, Rc = C / max(1, R), Fd / max(1, N)
            tp, fn, fp, tn = (sum(v[k] for v in vs) for k in ("tp", "fn", "fp", "tn"))
            cr, ce = tp / max(1, tp + fn), tn / max(1, tn + fp)
            out[name] = {"facades": len(vs), "gt": N, "render": R, "recall": Rc, "precision": P, "f1": 2 * P * Rc / max(1e-9, P + Rc),
                         "cells": {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "balcony_recall": cr, "empty_recall": ce,
                                   "balanced_acc": (cr + ce) / 2, "acc": (tp + tn) / max(1, tp + fn + fp + tn)}}
            print(f"{m:9s} {name:4s} n={len(vs):2d}  boxes: recall {Rc:.3f} precision {P:.3f} F1 {out[name]['f1']:.3f} | "
                  f"cells TP {tp} FN {fn} FP {fp} TN {tn}: balcony cells found {cr:.3f}, empty cells kept empty {ce:.3f}, "
                  f"balanced {(cr + ce) / 2:.3f}")
        res[m] = {"summary": out, "per_facade": per}
    a.out.write_text(json.dumps(res, indent=1, default=int) + "\n")


if __name__ == "__main__":
    main()
