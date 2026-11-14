#!/usr/bin/env python3
"""Score Astra's façades with the test-set protocol of tools/eval_testset.py (the numbers in the main table).

Each labelled Astra façade (render frame) is aligned to the photo as in eval_astra.py (window-set extents, mutual
nearest centres, then the summed Hungarian IoU maximised: the placement most favourable to Astra), then turned into
a pseudo recovery DSL: one instance per Astra window (aligned box), type = its Astra cluster, the type's structure =
the cluster's stage-2 label. Unusable Astra windows are instances with no structure. ``eval_testset.score_facade`` /
``summarize`` then score it exactly like a pipeline run: greedy one-to-one IoU >= 0.5, unmatched test windows score
0 on exact / TED, shape = macro F1 over rectangle / arch_head.

``--ours labels.json`` (eval_testset.py --out of a pipeline run) also summarises that run on the same façades.

Usage: python eval_astra_protocol.py --labels light_labels/labels light_labels2/labels \\
           [--ours ../../experiments/facade_e2e_combined/runs_test_cur_eval/labels.json] [--out DIR]
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
from eval_astra import apply, init_alignment, match_centre, refine_alignment  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
import soft_structure as ss  # noqa: E402

KEYS = ("facades", "windows", "precision", "recall", "type", "type_ted", "shape_macro_f1", "pair_p", "same1", "ari")


def _bands(boxes: list, lo: int, gap: float) -> list[list[int]]:
    """Group box indices into rows (lo=1: by centre y) or columns (lo=0: by centre x); a new band at a centre gap > gap."""
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][lo] + boxes[i][lo + 2])
    out, last = [], None
    for i in order:
        c = (boxes[i][lo] + boxes[i][lo + 2]) / 2
        if last is None or c - last > gap:
            out.append([])
        out[-1].append(i)
        last = c
    return out


def _piecewise(knots: list[tuple[float, float]], scale: float, shift: float):
    """Monotone piecewise-linear map through (astra, photo) knots; outside them the global scale continues."""
    knots = sorted(knots)
    mono = []
    for u, v in knots:
        if not mono or (u > mono[-1][0] + 1e-6 and v > mono[-1][1] + 1e-6):
            mono.append((u, v))
    if len(mono) < 2:
        return lambda x: x * scale + shift
    us, vs = np.array([k[0] for k in mono]), np.array([k[1] for k in mono])

    def f(x):
        if x < us[0]:
            return vs[0] + (x - us[0]) * scale
        if x > us[-1]:
            return vs[-1] + (x - us[-1]) * scale
        return float(np.interp(x, us, vs))
    return f


def grid_map(ab: list, gb: list, s, t, iters: int = 3):
    """Floor / bay alignment: after the global fit, each photo floor (row of test windows) and bay (column) pulls the
    Astra windows centre-matched into it; the Astra band's median top / bottom (left / right) edges map to the photo
    band's, piecewise linear between bands, as the per-cell warp of our renders in sam_iou_testset.py."""
    fx = lambda x: x * s[0] + t[0]  # noqa: E731
    fy = lambda y: y * s[1] + t[1]  # noqa: E731
    mh = float(np.median([b[3] - b[1] for b in gb]))
    mw = float(np.median([b[2] - b[0] for b in gb]))
    rows, cols = _bands(gb, 1, 0.5 * mh), _bands(gb, 0, 0.5 * mw)
    for _ in range(iters):
        mapped = [[fx(b[0]), fy(b[1]), fx(b[2]), fy(b[3])] for b in ab]
        m = {j: i for i, j in match_centre(mapped, gb).items()}  # photo j -> astra i
        kx, ky = [], []
        for bands, lo, knots in ((rows, 1, ky), (cols, 0, kx)):
            for band in bands:
                pairs = [(m[j], j) for j in band if j in m]
                if not pairs:
                    continue
                for e in (lo, lo + 2):
                    knots.append((float(np.median([ab[i][e] for i, _ in pairs])), float(np.median([gb[j][e] for _, j in pairs]))))
        fx, fy = _piecewise(kx, s[0], t[0]), _piecewise(ky, s[1], t[1])
    def fmap(b):
        return [fx(b[0]), fy(b[1]), fx(b[2]), fy(b[3])]
    fmap.fx, fmap.fy = fx, fy  # the per-axis maps (render -> photo), for warping the render image
    return fmap


def pseudo_dsl(awins: list[dict], fmap) -> dict:
    inst, types, nxt = [], {}, 1000
    for i, w in enumerate(awins):
        if w["unusable"] or w["tokens"] is None:
            tid, toks = nxt, []
            nxt += 1
        else:
            tid, toks = w["cluster"], w["tokens"]
        types[tid] = toks
        inst.append({"unit_id": i, "type_id": tid, "kind": "window", "box_xyxy": fmap(w["box"]), "pred_ir": None})
    return {"instances": inst, "window_types": [{"type_id": k, "structure_tokens": v} for k, v in types.items()]}


def panel(G: dict, dsl: dict) -> Image.Image:
    im = Image.open(TESTSET / "photos" / f"{G['stem']}.jpg").convert("RGB")
    d = ImageDraw.Draw(im)
    for w in G["windows"]:
        if not w["unusable"]:
            d.rectangle(w["box"], outline=(0, 230, 90), width=3)
    for u in dsl["instances"]:
        d.rectangle(u["box_xyxy"], outline=(255, 0, 200), width=2)
    im = im.resize((int(im.width * 300 / im.height), 300))
    c = Image.new("RGB", (im.width, 320), (20, 20, 20))
    c.paste(im, (0, 20))
    ImageDraw.Draw(c).text((4, 4), G["stem"][:40], fill=(255, 255, 255))
    return c


def pick(s: dict) -> dict:
    return {k: s[k] for k in KEYS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, nargs="+", default=[HERE / "light_labels" / "labels"])
    ap.add_argument("--ours", type=Path, help="eval_testset.py per-window json of a pipeline run")
    ap.add_argument("--min-iou", type=float, default=0.5)
    ap.add_argument("--match", choices=["iou", "centre"], default="iou",
                    help="centre: Hungarian on centre distance < half the smaller width (eval_astra.py), Astra only")
    ap.add_argument("--align", choices=["global", "grid"], default="global",
                    help="grid: then floor / bay bands of the photo, piecewise linear (grid_map)")
    ap.add_argument("--soft-tol", type=float, default=0.03, help="soft line F1 tolerance (soft_structure.py)")
    ap.add_argument("--out", type=Path, default=HERE / "eval_protocol")
    a = ap.parse_args()
    gt_all = ev.load_gt(TESTSET / "labels")
    per, align, panels, soft = {}, {}, [], []
    for lab in a.labels:
        for tid, A in sorted(ev.load_gt(lab).items()):
            if tid in per:
                raise SystemExit(f"{tid} labelled twice")
            if tid not in gt_all:  # left the test set (2026-11-10)
                continue
            G = gt_all[tid]
            gb = [w["box"] for w in G["windows"] if not w["unusable"]]
            ab = [w["box"] for w in A["windows"]]
            s, t = init_alignment(ab, gb)
            s, t = refine_alignment(ab, gb, s, t)
            align[tid] = {"sx": float(s[0]), "sy": float(s[1]), "tx": float(t[0]), "ty": float(t[1]), "labels": str(lab)}
            fmap = grid_map(ab, gb, s, t) if a.align == "grid" else (lambda b, s=s, t=t: apply(b, s, t))
            dsl = pseudo_dsl(A["windows"], fmap)
            if a.match == "centre":
                ev.match = lambda units, wins, _m: {j: i for i, j in match_centre(
                    [u["box_xyxy"] for u in units], [w["box"] for w in wins]).items()}
            per[tid] = ev.score_facade(dsl, G, a.min_iou)
            soft += ss.score_facade(dsl, G, a.min_iou, a.soft_tol)
            panels.append(panel(G, dsl))
    res = {"astra": dict(pick(ev.summarize(per)), soft_line_f1=float(np.mean(soft))), "tids": sorted(per)}
    if a.ours:
        ours = json.loads(a.ours.read_text())["per_facade"]
        missing = [t for t in per if t not in ours]
        res["ours_same_facades"] = pick(ev.summarize({t: ours[t] for t in per if t in ours}))
        res["ours_all"] = pick(ev.summarize(ours))
        res["ours_missing"] = missing
    a.out.mkdir(parents=True, exist_ok=True)
    rows = [panels[i:i + 10] for i in range(0, len(panels), 10)]
    sheet = Image.new("RGB", (max(sum(p.width + 4 for p in r) for r in rows), 324 * len(rows)), (20, 20, 20))
    for y, r in enumerate(rows):
        x = 0
        for p in r:
            sheet.paste(p, (x, y * 324))
            x += p.width + 4
    sheet.save(a.out / "alignment_check.jpg", quality=85)
    (a.out / "eval_astra_protocol.json").write_text(json.dumps(
        dict(res, per_facade={t: {k: v for k, v in f.items() if k != "recs"} for t, f in per.items()}, align=align),
        indent=1, default=str) + "\n")
    for k in ("astra", "ours_same_facades", "ours_all"):
        if k in res:
            s = res[k]
            print(f"{k:18s} n={s['facades']:3d} win={s['windows']:4d}  det P {s['precision']:.3f} R {s['recall']:.3f}  "
                  f"shape {s['shape_macro_f1']:.3f}  TED {s['type_ted']:.3f}  exact {s['type']:.3f}  "
                  f"pairP {s['pair_p']:.3f} same1 {s['same1']:.3f}" + (f"  soft {s['soft_line_f1']:.3f}" if "soft_line_f1" in s else ""))


if __name__ == "__main__":
    main()
