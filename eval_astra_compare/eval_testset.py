#!/usr/bin/env python3
"""Score facade_e2e runs (one ``<tid>/facade_dsl.json`` per façade) against the test-set labels.

Ground truth: stage-1 windows (labels/clusters, status ``done``; deleted ones dropped) and the stage-2 structure of
each cluster (labels/structure/<tid>_cKK/tokens.json), given to every member of the cluster. Windows marked
unusable are neither scored nor counted as false positives.

Pipeline window units are matched one-to-one to labelled windows by greedy IoU (>= --min-iou).

* detection   – precision / recall of units against clustered windows (GT boxes are SAM3 proposals the
                labeller kept or fixed, so this is optimistic)
* structure   – pane dissection only (boundary shape is scored separately). Each structure becomes a canonical
                pane tree (``pane_tree``: split positions ignored, through-cuts flattened, a head/body seam is a
                horizontal cut of the glass). ``exact`` = share of windows whose tree equals the label's, ``TED`` =
                mean 1 - tree edit distance / max(tree sizes), floored at 0. Unmatched windows score 0. ``dist`` =
                mean raw tree edit distance (unit costs); unmatched / unparseable windows cost the label tree's size.
                ``member`` = the unit's own prediction, ``type`` = its type's structure in facade_dsl.json.
                ``exemplar`` rows use only the window each cluster was labelled on.
* shape       – per-class F1 of the type's boundary shape (round heads count as ``arch_head``); the macro F1 averages
                the classes in ``MACRO_SHAPES`` (the only ones any method outputs). Windows of rarer labelled shapes
                (ellipse, circle, …) still count as false positives of whatever class they are predicted as.
* clustering  – over matched clustered windows per façade: ARI of types vs labelled clusters, share of same-cluster
                pairs put in one type (``same→1``) and of different-cluster pairs kept apart (``diff→split``).
                ``ACC`` = share of windows in the cluster their type is matched to (one-to-one Hungarian matching per
                façade, pooled over windows); pairwise P / R / F1 over window pairs put in one type.

Example:
  python tools/eval_testset.py --run ../../experiments/facade_e2e/runs_testset_main_4ec7868
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent)); import paths  # noqa: E402
TESTSET = paths.TESTSET_DIR
WAP = paths.WINDOW_AST_DIR
sys.path.insert(0, str(WAP))
from window_ast.metrics import try_parse_tokens  # noqa: E402
from window_ast.structure import ir_to_structure_tokens  # noqa: E402


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def canon(tokens: list[str]) -> list[str]:
    """Fold round-headed ``head_body`` into ``arch_head``: a segmental arch covers the semicircle case."""
    if "shape=head_body" in tokens and ("head=semicircle" in tokens or "head=arch" in tokens):
        return ["shape=arch_head" if t == "shape=head_body" else t for t in tokens if not t.startswith("head=")]
    return tokens


MACRO_SHAPES = ("shape=rectangle", "shape=arch_head")


def shape_of(tokens: list[str]) -> str:
    """Boundary shape (the first ``shape=``; later ones belong to program ops such as insert_shape) and head."""
    return " ".join([t for t in tokens if t.startswith("shape=")][:1] + [t for t in tokens if t.startswith("head=")])


def _leaves(rules: dict, rid: str, box: tuple) -> list[tuple]:
    """Pane rectangles (x0, y0, x1, y1) of rule ``rid`` in ``box``; splits at the middle, grids even (y up)."""
    rule = rules[rid]
    x0, y0, x1, y1 = box
    if rule["op"] == "grid":
        nx, ny = int(rule.get("vertical") or 0) + 1, int(rule.get("horizontal") or 0) + 1
        xs = [x0 + (x1 - x0) * Fraction(i, nx) for i in range(nx + 1)]
        ys = [y0 + (y1 - y0) * Fraction(i, ny) for i in range(ny + 1)]
        return [(xs[i], ys[j], xs[i + 1], ys[j + 1]) for i in range(nx) for j in range(ny)]
    if rule["op"] == "split":
        a, b = rule["children"]
        if rule["axis"] == "v":
            xm = (x0 + x1) / 2
            return _leaves(rules, a, (x0, y0, xm, y1)) + _leaves(rules, b, (xm, y0, x1, y1))
        ym = (y0 + y1) / 2
        return _leaves(rules, a, (x0, y0, x1, ym)) + _leaves(rules, b, (x0, ym, x1, y1))
    return [box]


def _guillotine(rects: list[tuple], box: tuple) -> tuple:
    """Canonical tree of a dissection: cut at every through line, vertical lines first."""
    if len(rects) == 1:
        return ("P", ())
    for axis, lo, hi in (("V", 0, 2), ("H", 1, 3)):
        cuts = sorted({r[lo] for r in rects} - {box[lo]})
        cuts = [c for c in cuts if not any(r[lo] < c < r[hi] for r in rects)]
        if cuts:
            edges = [box[lo], *cuts, box[hi]]
            kids = []
            for a, b in zip(edges, edges[1:]):
                sub = list(box)
                sub[lo], sub[hi] = a, b
                kids.append(_guillotine([r for r in rects if a <= r[lo] < b], tuple(sub)))
            return (axis, tuple(kids))
    raise ValueError("non-guillotine dissection")


def pane_tree(ir: dict | None) -> tuple | None:
    """Canonical pane tree of a structure IR, ignoring the boundary shape. A head/body seam cuts the glass at the
    middle: panes on ``body`` fill the lower half under one head pane, panes on ``glass`` are cut by the seam."""
    if not ir:
        return None
    try:
        sets = ir.get("panes") or []
        sets = [sets] if isinstance(sets, dict) else sets
        seam = any(op.get("id") == "seam" or op.get("region") == "body" for op in ir.get("program") or [])
        one, half = Fraction(1), Fraction(1, 2)
        lower, upper = (0, 0, one, half), (0, half, one, one)
        # every pane set (and the seam) partitions the glass; the pane tree is their common refinement
        rects = [(0, 0, one, one)]
        parts = [[lower, upper]] if seam else []
        for panes in sets:
            region = {"body": lower, "head": upper}.get(panes.get("region")) if seam else None
            leaves = _leaves(panes.get("rules") or {"root": {"op": "eps"}}, panes.get("root", "root"),
                             region or (0, 0, one, one))
            parts.append(leaves + ([upper if region == lower else lower] if region else []))
        for part in parts:
            rects = [c for r in rects for p in part
                     if (c := (max(r[0], p[0]), max(r[1], p[1]), min(r[2], p[2]), min(r[3], p[3])))[0] < c[2]
                     and c[1] < c[3]]
        return _guillotine(rects, (0, 0, one, one))
    except (KeyError, TypeError, ValueError, RecursionError):
        return None


def tokens_tree(tokens: list[str] | None) -> tuple | None:
    if not tokens:
        return None
    ok, ir = try_parse_tokens(tokens, structure_only=True)
    return pane_tree(ir) if ok else None


def _postorder(t: tuple) -> tuple[list[str], list[int]]:
    labels, lml = [], []

    def walk(n: tuple) -> int:
        first = None
        for c in n[1]:
            leftmost = walk(c)
            first = leftmost if first is None else first
        labels.append(n[0])
        lml.append(len(labels) - 1 if first is None else first)
        return lml[-1]

    walk(t)
    return labels, lml


def ted(t1: tuple, t2: tuple) -> int:
    """Zhang–Shasha tree edit distance, unit insert / delete / relabel costs."""
    l1, m1 = _postorder(t1)
    l2, m2 = _postorder(t2)
    kr = lambda m: sorted({max(i for i, v in enumerate(m) if v == x) for x in set(m)})  # noqa: E731
    td = [[0] * len(l2) for _ in l1]
    for i in kr(m1):
        for j in kr(m2):
            a, b = m1[i], m2[j]
            fd = [[0] * (j - b + 2) for _ in range(i - a + 2)]
            for x in range(1, i - a + 2):
                fd[x][0] = x
            for y in range(1, j - b + 2):
                fd[0][y] = y
            for x in range(1, i - a + 2):
                for y in range(1, j - b + 2):
                    p, q = a + x - 1, b + y - 1
                    if m1[p] == a and m2[q] == b:
                        fd[x][y] = min(fd[x - 1][y] + 1, fd[x][y - 1] + 1, fd[x - 1][y - 1] + (l1[p] != l2[q]))
                        td[p][q] = fd[x][y]
                    else:
                        fd[x][y] = min(fd[x - 1][y] + 1, fd[x][y - 1] + 1, fd[m1[p] - a][m2[q] - b] + td[p][q])
    return td[-1][-1]


def pane_score(pred: tuple | None, gold: tuple | None) -> tuple[bool, float, int]:
    """(exact match, 1 - TED / max size floored at 0, raw TED) of two pane trees. A missing prediction costs the
    size of the gold tree (every node inserted)."""
    if gold is None:
        return False, 0.0, 0
    if pred is None:
        return False, 0.0, len(_postorder(gold)[0])
    n = max(len(_postorder(pred)[0]), len(_postorder(gold)[0]))
    d = ted(pred, gold)
    return pred == gold, max(0.0, 1 - d / n), d


def pred_ir(u: dict) -> dict | None:
    """A unit's own structure prediction (``pred_ir``; ``structure_ir_member`` in older runs)."""
    return u.get("pred_ir", u.get("structure_ir_member"))


def load_gt(labels: Path) -> dict[str, dict]:
    struct = {}
    for line in (labels / "structure" / "labels.jsonl").open():
        r = json.loads(line)
        if r.get("status") == "done":
            struct[r["id"]] = canon(json.loads((labels / "structure" / r["id"] / "tokens.json").read_text())["tokens"])
    gt = {}
    for p in sorted((labels / "clusters").glob("*.json")):
        lab = json.loads(p.read_text())
        if lab.get("status") != "done":
            continue
        wins = []
        for w in lab["windows"]:
            if w.get("deleted"):
                continue
            k = w.get("cluster")
            sid = f"{lab['tid']}_c{int(k):02d}" if k is not None else None
            wins.append({"id": w["id"], "box": w["box_xyxy"], "unusable": bool(w.get("unusable")) or k is None,
                         "cluster": None if k is None else int(k), "tokens": struct.get(sid),
                         "pane": tokens_tree(struct.get(sid)),
                         "exemplar": k is not None and lab["clusters"][str(k)]["exemplar"] == w["id"]})
        gt[lab["tid"]] = {"stem": lab["stem"], "windows": wins}
    return gt


def match(units: list[dict], wins: list[dict], min_iou: float) -> dict[int, int]:
    pairs = sorted(((iou(u["box_xyxy"], w["box"]), i, j) for i, u in enumerate(units) for j, w in enumerate(wins)),
                   reverse=True)
    used_u, used_w, out = set(), set(), {}
    for v, i, j in pairs:
        if v < min_iou:
            break
        if i in used_u or j in used_w:
            continue
        used_u.add(i)
        used_w.add(j)
        out[j] = i
    return out


def singleton_types(dsl: dict) -> dict:
    """No type clustering: every window is its own type, with its own structure prediction."""
    units = [dict(u, type_id=i) for i, u in enumerate(dsl["instances"])]
    types = [{"type_id": i, "structure_tokens": ir_to_structure_tokens(pred_ir(u))
              if pred_ir(u) else []} for i, u in enumerate(units)]
    return dict(dsl, instances=units, window_types=types)


def score_facade(dsl: dict, gt: dict, min_iou: float) -> dict:
    units = [u for u in dsl["instances"] if u.get("kind", "window") == "window"]
    types = {int(t["type_id"]): canon(t["structure_tokens"]) for t in dsl["window_types"]}
    type_trees = {k: tokens_tree(t) for k, t in types.items()}
    wins = gt["windows"]
    m = match(units, wins, min_iou)
    recs = []
    for j, w in enumerate(wins):
        if w["unusable"]:
            continue
        r = {"id": w["id"], "cluster": w["cluster"], "exemplar": w["exemplar"], "matched": j in m,
             "gt_shape": shape_of(w["tokens"])}
        if j in m:
            u = units[m[j]]
            t = types.get(int(u["type_id"])) or []
            member, member_ted, member_dist = pane_score(pane_tree(pred_ir(u)), w["pane"])
            type_ok, type_ted, type_dist = pane_score(type_trees.get(int(u["type_id"])), w["pane"])
            r.update(unit_id=u["unit_id"], type=int(u["type_id"]), iou=round(iou(u["box_xyxy"], w["box"]), 3),
                     member=member, member_ted=member_ted, type_ok=type_ok, type_ted=type_ted,
                     member_dist=member_dist, type_dist=type_dist, shape=shape_of(t) == r["gt_shape"], pred_shape=shape_of(t))
        else:
            miss = pane_score(None, w["pane"])[2]
            r.update(member=False, member_ted=0.0, type_ok=False, type_ted=0.0, member_dist=miss, type_dist=miss,
                     shape=False)
        recs.append(r)
    matched_u = {m[j] for j, w in enumerate(wins) if j in m and not w["unusable"]}
    ignored_u = {m[j] for j, w in enumerate(wins) if j in m and w["unusable"]}
    mm = [r for r in recs if r["matched"]]
    same = [(a["type"] == b["type"]) for a, b in combinations(mm, 2) if a["cluster"] == b["cluster"]]
    diff = [(a["type"] != b["type"]) for a, b in combinations(mm, 2) if a["cluster"] != b["cluster"]]
    gt_k = len({r["cluster"] for r in recs})
    ari = adjusted_rand_score([r["cluster"] for r in mm], [r["type"] for r in mm]) if len(mm) > 1 else float("nan")
    hits = 0
    if mm:
        cs, ts = sorted({r["cluster"] for r in mm}), sorted({r["type"] for r in mm})
        cont = np.zeros((len(cs), len(ts)), dtype=int)
        for r in mm:
            cont[cs.index(r["cluster"]), ts.index(r["type"])] += 1
        hits = int(cont[linear_sum_assignment(-cont)].sum())
    joined = sum(a["type"] == b["type"] for a, b in combinations(mm, 2))
    return {"recs": recs, "tp": len(matched_u), "fp": len(units) - len(matched_u) - len(ignored_u),
            "fn": sum(not r["matched"] for r in recs), "ignored": len(ignored_u),
            "same": (sum(same), len(same)), "diff": (sum(diff), len(diff)), "ari": ari,
            "acc": (hits, len(mm)), "joined": joined,
            "n_types": len({int(u["type_id"]) for u in units}), "n_clusters": gt_k}


def shape_f1(recs: list[dict]) -> dict[str, float]:
    """Per-class F1 of the type's boundary shape; unmatched windows are false negatives of their GT shape."""
    classes = sorted({r["gt_shape"] for r in recs} | {r["pred_shape"] for r in recs if r["matched"]})
    out = {}
    for c in classes:
        tp = sum(r["matched"] and r["gt_shape"] == c and r["pred_shape"] == c for r in recs)
        fp = sum(r["matched"] and r["gt_shape"] != c and r["pred_shape"] == c for r in recs)
        fn = sum(r["gt_shape"] == c and not (r["matched"] and r["pred_shape"] == c) for r in recs)
        out[c] = 2 * tp / max(1, 2 * tp + fp + fn)
    return out


def summarize(per: dict) -> dict:
    recs = [r for v in per.values() for r in v["recs"]]
    ex = [r for r in recs if r["exemplar"]]
    tp, fp, fn = (sum(v[k] for v in per.values()) for k in ("tp", "fp", "fn"))
    aris = [v["ari"] for v in per.values() if v["ari"] == v["ari"]]
    f = lambda rs, k: sum(r[k] for r in rs) / max(1, len(rs))  # noqa: E731
    f1 = shape_f1(recs)
    return {
        "shape_f1": f1, "shape_macro_f1": sum(f1.get(c, 0.0) for c in MACRO_SHAPES) / len(MACRO_SHAPES),
        "facades": len(per), "windows": len(recs), "clusters": len(ex),
        "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "fp": fp, "fn": fn,
        "member": f(recs, "member"), "type": f(recs, "type_ok"), "shape": f(recs, "shape"),
        "member_ted": f(recs, "member_ted"), "type_ted": f(recs, "type_ted"),
        "member_dist": f(recs, "member_dist"), "type_dist": f(recs, "type_dist"),
        "ex_member": f(ex, "member"), "ex_type": f(ex, "type_ok"),
        "ex_member_ted": f(ex, "member_ted"), "ex_type_ted": f(ex, "type_ted"),
        "same1": sum(v["same"][0] for v in per.values()) / max(1, sum(v["same"][1] for v in per.values())),
        "diff_split": sum(v["diff"][0] for v in per.values()) / max(1, sum(v["diff"][1] for v in per.values())),
        "ari": sum(aris) / max(1, len(aris)),
        "acc": sum(v["acc"][0] for v in per.values()) / max(1, sum(v["acc"][1] for v in per.values())),
        "pair_p": sum(v["same"][0] for v in per.values()) / max(1, sum(v["joined"] for v in per.values())),
        "types_vs_clusters": (sum(v["n_types"] for v in per.values()), sum(v["n_clusters"] for v in per.values())),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="dir of <tid>/facade_dsl.json")
    ap.add_argument("--labels", type=Path, default=TESTSET / "labels")
    ap.add_argument("--min-iou", type=float, default=0.5)
    ap.add_argument("--no-types", action="store_true", help="ablation: score each window as its own type")
    ap.add_argument("--out", type=Path, default=None, help="per-window json (default: RUN/eval_testset.json)")
    args = ap.parse_args()
    gt = load_gt(args.labels)
    per = {}
    for tid, g in gt.items():
        p = args.run / tid / "facade_dsl.json"
        if not p.is_file():
            print(f"missing run: {tid}")
            continue
        dsl = json.loads(p.read_text())
        per[tid] = score_facade(singleton_types(dsl) if args.no_types else dsl, g, args.min_iou)

    print(f"{'tid':5s} {'win':>4s} {'P':>5s} {'R':>5s} {'member':>7s} {'type':>5s} {'typeTED':>7s} {'shpF1':>6s} "
          f"{'ARI':>5s} {'ACC':>5s} {'same→1':>9s} {'diff→split':>11s} {'types/GT':>8s}")
    for tid, v in per.items():
        s = summarize({tid: v})
        print(f"{tid:5s} {s['windows']:4d} {s['precision']:5.0%} {s['recall']:5.0%} {s['member']:7.0%} "
              f"{s['type']:5.0%} {s['type_ted']:7.0%} {s['shape_macro_f1']:6.0%} {v['ari']:5.2f} {s['acc']:5.0%} {v['same'][0]:4d}/{v['same'][1]:<4d} "
              f"{v['diff'][0]:5d}/{v['diff'][1]:<5d} {v['n_types']:3d}/{v['n_clusters']:<3d}")
    s = summarize(per)
    s["pair_f1"] = 2 * s["pair_p"] * s["same1"] / max(1e-9, s["pair_p"] + s["same1"])
    print(f"\n{s['facades']} façades, {s['windows']} labelled windows in {s['clusters']} clusters")
    print(f"detection   precision {s['precision']:.1%}  recall {s['recall']:.1%}  (FP {s['fp']}, FN {s['fn']})")
    print(f"panes exact member {s['member']:.1%}  type {s['type']:.1%}   "
          f"exemplars only: member {s['ex_member']:.1%}  type {s['ex_type']:.1%}")
    print(f"panes TED   member {s['member_ted']:.1%}  type {s['type_ted']:.1%}   "
          f"exemplars only: member {s['ex_member_ted']:.1%}  type {s['ex_type_ted']:.1%}")
    print(f"shape       macro F1 {s['shape_macro_f1']:.1%}  ("
          + "  ".join(f"{c.removeprefix('shape=')} {v:.1%}" for c, v in s["shape_f1"].items()) + ")")
    print(f"clustering  ARI {s['ari']:.3f} (mean over façades)  same→1 {s['same1']:.1%}  "
          f"diff→split {s['diff_split']:.1%}  types {s['types_vs_clusters'][0]} vs GT clusters "
          f"{s['types_vs_clusters'][1]}")
    print(f"            ACC {s['acc']:.1%}  pairwise P {s['pair_p']:.1%}  R {s['same1']:.1%}  F1 {s['pair_f1']:.1%}")
    out = args.out or args.run / "eval_testset.json"
    out.write_text(json.dumps({"summary": s, "per_facade": per}, indent=1, default=str) + "\n")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
