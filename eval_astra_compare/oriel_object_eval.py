"""Element-wise oriel comparison on Astra's 46 facades. Labelled oriels: connected groups of labelled oriel cells
(photo boxes). Predicted oriels: each method's oriel boxes read from its 3D model exactly as in layout_grid_eval.py
(upper floors only; Astra = depth > 0.3 m + oriel-named objects), boxes within half a window of each other merged, placed on the photo with
the same alignment. Match one-to-one (greedy by overlap): horizontal overlap >= 50 % of the narrower width and
vertical overlap >= 25 % of the shorter height. Reports object P / R / F1 and, over matched pairs, the IoU of the
photo window rows (floors) each covers."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.argv = sys.argv[:1]
import layout_grid_eval as G
import numpy as np

def merge(B, gx=0.0, gy=0.0):
    """Merge boxes that overlap after growing each by gx / gy (half a window), i.e. pieces of one element."""
    B = [list(b) for b in B]; m = True
    while m:
        m = False
        for i in range(len(B)):
            for j in range(i + 1, len(B)):
                a, b = B[i], B[j]
                if a[0] - gx <= b[2] and b[0] - gx <= a[2] and a[1] - gy <= b[3] and b[1] - gy <= a[3]:
                    B[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]; B.pop(j); m = True; break
            if m: break
    return B

def cell_objects(cells, fy, bx):
    cells = set(map(tuple, cells)); seen, out = set(), []
    for c in cells:
        if c in seen: continue
        comp, st = set(), [c]
        while st:
            f, b = st.pop()
            if (f, b) in comp or (f, b) not in cells: continue
            comp.add((f, b)); st += [(f + 1, b), (f - 1, b), (f, b + 1), (f, b - 1)]
        seen |= comp
        out.append([min(bx[b][0] for _, b in comp), min(fy[f][0] for f, _ in comp), max(bx[b][1] for _, b in comp), max(fy[f][1] for f, _ in comp)])
    return out

def ov1(a0, a1, b0, b1): return max(0.0, min(a1, b1) - max(a0, b0)) / max(1e-6, min(a1 - a0, b1 - b0))

def main():
    gt = G.ev.load_gt(G.ev.TESTSET / "labels")
    ol = {json.loads(l)["stem"]: json.loads(l) for l in (G.LBL / "oriel_labels.jsonl").open()}
    maps = {r["test_id"]: r for r in json.load(open(G.A / "light/mapping.json")) + json.load(open(G.A / "light/output/mapping.json"))}
    okw = re.compile(r"oriel|projecting bay|erker|bow window", re.I)
    key = lambda n: re.match(r"(Balc_\d+)", n).group(1) if re.match(r"Balc_\d+", n) else n
    tids = json.load(open(G.A / "tids46.json"))["tids"]
    tot = {m: dict(tp=0, fp=0, fn=0, fiou=[]) for m in ("ours_cur", "astra")}; per = {}
    for t in tids:
        stem = gt[t]["stem"]; ws = gt[t]["windows"]; gb = [w["box"] for w in ws if not w["unusable"]]; gW = [w["box"] for w in ws]
        fy, bx = G.dsl_bands(json.loads((G.LBL / "window_grid" / stem / "facade_dsl.json").read_text()))
        L = cell_objects([c for c in (ol[stem].get("cells") or []) if c[0] in fy and c[1] in bx], fy, bx)
        rows = sorted(G._bands(gW, 1, 0.5 * float(np.median([b[3] - b[1] for b in gW]))), key=lambda r: G.lf.ycentre(gW, r))
        rc = [G.lf.ycentre(gW, r) for r in rows]
        floors = lambda b: {k for k, y in enumerate(rc) if b[1] <= y <= b[3]}
        per[t] = {"labelled": len(L)}
        for m in ("ours_cur", "astra"):
            rw = [x["box_xyxy"] for x in json.loads((G.A / "eval_layout_dets" / m / f"{stem}.json").read_text())["windows"]]
            h = json.load(open(G.HM / ("ours" if m == "ours_cur" else "astra") / f"{t}.json"))
            if m == "ours_cur":
                ro = G.object_boxes(h, {i for i, n in enumerate(h["names"]) if "balc_enclosed" in n}, key=key)
            else:
                lab = G.A / ("light_labels2/labels" if maps[t]["astra_folder"].startswith("facade_") else "light_labels/labels")
                ro, _ = G.astra_oriels(t, stem, [w["box"] for w in G.ev.load_gt(lab)[t]["windows"]])
                ro = ro + G.object_boxes(h, {i for i, n in enumerate(h["names"]) if okw.search(n)})
            gx = 0.5 * float(np.median([b[2] - b[0] for b in rw])); gy = 0.5 * float(np.median([b[3] - b[1] for b in rw]))
            ro = merge(G.upper_only(ro, rw), gx, gy)
            a, b_ = G.init_alignment(rw, gb); a, b_ = G.refine_alignment(rw, gb, a, b_); fmap = G.grid_map(rw, gb, a, b_)
            Pd = [fmap(b) for b in ro]
            cand = sorted(((ov1(p[0], p[2], l[0], l[2]), i, j) for i, p in enumerate(Pd) for j, l in enumerate(L)
                           if ov1(p[0], p[2], l[0], l[2]) >= 0.5 and ov1(p[1], p[3], l[1], l[3]) >= 0.25), reverse=True)
            ui, uj, pairs = set(), set(), []
            for v, i, j in cand:
                if i in ui or j in uj: continue
                ui.add(i); uj.add(j); pairs.append((i, j))
            T = tot[m]; T["tp"] += len(pairs); T["fp"] += len(Pd) - len(pairs); T["fn"] += len(L) - len(pairs)
            for i, j in pairs:
                a_, b2 = floors(Pd[i]), floors(L[j]); T["fiou"].append(len(a_ & b2) / max(1, len(a_ | b2)))
            per[t][m] = {"pred": len(Pd), "matched": len(pairs)}
    for m, T in tot.items():
        P = T["tp"] / max(1, T["tp"] + T["fp"]); R = T["tp"] / max(1, T["tp"] + T["fn"])
        print(f"{m:9s} oriels TP {T['tp']} FP {T['fp']} FN {T['fn']}  P {P:.3f} R {R:.3f} F1 {2*P*R/max(1e-9,P+R):.3f}  floor IoU of matched {np.mean(T['fiou']) if T['fiou'] else float('nan'):.3f}")
    print("labelled oriels:", sum(v["labelled"] for v in per.values()), "in", sum(bool(v["labelled"]) for v in per.values()), "facades")
    json.dump(per, open(G.paths.OUT_DIR / "oriel_objects_46.json", "w"), indent=1)

if __name__ == "__main__":
    main()
