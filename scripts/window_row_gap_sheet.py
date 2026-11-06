#!/usr/bin/env python3
"""Rows where different window types on one floor have the most different bottoms.
Each facade: photo with that row's boxes | windows-only render with the same windows. One colour per type; the gap is
printed as a share of the row's median window height. Writes runs_val30_seated/compare/row_gaps_<n>.jpg."""
import collections, importlib, json, statistics as st, sys, types
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
E = Path(__file__).resolve().parents[1]; O = E / "runs_val30_seated/compare"; O.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(E / "vendor/window_compiler"), str(E / "scripts")]
for m in ("bpy", "bmesh", "mathutils"): sys.modules.setdefault(m, types.ModuleType(m))
sys.modules["mathutils"].Vector = object
sys.argv = ["x"]
import facade_spec as fs
rf = importlib.import_module("render_facade")
PAL = [(230,25,75),(60,180,75),(0,130,200),(255,200,25),(245,130,48),(145,30,180)]; H = 620; TOP = 8
Ft = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 21)
def worst(t):
    D = json.load(open(E / f"runs_val30_combined/{t}/facade_dsl.json")); spec = fs.normalize_facade_spec(D)
    rows = collections.defaultdict(list)
    for w in rf.planned_windows(spec):
        if w.get("kind") != "door": rows[w["row"]].append(w)
    best = (0.0, None)
    for r, v in rows.items():
        by = collections.defaultdict(list)
        for w in v: by[w["name"]].append(w)
        if len(by) < 2: continue
        med = [st.median(q["z0"] for q in m) for m in by.values()]
        g = (max(med) - min(med)) / st.median(q["z1"] - q["z0"] for q in v)
        if g > best[0]: best = (g, r)
    return best, D, spec, rows
res = sorted(((worst(f"v{i:02d}"), f"v{i:02d}") for i in range(30)), key=lambda x: -x[0][0][0])[:TOP]
rows_out = []
for (best, D, spec, rows), t in res:
    g, r = best; ph = Image.open(D["meta"]["image"]).convert("RGB"); s = H / ph.height; A = ph.resize((round(ph.width * s), H)); da = ImageDraw.Draw(A)
    names = sorted({w["name"] for w in rows[r]}); col = {n: PAL[i % len(PAL)] for i, n in enumerate(names)}
    fl = spec["floor_ids"][r]
    for u in D["instances"]:
        if u.get("kind", "window") == "window" and u.get("floor") == fl:
            da.rectangle([v * s for v in u["box_xyxy"]], outline=col.get(f"win_type_{int(u['type_id']):02d}", (255, 255, 255)), width=3)
    im = Image.open(E / f"runs_val30_seated/front/{t}_ours.png").convert("RGB"); im = im.crop(im.convert("L").point(lambda v: 255 if v > 8 else 0).getbbox()); B = im.resize((round(im.width * H / im.height), H)); db = ImageDraw.Draw(B)
    W, Z = fs.total_grid_size(spec["grid"])
    for w in rows[r]:
        x0, x1 = sorted(((W / 2 - w["x1"]) / W * B.width, (W / 2 - w["x0"]) / W * B.width))
        db.rectangle([x0, (1 - w["z1"] / Z) * H, x1, (1 - w["z0"] / Z) * H], outline=col[w["name"]], width=3)
    R = Image.new("RGB", (A.width + B.width + 30, H + 32), "white"); D2 = ImageDraw.Draw(R)
    D2.text((4, 4), f"{t} floor row {r}: gap {g:.2f}", fill="black", font=Ft); D2.text((A.width + 24, 4), "render", fill="black", font=Ft); R.paste(A, (0, 32)); R.paste(B, (A.width + 20, 32)); rows_out.append(R)
    print(t, "row", r, "gap %.2f" % g, {n: (round(st.median(q["z0"] for q in rows[r] if q["name"] == n), 2), round(st.median(q["z1"] - q["z0"] for q in rows[r] if q["name"] == n), 2), sum(q["name"] == n for q in rows[r])) for n in names})
for k in range(0, len(rows_out), 4):
    pg = rows_out[k:k+4]; P = Image.new("RGB", (max(q.width for q in pg), sum(q.height + 10 for q in pg)), "white"); y = 0
    for q in pg: P.paste(q, (0, y)); y += q.height + 10
    P.save(O / f"row_gaps_{k // 4 + 1}.jpg", quality=88)
