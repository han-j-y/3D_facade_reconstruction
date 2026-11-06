#!/usr/bin/env python3
"""test100 sample: photo (blocks magenta, flagged side windows yellow) | front off | front on | oblique off | oblique on.
Writes runs_test_side/compare/side_<n>.jpg and prints how much each render changed."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
E = Path(__file__).resolve().parents[1]; O = E / "runs_test_side"; F = O / "off"; C = O / "compare"; C.mkdir(exist_ok=True); H = 430
Ft = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 19)
GROUPS = [("punched, side windows", "t013 t014 t109 t110"), ("curtain wall, flagged", "t074 t100 t104 t105"), ("bay windows, no side", "t021 t067 t103 t031"), ("open balconies", "t005 t027 t032 t041"), ("no balconies", "t017 t026 t075")]
def load(p):
    im = Image.open(p)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, "white"); bg.paste(im, mask=im.split()[3]); return bg.crop(im.split()[3].getbbox())
    return im.convert("RGB")
fit = lambda im: im.resize((round(im.width * H / im.height), H))
def diff(a, b):
    a, b = load(a), load(b).resize(load(a).size); return float((np.abs(np.asarray(a, int) - np.asarray(b, int)).sum(2) > 30).mean() * 100)
def front(R, t):
    p = R / "front" / f"{t}_balc.png"; return p if p.exists() else R / "front" / f"{t}_ours.png"
for g, (name, ids) in enumerate(GROUPS, 1):
    rows = []
    for t in ids.split():
        p = O / t / "balcony/facade_dsl_with_balconies.json"; D = json.load(open(p if p.exists() else O / t / "facade_dsl.json"))
        ph = Image.open(D["meta"]["image"]).convert("RGB"); d = ImageDraw.Draw(ph); lw = max(2, ph.height // 300)
        B = D.get("layout", {}).get("balconies", []) or []; S = [u for u in D["instances"] if u.get("side_face")]
        for b in B:
            if "width_norm" not in b: d.rectangle(b["box_xyxy"], outline=(255, 0, 255), width=lw)
        for u in S: d.rectangle(u["box_xyxy"], outline=(255, 230, 0), width=lw * 2)
        df, dq = diff(front(F, t), front(O, t)), diff(F / "oblique" / f"{t}.png", O / "oblique" / f"{t}.png")
        print(name, t, "side", len(S), f"front {df:.2f}% oblique {dq:.2f}%")
        cells = [(f"{t} photo, {len(S)} side", fit(ph)), ("front OFF", fit(load(front(F, t)))), (f"front ON ({df:.1f}%)", fit(load(front(O, t)))), ("angle OFF", fit(load(F / "oblique" / f"{t}.png"))), (f"angle ON ({dq:.1f}%)", fit(load(O / "oblique" / f"{t}.png")))]
        R = Image.new("RGB", (sum(c.width for _, c in cells) + 60, H + 28), "white"); dr = ImageDraw.Draw(R); x = 0
        for s, c in cells: dr.text((x + 4, 3), s, fill="black", font=Ft); R.paste(c, (x, 28)); x += c.width + 15
        rows.append(R)
    P = Image.new("RGB", (max(q.width for q in rows), sum(q.height + 8 for q in rows)), "white"); y = 0
    for q in rows: P.paste(q, (0, y)); y += q.height + 8
    P.save(C / f"side_{g}.jpg", quality=88); print(C / f"side_{g}.jpg", P.size)
