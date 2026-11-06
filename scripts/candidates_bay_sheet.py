#!/usr/bin/env python3
"""Candidate facades with enclosed blocks: photo (blocks magenta, flagged side windows yellow) | front | oblique from the camera's side.
Writes runs_cand/compare/bays_<n>.jpg."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
E = Path(__file__).resolve().parents[1]; O = E / "runs_cand"; C = O / "compare"; C.mkdir(exist_ok=True); H = 520
Ft = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 21)
def load(p):
    im = Image.open(p)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, "white"); bg.paste(im, mask=im.split()[3]); return bg.crop(im.split()[3].getbbox())
    im = im.convert("RGB"); b = im.convert("L").point(lambda v: 255 if v > 8 else 0).getbbox(); return im.crop(b) if b else im
fit = lambda im: im.resize((round(im.width * H / im.height), H))
rows = []
for t in sys.argv[1:]:
    D = json.load(open(O / t / "balcony/facade_dsl_with_balconies.json")); ph = Image.open(D["meta"]["image"]).convert("RGB"); d = ImageDraw.Draw(ph); lw = max(2, ph.height // 300)
    E2 = [b for b in D["layout"]["balconies"] if "width_norm" not in b]
    for b in E2: d.rectangle(b["box_xyxy"], outline=(255, 0, 255), width=lw)
    S = [u for u in D["instances"] if u.get("side_face")]
    for u in S: d.rectangle(u["box_xyxy"], outline=(255, 230, 0), width=lw * 2)
    ir = next((x["structure_ir"] for x in D["balcony_types"] if x["name"] == "balc_enclosed"), {}); skin = "curtain" if (ir.get("walls") or {}).get("curtain_wall") else "punched"
    cells = [(f"{t} {skin}: {len(E2)} cells, {len(S)} side", fit(ph)), ("front", fit(load(O / "front" / f"{t}_balc.png"))), ("oblique", fit(load(O / "oblique" / f"{t}.png")))]
    R = Image.new("RGB", (sum(c.width for _, c in cells) + 40, H + 30), "white"); dr = ImageDraw.Draw(R); x = 0
    for s, c in cells: dr.text((x + 4, 3), s, fill="black", font=Ft); R.paste(c, (x, 30)); x += c.width + 20
    rows.append(R)
for i in range(0, len(rows), 4):
    pg = rows[i:i+4]; P = Image.new("RGB", (max(q.width for q in pg), sum(q.height + 10 for q in pg)), "white"); y = 0
    for q in pg: P.paste(q, (0, y)); y += q.height + 10
    P.save(C / f"bays_{i // 4 + 1}.jpg", quality=88); print(C / f"bays_{i // 4 + 1}.jpg", P.size)
