#!/usr/bin/env python3
"""Photo | grid-sized wall | wall from the facade region, with balconies where a facade has them. Args: ids.
Writes runs_region/compare/region_<n>.jpg."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
E = Path(__file__).resolve().parents[1]; F = E / "runs_region"; O = F / "compare"; O.mkdir(exist_ok=True); H = 500
Ft = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 21)
def load(p):
    im = Image.open(p).convert("RGB"); b = im.convert("L").point(lambda v: 255 if v > 8 else 0).getbbox(); return im.crop(b) if b else im
fit = lambda im: im.resize((round(im.width * H / im.height), H))
rows = []
for t in sys.argv[1:]:
    D = json.load(open(F / t / "facade_dsl.json")); kind = "balc" if (F / "front" / f"{t}_balc.png").exists() else "ours"
    before = (E / "runs_window_map/front" if kind == "balc" else E / "runs_val30_combined/front") / f"{t}_{kind}.png"
    ph = Image.open(D["meta"]["image"]).convert("RGB"); d = ImageDraw.Draw(ph); lw = max(2, ph.height // 250)
    d.rectangle(D["meta"]["facade_region_xyxy"], outline=(255, 140, 0), width=lw)
    cells = [(f"{t} photo, region", fit(ph)), ("before: grid-sized wall", fit(load(before))), ("after: region wall", fit(load(F / "front" / f"{t}_{kind}.png")))]
    R = Image.new("RGB", (sum(c.width for _, c in cells) + 40, H + 30), "white"); dr = ImageDraw.Draw(R); x = 0
    for s, c in cells: dr.text((x + 4, 3), s, fill="black", font=Ft); R.paste(c, (x, 30)); x += c.width + 20
    rows.append(R)
for i in range(0, len(rows), 4):
    pg = rows[i:i+4]; P = Image.new("RGB", (max(q.width for q in pg), sum(q.height + 10 for q in pg)), "white"); y = 0
    for q in pg: P.paste(q, (0, y)); y += q.height + 10
    P.save(O / f"region_{i // 4 + 1}.jpg", quality=88); print(O / f"region_{i // 4 + 1}.jpg", P.size)
