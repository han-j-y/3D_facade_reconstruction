#!/usr/bin/env python3
"""Validation facades with balconies: photo | scale fix only | also seated on window bottoms. Works on whatever has rendered so far.
Writes runs_val30_seated/compare/fix_<n>.jpg, five facades per page."""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont
E = Path(__file__).resolve().parents[1]; A = E / "runs_val30_combined"; S = E / "runs_val30_scalefix"; F = E / "runs_val30_seated"; O = F / "compare"; O.mkdir(exist_ok=True); H = 560
Ft = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
def fit(im, crop=False):
    if crop:
        b = im.convert("L").point(lambda v: 255 if v > 8 else 0).getbbox(); im = im.crop(b) if b else im
    return im.resize((round(im.width * H / im.height), H))
ids = sorted(p.name[:3] for p in (F / "front").glob("*_balc.png")); rows = []
for t in ids:
    ph = Image.open(json.load(open(A / t / "facade_dsl.json"))["meta"]["image"]).convert("RGB")
    cells = [(f"{t} photo", fit(ph)), ("scale fix only", fit(Image.open(S / "front" / f"{t}_balc.png").convert("RGB"), True)), ("dropped to window bottom", fit(Image.open(F / "front" / f"{t}_balc.png").convert("RGB"), True))]
    R = Image.new("RGB", (sum(a.width for _, a in cells) + 40, H + 32), "white"); D = ImageDraw.Draw(R); x = 0
    for c, a in cells: D.text((x + 4, 4), c, fill="black", font=Ft); R.paste(a, (x, 32)); x += a.width + 20
    rows.append(R)
for k in range(0, len(rows), 5):
    pg = rows[k:k+5]; P = Image.new("RGB", (max(r.width for r in pg), sum(r.height + 10 for r in pg)), "white"); y = 0
    for r in pg: P.paste(r, (0, y)); y += r.height + 10
    P.save(O / f"fix_{k // 5 + 1}.jpg", quality=88); print(O / f"fix_{k // 5 + 1}.jpg", P.size)
print(ids)
