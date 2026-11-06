#!/usr/bin/env python3
"""Validation facades: photo | before (sam3 enclosed) | merged (new enclosed model), balconies included where present.
Writes runs_val30_combined/compare/page_<n>.jpg for the facades whose enclosed result changed and all_<n>.jpg for all 30."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
E = Path(__file__).resolve().parents[1]; N = E / "runs_val30_combined"; M = E.parent / "facade_e2e/runs_val30"; O = N / "compare"; O.mkdir(exist_ok=True)
IDS = ["v07", "v08", "v12", "v14", "v16", "v17", "v19", "v24", "v26"]; H = 560
ALL = [f"v{i:02d}" for i in range(30)]
Ft = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
def fit(im, crop=False):
    if crop:
        b = im.convert("L").point(lambda v: 255 if v > 8 else 0).getbbox(); im = im.crop(b) if b else im
    return im.resize((round(im.width * H / im.height), H))
def front(run, t):
    b = run / "front" / f"{t}_balc.png"; return Image.open(b if b.is_file() else run / "front" / f"{t}_ours.png").convert("RGB")
def sheets(ids, name, per):
    rows = []
    for t in ids:
        ph = Image.open(json.load(open(N / t / "facade_dsl.json"))["meta"]["image"]).convert("RGB")
        cells = [(f"{t} photo", fit(ph)), ("before", fit(front(M, t), True)), ("merged", fit(front(N, t), True))]
        R = Image.new("RGB", (sum(a.width for _, a in cells) + 40, H + 32), "white"); D = ImageDraw.Draw(R); x = 0
        for c, a in cells: D.text((x + 4, 4), c, fill="black", font=Ft); R.paste(a, (x, 32)); x += a.width + 20
        rows.append(R)
    for k in range(0, len(rows), per):
        pg = rows[k:k+per]; P = Image.new("RGB", (max(r.width for r in pg), sum(r.height + 10 for r in pg)), "white"); y = 0
        for r in pg: P.paste(r, (0, y)); y += r.height + 10
        P.save(O / f"{name}_{k // per + 1}.jpg", quality=88); print(O / f"{name}_{k // per + 1}.jpg", P.size)
sheets(IDS, "page", 3)
sheets(ALL, "all", 5)
