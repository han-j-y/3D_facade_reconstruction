#!/usr/bin/env python3
"""The 4 validation facades with enclosed units: photo | control | before the block fit | blocks fitted, with IoU.
Writes runs_val30_fit/compare/fit.jpg."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageChops
import numpy as np
E = Path(__file__).resolve().parents[1]; M = E.parent / "facade_e2e"; EV = E / "runs_val30_combined_eval"; F = E / "runs_val30_fit"; O = F / "compare"; O.mkdir(exist_ok=True); H = 600
Ft = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
iou = {k: json.load(open(EV / k / "iou/sam_iou.json"))["per_facade"] for k in ("balc_old", "seated", "fit")}
def fit(im, crop=False):
    if crop:
        b = im.convert("L").point(lambda v: 255 if v > 8 else 0).getbbox(); im = im.crop(b) if b else im
    return im.resize((round(im.width * H / im.height), H))
rows = []
for t in ("v14", "v16", "v24", "v26"):
    ph = Image.open(json.load(open(F / t / "facade_dsl.json"))["meta"]["image"]).convert("RGB")
    a = Image.open(E / "runs_val30_seated/front" / f"{t}_balc.png").convert("RGB"); b = Image.open(F / "front" / f"{t}_balc.png").convert("RGB")
    print(t, "changed %", round((np.asarray(ImageChops.difference(a, b)).sum(2) > 30).mean() * 100, 2) if a.size == b.size else "size", {k: round(v[t]["iou"], 3) for k, v in iou.items()})
    cells = [(f"{t} photo", fit(ph)), (f"control {iou['balc_old'][t]['iou']:.3f}", fit(Image.open(M / "runs_val30/front" / f"{t}_balc.png").convert("RGB"), True)), (f"before {iou['seated'][t]['iou']:.3f}", fit(a, True)), (f"fitted {iou['fit'][t]['iou']:.3f}", fit(b, True))]
    R = Image.new("RGB", (sum(c.width for _, c in cells) + 60, H + 32), "white"); D = ImageDraw.Draw(R); x = 0
    for s, c in cells: D.text((x + 4, 4), s, fill="black", font=Ft); R.paste(c, (x, 32)); x += c.width + 20
    rows.append(R)
P = Image.new("RGB", (max(r.width for r in rows), sum(r.height + 10 for r in rows)), "white"); y = 0
for r in rows: P.paste(r, (0, y)); y += r.height + 10
P.save(O / "fit.jpg", quality=88); print(O / "fit.jpg", P.size)
