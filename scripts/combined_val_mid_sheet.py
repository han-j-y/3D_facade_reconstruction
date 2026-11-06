#!/usr/bin/env python3
"""The 4 validation facades with enclosed units: photo | front, before and with the block between front and side windows | oblique, before and after.
Writes runs_val30_side/compare/mid.jpg."""
import json
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageFont
import numpy as np
E = Path(__file__).resolve().parents[1]; EV = E / "runs_val30_combined_eval"; F = E / "runs_val30_mid"; B = E / "runs_val30_side"; O = F / "compare"; O.mkdir(exist_ok=True); H = 560
Ft = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
iou = {k: json.load(open(EV / k / "iou/sam_iou.json"))["per_facade"] for k in ("side", "mid")}
def fit(im, crop=False):
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, "white"); bg.paste(im, mask=im.split()[3]); box = im.split()[3].getbbox(); im = bg.crop(box)
    elif crop:
        b = im.convert("L").point(lambda v: 255 if v > 8 else 0).getbbox(); im = im.crop(b) if b else im
    return im.resize((round(im.width * H / im.height), H))
rows = []
for t in ("v14", "v26"):
    ph = Image.open(json.load(open(F / t / "facade_dsl.json"))["meta"]["image"]).convert("RGB")
    a = Image.open(B / "front" / f"{t}_balc.png").convert("RGB"); b = Image.open(F / "front" / f"{t}_balc.png").convert("RGB")
    print(t, "front changed %", round((np.asarray(ImageChops.difference(a, b)).sum(2) > 30).mean() * 100, 2), {k: round(v[t]["iou"], 3) for k, v in iou.items()})
    cells = [(f"{t} photo", fit(ph)), (f"front on main {iou['side'][t]['iou']:.3f}", fit(a, True)), (f"front between {iou['mid'][t]['iou']:.3f}", fit(b, True)),
             ("oblique before", fit(Image.open(F / "oblique" / f"{t}_before.png"))), ("oblique side", fit(Image.open(F / "oblique" / f"{t}_side.png")))]
    R = Image.new("RGB", (sum(c.width for _, c in cells) + 80, H + 32), "white"); D = ImageDraw.Draw(R); x = 0
    for s, c in cells: D.text((x + 4, 4), s, fill="black", font=Ft); R.paste(c, (x, 32)); x += c.width + 20
    rows.append(R)
P = Image.new("RGB", (max(r.width for r in rows), sum(r.height + 10 for r in rows)), "white"); y = 0
for r in rows: P.paste(r, (0, y)); y += r.height + 10
P.save(O / "mid.jpg", quality=88); print(O / "mid.jpg", P.size)
a = json.load(open(EV / "seated/iou/sam_iou.json"))["per_facade"]; pool = lambda d: sum(v["inter"] for v in d.values()) / sum(v["union"] for v in d.values())
import statistics as st
for name, f in (("side", iou["side"]), ("mid", iou["mid"])):
    m = {**a, **f}; print(name, len(m), "mean", round(st.mean(v["iou"] for v in m.values()), 4), "pooled", round(pool(m), 4))
