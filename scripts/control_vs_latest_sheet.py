#!/usr/bin/env python3
"""Validation facades with balconies, all in one image: photo | control | latest, with each facade's IoU above its render.
Sorted by IoU change, biggest gain first; v12 goes last (no balconies in latest).
Writes runs_val30_seated/compare/control_vs_latest.jpg."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
E = Path(__file__).resolve().parents[1]; M = E.parent / "facade_e2e"; EV = E / "runs_val30_combined_eval"; H = 460
CTRL, LATE = M / "runs_val30/front", E / "runs_val30_seated/front"
Ft = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
a = json.load(open(EV / "balc_old/iou/sam_iou.json"))["per_facade"]; b = json.load(open(EV / "seated/iou/sam_iou.json"))["per_facade"]
def fit(im, crop=False):
    if crop:
        box = im.convert("L").point(lambda v: 255 if v > 8 else 0).getbbox(); im = im.crop(box) if box else im
    return im.resize((round(im.width * H / im.height), H))
rows = []
for t in sorted(a, key=lambda t: -(b[t]["iou"] - a[t]["iou"]) if t in b else 9):
    ph = Image.open(json.load(open(E / "runs_val30_combined" / t / "facade_dsl.json"))["meta"]["image"]).convert("RGB")
    lp = LATE / f"{t}_balc.png"; lt = LATE / f"{t}_ours.png"
    late = (f"latest {b[t]['iou']:.3f} ({b[t]['iou'] - a[t]['iou']:+.3f})" if t in b else "latest: no balconies", lp if lp.exists() else lt)
    cells = [(f"{t} photo", fit(ph)), (f"control {a[t]['iou']:.3f}", fit(Image.open(CTRL / f"{t}_balc.png").convert("RGB"), True)), (late[0], fit(Image.open(late[1]).convert("RGB"), True))]
    R = Image.new("RGB", (sum(c.width for _, c in cells) + 40, H + 30), "white"); D = ImageDraw.Draw(R); x = 0
    for s, c in cells: D.text((x + 4, 3), s, fill="black", font=Ft); R.paste(c, (x, 30)); x += c.width + 20
    rows.append(R)
P = Image.new("RGB", (max(r.width for r in rows), sum(r.height + 10 for r in rows)), "white"); y = 0
for r in rows: P.paste(r, (0, y)); y += r.height + 10
O = E / "runs_val30_seated/compare/control_vs_latest.jpg"; P.save(O, quality=88); print(O, P.size, len(rows))
print([ (r) for r in sorted(a, key=lambda t: -(b[t]["iou"] - a[t]["iou"]) if t in b else 9)])
