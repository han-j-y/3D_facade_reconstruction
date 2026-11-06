#!/usr/bin/env python3
"""Balconies on their windows, stacks moving together: photo | before (bay centres) | after. Args: facade ids.
Writes runs_window_map/compare/map_<n>.jpg."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
E = Path(__file__).resolve().parents[1]; F = E / "runs_window_map"; O = F / "compare"; O.mkdir(exist_ok=True); H = 520
for old in O.glob("map_*.jpg"): old.unlink()
Ft = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 21)
iou = {k: json.load(open(E / "runs_val30_combined_eval" / k / "iou/sam_iou.json"))["per_facade"] for k in ("seated", "window_map")}
def load(p):
    im = Image.open(p).convert("RGB"); b = im.convert("L").point(lambda v: 255 if v > 8 else 0).getbbox(); return im.crop(b) if b else im
fit = lambda im: im.resize((round(im.width * H / im.height), H))
rows = []
for t in sys.argv[1:]:
    before = E / ("runs_val30_seated" if t[0] == "v" else "runs_cand") / "front" / f"{t}_balc.png"
    D = json.load(open(F / t / "balcony/facade_dsl_with_balconies.json"))
    a, b = ("", "") if t not in iou["seated"] else (f" {iou['seated'][t]['iou']:.3f}", f" {iou['window_map'][t]['iou']:.3f}")
    cells = [(f"{t} photo", fit(Image.open(D["meta"]["image"]).convert("RGB"))), ("before" + a, fit(load(before))), ("after" + b, fit(load(F / "front" / f"{t}_balc.png")))]
    R = Image.new("RGB", (sum(c.width for _, c in cells) + 40, H + 30), "white"); d = ImageDraw.Draw(R); x = 0
    for s, c in cells: d.text((x + 4, 3), s, fill="black", font=Ft); R.paste(c, (x, 30)); x += c.width + 20
    rows.append(R)
for i in range(0, len(rows), 4):
    pg = rows[i:i+4]; P = Image.new("RGB", (max(q.width for q in pg), sum(q.height + 10 for q in pg)), "white"); y = 0
    for q in pg: P.paste(q, (0, y)); y += q.height + 10
    P.save(O / f"map_{i // 4 + 1}.jpg", quality=88); print(O / f"map_{i // 4 + 1}.jpg", P.size)
