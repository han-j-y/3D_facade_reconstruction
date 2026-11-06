#!/usr/bin/env python3
"""Validation facades with blocks, pushed code: photo (blocks magenta, side windows yellow) | front off | front on | angle off | angle on, with SAM IoU.
Writes runs_val_side_now/compare/side.jpg and prints per-facade and 20-facade IoU."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
E = Path(__file__).resolve().parents[1]; F = E / "runs_val_side_now"; OFF = E / "runs_outer_bay2"; C = F / "compare"; C.mkdir(exist_ok=True); H = 480
Ft = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
A = json.load(open(OFF / "eval/iou/sam_iou.json"))["per_facade"]; B = json.load(open(F / "eval/iou/sam_iou.json"))["per_facade"]
M = {**A, **B}
print("off", round(sum(v["iou"] for v in A.values()) / len(A), 3), round(sum(v["inter"] for v in A.values()) / sum(v["union"] for v in A.values()), 3))
print("on", round(sum(v["iou"] for v in M.values()) / len(M), 3), round(sum(v["inter"] for v in M.values()) / sum(v["union"] for v in M.values()), 3), len(M))
def load(p):
    im = Image.open(p)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, "white"); bg.paste(im, mask=im.split()[3]); return bg.crop(im.split()[3].getbbox())
    return im.convert("RGB")
fit = lambda im: im.resize((round(im.width * H / im.height), H))
rows = []
for t in "v14 v26 v16 v24".split():
    D = json.load(open(F / t / "balcony/facade_dsl_with_balconies.json")); ph = Image.open(D["meta"]["image"]).convert("RGB"); d = ImageDraw.Draw(ph); lw = max(2, ph.height // 300)
    for b in D["layout"]["balconies"]:
        if "width_norm" not in b: d.rectangle(b["box_xyxy"], outline=(255, 0, 255), width=lw)
    S = [u for u in D["instances"] if u.get("side_face")]
    for u in S: d.rectangle(u["box_xyxy"], outline=(255, 230, 0), width=lw * 2)
    print(t, "side", len(S), "off", round(A[t]["iou"], 3), "on", round(B[t]["iou"], 3))
    cells = [(f"{t} photo, {len(S)} side", fit(ph)), (f"front OFF {A[t]['iou']:.3f}", fit(load(OFF / "front" / f"{t}_balc.png"))), (f"front ON {B[t]['iou']:.3f}", fit(load(F / "front" / f"{t}_balc.png"))), ("angle OFF", fit(load(F / "oblique_off" / f"{t}.png"))), ("angle ON", fit(load(F / "oblique" / f"{t}.png")))]
    R = Image.new("RGB", (sum(c.width for _, c in cells) + 60, H + 28), "white"); dr = ImageDraw.Draw(R); x = 0
    for s, c in cells: dr.text((x + 4, 3), s, fill="black", font=Ft); R.paste(c, (x, 28)); x += c.width + 15
    rows.append(R)
P = Image.new("RGB", (max(q.width for q in rows), sum(q.height + 8 for q in rows)), "white"); y = 0
for q in rows: P.paste(q, (0, y)); y += q.height + 8
P.save(C / "side.jpg", quality=88); print(C / "side.jpg", P.size)
