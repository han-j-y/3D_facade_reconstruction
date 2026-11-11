"""Balcony grouping of a batch run: each photo with its balcony boxes coloured by type, plus a legend of the voted labels.
   python scripts/draw_balcony_types.py <out_dir> <photo_dir> [name]
Reads <out_dir>/<stem>/facade_dsl_with_balconies.json (layout.balconies: box_xyxy + type). Writes <out_dir>/types/."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

out = Path(sys.argv[1]); ph = Path(sys.argv[2]); name = sys.argv[3] if len(sys.argv) > 3 else out.name
dst = out / "types"; dst.mkdir(exist_ok=True)
COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
          (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40), (128, 0, 0), (0, 0, 128)]
def font(n):
    try: return ImageFont.truetype("arialbd.ttf", n)
    except OSError: return ImageFont.load_default()
F, FS = font(30), font(26)
SHORT = {"open_work": "OW", "solid_panel": "SP", "projecting": "P", "juliet": "J", "rectangle": "rect",
         "circle": "circ", "triangle": "tri"}
def label(t):
    ir = t.get("structure_ir") or {}
    if ir.get("enclosure") == "enclosed":
        return "enclosed / oriel"
    r = ir.get("railing") or {}
    parts = [SHORT.get(ir.get("structure"), ir.get("structure")), SHORT.get(r.get("kind"), r.get("kind"))]
    if r.get("kind") == "open_work": parts.append(r.get("material"))
    if ir.get("structure") == "projecting": parts.append(SHORT.get((ir.get("floor") or {}).get("shape"), "?"))
    return " / ".join(str(p) for p in parts if p)

panels = []
for d in sorted(p for p in out.iterdir() if (p / "facade_dsl_with_balconies.json").exists()):
    j = json.loads((d / "facade_dsl_with_balconies.json").read_text(encoding="utf-8"))
    types = {t["name"]: t for t in j.get("balcony_types", [])}
    order = list(dict.fromkeys(b["type"] for b in j["layout"].get("balconies", [])))
    col = {n: COLORS[i % len(COLORS)] for i, n in enumerate(order)}
    im = Image.open(ph / f"{d.name}.jpg").convert("RGB"); s = 900 / im.height
    im = im.resize((round(im.width * s), 900), Image.LANCZOS); dr = ImageDraw.Draw(im)
    cnt = {}
    for b in j["layout"]["balconies"]:
        x0, y0, x1, y1 = [v * s for v in b["box_xyxy"]]; c = col[b["type"]]; cnt[b["type"]] = cnt.get(b["type"], 0) + 1
        dr.rectangle([x0, y0, x1, y1], outline=c, width=5)
        tag = f"T{order.index(b['type'])}"; tb = dr.textbbox((x0 + 3, y0 + 2), tag, font=FS)
        dr.rectangle(tb, fill=c); dr.text((x0 + 3, y0 + 2), tag, fill="white", font=FS)
    leg = Image.new("RGB", (im.width, 60 + 40 * len(order)), "white"); dl = ImageDraw.Draw(leg)
    dl.text((8, 10), d.name[:60], fill="black", font=FS)
    for i, n in enumerate(order):
        y = 52 + 40 * i; dl.rectangle([8, y, 38, y + 30], fill=col[n])
        dl.text((48, y), f"T{i}  x{cnt[n]}   {label(types.get(n, {}))}", fill="black", font=FS)
    p = Image.new("RGB", (im.width, im.height + leg.height), "white"); p.paste(im, (0, 0)); p.paste(leg, (0, im.height))
    p.save(dst / f"{d.name}.png"); panels.append(p)

PER, COLS, PAD = 6, 3, 24
for k in range(0, len(panels), PER):
    ch = panels[k:k + PER]; rows = (len(ch) + COLS - 1) // COLS
    cw = max(p.width for p in ch); rh = [max(p.height for p in ch[r * COLS:(r + 1) * COLS]) for r in range(rows)]
    sh = Image.new("RGB", (COLS * (cw + PAD) + PAD, sum(rh) + PAD * (rows + 1)), (235, 235, 235))
    y = PAD
    for r in range(rows):
        for c, p in enumerate(ch[r * COLS:(r + 1) * COLS]):
            sh.paste(p, (PAD + c * (cw + PAD), y))
        y += rh[r] + PAD
    f = dst / f"{name}_types_{k // PER + 1:02d}.png"; sh.save(f, optimize=True); print(f, sh.size)
print(len(panels), "photos")
