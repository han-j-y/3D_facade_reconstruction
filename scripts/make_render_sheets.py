"""Photo | Blender oblique render sheets for a batch from scripts/test100_current.sh.
   python scripts/make_render_sheets.py <out_dir> <photo_dir> [name]"""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
out = Path(sys.argv[1]); ph = Path(sys.argv[2]); name = sys.argv[3] if len(sys.argv) > 3 else out.name
dst = out / "sheets"; dst.mkdir(exist_ok=True)
H, W, PAD, TH = 640, 900, 16, 44
try: font = ImageFont.truetype("arial.ttf", 22)
except OSError: font = ImageFont.load_default()
def crop_render(im):
    from PIL import ImageOps
    im = im.convert("RGB"); g = im.convert("L").point(lambda v: 255 if v > 12 else 0); b = g.getbbox()
    if b:
        m = 12; im = im.crop((max(0, b[0] - m), max(0, b[1] - m), min(im.width, b[2] + m), min(im.height, b[3] + m)))
    return im
def fit(im):
    im = im.convert("RGB"); s = min(H / im.height, W / im.width)
    return im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
stems = [p.stem for p in sorted((out / "oblique").glob("*.png"))]
PER, COLS = 10, 2
cell_w = 2 * W + PAD; cell_h = TH + H
for pg in range(0, len(stems), PER):
    chunk = stems[pg:pg + PER]; rows = (len(chunk) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * cell_w + (COLS + 1) * PAD * 2, rows * (cell_h + PAD * 2) + PAD), "white")
    d = ImageDraw.Draw(sheet)
    for i, s in enumerate(chunk):
        x = PAD * 2 + (i % COLS) * (cell_w + PAD * 2); y = PAD + (i // COLS) * (cell_h + PAD * 2)
        f = out / s / "facade_dsl_with_balconies.json"
        info = "no balconies"
        if f.exists():
            j = json.loads(f.read_text(encoding="utf-8"))
            info = f'{len(j.get("layout", {}).get("balconies", []))} balconies, {len(j.get("balcony_types", []))} types'
        d.text((x, y + 8), f"{pg + i + 1}. {s}   [{info}]", fill="black", font=font)
        for k, src in enumerate((ph / f"{s}.jpg", out / "oblique" / f"{s}.png")):
            im = Image.open(src); im = fit(crop_render(im) if k else im); bx = x + k * (W + PAD)
            d.rectangle([bx, y + TH, bx + W - 1, y + TH + H - 1], fill=(242, 242, 242) if k == 0 else (30, 30, 30))
            sheet.paste(im, (bx + (W - im.width) // 2, y + TH + (H - im.height) // 2))
    p = dst / f"{name}_{pg // PER + 1:02d}.png"; sheet.save(p, optimize=True); print(p.name, sheet.size)
