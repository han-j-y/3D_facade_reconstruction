"""Balcony Type Classification sheet (confusion + accuracy + P/R/F1) as Excel formulas from the v4 report."""
import re, sys
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter as L

md, out = Path(sys.argv[1]), Path(sys.argv[2])
text = md.read_text(encoding="utf-8")
sec = text.split("=====")[1]  # first checkpoint (v4)
ckpt = sec.splitlines()[0].strip()
block = sec.split("```")[1]
HEADS = [("kind", "Kind", ["open_work", "surface_panel", "solid"]),
         ("material", "Material", ["metal", "masonry"]),
         ("floor", "Floor", ["rectangle", "circle"]),
         ("structure", "Structure", ["projecting", "juliet"])]
conf = {}
lines = block.splitlines()
for key, _, names in HEADS:
    i = next(j for j, l in enumerate(lines) if l.startswith(key if key != "material" else "material("))
    conf[key] = [[int(v) for v in re.findall(r"\d+", lines[i + 1 + r].split(None, 1)[1])] for r in range(len(names))]

DARK, WHITE, GRID = "1E1E1E", "FFFFFF", "555555"
mono = Font(name="Consolas", color=WHITE, size=11)
mono_b = Font(name="Consolas", color=WHITE, size=11, bold=True)
dark = PatternFill("solid", fgColor=DARK)
side = Side(style="thin", color=GRID)
box = Border(left=side, right=side, top=side, bottom=side)
wb = Workbook(); ws = wb.active; ws.title = "Balcony Type Classification"
ws["A1"] = "Balcony Type Classification"; ws["A1"].font = Font(size=18)
ws["A2"] = f"Model: {ckpt}   test100, 606 GT-matched balcony crops"; ws["A2"].font = Font(italic=True, color="666666")

def style(c, bold=False, align="left"):
    c.fill, c.font = dark, (mono_b if bold else mono)
    c.alignment = Alignment(horizontal=align, vertical="center")

# confusion block, columns B..E
r = 4
cells = {}
for key, title, names in HEADS:
    n = len(names)
    hdr = ws.cell(r, 2, "kind  rows=true  cols=pred" if key == "kind" else f"{'material(open_work)' if key == 'material' else key}  rows=true  cols=pred")
    style(hdr, True)
    for j, nm in enumerate(names):
        style(ws.cell(r, 3 + j, nm), True, "right")
    for j in range(len(names), 3):
        style(ws.cell(r, 3 + j, None))
    cells[key] = (r + 1, 3, n)
    for i, nm in enumerate(names):
        style(ws.cell(r + 1 + i, 2, "  " + nm))
        for j in range(n):
            style(ws.cell(r + 1 + i, 3 + j, conf[key][i][j]), align="right")
        for j in range(n, 3):
            style(ws.cell(r + 1 + i, 3 + j, None))
    r += n + 1
for col, w in zip("ABCDEFGHI", (2, 40, 18, 18, 18, 10, 14, 14, 10)):
    ws.column_dimensions[col].width = w

def rng(key):
    r0, c0, n = cells[key]
    return r0, c0, n

# accuracy, columns G..H
ar = 4
style(ws.cell(ar, 7, "Accuracy"), True)
style(ws.cell(ar, 8, None)); style(ws.cell(ar, 9, None))
for k, (key, title, names) in enumerate(HEADS):
    r0, c0, n = rng(key)
    diag = "+".join(f"{L(c0 + i)}{r0 + i}" for i in range(n))
    total = f"SUM({L(c0)}{r0}:{L(c0 + n - 1)}{r0 + n - 1})"
    row = ar + 1 + k
    style(ws.cell(row, 7, f"Acc_{'struct' if key == 'structure' else key}"))
    c = ws.cell(row, 8, f'=({diag})&" / "&{total}'); style(c, align="right")
    c = ws.cell(row, 9, f"=({diag})/{total}"); style(c, True, "right"); c.number_format = "0.000"

# P/R/F1 table below
r = max(r, ar + 6) + 2
heads = ["Head", "class", "precision", "recall", "F1"]
for j, h in enumerate(heads):
    c = ws.cell(r, 2 + j, h); style(c, True, "center"); c.border = box
r += 1
for key, title, names in HEADS:
    r0, c0, n = rng(key)
    first = r
    for i, nm in enumerate(names):
        tp = f"{L(c0 + i)}{r0 + i}"
        col = f"SUM({L(c0 + i)}{r0}:{L(c0 + i)}{r0 + n - 1})"
        row = f"SUM({L(c0)}{r0 + i}:{L(c0 + n - 1)}{r0 + i})"
        p, rc = f"IF({col}=0,0,{tp}/{col})", f"IF({row}=0,0,{tp}/{row})"
        vals = [title if i == 0 else None, nm,
                f'={tp}&"/"&{col}&" = "&TEXT({p},"0.000")',
                f'={tp}&"/"&{row}&" = "&TEXT({rc},"0.000")',
                f"=IF(({p})+({rc})=0,0,2*({p})*({rc})/(({p})+({rc})))"]
        for j, v in enumerate(vals):
            c = ws.cell(r, 2 + j, v); style(c); c.border = box
        ws.cell(r, 6).number_format = "0.000"
        r += 1
    # macro row: averages recomputed from the matrix (text cells cannot be averaged)
    ps = ",".join(f"IF(SUM({L(c0 + i)}{r0}:{L(c0 + i)}{r0 + n - 1})=0,0,{L(c0 + i)}{r0 + i}/SUM({L(c0 + i)}{r0}:{L(c0 + i)}{r0 + n - 1}))" for i in range(n))
    rs = ",".join(f"IF(SUM({L(c0)}{r0 + i}:{L(c0 + n - 1)}{r0 + i})=0,0,{L(c0 + i)}{r0 + i}/SUM({L(c0)}{r0 + i}:{L(c0 + n - 1)}{r0 + i}))" for i in range(n))
    vals = [None, "macro", f"=AVERAGE({ps})", f"=AVERAGE({rs})", f"=AVERAGE(F{first}:F{r - 1})"]
    for j, v in enumerate(vals):
        c = ws.cell(r, 2 + j, v); style(c, True); c.border = box
        if j >= 2:
            c.number_format = "0.000"; c.alignment = Alignment(horizontal="left")
    r += 1
ws.sheet_view.showGridLines = False
wb.save(out)
print("saved", out)
