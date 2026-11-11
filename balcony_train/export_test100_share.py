r"""Share package for the test100 balcony classification: one Excel row per balcony + photos.

Input is a scored crop dir (``build_test100_final_set.py`` output, default the GT-matched
606 crops). Truth comes from the same loader as ``evaluate_test100.py`` (triangle counts as
rectangle; material only for open_work; juliet has no floor), predictions from one checkpoint.

Output ``<out>/``:

* ``balcony_classification.xlsx``: sheet ``Balconies`` with Facade file name, Balcony file
  name, then truth / prediction for kind, material, floor and structure (wrong predictions in
  red; ``-`` where the crop has no truth for that head). Sheet ``Summary``: accuracy and
  per-class precision / recall / F1 per head.
* ``photos/combined/<balcony>.jpg``: balcony crop beside its facade with the box drawn in red.
* ``photos/facade/<facade>.jpg``: raw facade photos (``data/test100/photos``).
* ``photos/balcony/<balcony>.png``: raw balcony crops as fed to the classifier.

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_train/export_test100_share.py checkpoints/railing_test100_v4.pt --out runs/test100/share_jiyeon
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.dataset import MultitaskCropDataset  # noqa: E402
from balcony_train.evaluate_test100 import HEADS, TEST_FLOORS  # noqa: E402
from balcony_train.labels import FLOOR_SHAPES, iter_multitask_samples  # noqa: E402
from balcony_train.model import load_classifier  # noqa: E402

T100 = ROOT / "runs" / "test100"
OUT_INDEX = {"kind": 0, "material": 1, "floor": 2, "structure": 5}
TITLES = {"kind": "Kind", "material": "Material", "floor": "Floor", "structure": "Structure"}
PANEL_H = 900


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.is_file() else []


def combined(crop: Image.Image, photo: Image.Image, box: list[int]) -> Image.Image:
    ph = photo.copy()
    d = ImageDraw.Draw(ph)
    lw = max(3, ph.width // 250)
    d.rectangle(box, outline=(255, 255, 255), width=lw + 2)
    d.rectangle(box, outline=(230, 30, 30), width=lw)
    ph = ph.resize((round(ph.width * PANEL_H / ph.height), PANEL_H), Image.LANCZOS)
    s = min(PANEL_H * 0.6 / crop.height, 600 / crop.width)
    c = crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))), Image.LANCZOS)
    out = Image.new("RGB", (c.width + ph.width + 60, PANEL_H + 40), "white")
    out.paste(c, (20, 20 + (PANEL_H - c.height) // 2))
    out.paste(ph, (c.width + 40, 20))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--dir", type=Path, default=T100 / "crops_final_gt")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    man = {Path(m["crop"]).stem: m for m in jl(T100 / "crops/manifest.jsonl") + jl(T100 / "crops_split/manifest.jsonl")}
    samples = iter_multitask_samples(args.dir, args.dir / "labels.jsonl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_classifier(args.ckpt, device)
    tri, rect = FLOOR_SHAPES.index("triangle"), FLOOR_SHAPES.index("rectangle")

    rows, k = [], 0
    with torch.no_grad():
        for images, ky, my, fy, _ey, _sy, sty in DataLoader(MultitaskCropDataset(samples), batch_size=32, shuffle=False):
            outs = model(images.to(device))
            for i in range(len(ky)):
                path = Path(samples[k + i][0])
                row = {"crop": path}
                for head, y in (("kind", ky), ("material", my), ("floor", fy), ("structure", sty)):
                    t, p = int(y[i]), int(outs[OUT_INDEX[head]][i].argmax())
                    if head == "floor":
                        p = TEST_FLOORS.index(FLOOR_SHAPES[rect if p == tri else p])
                        t = TEST_FLOORS.index(FLOOR_SHAPES[rect if t == tri else t]) if t >= 0 else t
                    names = HEADS[head]
                    row[head] = (names[t] if t >= 0 else None, names[p])
                rows.append(row)
            k += len(ky)

    if args.out.exists():
        shutil.rmtree(args.out)
    dirs = {n: args.out / "photos" / n for n in ("combined", "facade", "balcony")}
    for d in dirs.values():
        d.mkdir(parents=True)
    for r in rows:
        m = man[r["crop"].stem]
        r["facade"] = m["photo"]
        r["box"] = m["box_xyxy"]
    rows.sort(key=lambda r: (r["facade"], r["crop"].name))

    photos: dict[str, Image.Image] = {}
    for r in rows:
        f = r["facade"]
        if f not in photos:
            src = ROOT / "data" / "test100" / "photos" / f
            shutil.copy2(src, dirs["facade"] / f)
            photos = {f: Image.open(src).convert("RGB")}
        crop = Image.open(r["crop"]).convert("RGB")
        shutil.copy2(r["crop"], dirs["balcony"] / r["crop"].name)
        combined(crop, photos[f], r["box"]).save(dirs["combined"] / f"{r['crop'].stem}.jpg", quality=90)

    wb = Workbook()
    ws = wb.active
    ws.title = "Balconies"
    thin = Side(style="thin", color="999999")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    bold = Font(bold=True)
    center = Alignment(horizontal="center", vertical="center")
    head_fill = PatternFill("solid", fgColor="DDE5F0")
    red = PatternFill("solid", fgColor="F8CBAD")
    ws.cell(1, 1, "Facade file name")
    ws.cell(1, 2, "Balcony file name")
    ws.merge_cells(start_row=1, start_column=1, end_row=2, end_column=1)
    ws.merge_cells(start_row=1, start_column=2, end_row=2, end_column=2)
    for j, head in enumerate(HEADS):
        c = 3 + 2 * j
        ws.cell(1, c, TITLES[head])
        ws.merge_cells(start_row=1, start_column=c, end_row=1, end_column=c + 1)
        ws.cell(2, c, "Truth")
        ws.cell(2, c + 1, "Prediction")
    for rr in (1, 2):
        for cc in range(1, 3 + 2 * len(HEADS)):
            cell = ws.cell(rr, cc)
            cell.font, cell.alignment, cell.fill, cell.border = bold, center, head_fill, border
    for i, r in enumerate(rows, start=3):
        ws.cell(i, 1, r["facade"]).border = border
        ws.cell(i, 2, f"{r['crop'].stem}.png").border = border
        for j, head in enumerate(HEADS):
            t, p = r[head]
            c = 3 + 2 * j
            ct = ws.cell(i, c, t if t is not None else "-")
            cp = ws.cell(i, c + 1, p if t is not None else "-")
            for cell in (ct, cp):
                cell.border, cell.alignment = border, center
            if t is not None and t != p:
                cp.fill = red
    ws.freeze_panes = "C3"
    ws.column_dimensions["A"].width = 62
    ws.column_dimensions["B"].width = 72
    for j in range(len(HEADS) * 2):
        ws.column_dimensions[ws.cell(2, 3 + j).column_letter].width = 14
    ws.auto_filter.ref = f"A2:{ws.cell(2, 2 + 2 * len(HEADS)).column_letter}{len(rows) + 2}"

    sm = wb.create_sheet("Summary")
    sha = hashlib.sha256(args.ckpt.read_bytes()).hexdigest()[:12]
    sm.append(["Model", args.ckpt.name, f"sha256 {sha}"])
    sm.append(["Balconies", len(rows), f"facades {len({r['facade'] for r in rows})}"])
    sm.append(["Rules", "triangle counts as rectangle; material only for open_work; juliet has no floor; '-' = no truth"])
    sm.append([])
    for head, names in HEADS.items():
        pairs = [r[head] for r in rows if r[head][0] is not None]
        n = len(pairs)
        acc = sum(t == p for t, p in pairs) / max(1, n)
        sm.append([TITLES[head], f"n = {n}", f"accuracy = {sum(t == p for t, p in pairs)}/{n} = {acc:.3f}"])
        sm.append(["class", "precision", "recall", "F1"])
        for cell in sm[sm.max_row]:
            cell.font = bold
        ps, rs, fs = [], [], []
        for nm in names:
            tp = sum(t == nm and p == nm for t, p in pairs)
            col = sum(p == nm for _, p in pairs)
            row = sum(t == nm for t, _ in pairs)
            pr, rc = (tp / col if col else 0.0), (tp / row if row else 0.0)
            f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
            ps.append(pr); rs.append(rc); fs.append(f1)
            sm.append([nm, f"{tp}/{col} = {pr:.3f}", f"{tp}/{row} = {rc:.3f}", round(f1, 3)])
        sm.append(["macro", round(sum(ps) / len(ps), 3), round(sum(rs) / len(rs), 3), round(sum(fs) / len(fs), 3)])
        for cell in sm[sm.max_row]:
            cell.font = bold
        sm.append([])
    for col, w in zip("ABCD", (16, 24, 24, 10)):
        sm.column_dimensions[col].width = w
    wb.save(args.out / "balcony_classification.xlsx")
    print(f"{len(rows)} balconies, {len(photos) and len({r['facade'] for r in rows})} facades -> {args.out}")


if __name__ == "__main__":
    main()
