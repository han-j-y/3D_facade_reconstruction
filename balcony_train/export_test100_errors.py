r"""Save the misclassified test100 crops of each head, for viewing.

Runs one railing checkpoint on a labeled crop dir (``evaluate_test100.py`` input,
e.g. the final-box set from ``build_test100_final_set.py``) and writes, per head
(kind, material on open_work, floor with triangle as rectangle, structure):

* ``<out>/<head>/<true>_as_<pred>/<crop>.jpg``: the crop (left) beside the photo with its
  red box (right; from ``runs/test100/crops*/context``), header true / pred / probability;
* ``<out>/<head>/<true>_as_<pred>/sheet_NN.jpg``: 24 crops per page;
* ``<out>/errors.csv``: one row per error.

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_train/export_test100_errors.py checkpoints/railing_test100_juliet.pt --dir <final set> --out runs/test100/errors_final_juliet
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.dataset import MultitaskCropDataset  # noqa: E402
from balcony_train.evaluate_test100 import HEADS, TEST_FLOORS  # noqa: E402
from balcony_train.labels import FLOOR_SHAPES, iter_multitask_samples  # noqa: E402
from balcony_train.model import load_classifier  # noqa: E402

T100 = ROOT / "runs" / "test100"
CONTEXT_DIRS = (T100 / "crops" / "context", T100 / "crops_split" / "context")
OUT_INDEX = {"kind": 0, "material": 1, "floor": 2, "structure": 5}
PANEL_H = 420


def font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def panel(crop_path: Path, header: str) -> Image.Image:
    crop = Image.open(crop_path).convert("RGB")
    s = min(PANEL_H / crop.height, 360 / crop.width)
    crop = crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))), Image.BILINEAR)
    ctx = next((d / f"{crop_path.stem}.jpg" for d in CONTEXT_DIRS if (d / f"{crop_path.stem}.jpg").is_file()), None)
    ctx_im = Image.open(ctx).convert("RGB") if ctx else Image.new("RGB", (PANEL_H, PANEL_H), "grey")
    ctx_im = ctx_im.resize((round(ctx_im.width * PANEL_H / ctx_im.height), PANEL_H), Image.BILINEAR)
    head = 56
    out = Image.new("RGB", (crop.width + ctx_im.width + 30, PANEL_H + head), "white")
    out.paste(crop, (10, head + (PANEL_H - crop.height) // 2))
    out.paste(ctx_im, (crop.width + 20, head))
    d = ImageDraw.Draw(out)
    d.text((10, 4), header, fill="black", font=font(20))
    d.text((10, 30), crop_path.stem[:90], fill=(90, 90, 90), font=font(14))
    return out


def sheet(paths: list[Path], dest: Path, per: int = 24, cols: int = 6, cell: int = 220) -> None:
    for k in range(0, len(paths), per):
        chunk = paths[k:k + per]
        rows = (len(chunk) + cols - 1) // cols
        im = Image.new("RGB", (cols * cell, rows * (cell + 20)), "white")
        d = ImageDraw.Draw(im)
        for i, p in enumerate(chunk):
            c = Image.open(p).convert("RGB")
            c.thumbnail((cell - 8, cell - 8))
            x, y = (i % cols) * cell, (i // cols) * (cell + 20)
            im.paste(c, (x + 4, y + 4))
            d.text((x + 4, y + cell - 2), p.stem[-28:], fill="black", font=font(11))
        im.save(dest / f"sheet_{k // per:02d}.jpg", quality=88)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--dir", type=Path, required=True, help="labeled crop dir (labels.jsonl + kind folders)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--all", action="store_true",
                    help="every crop, not only errors: <head>/<true>/, wrong ones prefixed ERR_; all.csv")
    args = ap.parse_args()

    samples = iter_multitask_samples(args.dir, args.dir / "labels.jsonl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_classifier(args.ckpt, device)
    tri, rect = FLOOR_SHAPES.index("triangle"), FLOOR_SHAPES.index("rectangle")
    if args.out.exists():
        shutil.rmtree(args.out)
    rows, k = [], 0
    loader = DataLoader(MultitaskCropDataset(samples), batch_size=32, shuffle=False)
    with torch.no_grad():
        for images, ky, my, fy, _ey, _sy, sty in loader:
            outs = model(images.to(device))
            paths = [Path(samples[k + i][0]) for i in range(len(ky))]
            k += len(ky)
            for head, y in (("kind", ky), ("material", my), ("floor", fy), ("structure", sty)):
                prob = outs[OUT_INDEX[head]].softmax(1).cpu()
                for path, t, pr in zip(paths, y.tolist(), prob):
                    if t < 0:
                        continue
                    p = int(pr.argmax())
                    conf = float(pr[p])
                    if head == "floor":
                        t = TEST_FLOORS.index(FLOOR_SHAPES[rect if t == tri else t])
                        p = TEST_FLOORS.index(FLOOR_SHAPES[rect if p == tri else p])
                    if t == p and not args.all:
                        continue
                    names = HEADS[head]
                    rows.append({"head": head, "true": names[t], "pred": names[p], "prob": round(conf, 3),
                                 "correct": int(t == p), "crop": path.name,
                                 "source": "cmp" if path.name.startswith("cmp_") else "lsaa"})
                    if args.all:
                        dest = args.out / head / names[t]
                        fname = f"{'' if t == p else 'ERR_'}{path.stem}.jpg"
                    else:
                        dest = args.out / head / f"{names[t]}_as_{names[p]}"
                        fname = f"{path.stem}.jpg"
                    dest.mkdir(parents=True, exist_ok=True)
                    mark = "" if t == p else "   WRONG"
                    panel(path, f"{head}: true {names[t]}  pred {names[p]} ({conf:.2f}){mark}").save(
                        dest / fname, quality=88)
    for d in sorted(args.out.glob("*/*")):
        tiles = sorted(p for p in d.glob("*.jpg"))
        if args.all:
            sheet(tiles, d, per=20, cols=4, cell=420)
        else:
            sheet(tiles, d)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ("all.csv" if args.all else "errors.csv")).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["head", "true", "pred", "prob", "correct", "crop", "source"])
        w.writeheader()
        w.writerows(rows)
    n_err = sum(not r["correct"] for r in rows)
    print(f"{args.ckpt}: {len(samples)} crops, {n_err} errors, {len(rows)} images -> {args.out}")
    for d in sorted(args.out.glob("*/*")):
        print(f"  {d.relative_to(args.out)}: {sum(1 for p in d.glob('*.jpg') if not p.name.startswith('sheet_'))}")


if __name__ == "__main__":
    main()
