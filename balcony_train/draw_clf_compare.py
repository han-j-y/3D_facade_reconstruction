"""Compare images for the pure cell classifier (no gate) on the eval set.

The classifier is trained on cmp_b only. Every eval photo is decoded with the
rectangle + mirror decoder at threshold t, whatever the gate says. Colors match
``evaluate_enclosed_grid`` (blue label only, red prediction only, purple both);
each cell shows its classifier probability, and a header shows gate max,
exact / IoU for the classifier and for OP1 (Stage 6). Photos are saved as
``{category}__{stem}.jpg`` with a ``_board_{category}.jpg`` per category
(categories as in ``draw_cell_clf_compare``).

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/draw_clf_compare.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.cell_clf_experiments import DEFAULT_CELLS, Data, line, score_sets  # noqa: E402
from balcony_train.cell_clf_search import SPECS, Model  # noqa: E402
from balcony_train.cell_clf_train import DEFAULT_TRAIN_CELLS, DEFAULT_TRAIN_LABELS  # noqa: E402
from balcony_train.cell_clf_v2 import mirror_index  # noqa: E402
from balcony_train.cell_clf_v3 import rect_decode  # noqa: E402
from balcony_train.draw_cell_clf_compare import CATEGORIES, board, category  # noqa: E402
from balcony_train.enclosed_grid_store import (  # noqa: E402
    DEFAULT_DSL_DIR,
    DEFAULT_EVAL_LABELS_JSONL,
    DEFAULT_EVAL_PHOTO_DIR,
    find_photo,
    load_grid,
)
from balcony_train.evaluate_enclosed_grid import render_compare  # noqa: E402

DEFAULT_OUT = ROOT / "runs" / "cell_clf" / "compare_clf"


def _font(size: int) -> ImageFont.ImageFont:
    for path in ("C:/Windows/Fonts/meiryob.ttc", "C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/YuGothM.ttc"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def iou(a: set, b: set) -> float:
    return 1.0 if not (a | b) else len(a & b) / len(a | b)


def draw_probs(image: Image.Image, floor_y, bay_x, probs: dict, t: float) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = _font(max(14, image.width // 40))
    for (f, b), p in probs.items():
        x0, x1 = bay_x[b]
        y0, y1 = floor_y[f]
        text = f"{p:.2f}"
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        x, y = (x0 + x1 - tw) // 2, y1 - th - 6
        draw.rectangle((x - 3, y - 1, x + tw + 3, y + th + 4), fill=(0, 0, 0))
        draw.text((x, y), text, fill=(255, 230, 0) if p >= t else (220, 220, 220), font=font)
    return image


def add_header(image: Image.Image, lines: list[str]) -> Image.Image:
    font = _font(max(14, image.width // 32))
    h = (font.size + 8) * len(lines) + 10
    out = Image.new("RGB", (image.width, image.height + h), (20, 20, 20))
    out.paste(image, (0, h))
    draw = ImageDraw.Draw(out)
    for k, text in enumerate(lines):
        draw.text((10, 6 + k * (font.size + 8)), text, fill=(255, 255, 255), font=font)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-cells", type=Path, default=DEFAULT_TRAIN_CELLS)
    ap.add_argument("--train-labels", type=Path, default=DEFAULT_TRAIN_LABELS)
    ap.add_argument("--eval-cells", type=Path, default=DEFAULT_CELLS)
    ap.add_argument("--eval-labels", type=Path, default=DEFAULT_EVAL_LABELS_JSONL)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_EVAL_PHOTO_DIR)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--t", type=float, default=0.3, help="cell threshold for the rectangle decoder")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    train = Data(args.train_cells, args.train_labels)
    test = Data(args.eval_cells, args.eval_labels)
    model = Model(train, train.stems, SPECS["base"], None)
    p_gate, p_cell = model.predict(test, test.stems)
    mir = mirror_index(test, args.dsl_dir)
    cur = test.f("in_cells_6")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    preds = {}
    report = []
    tiles: dict[str, list[Image.Image]] = {c: [] for c in CATEGORIES}
    for s in test.stems:
        ii = test.idx[s]
        gt = set(test.gt[s])
        clf = rect_decode(test, s, p_cell, args.t, mir)
        op1 = {test.cells[i] for i in ii if cur[i] > 0}
        preds[s] = clf
        gmax = float(p_gate[ii].max())
        cat = category(gt, clf)
        lines = [
            f"{s}  [{cat}]  label cells={len(gt)}",
            f"pure classifier t={args.t}  gate max={gmax:.2f}",
            f"classifier: exact={int(clf == gt)}  IoU={iou(clf, gt):.2f}  cells={len(clf)}",
            f"OP1:        exact={int(op1 == gt)}  IoU={iou(op1, gt):.2f}  cells={len(op1)}",
        ]
        floor_y, bay_x = load_grid(args.dsl_dir / s / "facade_dsl.json")
        photo = Image.open(find_photo(args.in_dir, s)).convert("RGB")
        img = render_compare(photo, floor_y, bay_x, gt, clf)
        img = draw_probs(img, floor_y, bay_x, {test.cells[i]: float(p_cell[i]) for i in ii}, args.t)
        panel = add_header(img, lines)
        panel.save(args.out_dir / f"{cat}__{s}.jpg", quality=92)
        tiles[cat].append(panel)
        report.append(
            f"{s}  {cat:<9}  gate={gmax:.2f}  clf exact={int(clf == gt)} iou={iou(clf, gt):.2f}  "
            f"op1 exact={int(op1 == gt)} iou={iou(op1, gt):.2f}"
        )
    for cat in CATEGORIES:
        if tiles[cat]:
            board(tiles[cat]).save(args.out_dir / f"_board_{cat}.jpg", quality=85)
    counts = "  ".join(f"{c}={len(tiles[c])}" for c in CATEGORIES)
    summary = line(f"eval pure classifier (t={args.t})", score_sets(test, preds))
    (args.out_dir / "summary.txt").write_text(
        summary + "\n" + counts + "\n" + "\n".join(sorted(report, key=lambda r: r.split()[1])) + "\n",
        encoding="utf-8",
    )
    print(summary)
    print(counts)
    print(f"{len(test.stems)} images -> {args.out_dir}")


if __name__ == "__main__":
    main()
