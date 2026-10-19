"""Compare images for a cell-level enclosure method on every eval photo.

Reads ``{"eval": {"preds": {stem: [[floor, bay], ...]}}}``: the cell
classifier (OP3, ``cell_clf_train.py``) by default, or the depth-first rule
(OP2, ``evaluate_depth_cells.py`` writes ``runs/depth_cells/results.json``).

Same colors as ``evaluate_enclosed_grid --compare-dir``: blue = label only,
red = prediction only, purple = both. Each photo is saved as
``{category}__{stem}.jpg``, every category also gets a tiled board, and
``_report.txt`` holds the enclosure / none / total rates next to OP1.

Categories:
    exact      enclosure photo, predicted cells match the label
    wrong      enclosure photo, some cells predicted but not the label set
    missed     enclosure photo, nothing predicted
    none_ok    no-enclosure photo, nothing predicted
    false_pos  no-enclosure photo, cells predicted

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/draw_cell_clf_compare.py
    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/draw_cell_clf_compare.py ^
      --results runs/depth_cells/results.json --out-dir runs/depth_cells/compare_board --name "OP2 depth-first rule"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.enclosed_grid_store import (  # noqa: E402
    DEFAULT_DSL_DIR,
    DEFAULT_EVAL_LABELS_JSONL,
    DEFAULT_EVAL_PHOTO_DIR,
    find_photo,
    load_grid,
    load_labels,
)
from balcony_train.evaluate_enclosed_grid import format_report, render_compare, score_photo, summarize  # noqa: E402
from balcony_train.try_sam3_da3_volumes import _stage_font  # noqa: E402

DEFAULT_RESULTS = ROOT / "runs" / "cell_clf" / "results_base.json"
DEFAULT_OUT_DIR = ROOT / "runs" / "cell_clf" / "compare"
DEFAULT_OP1 = ROOT / "runs" / "enclosed_grid" / "volume_cells.json"
CATEGORIES = ("exact", "wrong", "missed", "none_ok", "false_pos")
TILE_H = 480
BOARD_COLS = 4


def category(gt: set, pred: set) -> str:
    if gt:
        if pred == gt:
            return "exact"
        return "wrong" if pred else "missed"
    return "false_pos" if pred else "none_ok"


def titled(panel: Image.Image, title: str) -> Image.Image:
    font = _stage_font(max(14, panel.width // 40))
    bar_h = max(28, panel.height // 24)
    d = ImageDraw.Draw(panel)
    d.rectangle((0, 0, panel.width, bar_h), fill=(12, 16, 22))
    d.text((6, 3), title, fill=(240, 240, 240), font=font)
    return panel


def board(tiles: list[Image.Image], cols: int = BOARD_COLS, height: int = TILE_H, gap: int = 12) -> Image.Image:
    scaled = [t.resize((max(1, round(t.width * height / t.height)), height)) for t in tiles]
    rows = [scaled[i : i + cols] for i in range(0, len(scaled), cols)]
    width = max(sum(t.width for t in r) + gap * (len(r) + 1) for r in rows)
    out = Image.new("RGB", (width, gap + len(rows) * (height + gap)), (245, 245, 245))
    for r, row in enumerate(rows):
        x = gap
        for t in row:
            out.paste(t, (x, gap + r * (height + gap)))
            x += t.width + gap
    return out


def report(name: str, labels: dict, preds: dict[str, set], stems: list[str]) -> str:
    rows = []
    for stem in stems:
        row = score_photo(labels[stem], preds.get(stem, set()))
        row["stem"] = stem
        rows.append(row)
    head = "\n".join(format_report(summarize(rows), rows).splitlines()[:4])
    return f"=== {name} ({len(stems)} photos) ===\n{head}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--labels", type=Path, default=DEFAULT_EVAL_LABELS_JSONL)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_EVAL_PHOTO_DIR)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--name", default="OP3 cell classifier")
    ap.add_argument("--op1", type=Path, default=DEFAULT_OP1, help="volume_cells.json with OP1 cells_6")
    args = ap.parse_args()

    res = json.loads(args.results.read_text(encoding="utf-8"))
    preds = {s: {tuple(c) for c in cells} for s, cells in res["eval"]["preds"].items()}
    labels = load_labels(args.labels)
    gate = res.get("gate_max", {})
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tiles: dict[str, list[Image.Image]] = {c: [] for c in CATEGORIES}
    counts = {c: 0 for c in CATEGORIES}
    skipped: list[str] = []
    for stem in sorted(preds):
        if stem not in labels:
            skipped.append(f"{stem} (not in labels)")
            continue
        photo = find_photo(args.in_dir, stem)
        dsl = args.dsl_dir / stem / "facade_dsl.json"
        if photo is None or not dsl.is_file():
            skipped.append(f"{stem} (no photo or grid)")
            continue
        floor_y, bay_x = load_grid(dsl)
        gt = {tuple(c) for c in labels[stem]}
        pred = preds[stem]
        cat = category(gt, pred)
        tp, fp, fn = len(gt & pred), len(pred - gt), len(gt - pred)
        g = f"  gate {gate[stem]:.2f}" if stem in gate else ""
        image = Image.open(photo).convert("RGB")
        panel = titled(
            render_compare(image, floor_y, bay_x, gt, pred),
            f"{stem}  {cat}  tp{tp} fp{fp} fn{fn}{g}",
        )
        panel.save(args.out_dir / f"{cat}__{stem}.jpg", quality=90)
        tiles[cat].append(panel)
        counts[cat] += 1
        print(f"{stem}  {cat}  tp{tp} fp{fp} fn{fn}{g}", flush=True)

    for cat in CATEGORIES:
        if tiles[cat]:
            board(tiles[cat]).save(args.out_dir / f"_board_{cat}.jpg", quality=85)
    missing = sorted(set(labels) - set(preds))
    if skipped:
        print("skipped: " + ", ".join(skipped))
    if missing:
        print("labeled but not in results (rerun cell_clf_train.py): " + ", ".join(missing))
    print("  ".join(f"{c}={counts[c]}" for c in CATEGORIES))

    scored = sorted(s for s in preds if s in labels)
    blocks = [report(args.name, labels, preds, scored)]
    if args.op1.is_file():
        vol = json.loads(args.op1.read_text(encoding="utf-8"))
        op1 = {s: {tuple(c) for c in vol[s].get("cells_6", [])} for s in scored if s in vol}
        blocks.append(report("OP1 rule-based stage6", labels, op1, scored))
    text = "\n\n".join(blocks)
    print("\n" + text)
    (args.out_dir / "_report.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\ncompare -> {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
