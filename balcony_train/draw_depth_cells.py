"""DA3 depth analysis on the window grid, one board row per photo.

Panels: photo with labeled cells (blue) | DA3 depth (warm = near) |
per-cell "closer than wall" percentages.

Each cell shows two numbers:
    L = Stage 2 metric, cell vs the wall strips beside it (``box_rel_delta``)
    R = cell vs the median of its own storey row
Cell fill uses L: green >= 1.5%, yellow >= 0.6%, gray below, red when behind.
Labeled cells get a thick blue outline.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/draw_depth_cells.py ^
      --stems cmp_x0097 cmp_x0137 --out-dir runs\\stage_board_miss\\depth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
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
from balcony_train.extract_enclosed import foliage_mask  # noqa: E402
from balcony_train.try_da3_enclosure import colorize_depth, resize_depth  # noqa: E402
from balcony_train.try_sam3_da3_volumes import (  # noqa: E402
    DEFAULT_DEPTH_DIR,
    GROW_REL,
    MIN_REL,
    _stage_font,
    box_rel_delta,
    compose_row,
    robust_median,
)

LABEL_COLOR = (60, 130, 255)


def grid_depth_mask(
    height: int,
    width: int,
    floor_y: dict[int, tuple[int, int]],
    bay_x: dict[int, tuple[int, int]],
) -> np.ndarray:
    """True on window-grid cells, so depth color ignores sky and gray pads."""
    mask = np.zeros((height, width), dtype=bool)
    for y0, y1 in floor_y.values():
        for x0, x1 in bay_x.values():
            mask[int(y0) : int(y1), int(x0) : int(x1)] = True
    return mask


def cell_rel_table(
    depth: np.ndarray,
    veg: np.ndarray | None,
    floor_y: dict[int, tuple[int, int]],
    bay_x: dict[int, tuple[int, int]],
) -> dict[tuple[int, int], tuple[float | None, float | None]]:
    """(floor, bay) -> (local wall rel, storey row rel)."""
    out: dict[tuple[int, int], tuple[float | None, float | None]] = {}
    for fid, (y0, y1) in floor_y.items():
        meds: dict[int, float | None] = {}
        for bid, (x0, x1) in bay_x.items():
            meds[bid] = robust_median(depth, [int(x0), int(y0), int(x1), int(y1)])
        row_vals = [m for m in meds.values() if m is not None]
        row_med = float(np.median(row_vals)) if row_vals else None
        for bid, (x0, x1) in bay_x.items():
            box = [int(x0), int(y0), int(x1), int(y1)]
            local = box_rel_delta(depth, box, veg=veg)
            m = meds[bid]
            row = (row_med - m) / row_med if (m is not None and row_med) else None
            out[(int(fid), int(bid))] = (local, row)
    return out


def _fill(rel: float | None) -> tuple[int, int, int, int]:
    if rel is None:
        return (0, 0, 0, 0)
    if rel >= MIN_REL:
        return (40, 210, 90, 120)
    if rel >= GROW_REL:
        return (240, 210, 40, 110)
    if rel < 0:
        return (230, 60, 60, 90)
    return (120, 120, 120, 60)


def _pct(v: float | None) -> str:
    return "--" if v is None else f"{100 * v:+.1f}"


def draw_panels(
    image: Image.Image,
    depth: np.ndarray,
    floor_y: dict[int, tuple[int, int]],
    bay_x: dict[int, tuple[int, int]],
    gt: set[tuple[int, int]],
    table: dict[tuple[int, int], tuple[float | None, float | None]],
    stem: str,
) -> list[Image.Image]:
    iw, ih = image.size
    line = max(2, iw // 300)
    font = _stage_font(max(11, iw // 70))
    title_font = _stage_font(max(14, iw // 45))
    bar_h = max(26, ih // 28)

    def grid(base: Image.Image, *, values: bool) -> Image.Image:
        canvas = base.convert("RGBA")
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for (fid, bid), (local, row) in sorted(table.items()):
            y0, y1 = floor_y[fid]
            x0, x1 = bay_x[bid]
            if values:
                draw.rectangle((x0, y0, x1, y1), fill=_fill(local))
            labeled = (fid, bid) in gt
            draw.rectangle(
                (x0, y0, x1, y1),
                outline=LABEL_COLOR + (255,) if labeled else (255, 255, 255, 110),
                width=line * 2 if labeled else 1,
            )
            if values:
                draw.text((x0 + 3, y0 + 2), f"L{_pct(local)}", fill=(255, 255, 255, 255), font=font)
                draw.text((x0 + 3, y0 + 2 + font.size + 1), f"R{_pct(row)}", fill=(255, 255, 160, 255), font=font)
        return Image.alpha_composite(canvas, layer).convert("RGB")

    photo = grid(image, values=False)
    mask = grid_depth_mask(ih, iw, floor_y, bay_x)
    depth_img = grid(Image.fromarray(colorize_depth(depth, mask=mask)), values=False)
    cells = grid(image.convert("L").convert("RGB"), values=True)
    titles = (
        f"{stem}  labeled cells (blue)",
        "DA3 depth (warm = near)",
        f"L = vs side wall, R = vs storey row  green>={100 * MIN_REL:.1f}%  yellow>={100 * GROW_REL:.1f}%",
    )
    for panel, title in zip((photo, depth_img, cells), titles):
        d = ImageDraw.Draw(panel)
        d.rectangle((0, 0, iw, bar_h), fill=(12, 16, 22))
        d.text((6, 3), title, fill=(240, 240, 240), font=title_font)
    return [photo, depth_img, cells]


def summarize_stem(table, gt) -> str:
    def stats(keys, idx):
        vals = [table[k][idx] for k in keys if table.get(k) and table[k][idx] is not None]
        return f"{100 * float(np.median(vals)):+.1f}%" if vals else "--"

    others = [k for k in table if k not in gt]
    labeled = [k for k in gt if k in table]
    passing = sum(1 for k in labeled if (table[k][0] or -1) >= MIN_REL)
    return (
        f"labeled median L={stats(labeled, 0)} R={stats(labeled, 1)}  "
        f"other median L={stats(others, 0)} R={stats(others, 1)}  "
        f"labeled L>=1.5%: {passing}/{len(labeled)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stems", nargs="+", required=True)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_EVAL_PHOTO_DIR)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--depth-dir", type=Path, default=DEFAULT_DEPTH_DIR)
    ap.add_argument("--labels", type=Path, default=DEFAULT_EVAL_LABELS_JSONL)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    labels = load_labels(args.labels)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[Image.Image] = []
    stems = [s for raw in args.stems for s in raw.replace(",", " ").split()]
    for stem in stems:
        photo = find_photo(args.in_dir, stem)
        depth_path = args.depth_dir / f"{stem}_depth.npy"
        dsl = args.dsl_dir / stem / "facade_dsl.json"
        if photo is None or not depth_path.is_file() or not dsl.is_file():
            print(f"{stem}: missing photo, depth or grid")
            continue
        image = Image.open(photo).convert("RGB")
        depth = resize_depth(np.load(depth_path), image.width, image.height)
        floor_y, bay_x = load_grid(dsl)
        gt = {tuple(c) for c in labels.get(stem, [])}
        veg = foliage_mask(np.asarray(image, dtype=np.uint8))
        table = cell_rel_table(depth, veg, floor_y, bay_x)
        panels = draw_panels(image, depth, floor_y, bay_x, gt, table, stem)
        row = compose_row(panels, height=560)
        row.save(args.out_dir / f"{stem}_depth.jpg", quality=92)
        rows.append(row)
        print(f"{stem}  {summarize_stem(table, gt)}", flush=True)
    if not rows:
        return
    width = max(r.width for r in rows)
    board = Image.new("RGB", (width, sum(r.height + 12 for r in rows)), (24, 24, 24))
    y = 0
    for r in rows:
        board.paste(r, (0, y))
        y += r.height + 12
    path = args.out_dir / "depth_board.jpg"
    board.save(path, quality=88)
    print(f"board -> {path}")


if __name__ == "__main__":
    main()
