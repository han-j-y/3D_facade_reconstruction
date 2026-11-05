"""Carry oriel cell labels from one window grid to another by pixel overlap.

The labels (``enclosed_grid_ui.py`` jsonl) are indexed on the grids in
``--old-dsl-dir``. Each cell of the grid in ``--new-dsl-dir`` is labeled when
at least ``--min-cover`` (0.5) of its area lies on old labeled cells. Ignore
cells carry over the same way. Floor shape, skin and grid failures are copied.

A photo goes on the review list when the remap is not clean:

* a new cell is partly covered (between ``--review-lo`` and ``--review-hi``);
* an old labeled cell is not mostly covered by new labeled cells;
* the new grid is missing or empty.

Review those photos from the photo only (``enclosed_grid_ui.py --stems``).

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/remap_grid_labels.py ^
      --labels runs/enclosed_grid/labels_train.grid1.jsonl --out runs/enclosed_grid/labels_train.jsonl ^
      --old-dsl-dir runs/batch --new-dsl-dir runs/batch_grid2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.depth_cells import cell_box  # noqa: E402
from balcony_train.enclosed_grid_store import (  # noqa: E402
    GRID_FAILURE,
    Cell,
    GridRecord,
    load_grid,
    load_records,
    write_records,
)


def _mask(cells: list[Cell], floor_y, bay_x, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    for c in cells:
        if c[0] in floor_y and c[1] in bay_x:
            x0, y0, x1, y1 = cell_box(c, floor_y, bay_x)
            m[max(0, y0) : y1, max(0, x0) : x1] = True
    return m


def _cover(mask: np.ndarray, box: list[int]) -> float:
    x0, y0, x1, y1 = box
    area = max(1, (y1 - y0) * (x1 - x0))
    return float(mask[max(0, y0) : y1, max(0, x0) : x1].sum()) / area


def remap(
    old_cells: list[Cell],
    old_ignore: list[Cell],
    old_grid: tuple[dict, dict],
    new_grid: tuple[dict, dict],
    *,
    min_cover: float = 0.5,
    review_lo: float = 0.25,
    review_hi: float = 0.75,
) -> tuple[list[Cell], list[Cell], list[str]]:
    """New cells, new ignore cells, and the reasons to review (empty when clean)."""
    (ofy, obx), (nfy, nbx) = old_grid, new_grid
    h = int(max([y1 for _y0, y1 in list(ofy.values()) + list(nfy.values())] + [1])) + 1
    w = int(max([x1 for _x0, x1 in list(obx.values()) + list(nbx.values())] + [1])) + 1
    pos = _mask(old_cells, ofy, obx, (h, w))
    ign = _mask(old_ignore, ofy, obx, (h, w))
    cells: list[Cell] = []
    ignore: list[Cell] = []
    reasons: list[str] = []
    for f in nfy:
        for b in nbx:
            box = cell_box((f, b), nfy, nbx)
            cov = _cover(pos, box)
            if cov >= min_cover:
                cells.append((f, b))
            if review_lo <= cov < review_hi:
                reasons.append(f"partial f{f}b{b} {cov:.2f}")
            if _cover(ign, box) >= min_cover:
                ignore.append((f, b))
    new_pos = _mask(cells, nfy, nbx, (h, w))
    for c in old_cells:
        if c[0] in ofy and c[1] in obx and _cover(new_pos, cell_box(c, ofy, obx)) < min_cover:
            reasons.append(f"lost old f{c[0]}b{c[1]}")
    return sorted(cells), sorted(ignore), reasons


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, required=True, help="labels on the old grids")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--old-dsl-dir", type=Path, required=True)
    ap.add_argument("--new-dsl-dir", type=Path, required=True)
    ap.add_argument("--review-out", type=Path, default=None, help="default: <out>.review.txt")
    ap.add_argument("--min-cover", type=float, default=0.5)
    ap.add_argument("--review-lo", type=float, default=0.25)
    ap.add_argument("--review-hi", type=float, default=0.75)
    args = ap.parse_args()
    if args.out.resolve() == args.labels.resolve():
        raise SystemExit("--out must differ from --labels")

    old = load_records(args.labels)
    new: dict[str, GridRecord] = {}
    review: list[str] = []
    missing: list[str] = []
    n_old = n_new = 0
    for stem, rec in sorted(old.items()):
        new_dsl = args.new_dsl_dir / stem / "facade_dsl.json"
        if rec.status == GRID_FAILURE:
            new[stem] = rec.copy()
            continue
        if not new_dsl.is_file():
            missing.append(stem)
            continue
        new_grid = load_grid(new_dsl)
        if not rec.cells and not rec.ignore:
            new[stem] = GridRecord(cells=[])
            continue
        old_grid = load_grid(args.old_dsl_dir / stem / "facade_dsl.json")
        if not new_grid[0] or not new_grid[1]:
            review.append(f"{stem}\tno new grid")
            new[stem] = GridRecord(cells=[], floor_shape=rec.floor_shape, skin=rec.skin)
            continue
        cells, ignore, reasons = remap(
            rec.cells, rec.ignore, old_grid, new_grid,
            min_cover=args.min_cover, review_lo=args.review_lo, review_hi=args.review_hi,
        )
        n_old += len(rec.cells)
        n_new += len(cells)
        new[stem] = GridRecord(
            cells=cells,
            floor_shape=rec.floor_shape if cells else None,
            skin=rec.skin if cells else None,
            ignore=ignore,
        )
        if reasons:
            review.append(f"{stem}\t" + "; ".join(reasons))
    write_records(args.out, new)
    review_out = args.review_out or args.out.with_suffix(".review.txt")
    review_out.write_text("".join(r + "\n" for r in review), encoding="utf-8")
    print(f"photos {len(old)}  written {len(new)}  missing new grid {len(missing)}")
    print(f"labeled cells old {n_old} -> new {n_new}")
    print(f"review {len(review)} photos -> {review_out}")
    if missing:
        print("missing: " + " ".join(missing))


if __name__ == "__main__":
    main()
