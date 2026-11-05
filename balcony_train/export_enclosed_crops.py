"""Crop enclosed volumes from train facades into the railing label folders.

Reads ``runs/enclosed_grid/labels_train.jsonl``. Cells that share a side
become one volume. The crop is the union of those cells plus 10% context.
Images land in ``runs/balcony_clf/crops/unlabeled/<stem>_enclosed_<NN>.png``
so ``label_ui.py`` can set floor shape and enclosure skin.

Photos are ``data/base_jpg``. Eval stems (``cmp_x``) are refused.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/export_enclosed_crops.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.depth_cells import cell_box  # noqa: E402
from balcony_train.enclosed_grid_store import (  # noqa: E402
    DEFAULT_DSL_DIR,
    Cell,
    find_photo,
    load_grid,
    load_labels,
)

DEFAULT_LABELS = ROOT / "runs" / "enclosed_grid" / "labels_train.jsonl"
DEFAULT_PHOTO_DIR = ROOT / "data" / "base_jpg"
DEFAULT_OUT_DIR = ROOT / "runs" / "balcony_clf" / "crops" / "unlabeled"
CONTEXT = 0.10


def connected_components(cells: list[Cell]) -> list[list[Cell]]:
    """4-connected groups. Diagonal cells stay apart. Order is stable."""
    remaining = {(int(f), int(b)) for f, b in cells}
    groups: list[list[Cell]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        group = [start]
        while stack:
            floor, bay = stack.pop()
            for nb in (
                (floor - 1, bay),
                (floor + 1, bay),
                (floor, bay - 1),
                (floor, bay + 1),
            ):
                if nb in remaining:
                    remaining.remove(nb)
                    stack.append(nb)
                    group.append(nb)
        groups.append(sorted(group))
    groups.sort(key=lambda group: group[0])
    return groups


def union_box(
    cells: list[Cell],
    floor_y: dict[int, tuple[int, int]],
    bay_x: dict[int, tuple[int, int]],
    width: int,
    height: int,
    *,
    margin: float = CONTEXT,
) -> tuple[int, int, int, int] | None:
    """Union of cell boxes, expanded by ``margin`` of that union, clamped to the photo."""
    boxes = [
        cell_box(cell, floor_y, bay_x)
        for cell in cells
        if int(cell[0]) in floor_y and int(cell[1]) in bay_x
    ]
    if not boxes:
        return None
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    pad_x = (x1 - x0) * float(margin)
    pad_y = (y1 - y0) * float(margin)
    left = max(0, int(round(x0 - pad_x)))
    top = max(0, int(round(y0 - pad_y)))
    right = min(int(width), int(round(x1 + pad_x)))
    bottom = min(int(height), int(round(y1 + pad_y)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def is_eval_stem(stem: str) -> bool:
    return str(stem).startswith("cmp_x")


def export_stem(
    stem: str,
    cells: list[Cell],
    *,
    photo_dir: Path,
    dsl_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
) -> list[Path]:
    """Write one png per connected enclosed volume. Existing files are kept."""
    if is_eval_stem(stem):
        raise ValueError(f"refusing eval stem {stem}")
    if not cells:
        return []
    photo_path = find_photo(photo_dir, stem)
    dsl = Path(dsl_dir) / stem / "facade_dsl.json"
    if photo_path is None or not dsl.is_file():
        return []
    photo = Image.open(photo_path).convert("RGB")
    floor_y, bay_x = load_grid(dsl)
    written: list[Path] = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, group in enumerate(connected_components(cells)):
        box = union_box(group, floor_y, bay_x, photo.width, photo.height)
        if box is None:
            continue
        dest = out_dir / f"{stem}_enclosed_{index:02d}.png"
        if dest.is_file() and not overwrite:
            written.append(dest)
            continue
        photo.crop(box).save(dest)
        written.append(dest)
    return written


def export_all(
    labels_path: Path,
    *,
    photo_dir: Path,
    dsl_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Export every train facade that has enclosed cells. Returns (photos, crops)."""
    labels = load_labels(labels_path)
    photos = 0
    crops = 0
    for stem, cells in labels.items():
        if is_eval_stem(stem):
            print(f"refuse eval stem {stem}")
            continue
        if not cells:
            continue
        paths = export_stem(
            stem,
            cells,
            photo_dir=photo_dir,
            dsl_dir=dsl_dir,
            out_dir=out_dir,
            overwrite=overwrite,
        )
        if not paths:
            print(f"skip {stem}: no photo, grid, or cells on the grid")
            continue
        photos += 1
        crops += len(paths)
        print(f"{stem}  {len(paths)} crop(s)")
    return photos, crops


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--photo-dir", type=Path, default=DEFAULT_PHOTO_DIR)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    photos, crops = export_all(
        args.labels,
        photo_dir=args.photo_dir,
        dsl_dir=args.dsl_dir,
        out_dir=args.out_dir,
        overwrite=args.overwrite,
    )
    print(f"exported {crops} crops from {photos} photos -> {args.out_dir}")


if __name__ == "__main__":
    main()
