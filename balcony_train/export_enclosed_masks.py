"""Write binary GT masks for the enclosed-balcony train set.

Reads ``runs/enclosed_grid/labels_train.jsonl``. Each 4-connected group of
window cells becomes one mask, the same size as the facade photo. Labeled
cells are white (255). Everything else is black (0). Photos with no enclosure
are skipped. Eval stems (``cmp_x``) are refused.

Output::

    runs/enclosed_grid/gt_masks/images/<stem>.jpg
    runs/enclosed_grid/gt_masks/masks/<stem>_enclosed_00.png
    runs/enclosed_grid/gt_masks/check/<stem>.jpg
    runs/enclosed_grid/gt_masks/manifest.jsonl

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/export_enclosed_masks.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
from balcony_train.export_enclosed_crops import connected_components, is_eval_stem  # noqa: E402

DEFAULT_LABELS = ROOT / "runs" / "enclosed_grid" / "labels_train.jsonl"
DEFAULT_PHOTO_DIR = ROOT / "data" / "base_jpg"
DEFAULT_OUT_DIR = ROOT / "runs" / "enclosed_grid" / "gt_masks"
# Window grid does not cover the enclosure, so these stay out of the shared masks.
EXCLUDED_STEMS = frozenset({"cmp_b0291", "cmp_b0304", "cmp_b0305"})
# First color matches the Compare label blue. Later colors separate extra balconies.
INSTANCE_COLORS: tuple[tuple[int, int, int], ...] = (
    (40, 110, 255),
    (255, 170, 40),
    (40, 190, 110),
    (170, 70, 220),
)


def paint_mask(
    width: int,
    height: int,
    cells: list[Cell],
    floor_y: dict[int, tuple[int, int]],
    bay_x: dict[int, tuple[int, int]],
) -> Image.Image | None:
    """White rectangles on black. Returns None when no cell lands on the grid."""
    mask = Image.new("L", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(mask)
    painted = 0
    for cell in cells:
        if int(cell[0]) not in floor_y or int(cell[1]) not in bay_x:
            continue
        x0, y0, x1, y1 = cell_box(cell, floor_y, bay_x)
        left = max(0, min(int(width), int(x0)))
        top = max(0, min(int(height), int(y0)))
        right = max(0, min(int(width), int(x1)))
        bottom = max(0, min(int(height), int(y1)))
        if right <= left or bottom <= top:
            continue
        draw.rectangle([left, top, right - 1, bottom - 1], fill=255)
        painted += 1
    if painted == 0:
        return None
    return mask


def export_all(
    labels_path: Path,
    *,
    photo_dir: Path,
    dsl_dir: Path,
    out_dir: Path,
) -> tuple[int, int]:
    """Copy each enclosed facade and write one mask per balcony. Returns (photos, masks)."""
    labels = load_labels(labels_path)
    image_dir = Path(out_dir) / "images"
    mask_dir = Path(out_dir) / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    photos = 0
    masks = 0
    for stem in sorted(labels):
        cells = labels[stem]
        if is_eval_stem(stem):
            print(f"refuse eval stem {stem}")
            continue
        if stem in EXCLUDED_STEMS:
            print(f"exclude {stem}: window grid does not cover the enclosure")
            continue
        if not cells:
            continue
        photo_path = find_photo(photo_dir, stem)
        dsl = Path(dsl_dir) / stem / "facade_dsl.json"
        if photo_path is None or not dsl.is_file():
            print(f"skip {stem}: no photo or grid")
            continue
        photo = Image.open(photo_path)
        floor_y, bay_x = load_grid(dsl)
        groups = connected_components(cells)
        written = 0
        for index, group in enumerate(groups):
            mask = paint_mask(photo.width, photo.height, group, floor_y, bay_x)
            if mask is None:
                print(f"skip {stem} group {index}: cells are off the grid")
                continue
            mask_name = f"{stem}_enclosed_{index:02d}.png"
            mask.save(mask_dir / mask_name)
            rows.append(
                {
                    "stem": stem,
                    "image": f"images/{stem}{photo_path.suffix.lower()}",
                    "mask": f"masks/{mask_name}",
                    "cells": [list(cell) for cell in group],
                }
            )
            written += 1
        if written == 0:
            continue
        dest = image_dir / f"{stem}{photo_path.suffix.lower()}"
        shutil.copy2(photo_path, dest)
        photos += 1
        masks += written
        print(f"{stem}  {written} mask(s)  {len(cells)} cells")
    manifest = Path(out_dir) / "manifest.jsonl"
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    manifest.write_text(text, encoding="utf-8")
    return photos, masks


def _edge(selected: np.ndarray) -> np.ndarray:
    """Pixels of ``selected`` that touch the background."""
    inner = selected.copy()
    inner[1:, :] &= selected[:-1, :]
    inner[:-1, :] &= selected[1:, :]
    inner[:, 1:] &= selected[:, :-1]
    inner[:, :-1] &= selected[:, 1:]
    return selected & ~inner


def write_checks(out_dir: Path) -> int:
    """Paint each saved mask on its photo so a person can see the selection.

    One jpg per facade. The first balcony uses the Compare label blue. Extra
    balconies on the same photo use a different color, with the mask name written
    on the region. The binary masks are left unchanged.
    """
    image_dir = Path(out_dir) / "images"
    mask_dir = Path(out_dir) / "masks"
    check_dir = Path(out_dir) / "check"
    check_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        masks = sorted(mask_dir.glob(f"{image_path.stem}_enclosed_*.png"))
        if not masks:
            continue
        photo = Image.open(image_path).convert("RGB")
        painted = np.asarray(photo, dtype=np.float32)
        labels: list[tuple[int, int, str]] = []
        for index, mask_path in enumerate(masks):
            selected = np.asarray(Image.open(mask_path).convert("L")) >= 128
            if not bool(selected.any()):
                continue
            color = np.asarray(INSTANCE_COLORS[index % len(INSTANCE_COLORS)], dtype=np.float32)
            painted[selected] = painted[selected] * 0.49 + color * 0.51
            edge = _edge(selected)
            painted[edge] = color
            ys, xs = np.nonzero(selected)
            labels.append((int(xs.min()), int(ys.min()), mask_path.stem))
        image = Image.fromarray(np.clip(painted, 0, 255).astype(np.uint8), "RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for path in ("C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/meiryo.ttc"):
            if Path(path).is_file():
                font = ImageFont.truetype(path, max(14, image.width // 42))
                break
        for x, y, name in labels:
            short = name.removeprefix(f"{image_path.stem}_")
            draw.text((x + 6, y + 4), short, fill=(255, 255, 255), font=font)
        image.save(check_dir / f"{image_path.stem}.jpg", quality=90)
        count += 1
        print(f"check {image_path.stem}  {len(labels)} region(s)")
    return count


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--photo-dir", type=Path, default=DEFAULT_PHOTO_DIR)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    photos, masks = export_all(
        args.labels,
        photo_dir=args.photo_dir,
        dsl_dir=args.dsl_dir,
        out_dir=args.out_dir,
    )
    print(f"wrote {masks} masks for {photos} photos -> {args.out_dir}")
    checks = write_checks(args.out_dir)
    print(f"wrote {checks} check images -> {args.out_dir / 'check'}")


if __name__ == "__main__":
    main()
