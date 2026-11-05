"""Draw every cached SAM3 box before any Stage 1 filter.

Color = prompt. Thick outline = the box passes the Stage 1 seed filter
(height > 0.85 storey, size limits, not under a fire escape). Thin outline =
dropped at Stage 1. Gray = exclude prompt (fire escape). Label = score.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/draw_sam3_raw.py ^
      --stems cmp_x0028 cmp_x0116 --out-dir runs\\stage_board_miss\\sam3_raw
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

from balcony_train.enclosed_grid_store import DEFAULT_EVAL_PHOTO_DIR, find_photo  # noqa: E402
from balcony_train.try_sam3_da3_volumes import (  # noqa: E402
    DEFAULT_DSL_DIR,
    SAM3_PROMPTS,
    SEED_MIN_FLOOR_FRAC,
    _stage_font,
    compose_row,
    covered_by_exclude,
    resolve_floor_h,
    seed_ok,
    split_seed_records,
)
from balcony_train.recrop_sam3 import clamp_box_xyxy  # noqa: E402

DEFAULT_CACHE = ROOT / "runs" / "sam3_da3_volumes" / "sam3_seeds.json"
PROMPT_COLORS = (
    (255, 80, 80),
    (255, 170, 40),
    (60, 220, 90),
    (60, 170, 255),
    (220, 90, 255),
)
EXCLUDE_COLOR = (150, 150, 150)


def _color(prompt: str | None) -> tuple[int, int, int]:
    if prompt in SAM3_PROMPTS:
        return PROMPT_COLORS[SAM3_PROMPTS.index(prompt) % len(PROMPT_COLORS)]
    return (255, 255, 255)


def draw_raw(
    image: Image.Image,
    records: list[dict],
    *,
    floor_h: float | None,
) -> tuple[Image.Image, int, int]:
    """Return (overlay, raw count, Stage 1 pass count)."""
    iw, ih = image.size
    min_h = SEED_MIN_FLOOR_FRAC * floor_h if floor_h else None
    seeds, excludes = split_seed_records(records)
    ex_boxes = [list(clamp_box_xyxy(r["box_xyxy"], width=iw, height=ih)) for r in excludes]
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    thick = max(3, iw // 200)
    font = _stage_font(max(12, iw // 60))
    for r in excludes:
        draw.rectangle(r["box_xyxy"], outline=EXCLUDE_COLOR, width=max(1, thick // 2))
    n_pass = 0
    for r in sorted(seeds, key=lambda r: float(r.get("score") or 0.0)):
        box = list(clamp_box_xyxy(r["box_xyxy"], width=iw, height=ih))
        ok = seed_ok(box, iw=iw, ih=ih, min_height_px=min_h) and not covered_by_exclude(box, ex_boxes)
        n_pass += int(ok)
        color = _color(r.get("prompt"))
        draw.rectangle(box, outline=color, width=thick if ok else max(1, thick // 3))
        draw.text((box[0] + 3, box[1] + 2), f"{float(r.get('score') or 0):.2f}", fill=color, font=font)
    bar_h = max(28, ih // 26)
    draw.rectangle((0, 0, iw, bar_h), fill=(12, 16, 22))
    draw.text((8, 4), f"SAM3 raw n={len(seeds)}  stage1 pass={n_pass}", fill=(240, 240, 240), font=font)
    return out, len(seeds), n_pass


def legend(width: int) -> Image.Image:
    font = _stage_font(16)
    img = Image.new("RGB", (width, 30), (20, 20, 24))
    draw = ImageDraw.Draw(img)
    x = 8
    for prompt in SAM3_PROMPTS:
        draw.rectangle((x, 8, x + 16, 22), fill=_color(prompt))
        draw.text((x + 22, 6), prompt, fill=(235, 235, 235), font=font)
        x += 30 + int(draw.textlength(prompt, font=font))
    draw.text((x, 6), "thick=pass stage1  thin=dropped  gray=fire escape", fill=(200, 200, 200), font=font)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stems", nargs="+", required=True)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_EVAL_PHOTO_DIR)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tiles: list[Image.Image] = []
    stems = [s for raw in args.stems for s in raw.replace(",", " ").split()]
    for stem in stems:
        photo = find_photo(args.in_dir, stem)
        if photo is None:
            print(f"{stem}: no photo")
            continue
        if stem not in cache:
            print(f"{stem}: not in SAM3 cache")
            continue
        image = Image.open(photo).convert("RGB")
        overlay, n_raw, n_pass = draw_raw(image, cache[stem], floor_h=resolve_floor_h(stem, args.dsl_dir))
        overlay.save(args.out_dir / f"{stem}_sam3_raw.jpg", quality=92)
        tiles.append(overlay)
        print(f"{stem}  raw={n_raw}  stage1_pass={n_pass}", flush=True)
    if not tiles:
        return
    per_row = 5
    rows = [compose_row(tiles[i : i + per_row], height=480) for i in range(0, len(tiles), per_row)]
    width = max(r.width for r in rows)
    key = legend(width)
    board = Image.new("RGB", (width, key.height + sum(r.height + 8 for r in rows)), (24, 24, 24))
    board.paste(key, (0, 0))
    y = key.height
    for row in rows:
        board.paste(row, (0, y))
        y += row.height + 8
    path = args.out_dir / "sam3_raw_board.jpg"
    board.save(path, quality=90)
    print(f"board -> {path}")


if __name__ == "__main__":
    main()
