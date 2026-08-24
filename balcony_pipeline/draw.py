"""Stage overlays for the balcony track."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _banner(draw: ImageDraw.ImageDraw, text: str, fill: tuple[int, int, int]) -> None:
    draw.rectangle([8, 8, 8 + 11 * min(len(text), 80), 40], fill=(20, 20, 20))
    draw.text((16, 12), text[:80], fill=fill, font=_font(18))


def draw_boxes(
    image: Image.Image,
    boxes: list[list[int]],
    *,
    title: str,
    color: tuple[int, int, int] = (220, 40, 40),
) -> Image.Image:
    im = image.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    for box in boxes:
        d.rectangle(box, outline=color, width=3)
    _banner(d, title, color)
    return im


def draw_snap(image: Image.Image, units: list[dict[str, Any]], title: str) -> Image.Image:
    im = image.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    font = _font(14)
    for u in units:
        box = u["box_xyxy"]
        d.rectangle(box, outline=(40, 180, 255), width=3)
        lab = f"F{u['floor']} B{u['bay_start']}-{u['bay_end']}"
        d.text((box[0] + 2, box[1] + 2), lab, fill=(40, 180, 255), font=font)
    _banner(d, title, (40, 180, 255))
    return im


def draw_cluster_overlay(
    facade: Image.Image,
    boxes: list[list[int]],
    labels: np.ndarray,
    title: str,
) -> Image.Image:
    palette = [
        (230, 70, 70),
        (70, 160, 230),
        (70, 200, 120),
        (230, 180, 50),
        (180, 90, 220),
    ]
    im = facade.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    for box, lab in zip(boxes, labels.tolist()):
        col = palette[int(lab) % len(palette)]
        d.rectangle(box, outline=col, width=4)
    _banner(d, title, (240, 240, 240))
    return im


def draw_vote(types: list[dict], out_dir: Path, note: str) -> Image.Image:
    cell = 140
    rows = max(1, len(types))
    im = Image.new("RGB", (1100, 48 + rows * (cell + 16)), (18, 18, 18))
    d = ImageDraw.Draw(im)
    d.text((12, 10), f"4. Majority vote + balcony IR  {note}", fill=(240, 240, 240), font=_font(18))
    font = _font(13)
    for i, t in enumerate(types):
        y = 44 + i * (cell + 16)
        tid = int(t["type_id"])
        d.rectangle([8, y, 8 + cell, y + cell], outline=(230, 70, 70), width=3)
        p = out_dir / t.get("exemplar_asset", "")
        if p.is_file():
            crop = Image.open(p).convert("RGB")
            crop.thumbnail((cell - 8, cell - 8), Image.Resampling.LANCZOS)
            im.paste(crop, (12, y + 4))
        ir = t.get("structure_ir") or {}
        vote = t.get("structure_vote") or {}
        toks = " ".join(str(x) for x in (t.get("structure_tokens") or [])[:16])
        lines = [
            f"T{tid}  n={t.get('n_instances')}  structure={ir.get('structure')}  "
            f"enc={ir.get('enclosure')}",
            f"vote {vote.get('winner_count', 0)}/{vote.get('n_valid', 0)} "
            f"unique={vote.get('n_unique', 0)}",
            toks[:90],
        ]
        for li, line in enumerate(lines):
            d.text((8 + cell + 16, y + 8 + li * 22), line, fill=(220, 220, 220), font=font)
    return im


def save(im: Image.Image, path: Path, *, force: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if force and path.is_file():
        path.unlink()
    im.save(path)
    print(f"  -> {path}")
    return path
