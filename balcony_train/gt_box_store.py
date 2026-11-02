"""Open-balcony ground-truth boxes for the test100 photos.

One JSONL line per reviewed photo, rewritten in place on every save::

    {"stem": "cmp_b0171", "image": "cmp_b0171.jpg", "width": 1024, "height": 1100,
     "boxes": [[x0, y0, x1, y1], ...], "ignore": [[x0, y0, x1, y1], ...],
     "prefill": "cmp_label", "prefill_boxes": 8,
     "edited": true, "updated": "2026-11-01T12:00:00"}

Boxes are integer photo pixels, x1/y1 exclusive. A photo counts as ground truth
only once it has a line here (the user has looked at it), even with no boxes.

``ignore`` regions are areas that cannot be judged (balconies cut by the image
border and too small, or hidden by glare). Scoring rule, fixed before scoring:
balcony boxes are matched first as usual; an unmatched prediction with at least
half of its own area inside one ignore region is dropped (neither TP nor FP);
ignore regions never count as false negatives.

Prefill (user, 2026-11-01): cmp_b / cmp_x photos start from the CMP label map
(palette 7, 8-connected component boxes, as in the box score); LSAA photos start
from the pipeline boxes (``filtered`` stage of the test100 SAM3 cache).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from PIL import Image

from balcony_train.crop_balconies_xml import balcony_pixel_boxes, scale_box

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHOTOS = ROOT / "runs" / "test100" / "photos"
DEFAULT_OUT = ROOT / "runs" / "test100" / "balcony_gt_boxes.jsonl"
DEFAULT_CACHE = ROOT / "runs" / "batch" / "_rows" / "test100_sam3_stage_boxes.json"
CMP_LABEL_DIRS = {"cmp_b": ROOT / "data" / "base", "cmp_x": ROOT / "data" / "extended"}

Box = list[int]


def is_cmp(stem: str) -> bool:
    return stem.startswith(("cmp_b", "cmp_x"))


def cmp_label_path(stem: str) -> Path:
    return CMP_LABEL_DIRS[stem[:5]] / f"{stem}.png"


def clean_box(box, width: int, height: int) -> Box | None:
    """Sorted, clipped integer box; None if under 2 px on a side."""
    x0, x1 = sorted((float(box[0]), float(box[2])))
    y0, y1 = sorted((float(box[1]), float(box[3])))
    x0, x1 = max(0, round(x0)), min(width, round(x1))
    y0, y1 = max(0, round(y0)), min(height, round(y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return [int(x0), int(y0), int(x1), int(y1)]


def prefill_boxes(stem: str, size: tuple[int, int], cache: dict) -> tuple[str, list[Box]]:
    """(source name, boxes) used before the user edits a photo."""
    w, h = size
    if is_cmp(stem):
        label = Image.open(cmp_label_path(stem))
        raw = [scale_box(b, src_w=label.size[0], src_h=label.size[1], dst_w=w, dst_h=h)
               for b in balcony_pixel_boxes(label)]
        source = "cmp_label"
    else:
        raw = cache.get(stem, {}).get("filtered", [])
        source = "pipeline"
    boxes = [b for b in (clean_box(r, w, h) for r in raw) if b is not None]
    return source, boxes


def load_gt(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["stem"]] = row
    return rows


def save_gt(path: Path, rows: dict[str, dict]) -> None:
    """Rewrite the whole file through a temp file, sorted by stem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for stem in sorted(rows):
            f.write(json.dumps(rows[stem], ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def make_row(stem: str, image: str, size: tuple[int, int], boxes: list[Box],
             prefill: str, prefill_list: list[Box], ignore: list[Box] | None = None) -> dict:
    def order(bs: list[Box]) -> list[Box]:
        return sorted(bs, key=lambda b: (b[1], b[0], b[3], b[2]))

    norm = order(boxes)
    ign = order(ignore or [])
    return {
        "stem": stem,
        "image": image,
        "width": size[0],
        "height": size[1],
        "boxes": norm,
        "ignore": ign,
        "prefill": prefill,
        "prefill_boxes": len(prefill_list),
        "edited": norm != order(prefill_list) or bool(ign),
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
