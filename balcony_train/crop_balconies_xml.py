"""Crop balcony regions from CMP label PNGs.

The Extended ``.png`` is the label map. Balcony pixels are palette index 7
(RGB 170, 255, 85). Each connected region is cropped from the matching ``.jpg``.
XML boxes are not used: their x/y axes are swapped relative to the photo, so a
box read as written lands on the wrong part of the facade. Photos with no
balcony pixels are recorded in ``none.jsonl``.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/crop_balconies_xml.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.labels import ensure_crop_dirs  # noqa: E402
from balcony_train.paths import (  # noqa: E402
    EXT_PHOTO_DIR,
    EXT_XML_CROPS_DIR,
    EXT_XML_MANIFEST,
    EXT_XML_NONE,
)

_OBJECT = re.compile(r"<object>(.*?)</object>", re.IGNORECASE | re.DOTALL)
_LABELNAME = re.compile(r"<labelname>\s*([^<]+?)\s*</labelname>", re.IGNORECASE)
_X = re.compile(r"<x>\s*([^<]+?)\s*</x>", re.IGNORECASE)
_Y = re.compile(r"<y>\s*([^<]+?)\s*</y>", re.IGNORECASE)

# CMP label map: index 7 is balcony (light green, 170, 255, 85).
BALCONY_INDEX = 7


def clamp_box_xyxy(
    box: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(v) for v in box]
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return x0, y0, x1, y1


def parse_balcony_norm_boxes(xml_text: str) -> list[tuple[float, float, float, float]]:
    """Normalized (x0, y0, x1, y1) for each ``balcony`` object, top then left."""
    text = xml_text.replace("\r\n", "\n").replace("\r", "\n")
    boxes: list[tuple[float, float, float, float]] = []
    for block in _OBJECT.findall(text):
        name_m = _LABELNAME.search(block)
        if name_m is None or name_m.group(1).strip().lower() != "balcony":
            continue
        xs = [float(v) for v in _X.findall(block)]
        ys = [float(v) for v in _Y.findall(block)]
        if len(xs) < 2 or len(ys) < 2:
            continue
        x0, x1 = sorted((xs[0], xs[1]))
        y0, y1 = sorted((ys[0], ys[1]))
        if x1 <= x0 or y1 <= y0:
            continue
        boxes.append((x0, y0, x1, y1))
    boxes.sort(key=lambda b: (b[1], b[0], b[3], b[2]))
    return boxes


def norm_box_to_pixels(
    box: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    pad_frac: float = 0.12,
) -> tuple[int, int, int, int]:
    """Pixel box, expanded by ``pad_frac`` of the box size, clamped to the image."""
    x0, y0, x1, y1 = box
    left = x0 * width
    top = y0 * height
    right = x1 * width
    bottom = y1 * height
    pad_x = pad_frac * (right - left)
    pad_y = pad_frac * (bottom - top)
    return clamp_box_xyxy(
        (
            int(round(left - pad_x)),
            int(round(top - pad_y)),
            int(round(right + pad_x)),
            int(round(bottom + pad_y)),
        ),
        width=width,
        height=height,
    )


def balcony_pixel_boxes(label_image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Tight boxes of 8-connected balcony regions, top then left.

    ``label_image`` is a CMP palette PNG. Index 7 is the balcony class.
    """
    if label_image.mode != "P":
        rgb = np.asarray(label_image.convert("RGB"))
        mask = (
            (rgb[:, :, 0] == 170) & (rgb[:, :, 1] == 255) & (rgb[:, :, 2] == 85)
        )
    else:
        mask = np.asarray(label_image) == BALCONY_INDEX
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(1, int(n) + 1):
        ys, xs = np.nonzero(labeled == i)
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    boxes.sort(key=lambda b: (b[1], b[0], b[3], b[2]))
    return boxes


def scale_box(
    box: tuple[int, int, int, int],
    *,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> tuple[int, int, int, int]:
    """Map a label-map box onto a photo of a different size."""
    if (src_w, src_h) == (dst_w, dst_h):
        return box
    x0, y0, x1, y1 = box
    return (
        int(round(x0 * dst_w / src_w)),
        int(round(y0 * dst_h / src_h)),
        int(round(x1 * dst_w / src_w)),
        int(round(y1 * dst_h / src_h)),
    )


def crop_output_name(stem: str, index: int) -> str:
    return f"{stem}_bal_{index:03d}.png"


def list_jpg_photos(image_dir: Path) -> list[Path]:
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        return []
    return sorted(p for p in image_dir.glob("*.jpg") if p.is_file())


def crop_xml_balconies(
    image_dir: Path,
    *,
    xml_dir: Path | None = None,
    out_dir: Path,
    manifest_path: Path,
    none_path: Path,
    pad_frac: float = 0.12,
    min_side: int = 8,
) -> dict[str, int]:
    """Write one crop per balcony region in the label PNG, plus manifest and none lists.

    ``out_dir`` is the unlabeled crop folder. Existing crop names are overwritten
    so a re-run keeps the same stem order. ``xml_dir`` is ignored.
    """
    image_dir = Path(image_dir)
    _ = xml_dir
    out_dir = Path(out_dir)
    ensure_crop_dirs(out_dir.parent if out_dir.name == "unlabeled" else out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    photos = list_jpg_photos(image_dir)
    manifest: list[dict] = []
    none_rows: list[dict] = []
    n_missing_png = 0
    n_tiny = 0

    search_root = out_dir.parent if out_dir.name == "unlabeled" else out_dir
    folders = [p for p in search_root.iterdir() if p.is_dir()] if search_root.is_dir() else []
    for photo in photos:
        for folder in folders:
            for old in folder.glob(f"{photo.stem}_bal_*.png"):
                old.unlink()
        label_path = image_dir / f"{photo.stem}.png"
        if not label_path.is_file():
            n_missing_png += 1
            none_rows.append(
                {"photo": photo.name, "n_balcony": 0, "missing_png": True}
            )
            continue
        label = Image.open(label_path)
        raw_boxes = balcony_pixel_boxes(label)
        if not raw_boxes:
            none_rows.append({"photo": photo.name, "n_balcony": 0})
            continue
        image = Image.open(photo).convert("RGB")
        width, height = image.size
        src_w, src_h = label.size
        kept = 0
        for raw in raw_boxes:
            scaled = scale_box(raw, src_w=src_w, src_h=src_h, dst_w=width, dst_h=height)
            bw = max(1, scaled[2] - scaled[0])
            bh = max(1, scaled[3] - scaled[1])
            pad_x = pad_frac * bw
            pad_y = pad_frac * bh
            pix = clamp_box_xyxy(
                (
                    int(round(scaled[0] - pad_x)),
                    int(round(scaled[1] - pad_y)),
                    int(round(scaled[2] + pad_x)),
                    int(round(scaled[3] + pad_y)),
                ),
                width=width,
                height=height,
            )
            if pix[2] - pix[0] < min_side or pix[3] - pix[1] < min_side:
                n_tiny += 1
                continue
            name = crop_output_name(photo.stem, kept)
            image.crop(pix).save(out_dir / name)
            manifest.append(
                {
                    "crop": name,
                    "photo": photo.name,
                    "label": label_path.name,
                    "box_xyxy": list(pix),
                    "mask_box_xyxy": list(scaled),
                }
            )
            kept += 1
        if kept == 0:
            none_rows.append({"photo": photo.name, "n_balcony": 0, "all_tiny": True})

    _write_jsonl(manifest_path, manifest)
    _write_jsonl(none_path, none_rows)
    return {
        "n_photos": len(photos),
        "n_with_balcony": len({row["photo"] for row in manifest}),
        "n_crops": len(manifest),
        "n_none": len(none_rows),
        "n_missing_xml": n_missing_png,
        "n_missing_png": n_missing_png,
        "n_tiny": n_tiny,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-dir", type=Path, default=EXT_PHOTO_DIR)
    ap.add_argument(
        "--xml-dir",
        type=Path,
        default=None,
        help="defaults to --image-dir (cmp_x####.xml next to the jpg)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=EXT_XML_CROPS_DIR / "unlabeled",
    )
    ap.add_argument("--manifest", type=Path, default=EXT_XML_MANIFEST)
    ap.add_argument("--none", type=Path, default=EXT_XML_NONE)
    ap.add_argument("--pad-frac", type=float, default=0.12)
    ap.add_argument("--min-side", type=int, default=8)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    stats = crop_xml_balconies(
        args.image_dir,
        xml_dir=args.xml_dir,
        out_dir=args.out_dir,
        manifest_path=args.manifest,
        none_path=args.none,
        pad_frac=args.pad_frac,
        min_side=args.min_side,
    )
    print(
        f"photos={stats['n_photos']}  with_balcony={stats['n_with_balcony']}  "
        f"crops={stats['n_crops']}  none={stats['n_none']}  "
        f"missing_xml={stats['n_missing_xml']}  tiny={stats['n_tiny']}",
        flush=True,
    )
    print(f"crops={args.out_dir}", flush=True)
    print(f"manifest={args.manifest}", flush=True)
    print(f"none={args.none}", flush=True)


if __name__ == "__main__":
    main()
