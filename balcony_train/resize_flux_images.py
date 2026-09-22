"""Crop ``flux2_*`` training images to a real-crop-shaped horizontal band.

Originals stay in ``crops/_Flux Original/{subfolder}/``. Each training copy
is rewritten from that original: a centered horizontal band at the real
median aspect (120×44), then a JPEG quality-60 round trip, saved back under
the same filename. A second run does not touch the archive.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/resize_flux_images.py
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.labels import CLASSES, LABELED_IMAGE_SUFFIXES  # noqa: E402
from balcony_train.paths import DEFAULT_CROPS_DIR  # noqa: E402

DEFAULT_WIDTH = 120
DEFAULT_HEIGHT = 44
DEFAULT_JPEG_QUALITY = 60
FLUX_PREFIX = "flux2_"
FLUX_ORIGINAL_DIRNAME = "_Flux Original"
SOURCE_SUBDIRS: tuple[str, ...] = ("unlabeled", *CLASSES)


def iter_flux_images(crops_dir: Path, *, prefix: str = FLUX_PREFIX) -> list[Path]:
    """``flux2_*`` images one level under unlabeled/ and kind folders."""
    crops_dir = Path(crops_dir)
    needle = str(prefix).lower()
    found: list[Path] = []
    for sub in SOURCE_SUBDIRS:
        folder = crops_dir / sub
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in LABELED_IMAGE_SUFFIXES:
                continue
            if path.name.lower().startswith(needle):
                found.append(path)
    return found


def horizontal_band_box(
    size: tuple[int, int],
    target: tuple[int, int] = (DEFAULT_WIDTH, DEFAULT_HEIGHT),
) -> tuple[int, int, int, int]:
    """Centered crop with the same aspect as ``target`` (left, top, right, bottom).

    Images taller than the target aspect lose top and bottom. Images wider
    than it lose the left and right edges.
    """
    width, height = int(size[0]), int(size[1])
    target_w, target_h = int(target[0]), int(target[1])
    if width < 1 or height < 1 or target_w < 1 or target_h < 1:
        raise ValueError(f"invalid size {size} or target {target}")
    aspect = target_w / target_h
    if (width / height) >= aspect:
        crop_h = height
        crop_w = min(width, max(1, int(round(height * aspect))))
        left = (width - crop_w) // 2
        top = 0
    else:
        crop_w = width
        crop_h = min(height, max(1, int(round(width / aspect))))
        left = 0
        top = (height - crop_h) // 2
    return left, top, left + crop_w, top + crop_h


def write_matched_crop(
    src: Path,
    dest: Path,
    *,
    target: tuple[int, int],
    jpeg_quality: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Crop a horizontal band, resize to ``target``, JPEG round trip.

    Returns (source size, output size).
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        source_size = image.size
        rgb = image.convert("RGB")
    band = rgb.crop(horizontal_band_box(source_size, target))
    resized = band.resize((int(target[0]), int(target[1])), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=int(jpeg_quality))
    buffer.seek(0)
    with Image.open(buffer) as jpeg:
        jpeg.convert("RGB").save(dest)
    return source_size, (int(target[0]), int(target[1]))


def replace_flux_with_resized(
    crops_dir: Path,
    *,
    target: tuple[int, int] = (DEFAULT_WIDTH, DEFAULT_HEIGHT),
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    prefix: str = FLUX_PREFIX,
    archive_dirname: str = FLUX_ORIGINAL_DIRNAME,
) -> dict[str, int | str]:
    """Archive each ``flux2_*`` file, then write a 120×44 band in its place.

    The archive is ``crops_dir / archive_dirname / {subfolder} / {filename}``.
    If that original is already archived, it is left untouched and the
    training copy is rewritten from it.
    """
    crops_dir = Path(crops_dir)
    archive_root = crops_dir / archive_dirname
    saved = 0
    for src in iter_flux_images(crops_dir, prefix=prefix):
        sub = src.parent.name
        archived = archive_root / sub / src.name
        if not archived.is_file():
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(archived))
        before, after = write_matched_crop(
            archived,
            src,
            target=target,
            jpeg_quality=jpeg_quality,
        )
        saved += 1
        print(
            f"{sub}/{src.name}  {before[0]}x{before[1]} -> {after[0]}x{after[1]}",
            flush=True,
        )
    print(
        f"resized {saved} images in {crops_dir.resolve()}  "
        f"originals -> {archive_root.resolve()}",
        flush=True,
    )
    return {"saved": saved, "archive": str(archive_root.resolve())}


def match_rgb(
    pixels: np.ndarray,
    source_mean: np.ndarray,
    source_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> np.ndarray:
    """Shift each RGB channel onto the target mean and standard deviation."""
    scale = np.maximum(np.asarray(source_std, dtype=np.float32), 1e-3)
    matched = (pixels - source_mean) / scale * target_std + target_mean
    return np.clip(matched, 0, 255).astype(np.uint8)


def _resize_rgb(path: Path, target: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize(
            (int(target[0]), int(target[1])),
            Image.Resampling.LANCZOS,
        )
    return np.asarray(rgb, dtype=np.float32)


def real_kind_stats(
    crops_dir: Path,
    kind: str,
    target: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean and std of real ``cmp*`` images in ``crops_dir/kind``, at ``target`` size."""
    folder = Path(crops_dir) / kind
    arrays: list[np.ndarray] = []
    if folder.is_dir():
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in LABELED_IMAGE_SUFFIXES:
                continue
            if not path.name.lower().startswith("cmp"):
                continue
            arrays.append(_resize_rgb(path, target))
    if not arrays:
        raise SystemExit(f"no cmp* images in {folder}")
    stacked = np.stack(arrays, axis=0)
    return (
        stacked.mean(axis=(0, 1, 2)),
        stacked.std(axis=(0, 1, 2)),
        len(arrays),
    )


def _save_jpeg_png(pixels: np.ndarray, dest: Path, jpeg_quality: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buffer, format="JPEG", quality=int(jpeg_quality))
    buffer.seek(0)
    with Image.open(buffer) as jpeg:
        jpeg.convert("RGB").save(dest)


def match_flux_kind_to_real(
    crops_dir: Path,
    kind: str,
    *,
    target: tuple[int, int] = (DEFAULT_WIDTH, DEFAULT_HEIGHT),
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    prefix: str = FLUX_PREFIX,
    archive_dirname: str = FLUX_ORIGINAL_DIRNAME,
) -> dict[str, int | str]:
    """Rewrite ``flux2_*`` in one kind folder to that kind's real cmp brightness.

    Originals stay in ``_Flux Original``. Each training copy is a 120×44 band
    whose RGB mean and standard deviation match the real ``cmp*`` images.
    """
    crops_dir = Path(crops_dir)
    real_mean, real_std, n_real = real_kind_stats(crops_dir, kind, target)
    print(
        f"real {kind} n={n_real}  mean={np.round(real_mean, 1).tolist()}  "
        f"std={np.round(real_std, 1).tolist()}",
        flush=True,
    )
    archive_root = crops_dir / archive_dirname
    saved = 0
    for src in iter_flux_images(crops_dir, prefix=prefix):
        if src.parent.name != kind:
            continue
        archived = archive_root / kind / src.name
        if not archived.is_file():
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(archived))
        with Image.open(archived) as image:
            source_size = image.size
            rgb = image.convert("RGB")
        band = rgb.crop(horizontal_band_box(source_size, target)).resize(
            (int(target[0]), int(target[1])),
            Image.Resampling.LANCZOS,
        )
        pixels = np.asarray(band, dtype=np.float32)
        matched = match_rgb(
            pixels,
            pixels.mean(axis=(0, 1)),
            pixels.std(axis=(0, 1)),
            real_mean,
            real_std,
        )
        _save_jpeg_png(matched, src, jpeg_quality)
        saved += 1
        print(
            f"{kind}/{src.name}  {source_size[0]}x{source_size[1]} -> "
            f"{target[0]}x{target[1]} matched",
            flush=True,
        )
    print(
        f"matched {saved} {kind} images in {crops_dir.resolve()}  "
        f"originals -> {archive_root.resolve()}",
        flush=True,
    )
    return {"saved": saved, "archive": str(archive_root.resolve()), "real_n": n_real}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS_DIR)
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help="JPEG round-trip quality before saving PNG (default: 60)",
    )
    ap.add_argument(
        "--prefix",
        default=FLUX_PREFIX,
        help="only filenames starting with this (default: flux2_)",
    )
    ap.add_argument(
        "--match-kind",
        default=None,
        help=(
            "rewrite flux2_* in this kind folder so RGB mean/std match its cmp* images "
            "(e.g. surface_panel)"
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    target = (int(args.width), int(args.height))
    if args.match_kind:
        match_flux_kind_to_real(
            args.crops_dir,
            str(args.match_kind),
            target=target,
            jpeg_quality=int(args.jpeg_quality),
            prefix=str(args.prefix),
        )
        return
    replace_flux_with_resized(
        args.crops_dir,
        target=target,
        jpeg_quality=int(args.jpeg_quality),
        prefix=str(args.prefix),
    )


if __name__ == "__main__":
    main()
