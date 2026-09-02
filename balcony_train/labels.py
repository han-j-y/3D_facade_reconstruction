"""Railing class names. No torch — used by collect.py and folder scans."""

from __future__ import annotations

from pathlib import Path

CLASSES: tuple[str, str] = ("baluster", "solid")

# Labeled crop image extensions (case-insensitive on Windows).
LABELED_IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")


def class_index(name: str) -> int:
    k = str(name).strip().lower()
    if k not in CLASSES:
        raise ValueError(f"unknown railing class {name!r}; expected {CLASSES}")
    return CLASSES.index(k)


def iter_labeled_samples(crops_dir: Path) -> list[tuple[Path, int]]:
    """Image paths under solid/ and baluster/ with class indices."""
    rows: list[tuple[Path, int]] = []
    for name in CLASSES:
        folder = Path(crops_dir) / name
        if not folder.is_dir():
            continue
        idx = class_index(name)
        paths: list[Path] = []
        for suffix in LABELED_IMAGE_SUFFIXES:
            paths.extend(folder.glob(f"*{suffix}"))
            paths.extend(folder.glob(f"*{suffix.upper()}"))
        for path in sorted(set(paths)):
            if path.is_file():
                rows.append((path, idx))
    return rows


def class_counts(samples: list[tuple[Path, int]]) -> dict[str, int]:
    counts = {name: 0 for name in CLASSES}
    for _, idx in samples:
        counts[CLASSES[idx]] += 1
    return counts


def ensure_crop_dirs(crops_dir: Path) -> None:
    crops_dir = Path(crops_dir)
    for name in (*CLASSES, "unlabeled"):
        (crops_dir / name).mkdir(parents=True, exist_ok=True)
