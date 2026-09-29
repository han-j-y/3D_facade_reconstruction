"""Railing class names. No torch — used by collect.py and folder scans."""

from __future__ import annotations

from pathlib import Path
from typing import Any

CLASSES: tuple[str, ...] = ("open_work", "surface_panel", "solid")
# Material only applies when kind == open_work (mesh proportions).
MATERIALS: tuple[str, ...] = ("metal", "masonry")
# Plan shape stored on labels and trained when set.
FLOOR_SHAPES: tuple[str, ...] = (
    "rectangle",
    "triangle",
    "circle",
    "hexagon",
    "trapezoid",
)
# Dataset / loss mask: material head ignored when not open_work.
MATERIAL_IGNORE_INDEX = -100
# Floor head ignored when the crop has no floor_shape label.
FLOOR_IGNORE_INDEX = -100

# Labeled crop image extensions (case-insensitive on Windows).
LABELED_IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")

# (path, kind_idx, material_idx, floor_idx). Ignore indexes skip that head.
MultitaskSample = tuple[Path, int, int, int]


def class_index(name: str) -> int:
    k = str(name).strip().lower()
    if k not in CLASSES:
        raise ValueError(f"unknown railing class {name!r}; expected {CLASSES}")
    return CLASSES.index(k)


def material_index(name: str) -> int:
    m = str(name).strip().lower()
    if m not in MATERIALS:
        raise ValueError(f"unknown railing material {name!r}; expected {MATERIALS}")
    return MATERIALS.index(m)


def floor_index(name: str) -> int:
    f = str(name).strip().lower()
    if f not in FLOOR_SHAPES:
        raise ValueError(f"unknown floor shape {name!r}; expected {FLOOR_SHAPES}")
    return FLOOR_SHAPES.index(f)


def sample_targets(row: tuple) -> tuple[int, int, int]:
    """Kind, material, floor indexes. Missing fields are ignore indexes."""
    kind_idx = int(row[1])
    mat_idx = int(row[2]) if len(row) >= 3 else MATERIAL_IGNORE_INDEX
    floor_idx = int(row[3]) if len(row) >= 4 else FLOOR_IGNORE_INDEX
    return kind_idx, mat_idx, floor_idx


def iter_labeled_samples(crops_dir: Path) -> list[tuple[Path, int]]:
    """Image paths under each CLASSES folder with kind indices (kind-only)."""
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


def iter_multitask_samples(
    crops_dir: Path,
    labels_jsonl: Path | None = None,
) -> list[MultitaskSample]:
    """Kind from folders; material and floor shape from labels.jsonl.

    open_work without material defaults to ``metal``. A missing floor_shape
    stays unlabeled (``FLOOR_IGNORE_INDEX``) and still trains kind.
    """
    from balcony_train.label_store import find_image, merge_label_sources
    from balcony_train.paths import DEFAULT_LABELS_JSONL

    crops_dir = Path(crops_dir)
    jsonl = Path(labels_jsonl) if labels_jsonl else DEFAULT_LABELS_JSONL
    merged = merge_label_sources(crops_dir, jsonl)
    rows: list[MultitaskSample] = []
    for stem, rec in sorted(merged.items()):
        kind = rec.get("kind")
        if kind not in CLASSES:
            continue
        path = find_image(crops_dir, stem)
        if path is None:
            continue
        kind_idx = class_index(kind)
        if kind == "open_work":
            mat = rec.get("material")
            if mat not in MATERIALS:
                mat = "metal"
            mat_idx = material_index(mat)
        else:
            mat_idx = MATERIAL_IGNORE_INDEX
        floor = rec.get("floor_shape")
        if floor in FLOOR_SHAPES:
            floor_idx = floor_index(str(floor))
        else:
            floor_idx = FLOOR_IGNORE_INDEX
        rows.append((path, kind_idx, mat_idx, floor_idx))
    return rows


def class_counts(samples: list[tuple[Any, int]] | list[MultitaskSample]) -> dict[str, int]:
    counts = {name: 0 for name in CLASSES}
    for row in samples:
        idx = int(row[1])
        counts[CLASSES[idx]] += 1
    return counts


def material_counts(samples: list[MultitaskSample] | list[tuple]) -> dict[str, int]:
    counts = {name: 0 for name in MATERIALS}
    for row in samples:
        kind_idx, mat_idx, _floor_idx = sample_targets(row)
        if CLASSES[kind_idx] != "open_work":
            continue
        if mat_idx < 0:
            continue
        counts[MATERIALS[mat_idx]] += 1
    return counts


def floor_counts(samples: list[MultitaskSample] | list[tuple]) -> dict[str, int]:
    counts = {name: 0 for name in FLOOR_SHAPES}
    for row in samples:
        _kind_idx, _mat_idx, floor_idx = sample_targets(row)
        if floor_idx < 0:
            continue
        counts[FLOOR_SHAPES[floor_idx]] += 1
    return counts


def strat_key(kind_idx: int, material_idx: int) -> str:
    """Stratification key: open_work|metal, open_work|masonry, or bare kind."""
    kind = CLASSES[int(kind_idx)]
    if kind == "open_work" and int(material_idx) >= 0:
        return f"{kind}|{MATERIALS[int(material_idx)]}"
    return kind


def ensure_crop_dirs(crops_dir: Path) -> None:
    crops_dir = Path(crops_dir)
    for name in (*CLASSES, "unlabeled"):
        (crops_dir / name).mkdir(parents=True, exist_ok=True)
