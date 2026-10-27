"""Queues and distribution for CMP-XML balcony crops.

Open UI: crops not yet sent to the enclosed queue.
Enclosed UI: crops marked ``enclosed`` or ``half_enclosed``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from balcony_train.labels import (
    CLASSES,
    ENCLOSURE_SKINS,
    ENCLOSURES,
    FLOOR_SHAPES,
    LABEL_STRUCTURES,
    MATERIALS,
)
from balcony_train.label_store import (
    _place_in_unlabeled,
    find_image,
    iter_crop_images,
    save_annotation,
    write_labels_jsonl,
)


def _complete_open(rec: dict[str, Any] | None) -> bool:
    if not rec or rec.get("enclosure") != "open":
        return False
    kind = rec.get("kind")
    if kind not in CLASSES:
        return False
    if rec.get("floor_shape") not in FLOOR_SHAPES:
        return False
    if kind == "open_work" and rec.get("material") not in MATERIALS:
        return False
    return True


def _complete_enclosed(rec: dict[str, Any] | None) -> bool:
    if not rec:
        return False
    enc = rec.get("enclosure")
    if rec.get("floor_shape") not in FLOOR_SHAPES:
        return False
    if enc == "enclosed":
        return rec.get("skin") in ENCLOSURE_SKINS and rec.get("kind") is None
    if enc == "half_enclosed":
        kind = rec.get("kind")
        if kind not in CLASSES:
            return False
        if kind == "open_work" and rec.get("material") not in MATERIALS:
            return False
        return True
    return False


def open_queue(
    crops_dir: Path,
    labels: dict[str, dict[str, Any]],
    *,
    unlabeled_only: bool = True,
) -> list[str]:
    """Stems still labeled as open, or not labeled yet."""
    images = {p.stem: p for p in iter_crop_images(crops_dir)}
    todo: list[str] = []
    done: list[str] = []
    for stem in sorted(images.keys()):
        rec = labels.get(stem)
        enc = rec.get("enclosure") if rec else None
        if enc in ("enclosed", "half_enclosed"):
            continue
        if _complete_open(rec):
            done.append(stem)
        else:
            todo.append(stem)
    if unlabeled_only:
        return todo
    return todo + done


def enclosed_queue(
    crops_dir: Path,
    labels: dict[str, dict[str, Any]],
    *,
    unlabeled_only: bool = True,
) -> list[str]:
    """Stems marked enclosed or half_enclosed from the open UI (key E)."""
    images = {p.stem: p for p in iter_crop_images(crops_dir)}
    todo: list[str] = []
    done: list[str] = []
    for stem in sorted(images.keys()):
        rec = labels.get(stem)
        enc = rec.get("enclosure") if rec else None
        if enc not in ("enclosed", "half_enclosed"):
            continue
        if _complete_enclosed(rec):
            done.append(stem)
        else:
            todo.append(stem)
    if unlabeled_only:
        return todo
    return todo + done


def send_to_enclosed(
    *,
    crops_dir: Path,
    jsonl_path: Path,
    image_path: Path,
    labels: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Park a crop for the enclosed UI. Clears open-balcony fields."""
    return save_annotation(
        crops_dir=crops_dir,
        jsonl_path=jsonl_path,
        image_path=image_path,
        kind=None,
        material=None,
        floor_shape=None,
        enclosure="enclosed",
        skin=None,
        labels=labels,
    )


def send_back_to_open(
    *,
    crops_dir: Path,
    jsonl_path: Path,
    image_path: Path,
    labels: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Drop the enclosed mark so the open UI shows the crop again."""
    placed = _place_in_unlabeled(crops_dir, image_path)
    labels.pop(placed.stem, None)
    write_labels_jsonl(jsonl_path, labels)
    return labels


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    by_crop: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return by_crop
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            crop = str(rec.get("crop") or "")
            if crop:
                by_crop[Path(crop).stem] = rec
    return by_crop


def load_none_count(path: Path) -> int:
    path = Path(path)
    if not path.is_file():
        return 0
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def distribution_counts(
    labels: dict[str, dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Instance counts for finished labels. Incomplete rows are omitted."""
    counts = {
        "enclosure": {name: 0 for name in ENCLOSURES},
        "floor_shape": {name: 0 for name in FLOOR_SHAPES},
        "kind": {name: 0 for name in CLASSES},
        "material": {name: 0 for name in MATERIALS},
        "skin": {name: 0 for name in ENCLOSURE_SKINS},
        "structure": {name: 0 for name in LABEL_STRUCTURES},
    }
    for rec in labels.values():
        if rec.get("enclosure") == "open" and not _complete_open(rec):
            continue
        if rec.get("enclosure") in ("enclosed", "half_enclosed") and not _complete_enclosed(rec):
            continue
        if rec.get("enclosure") not in ENCLOSURES:
            continue
        enc = rec["enclosure"]
        counts["enclosure"][enc] += 1
        floor = rec.get("floor_shape")
        if floor in FLOOR_SHAPES:
            counts["floor_shape"][floor] += 1
        kind = rec.get("kind")
        if kind in CLASSES:
            counts["kind"][kind] += 1
        mat = rec.get("material")
        if kind == "open_work" and mat in MATERIALS:
            counts["material"][mat] += 1
        skin = rec.get("skin")
        if enc == "enclosed" and skin in ENCLOSURE_SKINS:
            counts["skin"][skin] += 1
        if enc == "open":
            structure = rec.get("structure") or "projecting"
            if structure in LABEL_STRUCTURES:
                counts["structure"][structure] += 1
    return counts


def facade_stem_from_crop(crop_stem: str) -> str:
    marker = "_bal_"
    if marker in crop_stem:
        return crop_stem.rsplit(marker, 1)[0]
    return crop_stem


def find_crop_image(crops_dir: Path, stem: str) -> Path | None:
    return find_image(crops_dir, stem)
