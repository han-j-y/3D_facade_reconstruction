"""JSONL label store for railing kind (+ open_work material, floor shape).

Each line::

    {"path": "relative/or/name.png", "kind": "open_work", "material": "masonry",
     "floor_shape": "rectangle"}

``material`` is set only for ``kind=open_work``; otherwise ``null``.
``floor_shape`` is rectangle, triangle, or circle (legacy hexagon and
trapezoid load as rectangle).
``enclosure`` is open, half_enclosed, or enclosed (BDSL ``enclosure``).
Folder layout under ``crops/`` (``open_work/`` etc.) stays in sync for the
existing kind-only trainer. Records with an enclosure but no kind keep their
image in ``unlabeled/``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from balcony_train.labels import (
    CLASSES,
    ENCLOSURES,
    FLOOR_SHAPES,
    LABELED_IMAGE_SUFFIXES,
    LEGACY_FLOOR_SHAPES,
    MATERIALS,
    ensure_crop_dirs,
)


def normalize_kind(raw: Any) -> str | None:
    if raw is None:
        return None
    k = str(raw).strip().lower()
    if not k or k in ("null", "none", "unlabeled"):
        return None
    if k in ("baluster", "openwork", "lined_panel", "line_panel"):
        return "open_work"
    if k in CLASSES:
        return k
    raise ValueError(f"unknown kind {raw!r}; expected {CLASSES}")


def normalize_material(raw: Any, *, kind: str | None) -> str | None:
    if kind != "open_work":
        return None
    if raw is None:
        return None
    m = str(raw).strip().lower()
    if not m or m in ("null", "none"):
        return None
    if m not in MATERIALS:
        raise ValueError(f"unknown material {raw!r}; expected {MATERIALS}")
    return m


def normalize_floor_shape(raw: Any) -> str | None:
    """Plan label. Unknown names raise. Empty stays unset."""
    if raw is None:
        return None
    name = str(raw).strip().lower()
    if not name or name in ("null", "none"):
        return None
    name = LEGACY_FLOOR_SHAPES.get(name, name)
    if name not in FLOOR_SHAPES:
        raise ValueError(f"unknown floor shape {raw!r}; expected {FLOOR_SHAPES}")
    return name


def normalize_enclosure(raw: Any) -> str | None:
    if raw is None:
        return None
    name = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    if not name or name in ("null", "none"):
        return None
    if name == "half_enclosure":
        name = "half_enclosed"
    if name not in ENCLOSURES:
        raise ValueError(f"unknown enclosure {raw!r}; expected {ENCLOSURES}")
    return name


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in LABELED_IMAGE_SUFFIXES


def iter_crop_images(crops_dir: Path) -> list[Path]:
    """All images under unlabeled/ and kind folders."""
    crops_dir = Path(crops_dir)
    ensure_crop_dirs(crops_dir)
    out: list[Path] = []
    for sub in ("unlabeled", *CLASSES):
        folder = crops_dir / sub
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if _is_image(path):
                out.append(path.resolve())
    return out


def load_labels(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    """Map stem → record. Last line wins for duplicate stems."""
    jsonl_path = Path(jsonl_path)
    by_stem: dict[str, dict[str, Any]] = {}
    if not jsonl_path.is_file():
        return by_stem
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            stem = Path(str(rec.get("path") or "")).stem
            if not stem:
                continue
            kind = normalize_kind(rec.get("kind"))
            material = normalize_material(rec.get("material"), kind=kind)
            by_stem[stem] = {
                "path": str(rec.get("path") or f"{stem}.png"),
                "kind": kind,
                "material": material,
                "floor_shape": normalize_floor_shape(rec.get("floor_shape")),
                "enclosure": normalize_enclosure(rec.get("enclosure")),
            }
    return by_stem


def labels_from_folders(crops_dir: Path) -> dict[str, dict[str, Any]]:
    """Infer kind from folder name; material stays null."""
    crops_dir = Path(crops_dir)
    by_stem: dict[str, dict[str, Any]] = {}
    for kind in CLASSES:
        folder = crops_dir / kind
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if not _is_image(path):
                continue
            by_stem[path.stem] = {
                "path": path.name,
                "kind": kind,
                "material": None,
                "floor_shape": None,
                "enclosure": None,
            }
    return by_stem


def merge_label_sources(
    crops_dir: Path,
    jsonl_path: Path,
) -> dict[str, dict[str, Any]]:
    """Folder kinds as base; JSONL overrides (keeps material)."""
    merged = labels_from_folders(crops_dir)
    for stem, rec in load_labels(jsonl_path).items():
        prev = merged.get(stem) or {}
        kind = rec.get("kind") if rec.get("kind") is not None else prev.get("kind")
        material = normalize_material(rec.get("material"), kind=kind)
        if material is None and kind == "open_work":
            material = normalize_material(prev.get("material"), kind=kind)
        floor = rec.get("floor_shape")
        if floor is None:
            floor = normalize_floor_shape(prev.get("floor_shape"))
        enclosure = rec.get("enclosure") or prev.get("enclosure")
        merged[stem] = {
            "path": rec.get("path") or prev.get("path") or f"{stem}.png",
            "kind": kind,
            "material": material,
            "floor_shape": floor,
            "enclosure": enclosure,
        }
    return merged


def write_labels_jsonl(
    jsonl_path: Path,
    records: dict[str, dict[str, Any]],
) -> None:
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for stem in sorted(records.keys()):
        rec = records[stem]
        kind = normalize_kind(rec.get("kind"))
        enclosure = normalize_enclosure(rec.get("enclosure"))
        if kind is None and enclosure is None:
            continue
        material = normalize_material(rec.get("material"), kind=kind)
        lines.append(
            json.dumps(
                {
                    "path": str(rec.get("path") or f"{stem}.png"),
                    "kind": kind,
                    "material": material,
                    "floor_shape": normalize_floor_shape(rec.get("floor_shape")),
                    "enclosure": enclosure,
                },
                ensure_ascii=False,
            )
        )
    jsonl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def find_image(crops_dir: Path, stem: str) -> Path | None:
    crops_dir = Path(crops_dir)
    for sub in ("unlabeled", *CLASSES):
        folder = crops_dir / sub
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if path.stem == stem and _is_image(path):
                return path.resolve()
    return None


def place_image_in_kind_folder(
    crops_dir: Path,
    image_path: Path,
    kind: str,
) -> Path:
    """Move image into crops/{kind}/ (overwrite same name)."""
    crops_dir = Path(crops_dir)
    ensure_crop_dirs(crops_dir)
    kind = normalize_kind(kind)
    if kind is None:
        raise ValueError("kind required to place image")
    src = Path(image_path).resolve()
    dest = (crops_dir / kind / src.name).resolve()
    if src == dest:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.move(str(src), str(dest))
    return dest


def _place_in_unlabeled(crops_dir: Path, image_path: Path) -> Path:
    src = Path(image_path).resolve()
    dest = (Path(crops_dir) / "unlabeled" / src.name).resolve()
    if src == dest:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.move(str(src), str(dest))
    return dest


def kind_optional(enclosure: str | None) -> bool:
    """A fully enclosed balcony has no railing, so kind may stay unset."""
    return enclosure == "enclosed"


def queue_stems(
    crops_dir: Path,
    labels: dict[str, dict[str, Any]],
    *,
    unlabeled_only: bool = True,
    need_material: bool = True,
    need_floor_shape: bool = False,
    need_enclosure: bool = False,
) -> list[str]:
    """Stems to show in the UI, unlabeled / incomplete first."""
    images = {p.stem: p for p in iter_crop_images(crops_dir)}
    todo: list[str] = []
    done: list[str] = []
    for stem in sorted(images.keys()):
        rec = labels.get(stem)
        kind = rec.get("kind") if rec else None
        material = rec.get("material") if rec else None
        floor = rec.get("floor_shape") if rec else None
        enclosure = rec.get("enclosure") if rec else None
        incomplete = kind is None and not kind_optional(enclosure)
        if need_material and kind == "open_work" and material is None:
            incomplete = True
        if need_floor_shape and floor not in FLOOR_SHAPES:
            incomplete = True
        if need_enclosure and enclosure not in ENCLOSURES:
            incomplete = True
        if incomplete:
            todo.append(stem)
        elif not unlabeled_only:
            done.append(stem)
    return todo + done


def save_annotation(
    *,
    crops_dir: Path,
    jsonl_path: Path,
    image_path: Path,
    kind: str | None,
    material: str | None,
    floor_shape: str | None = None,
    enclosure: str | None = None,
    labels: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Update JSONL + move file into kind folder. Returns full label map.

    Without a kind (enclosure-only label) the image moves to ``unlabeled/``.
    """
    kind_n = normalize_kind(kind)
    enc_n = normalize_enclosure(enclosure)
    if kind_n is None and enc_n is None:
        raise ValueError("kind or enclosure is required")
    mat_n = normalize_material(material, kind=kind_n)
    if kind_n == "open_work" and mat_n is None:
        raise ValueError("material required for open_work (metal or masonry)")
    floor_n = normalize_floor_shape(floor_shape)

    crops_dir = Path(crops_dir)
    jsonl_path = Path(jsonl_path)
    if kind_n is not None:
        placed = place_image_in_kind_folder(crops_dir, image_path, kind_n)
    else:
        placed = _place_in_unlabeled(crops_dir, image_path)
    records = labels if labels is not None else merge_label_sources(crops_dir, jsonl_path)
    records[placed.stem] = {
        "path": placed.name,
        "kind": kind_n,
        "material": mat_n,
        "floor_shape": floor_n,
        "enclosure": enc_n,
    }
    write_labels_jsonl(jsonl_path, records)
    return records
