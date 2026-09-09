"""JSONL label store for railing kind (+ open_work material).

Each line::

    {"path": "relative/or/name.png", "kind": "open_work", "material": "masonry"}

``material`` is set only for ``kind=open_work``; otherwise ``null``.
Folder layout under ``crops/`` (``open_work/`` etc.) stays in sync for the
existing kind-only trainer.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from balcony_train.labels import (
    CLASSES,
    LABELED_IMAGE_SUFFIXES,
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
        merged[stem] = {
            "path": rec.get("path") or prev.get("path") or f"{stem}.png",
            "kind": kind,
            "material": material,
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
        if kind is None:
            continue
        material = normalize_material(rec.get("material"), kind=kind)
        lines.append(
            json.dumps(
                {
                    "path": str(rec.get("path") or f"{stem}.png"),
                    "kind": kind,
                    "material": material,
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


def queue_stems(
    crops_dir: Path,
    labels: dict[str, dict[str, Any]],
    *,
    unlabeled_only: bool = True,
    need_material: bool = True,
) -> list[str]:
    """Stems to show in the UI, unlabeled / incomplete first."""
    images = {p.stem: p for p in iter_crop_images(crops_dir)}
    todo: list[str] = []
    done: list[str] = []
    for stem in sorted(images.keys()):
        rec = labels.get(stem)
        kind = rec.get("kind") if rec else None
        material = rec.get("material") if rec else None
        incomplete = kind is None
        if need_material and kind == "open_work" and material is None:
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
    kind: str,
    material: str | None,
    labels: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Update JSONL + move file into kind folder. Returns full label map."""
    kind_n = normalize_kind(kind)
    if kind_n is None:
        raise ValueError("kind is required")
    mat_n = normalize_material(material, kind=kind_n)
    if kind_n == "open_work" and mat_n is None:
        raise ValueError("material required for open_work (metal or masonry)")

    crops_dir = Path(crops_dir)
    jsonl_path = Path(jsonl_path)
    placed = place_image_in_kind_folder(crops_dir, image_path, kind_n)
    records = labels if labels is not None else merge_label_sources(crops_dir, jsonl_path)
    records[placed.stem] = {
        "path": placed.name,
        "kind": kind_n,
        "material": mat_n,
    }
    write_labels_jsonl(jsonl_path, records)
    return records
