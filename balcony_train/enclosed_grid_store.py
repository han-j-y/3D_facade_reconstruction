"""Ground truth for enclosed volumes as window-grid cells.

A saved photo with an empty cell list has no enclosure. A photo that is not
in the file is still unlabeled. Cells are ``(floor_id, bay_id)`` from the
window facade DSL.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_PIPE = ROOT / "balcony_pipeline"
if str(_PIPE) not in sys.path:
    sys.path.insert(0, str(_PIPE))

from snap import _bands_from_windows_dsl  # noqa: E402

DEFAULT_PHOTO_DIR = ROOT / "data" / "test for enclosure"
DEFAULT_DSL_DIR = ROOT / "runs" / "batch"
DEFAULT_LABELS_JSONL = ROOT / "runs" / "enclosed_grid" / "labels.jsonl"
DEFAULT_EVAL_PHOTO_DIR = ROOT / "runs" / "balcony_clf" / "crops_ext_eval" / "Before Crop"
DEFAULT_EVAL_CROP_LABELS = ROOT / "runs" / "balcony_clf" / "labels_ext_eval_balanced.jsonl"
DEFAULT_EVAL_LABELS_JSONL = ROOT / "runs" / "enclosed_grid" / "labels_eval.jsonl"

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
Cell = tuple[int, int]
Bands = dict[int, tuple[int, int]]


def normalize_cells(raw: Any) -> list[Cell]:
    """Unique ``(floor, bay)`` pairs, sorted by floor then bay."""
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"cells must be a list, got {type(raw).__name__}")
    seen: set[Cell] = set()
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ValueError(f"cell must be [floor, bay], got {item!r}")
        seen.add((int(item[0]), int(item[1])))
    return sorted(seen)


def load_labels(path: Path) -> dict[str, list[Cell]]:
    """Stem → cells. Missing file is empty. An empty ``cells`` list is kept."""
    path = Path(path)
    if not path.is_file():
        return {}
    out: dict[str, list[Cell]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        stem = str(rec.get("stem") or "").strip()
        if not stem:
            raise ValueError(f"label row missing stem: {line}")
        out[stem] = normalize_cells(rec.get("cells"))
    return out


def write_labels(path: Path, labels: dict[str, list[Cell]]) -> None:
    """Rewrite the jsonl. Every key is a saved photo, including empty cells."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for stem in sorted(labels):
        cells = normalize_cells(labels[stem])
        lines.append(
            json.dumps(
                {"stem": stem, "cells": [list(cell) for cell in cells]},
                ensure_ascii=False,
            )
        )
    text = ("\n".join(lines) + "\n") if lines else ""
    path.write_text(text, encoding="utf-8")


def find_photo(photo_dir: Path, stem: str) -> Path | None:
    photo_dir = Path(photo_dir)
    for suffix in IMAGE_SUFFIXES:
        path = photo_dir / f"{stem}{suffix}"
        if path.is_file():
            return path
    return None


def facade_stems_from_crop_labels(path: Path) -> list[str]:
    """Facade ids from balcony-crop labels (``cmp_x0002_bal_002`` → ``cmp_x0002``)."""
    path = Path(path)
    if not path.is_file():
        return []
    stems: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        name = Path(str(rec.get("path") or "")).stem
        facade = name.split("_bal_")[0].strip()
        if facade and facade not in seen:
            seen.add(facade)
            stems.append(facade)
    return stems


def list_grid_stems(
    photo_dir: Path,
    dsl_dir: Path,
    *,
    stems: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return ``(ready, missing_dsl)`` for photos in ``photo_dir``.

    Ready stems have ``{dsl_dir}/{stem}/facade_dsl.json``.
    ``stems`` limits the queue (eval facades) and keeps that order.
    """
    photo_dir = Path(photo_dir)
    dsl_dir = Path(dsl_dir)
    ready: list[str] = []
    missing: list[str] = []
    if stems is None:
        if not photo_dir.is_dir():
            return ready, missing
        stems = sorted(
            {
                path.stem
                for path in photo_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            }
        )
    for stem in stems:
        has_photo = find_photo(photo_dir, stem) is not None
        has_dsl = (dsl_dir / stem / "facade_dsl.json").is_file()
        if has_photo and has_dsl:
            ready.append(stem)
        else:
            missing.append(stem)
    return ready, missing


def load_grid(dsl_path: Path) -> tuple[Bands, Bands]:
    """Floor bands and bay bands from a window ``facade_dsl.json``."""
    dsl = json.loads(Path(dsl_path).read_text(encoding="utf-8"))
    floor_y, bay_x = _bands_from_windows_dsl(dsl)
    return floor_y, bay_x


def _band_id(value: float, bands: Bands) -> int | None:
    hits = [bid for bid, (a0, a1) in bands.items() if float(a0) <= value <= float(a1)]
    if not hits:
        return None
    if len(hits) == 1:
        return int(hits[0])
    for bid in hits:
        if float(bands[bid][0]) == value:
            return int(bid)
    return int(sorted(hits)[0])


def cell_at(
    x: float,
    y: float,
    floor_y: Bands,
    bay_x: Bands,
) -> Cell | None:
    """Grid cell under a photo pixel, or None outside every band."""
    floor = _band_id(y, floor_y)
    bay = _band_id(x, bay_x)
    if floor is None or bay is None:
        return None
    return (floor, bay)
