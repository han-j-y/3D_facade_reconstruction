"""Ground truth for enclosed volumes as window-grid cells.

A saved photo with an empty cell list has no enclosure. A photo that is not
in the file is still unlabeled. ``status`` ``grid_failure`` is a photo whose
window grid cannot be labeled, so it is not a cell label and not "no
enclosure". Cells are ``(floor_id, bay_id)`` from the window facade DSL.

``bay_id`` is a column of the union grid (``layout.bays``). With the per-floor
layout (``meta.floor_bays``) one floor bay spans one or more of these columns.

``ignore`` lists cells whose label is ambiguous from the photo (the grid line
cuts through the oriel, or the cell mixes it with a neighbouring building).
It does not change ``cells``: the main score counts every cell, and only a
supplementary score drops the ignore cells from TP, FP and FN.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_PIPE = ROOT / "balcony_pipeline"
if str(_PIPE) not in sys.path:
    sys.path.insert(0, str(_PIPE))

from balcony_train.labels import ENCLOSURE_SKINS, FLOOR_SHAPES  # noqa: E402
from snap import _bands_from_windows_dsl  # noqa: E402

DEFAULT_PHOTO_DIR = ROOT / "data" / "test for enclosure"
DEFAULT_DSL_DIR = ROOT / "runs" / "batch"
DEFAULT_LABELS_JSONL = ROOT / "runs" / "enclosed_grid" / "labels.jsonl"
# 133-photo test: labeled rows in labels_ext_all whose photo is in extended_selected.
# Grid-failure rows are omitted by load_labels. The previous 61-photo set is labels_eval.jsonl.
DEFAULT_EVAL_PHOTO_DIR = ROOT / "data" / "extended_selected"
DEFAULT_EVAL_CROP_LABELS = ROOT / "runs" / "enclosed_grid" / "labels_ext_all.jsonl"
DEFAULT_EVAL_LABELS_JSONL = ROOT / "runs" / "enclosed_grid" / "labels_ext_all.jsonl"

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
Cell = tuple[int, int]
Bands = dict[int, tuple[int, int]]
# floor id -> that floor's bays as union-column spans [c0, c1)
FloorBays = dict[int, list[tuple[int, int]]]
GRID_FAILURE = "grid_failure"


@dataclass
class GridRecord:
    """One saved facade. Shape and skin are set only when ``cells`` is non-empty.

    ``status`` is ``grid_failure`` when the window grid cannot be labeled.
    That row has no cells. It is not a photo with no enclosure.
    ``ignore`` holds the ambiguous cells, oriel or not.
    """

    cells: list[Cell]
    floor_shape: str | None = None
    skin: str | None = None
    status: str | None = None
    ignore: list[Cell] = field(default_factory=list)

    def copy(self) -> GridRecord:
        return GridRecord(list(self.cells), self.floor_shape, self.skin, self.status, list(self.ignore))


def _plan_name(raw: Any, allowed: tuple[str, ...], field: str) -> str | None:
    if raw is None:
        return None
    name = str(raw).strip().lower()
    if name in ("", "none", "null"):
        return None
    if name not in allowed:
        raise ValueError(f"unknown {field} {raw!r}; expected {allowed}")
    return name


def _status_name(raw: Any) -> str | None:
    if raw is None:
        return None
    name = str(raw).strip().lower()
    if name in ("", "none", "null"):
        return None
    if name != GRID_FAILURE:
        raise ValueError(f"unknown status {raw!r}; expected {GRID_FAILURE}")
    return name


def normalize_floor_shape(raw: Any) -> str | None:
    return _plan_name(raw, FLOOR_SHAPES, "floor_shape")


def normalize_skin(raw: Any) -> str | None:
    return _plan_name(raw, ENCLOSURE_SKINS, "skin")


def record_for_save(
    cells: Any,
    shape: Any = None,
    skin: Any = None,
    *,
    plan: bool = False,
    previous: GridRecord | None = None,
    ignore: Any = None,
) -> GridRecord:
    """Cells plus one floor shape and one skin, and the ignore cells.

    An empty cell list is a photo with no enclosure, so shape and skin are
    cleared. A previous grid failure stays a grid failure until cells or
    ignore cells are chosen. With ``plan`` off, an existing shape and skin are
    kept.
    """
    norm = normalize_cells(cells)
    skip = normalize_cells(ignore)
    if not norm:
        if previous is not None and previous.status == GRID_FAILURE and not skip:
            return GridRecord([], status=GRID_FAILURE)
        return GridRecord([], ignore=skip)
    if not plan:
        return GridRecord(
            norm,
            previous.floor_shape if previous is not None else None,
            previous.skin if previous is not None else None,
            ignore=skip,
        )
    floor = normalize_floor_shape(shape)
    shell = normalize_skin(skin)
    if floor is None or shell is None:
        raise ValueError("enclosed photo needs floor_shape and skin")
    return GridRecord(norm, floor, shell, ignore=skip)


def normalize_cells(raw: Any) -> list[Cell]:
    """Unique ``(floor, bay)`` pairs, sorted by floor then bay.

    A set is accepted. The label UI keeps the current selection as a set.
    """
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple, set)):
        raise ValueError(f"cells must be a list or set, got {type(raw).__name__}")
    seen: set[Cell] = set()
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ValueError(f"cell must be [floor, bay], got {item!r}")
        seen.add((int(item[0]), int(item[1])))
    return sorted(seen)


def load_records(path: Path) -> dict[str, GridRecord]:
    """Stem → cells, and floor shape / skin when the row has them.

    A missing file is empty. An empty ``cells`` list is kept and has no plan.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    out: dict[str, GridRecord] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        stem = str(rec.get("stem") or "").strip()
        if not stem:
            raise ValueError(f"label row missing stem: {line}")
        status = _status_name(rec.get("status"))
        cells = normalize_cells(rec.get("cells"))
        skip = normalize_cells(rec.get("ignore"))
        if status == GRID_FAILURE:
            if cells or skip:
                raise ValueError(f"{stem} is a grid failure, so it cannot list cells")
            out[stem] = GridRecord([], status=GRID_FAILURE)
            continue
        if not cells:
            out[stem] = GridRecord([], ignore=skip)
            continue
        out[stem] = GridRecord(
            cells,
            normalize_floor_shape(rec.get("floor_shape")),
            normalize_skin(rec.get("skin")),
            ignore=skip,
        )
    return out


def load_labels(path: Path) -> dict[str, list[Cell]]:
    """Stem → cells. Missing file is empty. An empty ``cells`` list is kept.

    Grid failures are omitted. They are not photos with no enclosure.
    """
    return {
        stem: list(rec.cells)
        for stem, rec in load_records(path).items()
        if rec.status != GRID_FAILURE
    }


def load_ignore(path: Path) -> dict[str, list[Cell]]:
    """Stem → ignore cells, for photos that have any."""
    return {stem: list(rec.ignore) for stem, rec in load_records(path).items() if rec.ignore}


def write_records(path: Path, records: dict[str, GridRecord]) -> None:
    """Rewrite the jsonl. Shape and skin are omitted when the photo has no cells."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for stem in sorted(records):
        rec = records[stem]
        if rec.status == GRID_FAILURE:
            if rec.cells:
                raise ValueError(f"{stem} is a grid failure, so it cannot list cells")
            lines.append(json.dumps({"stem": stem, "status": GRID_FAILURE}, ensure_ascii=False))
            continue
        cells = normalize_cells(rec.cells)
        row: dict[str, Any] = {"stem": stem, "cells": [list(cell) for cell in cells]}
        if cells:
            floor = normalize_floor_shape(rec.floor_shape)
            skin = normalize_skin(rec.skin)
            if floor is not None:
                row["floor_shape"] = floor
            if skin is not None:
                row["skin"] = skin
        skip = normalize_cells(rec.ignore)
        if skip:
            row["ignore"] = [list(cell) for cell in skip]
        lines.append(json.dumps(row, ensure_ascii=False))
    text = ("\n".join(lines) + "\n") if lines else ""
    path.write_text(text, encoding="utf-8")


def write_labels(path: Path, labels: dict[str, list[Cell]]) -> None:
    """Rewrite cells only. A list value does not keep a floor shape or skin."""
    write_records(
        path,
        {stem: GridRecord(normalize_cells(cells)) for stem, cells in labels.items()},
    )


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
    return _ordered_bands(floor_y), _ordered_bands(bay_x)


def _ordered_bands(bands: Bands) -> Bands:
    """Each band runs from the smaller edge to the larger one."""
    ordered: Bands = {}
    for bid, (a0, a1) in bands.items():
        lo, hi = (a0, a1) if a0 <= a1 else (a1, a0)
        if hi <= lo:
            continue
        ordered[int(bid)] = (lo, hi)
    return ordered


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


def load_floor_bays(dsl_path: Path, floor_y: Bands, bay_x: Bands) -> FloorBays:
    """Per-floor bays from ``meta.floor_bays``; one bay per column without it."""
    dsl = json.loads(Path(dsl_path).read_text(encoding="utf-8"))
    rows = (dsl.get("meta") or {}).get("floor_bays")
    singles = [(c, c + 1) for c in sorted(bay_x)]
    if not rows or len(rows) != len(floor_y):
        return {f: list(singles) for f in floor_y}
    out: FloorBays = {}
    for f, row in zip(sorted(floor_y), rows):
        spans = [(int(b["c0"]), int(b["c1"])) for b in row if int(b["c1"]) > int(b["c0"])]
        out[f] = spans or list(singles)
    return out


def bay_cells(cell: Cell, floor_bays: FloorBays) -> list[Cell]:
    """Every cell of the floor bay that holds ``cell``."""
    floor, col = cell
    for c0, c1 in floor_bays.get(floor, []):
        if c0 <= col < c1:
            return [(floor, c) for c in range(c0, c1)]
    return [cell]


def partial_bays(cells: set[Cell] | list[Cell], floor_bays: FloorBays) -> list[tuple[int, int, int]]:
    """Floor bays ``(floor, c0, c1)`` with some but not all of their cells labeled."""
    chosen = set(cells)
    out = []
    for floor, spans in sorted(floor_bays.items()):
        for c0, c1 in spans:
            hit = sum((floor, c) in chosen for c in range(c0, c1))
            if 0 < hit < c1 - c0:
                out.append((floor, c0, c1))
    return out
