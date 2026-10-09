"""Bulk-label FLUX.2 synth crops from filename prefixes (no Label UI).

Maps::

    flux2_masonry_*  -> open_work + masonry
    flux2_metal_*    -> open_work + metal
    flux2_surface_*  -> surface_panel
    flux2_solid_*    -> solid

    flux2_triangle_{masonry,metal}_*  -> open_work + that material + triangle
    flux2_triangle_surface_*          -> surface_panel + triangle
    flux2_triangle_solid_*            -> solid + triangle
    flux2_{hexagon,trapezoid}_*       -> same kinds, floor rectangle

All of the above are enclosure ``open``.

    flux2_half_enclosed_*  -> enclosure half_enclosed (kind unset)
    flux2_enclosed_*       -> enclosure enclosed (kind unset)

Moves matched files into ``crops/{kind}/`` and appends ``labels.jsonl`` via
``save_annotation`` (same as Label UI). Enclosure-only files stay in
``unlabeled/``; set their kind and floor in the Label UI.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe ^
      balcony_train/label_from_prefix.py ^
      --unlabeled-dir runs/balcony_clf/crops/unlabeled
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.label_store import merge_label_sources, save_annotation  # noqa: E402
from balcony_train.labels import LABELED_IMAGE_SUFFIXES, LEGACY_FLOOR_SHAPES  # noqa: E402
from balcony_train.paths import DEFAULT_CROPS_DIR, DEFAULT_LABELS_JSONL  # noqa: E402

# Longest-prefix-first matching (order matters if names ever overlap).
# Each row is (prefix, kind, material, floor_shape, enclosure).
_FLOOR_PLAN_SHAPES: tuple[str, ...] = ("triangle", "hexagon", "trapezoid")
_FLOOR_PLAN_KINDS: tuple[tuple[str, str, str | None], ...] = (
    ("masonry", "open_work", "masonry"),
    ("metal", "open_work", "metal"),
    ("surface", "surface_panel", None),
    ("solid", "solid", None),
)
PrefixLabel = tuple[str, str | None, str | None, str | None, str]

PREFIX_LABELS: tuple[PrefixLabel, ...] = (
    *(
        (f"flux2_{shape}_{token}_", kind, material, LEGACY_FLOOR_SHAPES.get(shape, shape), "open")
        for shape in _FLOOR_PLAN_SHAPES
        for token, kind, material in _FLOOR_PLAN_KINDS
    ),
    ("flux2_half_enclosed_", None, None, None, "half_enclosed"),
    ("flux2_enclosed_", None, None, None, "enclosed"),
    ("flux2_masonry_", "open_work", "masonry", None, "open"),
    ("flux2_metal_", "open_work", "metal", None, "open"),
    ("flux2_surface_", "surface_panel", None, None, "open"),
    ("flux2_solid_", "solid", None, None, "open"),
)


def match_prefix(name: str) -> PrefixLabel | None:
    """Return (prefix, kind, material, floor_shape, enclosure) for a known filename."""
    lower = name.lower()
    for row in PREFIX_LABELS:
        if lower.startswith(row[0]):
            return row
    return None


def iter_unlabeled_images(unlabeled_dir: Path) -> list[Path]:
    unlabeled_dir = Path(unlabeled_dir)
    if not unlabeled_dir.is_dir():
        return []
    paths: list[Path] = []
    for suffix in LABELED_IMAGE_SUFFIXES:
        paths.extend(unlabeled_dir.glob(f"*{suffix}"))
        paths.extend(unlabeled_dir.glob(f"*{suffix.upper()}"))
    return sorted({p.resolve() for p in paths if p.is_file()})


def label_from_prefixes(
    *,
    crops_dir: Path,
    unlabeled_dir: Path,
    jsonl_path: Path,
    dry_run: bool = False,
) -> dict[str, int]:
    """Label matching files under unlabeled_dir. Returns per-rule counts."""
    crops_dir = Path(crops_dir)
    unlabeled_dir = Path(unlabeled_dir)
    jsonl_path = Path(jsonl_path)

    counts = {row[0]: 0 for row in PREFIX_LABELS}
    counts["skipped_no_prefix"] = 0
    counts["labeled"] = 0

    images = iter_unlabeled_images(unlabeled_dir)
    if not images:
        print(f"no images in {unlabeled_dir}")
        return counts

    labels = None if dry_run else merge_label_sources(crops_dir, jsonl_path)

    for path in images:
        matched = match_prefix(path.name)
        if matched is None:
            counts["skipped_no_prefix"] += 1
            print(f"skip (unknown prefix): {path.name}")
            continue
        prefix, kind, material, floor_shape, enclosure = matched
        detail = " + ".join(str(v) for v in (kind, material, floor_shape, enclosure) if v)
        if dry_run:
            print(f"DRY {path.name} -> {detail}")
        else:
            assert labels is not None
            labels = save_annotation(
                crops_dir=crops_dir,
                jsonl_path=jsonl_path,
                image_path=path,
                kind=kind,
                material=material,
                floor_shape=floor_shape,
                enclosure=enclosure,
                labels=labels,
            )
            print(f"{path.name} -> {detail}")
        counts[prefix] += 1
        counts["labeled"] += 1

    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Bulk-label flux2_* crops from filename prefixes"
    )
    ap.add_argument(
        "--crops-dir",
        type=Path,
        default=DEFAULT_CROPS_DIR,
        help="crops root with unlabeled/ and kind folders",
    )
    ap.add_argument(
        "--unlabeled-dir",
        type=Path,
        default=None,
        help="source folder (default: <crops-dir>/unlabeled)",
    )
    ap.add_argument(
        "--labels-jsonl",
        type=Path,
        default=DEFAULT_LABELS_JSONL,
        help="labels.jsonl path",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned labels; do not move or write jsonl",
    )
    args = ap.parse_args(argv)

    crops_dir = Path(args.crops_dir)
    unlabeled_dir = (
        Path(args.unlabeled_dir)
        if args.unlabeled_dir is not None
        else crops_dir / "unlabeled"
    )

    stats = label_from_prefixes(
        crops_dir=crops_dir,
        unlabeled_dir=unlabeled_dir,
        jsonl_path=Path(args.labels_jsonl),
        dry_run=bool(args.dry_run),
    )
    print("---")
    for row in PREFIX_LABELS:
        print(f"{row[0]}*: {stats[row[0]]}")
    print(f"skipped_no_prefix: {stats['skipped_no_prefix']}")
    print(f"labeled: {stats['labeled']}")
    if args.dry_run:
        print("dry-run only; no files moved")
    else:
        print(f"labels -> {Path(args.labels_jsonl).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
