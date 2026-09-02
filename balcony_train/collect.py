"""Copy balcony unit crops into runs/balcony_clf/crops/unlabeled for labeling."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.labels import ensure_crop_dirs  # noqa: E402
from balcony_train.paths import BATCH_RUNS_DIR, DEFAULT_CROPS_DIR  # noqa: E402


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _dest_name(src: Path) -> str:
    """cmp_b0082_type_00_unit_000.png from a typical balcony asset path."""
    parts = src.parts
    unit = src.stem
    type_name = src.parent.name if src.parent.name.startswith("type_") else "type"
    facade = "facade"
    for i, p in enumerate(parts):
        if p == "balcony" and i > 0:
            facade = parts[i - 1]
            break
        if p == "batch" and i + 1 < len(parts):
            facade = parts[i + 1]
    return f"{facade}_{type_name}_{unit}.png"


def iter_source_crops(src_root: Path) -> list[Path]:
    """Balcony unit PNGs under a batch (or any) run tree."""
    root = Path(src_root)
    if not root.is_dir():
        return []
    found: list[Path] = []
    for path in root.rglob("unit_*.png"):
        parts = set(path.parts)
        if "balcony" not in parts:
            continue
        if path.name.startswith("exemplar"):
            continue
        found.append(path)
    return sorted(set(found))


def collect_crops(
    src_root: Path,
    crops_dir: Path,
    *,
    dest_subdir: str = "unlabeled",
) -> dict[str, int]:
    ensure_crop_dirs(crops_dir)
    dest_dir = Path(crops_dir) / dest_subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    seen_hash: set[str] = set()
    for existing in dest_dir.glob("*.png"):
        seen_hash.add(_file_md5(existing))
    for name in ("baluster", "solid"):
        labeled = Path(crops_dir) / name
        if labeled.is_dir():
            for existing in labeled.glob("*.png"):
                seen_hash.add(_file_md5(existing))

    n_copy = 0
    n_skip = 0
    for src in iter_source_crops(src_root):
        digest = _file_md5(src)
        if digest in seen_hash:
            n_skip += 1
            continue
        dest = dest_dir / _dest_name(src)
        if dest.is_file():
            dest = dest_dir / f"{dest.stem}_{digest[:8]}{dest.suffix}"
        shutil.copy2(src, dest)
        seen_hash.add(digest)
        n_copy += 1
    return {"copied": n_copy, "skipped": n_skip, "dest": str(dest_dir)}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src",
        type=Path,
        default=BATCH_RUNS_DIR,
        help="run tree containing balcony/assets/**/unit_*.png",
    )
    ap.add_argument(
        "--crops-dir",
        type=Path,
        default=DEFAULT_CROPS_DIR,
        help="label folders: unlabeled/ solid/ baluster/",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    stats = collect_crops(args.src, args.crops_dir)
    print(
        f"copied={stats['copied']} skipped_dup={stats['skipped']} -> {stats['dest']}"
    )
    print("Move PNGs from unlabeled/ into solid/ or baluster/, then run train.py")


if __name__ == "__main__":
    main()
