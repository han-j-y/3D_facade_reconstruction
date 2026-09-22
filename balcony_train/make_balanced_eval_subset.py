"""Build a balanced held-out eval subset from crops_ext_eval labels.

Caps abundant classes so the eval set is not dominated by open_work.
Uses all surface_panel samples when scarce.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe ^
      balcony_train/make_balanced_eval_subset.py
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.label_store import find_image, write_labels_jsonl  # noqa: E402
from balcony_train.labels import CLASSES, ensure_crop_dirs  # noqa: E402

DEFAULT_SRC_CROPS = ROOT / "runs" / "balcony_clf" / "crops_ext_eval"
DEFAULT_SRC_LABELS = ROOT / "runs" / "balcony_clf" / "labels_ext_eval.jsonl"
DEFAULT_DST_CROPS = ROOT / "runs" / "balcony_clf" / "crops_ext_eval_balanced"
DEFAULT_DST_LABELS = ROOT / "runs" / "balcony_clf" / "labels_ext_eval_balanced.jsonl"


def load_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        stem = Path(str(rec.get("path") or "")).stem
        if not stem and "path" in rec:
            stem = str(rec["path"])
        # Prefer explicit stem from filename
        p = rec.get("path")
        if p:
            stem = Path(str(p)).stem
        kind = rec.get("kind")
        if kind not in CLASSES:
            continue
        rows[stem] = {
            "path": Path(str(p)).name if p else f"{stem}.png",
            "kind": kind,
            "material": rec.get("material"),
        }
    return rows


def sample_balanced(
    labels: dict[str, dict],
    *,
    target_total: int,
    seed: int,
) -> list[str]:
    """Return stems for a mixed eval set (~target_total)."""
    by_kind: dict[str, list[str]] = defaultdict(list)
    by_ow_mat: dict[str, list[str]] = defaultdict(list)
    for stem, rec in labels.items():
        kind = rec["kind"]
        by_kind[kind].append(stem)
        if kind == "open_work":
            mat = rec.get("material") or "unknown"
            by_ow_mat[str(mat)].append(stem)

    rng = random.Random(int(seed))
    for lst in by_kind.values():
        rng.shuffle(lst)
    for lst in by_ow_mat.values():
        rng.shuffle(lst)

    # Always take all surface_panel (usually scarce).
    chosen: list[str] = list(by_kind.get("surface_panel", []))
    n_surf = len(chosen)
    remain = max(0, int(target_total) - n_surf)

    # Split remainder ~55% open_work / 45% solid (adjust if empty).
    n_ow_target = int(round(remain * 0.55))
    n_solid_target = remain - n_ow_target

    solid_pool = list(by_kind.get("solid", []))
    n_solid = min(n_solid_target, len(solid_pool))
    chosen.extend(solid_pool[:n_solid])

    # Recompute open_work slots with leftover capacity.
    n_ow = min(n_ow_target + (n_solid_target - n_solid), len(by_kind.get("open_work", [])))

    # Prefer balanced metal/masonry within open_work.
    metals = list(by_ow_mat.get("metal", []))
    masons = list(by_ow_mat.get("masonry", []))
    half = n_ow // 2
    take_m = min(half, len(metals))
    take_n = min(n_ow - take_m, len(masons))
    # If one side short, fill from the other.
    if take_m + take_n < n_ow:
        need = n_ow - take_m - take_n
        if len(metals) > take_m:
            extra = min(need, len(metals) - take_m)
            take_m += extra
            need -= extra
        if need and len(masons) > take_n:
            take_n += min(need, len(masons) - take_n)
    chosen.extend(metals[:take_m])
    chosen.extend(masons[:take_n])

    # Deduplicate preserve order
    seen: set[str] = set()
    out: list[str] = []
    for stem in chosen:
        if stem not in seen:
            seen.add(stem)
            out.append(stem)
    return out


def materialize_subset(
    *,
    src_crops: Path,
    dst_crops: Path,
    labels: dict[str, dict],
    stems: list[str],
    dst_labels: Path,
) -> dict[str, int]:
    ensure_crop_dirs(dst_crops)
    # Clear previous kind folders (keep structure).
    for kind in CLASSES:
        folder = dst_crops / kind
        if folder.is_dir():
            for p in folder.iterdir():
                if p.is_file():
                    p.unlink()

    out_labels: dict[str, dict] = {}
    counts = {k: 0 for k in CLASSES}
    for stem in stems:
        rec = labels[stem]
        kind = rec["kind"]
        src = find_image(src_crops, stem)
        if src is None:
            print(f"skip missing image: {stem}", flush=True)
            continue
        dest = dst_crops / kind / src.name
        shutil.copy2(src, dest)
        out_labels[stem] = {
            "path": dest.name,
            "kind": kind,
            "material": rec.get("material") if kind == "open_work" else None,
        }
        counts[kind] += 1

    write_labels_jsonl(dst_labels, out_labels)
    return counts


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-crops", type=Path, default=DEFAULT_SRC_CROPS)
    ap.add_argument("--src-labels", type=Path, default=DEFAULT_SRC_LABELS)
    ap.add_argument("--dst-crops", type=Path, default=DEFAULT_DST_CROPS)
    ap.add_argument("--dst-labels", type=Path, default=DEFAULT_DST_LABELS)
    ap.add_argument("--target", type=int, default=100, help="approx eval set size")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    labels = load_jsonl(args.src_labels)
    if not labels:
        print(f"no labels in {args.src_labels}", file=sys.stderr)
        return 1
    avail = {k: 0 for k in CLASSES}
    for rec in labels.values():
        avail[rec["kind"]] += 1
    print(f"available: {avail}  total={sum(avail.values())}")
    if avail.get("surface_panel", 0) < 10:
        print(
            f"warn: only {avail.get('surface_panel', 0)} surface_panel labeled; "
            "balanced set will still be weak on that class"
        )

    stems = sample_balanced(labels, target_total=int(args.target), seed=int(args.seed))
    counts = materialize_subset(
        src_crops=args.src_crops,
        dst_crops=args.dst_crops,
        labels=labels,
        stems=stems,
        dst_labels=args.dst_labels,
    )
    print(f"balanced subset: {counts}  total={sum(counts.values())}")
    print(f"crops -> {args.dst_crops.resolve()}")
    print(f"labels -> {args.dst_labels.resolve()}")
    print(
        "Evaluate with:\n"
        f"  python balcony_train/evaluate.py --device cuda "
        f"--data-dir {args.dst_crops} --labels {args.dst_labels} "
        f"--split all --include-synth"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
