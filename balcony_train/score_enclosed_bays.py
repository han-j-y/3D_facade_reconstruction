"""Oriel scores per union column and per floor bay, with and without ignore cells.

Per column is the main protocol (the same count as ``clf_three_rules.py``).
The other rows are supplementary:

* per floor bay: a floor bay (``meta.floor_bays``) is one unit however many
  union columns it spans. It is labeled when any of its columns is labeled,
  and predicted when any of its columns is predicted.
* ignore: cells in a label row's ``ignore`` list (ambiguous from the photo)
  are dropped from TP, FP and FN. A floor bay whose columns are all ignore is
  dropped; otherwise its other columns decide.

TP / FP / FN are pooled over all photos.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/score_enclosed_bays.py ^
      --preds runs/cell_clf/compare_clf_rules_gate_ext_20261029/preds.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.enclosed_grid_store import (  # noqa: E402
    DEFAULT_DSL_DIR,
    DEFAULT_EVAL_LABELS_JSONL,
    Cell,
    FloorBays,
    load_floor_bays,
    load_grid,
    load_ignore,
    load_labels,
)

DEFAULT_DEV_LABELS = ROOT / "runs" / "enclosed_grid" / "labels_eval.jsonl"


def bay_units(cells: set[Cell], floor_bays: FloorBays) -> set[tuple[int, int, int]]:
    """Floor bays ``(floor, c0, c1)`` that hold at least one of ``cells``."""
    return {
        (f, c0, c1)
        for f, spans in floor_bays.items()
        for c0, c1 in spans
        if any((f, c) in cells for c in range(c0, c1))
    }


def score(
    preds: dict[str, set[Cell]],
    labels: dict[str, list[Cell]],
    stems: list[str],
    dsl_dir: Path,
    *,
    per_bay: bool,
    ignore: dict[str, list[Cell]] | None = None,
) -> dict[str, float]:
    tp = fp = fn = units = exact = dropped = 0
    for s in stems:
        dsl = Path(dsl_dir) / s / "facade_dsl.json"
        floor_y, bay_x = load_grid(dsl)
        skip = set(map(tuple, (ignore or {}).get(s, [])))
        g = {tuple(c) for c in labels[s]} - skip
        p = set(preds[s]) - skip
        if per_bay:
            fb = load_floor_bays(dsl, floor_y, bay_x)
            kept = {f: [(c0, c1) for c0, c1 in spans if any((f, c) not in skip for c in range(c0, c1))]
                    for f, spans in fb.items()}
            n = sum(len(v) for v in kept.values())
            dropped += sum(len(v) for v in fb.values()) - n
            g, p = bay_units(g, kept), bay_units(p, kept)
        else:
            n = len(floor_y) * len(bay_x) - len(skip)
            dropped += len(skip)
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        units += n
        exact += g == p
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {
        "photos": len(stems),
        "units": units,
        "dropped": dropped,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": pr,
        "recall": rc,
        "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "exact": exact,
    }


def _row(name: str, r: dict[str, float]) -> str:
    return (
        f"| {name} | {r['photos']} | {r['units']} | {r['dropped']} | {r['tp']} | {r['fp']} | {r['fn']} | "
        f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['iou']:.3f} | "
        f"{r['exact']}/{r['photos']} |"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", type=Path, required=True, help="preds.json from clf_three_rules.py")
    ap.add_argument("--labels", type=Path, default=DEFAULT_EVAL_LABELS_JSONL)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--minus-dev", action="store_true", help="also score without the dev photos")
    ap.add_argument("--dev-labels", type=Path, default=DEFAULT_DEV_LABELS)
    args = ap.parse_args()

    raw = json.loads(args.preds.read_text(encoding="utf-8"))
    preds = {s: {tuple(c) for c in v} for s, v in raw.items()}
    labels = load_labels(args.labels)
    ignore = load_ignore(args.labels)
    sets = [("test", sorted(s for s in preds if s in labels))]
    if args.minus_dev:
        dev = set(load_labels(args.dev_labels))
        sets.append(("test minus dev", [s for s in sets[0][1] if s not in dev]))
    n_ign = sum(len(ignore.get(s, [])) for s in sets[0][1])
    print(f"ignore cells: {n_ign} in {sum(bool(ignore.get(s)) for s in sets[0][1])} photos")
    print("| set | photos | units | ignored | TP | FP | FN | P | R | F1 | IoU | exact photos |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, stems in sets:
        for unit, per_bay in (("column", False), ("floor bay", True)):
            print(_row(f"{name}, per {unit}", score(preds, labels, stems, args.dsl_dir, per_bay=per_bay)))
            if n_ign:
                r = score(preds, labels, stems, args.dsl_dir, per_bay=per_bay, ignore=ignore)
                print(_row(f"{name}, per {unit}, ignore", r))


if __name__ == "__main__":
    main()
