"""Score enclosed floor shape and skin on the eval set.

Uses cells the detector already predicted (the official gate run by default).
Each predicted cell is cropped and classified, then one shape and one skin
are chosen by the same majority vote as the pipeline. Photos with no
enclosure, and enclosure photos that do not yet have both labels, are left
out of the shape and skin rates. A photo with no predicted cells is left
out of those rates too: floor and skin were never judged.

``cmp_x`` labels stay in ``labels_ext_all.jsonl``. This script does not write
the training set.

Wrong photos are copied like the open-balcony eval: one folder per
``true-{label}__pred-{label}`` under ``floor/`` and ``skin/``, plus
``mistakes.txt``. Triangle counts as rectangle, so it is not listed as a
floor mistake when the vote is rectangle.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/score_enclosed_plan.py ^
      --ckpt checkpoints/railing_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.enclosed_grid_store import (  # noqa: E402
    DEFAULT_DSL_DIR,
    DEFAULT_EVAL_LABELS_JSONL,
    DEFAULT_EVAL_PHOTO_DIR,
    GRID_FAILURE,
    find_photo,
    load_grid,
    load_records,
)
from balcony_train.evaluate_enclosed_grid import (  # noqa: E402
    apply_plan_score,
    format_report,
    load_pred_cells,
    score_photo,
    scored_floor_shape,
    summarize,
    vote_plan,
)
from balcony_train.labels import ENCLOSURE_SKINS, FLOOR_SHAPES  # noqa: E402
from balcony_train.paths import DEFAULT_RAILING_CKPT  # noqa: E402

DEFAULT_PREDS = ROOT / "runs" / "cell_clf" / "compare_clf_rules_gate_ext_relabel" / "preds.json"
DEFAULT_ERRORS = ROOT / "runs" / "enclosed_grid" / "plan_wrong"


def score_rows(
    records: dict,
    preds: dict,
    *,
    photo_dir: Path,
    dsl_dir: Path,
    predict_full,
) -> tuple[list[dict], list[str]]:
    """Cell scores for every saved photo, plus a plan score when both labels exist."""
    from PIL import Image

    rows: list[dict] = []
    skipped: list[str] = []
    for stem in sorted(records):
        rec = records[stem]
        if rec.status == GRID_FAILURE:
            continue
        pred_cells = preds.get(stem, set())
        row = score_photo(rec.cells, pred_cells)
        row["stem"] = stem
        if not rec.cells:
            rows.append(row)
            continue
        photo = find_photo(photo_dir, stem)
        dsl = Path(dsl_dir) / stem / "facade_dsl.json"
        if photo is None or not dsl.is_file():
            skipped.append(stem)
            rows.append(row)
            continue
        floor_y, bay_x = load_grid(dsl)
        row["photo"] = str(photo)
        pred_shape = pred_skin = None
        if rec.floor_shape and rec.skin:
            image = Image.open(photo).convert("RGB")
            pred_shape, pred_skin = vote_plan(image, pred_cells, floor_y, bay_x, predict_full)
        apply_plan_score(row, rec.floor_shape, rec.skin, pred_shape, pred_skin)
        rows.append(row)
        mark = ""
        if row.get("plan_labeled"):
            mark = f"  shape={int(row['shape_ok'])}  skin={int(row['skin_ok'])}"
        print(
            f"{stem}  detect={int(row['detected'])}  bay_floor={int(row['exact'])}{mark}",
            flush=True,
        )
    return rows, skipped


def copy_plan_mistakes(
    rows: list[dict],
    dest_dir: Path,
    *,
    stamp: str | None = None,
) -> int:
    """Copy facade photos whose floor shape or skin vote disagrees with the label.

    Same layout as ``evaluate.py --errors-to``:

        dest/YYYY-MM-DD_HHMMSS/floor/true-{label}__pred-{label}/<photo>
        dest/YYYY-MM-DD_HHMMSS/skin/true-{label}__pred-{label}/<photo>

    A missing prediction is the folder name ``none``. ``mistakes.txt`` lists
    ``head true pred filename``. Returns the number of file copies.
    """
    from balcony_train.evaluate import copy_eval_images, error_run_dir, misclass_folder_name
    from balcony_train.evaluate_enclosed_grid import scored_floor_shape

    buckets: dict[Path, list[Path]] = {}
    lines: list[str] = []

    def take(head: str, true_name: str, pred_name: str, path: Path) -> None:
        rel = Path(head) / misclass_folder_name(true_name, pred_name)
        buckets.setdefault(rel, []).append(path)
        lines.append(f"{head}\t{true_name}\t{pred_name}\t{path.name}")

    for row in rows:
        if not row.get("plan_labeled") or not row.get("photo") or not row.get("pred"):
            continue
        path = Path(row["photo"])
        if row.get("shape_ok") is False:
            take(
                "floor",
                scored_floor_shape(row.get("gt_shape")) or "none",
                scored_floor_shape(row.get("pred_shape")) or "none",
                path,
            )
        if row.get("skin_ok") is False:
            take(
                "skin",
                str(row.get("gt_skin") or "none"),
                str(row.get("pred_skin") or "none"),
                path,
            )

    run_dir = error_run_dir(dest_dir, stamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for rel in sorted(buckets):
        n += copy_eval_images(buckets[rel], run_dir / rel)
    list_path = run_dir / "mistakes.txt"
    list_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if not buckets:
        print(f"no plan mistakes -> {run_dir.resolve()}", flush=True)
    else:
        print(f"plan mistake copies={n} -> {run_dir.resolve()}", flush=True)
        print(f"list -> {list_path.resolve()}", flush=True)
    return n


def _confusion(rows: list[dict], classes: tuple[str, ...], true_of, pred_of) -> list[list[int]]:
    """Counts for judged photos. A photo with no predicted cells is left out."""
    index = {name: i for i, name in enumerate(classes)}
    conf = [[0 for _ in classes] for _ in classes]
    for row in rows:
        if not row.get("plan_labeled") or not row.get("pred"):
            continue
        true_name = true_of(row)
        pred_name = pred_of(row)
        if true_name not in index or pred_name not in index:
            continue
        conf[index[true_name]][index[pred_name]] += 1
    return conf


def format_confusion(title: str, classes: tuple[str, ...], conf: list[list[int]]) -> str:
    """Same text table as the open-balcony eval: rows are labels, columns are predictions."""
    lines = [f"{title}  " + "  ".join(classes)]
    for name, counts in zip(classes, conf):
        row = " ".join(f"{value:4d}" for value in counts)
        lines.append(f"  {name:16s} {row}")
    return "\n".join(lines)


def format_plan_confusion(rows: list[dict]) -> str:
    """Floor and skin tables. Triangle counts as rectangle, matching the photo vote."""
    floor = _confusion(
        rows,
        FLOOR_SHAPES,
        lambda row: scored_floor_shape(row.get("gt_shape")),
        lambda row: scored_floor_shape(row.get("pred_shape")),
    )
    skin = _confusion(
        rows,
        ENCLOSURE_SKINS,
        lambda row: row.get("gt_skin"),
        lambda row: row.get("pred_skin"),
    )
    return "\n".join(
        [
            format_confusion("floor rows=true cols=pred", FLOOR_SHAPES, floor),
            format_confusion("skin rows=true cols=pred", ENCLOSURE_SKINS, skin),
        ]
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=DEFAULT_EVAL_LABELS_JSONL)
    ap.add_argument("--preds", type=Path, default=DEFAULT_PREDS)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_RAILING_CKPT)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_EVAL_PHOTO_DIR)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument(
        "--device",
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    ap.add_argument(
        "--errors-to",
        type=Path,
        default=DEFAULT_ERRORS,
        help="Copy wrong floor/skin photos here, grouped like the open-balcony eval. Empty skips.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.labels)
    if not records:
        print(f"no labeled photos in {args.labels}")
        return
    failures = sorted(stem for stem, rec in records.items() if rec.status == GRID_FAILURE)
    if failures:
        print(f"grid failure ({len(failures)}): " + ", ".join(failures))
    if not args.preds.is_file():
        print(f"no predicted cells at {args.preds}")
        return
    labeled = [
        stem
        for stem, rec in records.items()
        if rec.cells and rec.floor_shape and rec.skin
    ]
    pending = [stem for stem, rec in records.items() if rec.cells and stem not in labeled]
    if pending:
        print(f"enclosure photos still missing shape or skin: {len(pending)}")
    if not labeled:
        print("no enclosure plan labels; shape and skin not scored")
        return
    if not args.ckpt.is_file():
        print(f"no railing checkpoint at {args.ckpt}")
        return
    import torch

    from balcony_train.model import RailingPredictor

    predictor = RailingPredictor(args.ckpt, torch.device(args.device))
    preds = load_pred_cells(args.preds)
    rows, skipped = score_rows(
        records,
        preds,
        photo_dir=args.in_dir,
        dsl_dir=args.dsl_dir,
        predict_full=predictor.predict_full,
    )
    if skipped:
        print("skipped (no photo or window grid): " + ", ".join(skipped))
    print(format_report(summarize(rows), rows))
    print(format_plan_confusion(rows))
    if args.errors_to is not None and str(args.errors_to) != "":
        copy_plan_mistakes(rows, args.errors_to)


if __name__ == "__main__":
    main()
