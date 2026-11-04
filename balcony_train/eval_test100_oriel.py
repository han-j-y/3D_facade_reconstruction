"""Oriel cell scores on the 100-photo test set (``runs/test100``).

Runs the official enclosed detector (``enclosed_detect.py``: classifier
rectangle, four cell rules, photo gate) on every photo in ``--photos-dir``,
using the window grids in ``--dsl-dir``, and scores the cells against
``--labels`` (``enclosed_grid_ui.py`` output).

One cell model is scored: the pipeline's cached ``--model``
(``checkpoints/oriel_cell_gate.joblib``). The script stops if any labeled
test photo is in its training features (``--train-cells``), so the score is
never from a model that saw a test photo. (Until 2026-11-03 a second
``heldout`` refit without the test photos was also scored; once the test
photos left oriel training it gave identical predictions.)

Outputs in ``--out-dir``: ``summary.md``, ``preds_official.json`` and one
``qualitative/`` folder (one panel per photo + a board per category).

TP / FP / FN / TN are pooled over all cells (per union column, the main
protocol). Rows without and with the ignore cells are printed; per floor bay
is supplementary. SAM3 boxes and DA3 depth are cached under ``runs/test100``.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/eval_test100_oriel.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_PIPE = ROOT / "balcony_pipeline"
if str(_PIPE) not in sys.path:
    sys.path.insert(0, str(_PIPE))

from balcony_train.cell_clf_train import DEFAULT_TRAIN_CELLS, DEFAULT_TRAIN_LABELS  # noqa: E402
from balcony_train.enclosed_detect import (  # noqa: E402
    MODEL_PATH,
    CELL_T,
    PHOTO_GATE,
    SAM3_THRESHOLD,
    PhotoData,
    _predict,
    decode_cells,
    photo_cell_data,
)
from balcony_train.draw_cell_clf_compare import CATEGORIES, board, category  # noqa: E402
from balcony_train.draw_clf_compare import add_header  # noqa: E402
from balcony_train.enclosed_grid_store import load_floor_bays, load_grid, load_ignore, load_labels  # noqa: E402
from balcony_train.evaluate_enclosed_grid import render_compare  # noqa: E402
from balcony_train.score_enclosed_bays import score  # noqa: E402
from balcony_train.try_da3_enclosure import resize_depth  # noqa: E402
from balcony_train.try_sam3_da3_volumes import DEFAULT_DEPTH_DIR, run_sam3  # noqa: E402
from enclosed import ensure_da3_depth  # noqa: E402
from snap import _bands_from_windows_dsl  # noqa: E402

TEST100 = ROOT / "runs" / "test100"


def photo_data(stem: str, photo: Path, dsl: dict, seeds: list, depth_path: Path, device: str) -> PhotoData | None:
    """Per-cell features for one photo, as in ``detect_enclosed_cells``."""
    image = Image.open(photo).convert("RGB")
    depth = resize_depth(np.load(depth_path), image.width, image.height)
    return photo_cell_data(stem, image, dsl, depth, seeds, device=device)


def draw_panels(out_dir: Path, photos: dict[str, Path], dsl_dir: Path, labels: dict, ignore: dict, name: str) -> None:
    """One panel per photo (blue = label only, red = prediction only, purple = both) plus boards per category."""
    preds = json.loads((out_dir / f"preds_{name}.json").read_text(encoding="utf-8"))
    qdir = out_dir / "qualitative"
    qdir.mkdir(parents=True, exist_ok=True)
    tiles: dict[str, list[Image.Image]] = {c: [] for c in CATEGORIES}
    gone = sorted(s for s in preds if s not in photos)
    if gone:
        print(f"no photo in the photo folder, not drawn: {' '.join(gone)}")
    for s in sorted(set(preds) - set(gone)):
        dsl = dsl_dir / s / "facade_dsl.json"
        floor_y, bay_x = load_grid(dsl)
        gt = {tuple(c) for c in labels.get(s, [])}
        pred = {tuple(c) for c in preds[s]}
        cat = category(gt, pred)
        img = render_compare(Image.open(photos[s]).convert("RGB"), floor_y, bay_x, gt, pred, load_floor_bays(dsl, floor_y, bay_x))
        lines = [
            f"{s[:60]}  [{cat}]  model={name}",
            f"label={len(gt)}  pred={len(pred)}  TP={len(gt & pred)}  FP={len(pred - gt)}  FN={len(gt - pred)}"
            + (f"  ignore={len(ignore[s])}" if ignore.get(s) else ""),
        ]
        panel = add_header(img, lines)
        panel.save(qdir / f"{cat}__{s}.jpg", quality=90)
        tiles[cat].append(panel)
    for cat, ts in tiles.items():
        if ts:
            board(ts).save(qdir / f"board_{cat}.jpg", quality=85)
    print(f"qualitative -> {qdir}  " + "  ".join(f"{c} {len(t)}" for c, t in tiles.items()))


def check_no_leak(test_stems: set[str], cells: Path = DEFAULT_TRAIN_CELLS) -> None:
    """Stop when a test photo is in the cell model's training features."""
    train = set(np.load(cells, allow_pickle=False)["groups"].astype(str))
    leak = sorted(train & set(test_stems))
    if leak:
        raise SystemExit(f"{len(leak)} test photos are in {cells}: {leak[:5]} ... refit the cell model without them")
    print(f"leak check: 0 of {len(test_stems)} test photos in {Path(cells).name} ({len(train)} training photos)")


def table(name: str, r: dict) -> str:
    tn = r["units"] - r["tp"] - r["fp"] - r["fn"]
    return (
        f"| {name} | {r['photos']} | {r['units']} | {r['tp']} | {r['fp']} | {r['fn']} | {tn} | "
        f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['iou']:.3f} | {r['exact']}/{r['photos']} |"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--photos-dir", type=Path, default=TEST100 / "photos")
    ap.add_argument("--dsl-dir", type=Path, default=TEST100 / "batch_grid2")
    ap.add_argument("--labels", type=Path, default=TEST100 / "oriel_labels.jsonl")
    ap.add_argument("--depth-dir", type=Path, default=TEST100 / "da3_depth")
    ap.add_argument("--sam3-cache", type=Path, default=TEST100 / "enclosed_sam3_cache.json")
    ap.add_argument("--out-dir", type=Path, default=TEST100 / "oriel_eval")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train-cells", type=Path, default=DEFAULT_TRAIN_CELLS, help="training features of --model, for the leak check")
    ap.add_argument("--train-labels", type=Path, default=DEFAULT_TRAIN_LABELS, help="unused (kept for old scripts)")
    ap.add_argument("--model", type=Path, default=MODEL_PATH, help="official cell model (joblib)")
    ap.add_argument("--skip-stems", type=Path, default=None, help="text file of photos left out of the score")
    ap.add_argument("--gate", type=float, default=PHOTO_GATE, help="photo gate (pipeline default %(default)s)")
    ap.add_argument("--t", type=float, default=CELL_T, help="cell threshold (pipeline default %(default)s)")
    ap.add_argument("--draw-only", action="store_true", help="only redraw the panels from the saved preds_*.json")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    ignore = load_ignore(args.labels)
    if args.skip_stems:
        skip = set(args.skip_stems.read_text(encoding="utf-8").split())
        labels = {s: v for s, v in labels.items() if s not in skip}
    photos: dict[str, Path] = {}
    for p in sorted(args.photos_dir.iterdir()):  # .jpg sorts before .png: a CMP label map never replaces its photo
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            photos.setdefault(p.stem, p)
    if args.draw_only:
        draw_panels(args.out_dir, photos, args.dsl_dir, labels, ignore, "official")
        return

    import torch

    stems = [s for s in sorted(photos) if s in labels and (args.dsl_dir / s / "facade_dsl.json").is_file()]
    print(f"photos {len(photos)}  labeled with grid {len(stems)}")

    seeds = run_sam3(
        [photos[s] for s in stems], device=torch.device(args.device), cache_path=args.sam3_cache, threshold=SAM3_THRESHOLD
    )
    torch.cuda.empty_cache()
    args.depth_dir.mkdir(parents=True, exist_ok=True)
    for s in stems:
        old = DEFAULT_DEPTH_DIR / f"{s}_depth.npy"
        new = args.depth_dir / f"{s}_depth.npy"
        if old.is_file() and not new.is_file():
            shutil.copy2(old, new)

    data: dict[str, PhotoData] = {}
    grids: dict[str, tuple[dict, dict]] = {}
    for s in stems:
        depth_path = ensure_da3_depth(photos[s], args.depth_dir, device=args.device)
        dsl = json.loads((args.dsl_dir / s / "facade_dsl.json").read_text(encoding="utf-8"))
        d = photo_data(s, photos[s], dsl, seeds.get(s, []), depth_path, args.device) if depth_path else None
        if d is None:
            print(f"skip {s}: no depth or no cells")
            continue
        data[s] = d
        grids[s] = _bands_from_windows_dsl(dsl)

    import joblib

    check_no_leak(set(labels), args.train_cells)
    models = {"official": joblib.load(args.model)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Oriel cells: {len(data)} photos from {args.photos_dir}, labels {args.labels.name}, gate {args.gate}, t {args.t}",
        "",
        "| model, unit | photos | cells | TP | FP | FN | TN | P | R | F1 | IoU | exact photos |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, model in models.items():
        preds: dict[str, set] = {}
        for s, d in data.items():
            p_gate, p_cell = _predict(model, d)
            preds[s], _gmax, _off = decode_cells(d, s, p_gate, p_cell, *grids[s], gate=args.gate, t=args.t)
        (args.out_dir / f"preds_{name}.json").write_text(
            json.dumps({s: [list(c) for c in sorted(v)] for s, v in preds.items()}) + "\n", encoding="utf-8"
        )
        scored = sorted(preds)
        for unit, per_bay in (("column", False), ("floor bay", True)):
            lines.append(table(f"{name}, per {unit}", score(preds, labels, scored, args.dsl_dir, per_bay=per_bay)))
            lines.append(
                table(f"{name}, per {unit}, ignore", score(preds, labels, scored, args.dsl_dir, per_bay=per_bay, ignore=ignore))
            )
    text = "\n".join(lines)
    print(text)
    (args.out_dir / "summary.md").write_text(text + "\n", encoding="utf-8")
    for name in models:
        draw_panels(args.out_dir, photos, args.dsl_dir, labels, ignore, name)


if __name__ == "__main__":
    main()
