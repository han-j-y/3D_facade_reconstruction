"""Oriel skin model: punched window vs curtain wall, per grid cell.

Kept apart from the railing classifier (user, 2026-11-03): the oriel track owns
its skin. Features are the per-cell DINOv2 vectors the oriel cell model already
uses (cell box + 25 % context, ``cell_features_build.dino_embed``), so training
reads ``runs/cell_clf/cells_train.npz`` and the pipeline embeds the same boxes.

Training rows: the labeled oriel cells of every photo in
``labels_train_oriel.jsonl`` that has a photo-level ``skin``; each cell takes
that skin. A photo is decided by ``enclosed.vote_enclosed_skin`` over its cells,
as in the pipeline. ``C`` is chosen on grouped (by photo) K-fold out-of-fold
photo balanced accuracy inside the training set. No test photo is read.

Usage::

    python balcony_train/oriel_skin.py train
    python balcony_train/oriel_skin.py score-test100 --preds <preds_official.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "balcony_pipeline"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DEFAULT_CELLS = ROOT / "runs" / "cell_clf" / "cells_train.npz"
DEFAULT_LABELS = ROOT / "runs" / "enclosed_grid" / "labels_train_oriel.jsonl"
MODEL_PATH = ROOT / "checkpoints" / "oriel_skin.joblib"  # Git LFS; until 2026-11-03 runs/cell_clf/
TEST100 = ROOT / "runs" / "test100"
SKINS = ("punched_window", "curtain_wall")
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)
FOLDS = 5

_MODEL = None


def _read_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[str(row["stem"])] = row
    return out


def _make(c: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", max_iter=5000),
    )


def vote(labels: list[str]) -> str:
    from enclosed import vote_enclosed_skin

    return vote_enclosed_skin(labels)[0]


def training_rows(cells_npz: Path, labels_path: Path, test_stems: set[str]):
    z = np.load(cells_npz, allow_pickle=False)
    groups = z["groups"].astype(str)
    cells = [tuple(map(int, c)) for c in z["cells"]]
    dino = z["dino"].astype(np.float64)
    labels = _read_jsonl(labels_path)
    leak = sorted(set(labels) & test_stems)
    if leak:
        raise SystemExit(f"test photos in training labels: {leak[:5]}")
    keep, y, g = [], [], []
    for i, (stem, cell) in enumerate(zip(groups, cells)):
        rec = labels.get(stem)
        if not rec or rec.get("skin") not in SKINS:
            continue
        if cell not in {tuple(map(int, c)) for c in rec.get("cells") or []}:
            continue
        keep.append(i)
        y.append(SKINS.index(rec["skin"]))
        g.append(stem)
    return dino[keep], np.asarray(y), np.asarray(g)


def photo_scores(stems: np.ndarray, y: np.ndarray, pred: np.ndarray) -> dict:
    gt, pr = {}, {}
    for s in sorted(set(stems)):
        m = stems == s
        gt[s] = SKINS[int(y[m][0])]
        pr[s] = vote([SKINS[int(v)] for v in pred[m]])
    return confusion(gt, pr)


def confusion(gt: dict[str, str], pr: dict[str, str]) -> dict:
    conf = {a: {b: 0 for b in SKINS} for a in SKINS}
    for s in gt:
        conf[gt[s]][pr[s]] += 1
    recalls = [conf[a][a] / max(1, sum(conf[a].values())) for a in SKINS]
    n = len(gt)
    acc = sum(conf[a][a] for a in SKINS) / max(1, n)
    return {"n": n, "acc": acc, "bal_acc": float(np.mean(recalls)), "conf": conf}


def fmt(score: dict) -> str:
    c = score["conf"]
    return (
        f"n={score['n']} acc={score['acc']:.3f} bal_acc={score['bal_acc']:.3f}  "
        f"punched->punched {c['punched_window']['punched_window']} punched->curtain {c['punched_window']['curtain_wall']}  "
        f"curtain->punched {c['curtain_wall']['punched_window']} curtain->curtain {c['curtain_wall']['curtain_wall']}"
    )


def cmd_train(args) -> None:
    import joblib
    from sklearn.model_selection import StratifiedGroupKFold

    test_stems = set(_read_jsonl(TEST100 / "oriel_labels.jsonl"))
    X, y, g = training_rows(args.cells, args.labels, test_stems)
    photos = sorted(set(g))
    n_cw = len({s for s, v in zip(g, y) if v == 1})
    print(f"train cells={len(y)} (curtain {int(y.sum())})  photos={len(photos)} (curtain {n_cw})")
    best = None
    for c in C_GRID:
        oof = np.zeros(len(y), dtype=int)
        for tr, te in StratifiedGroupKFold(FOLDS, shuffle=True, random_state=0).split(X, y, g):
            oof[te] = _make(c).fit(X[tr], y[tr]).predict(X[te])
        sc = photo_scores(g, y, oof)
        print(f"  C={c:<5} OOF photos {fmt(sc)}")
        if best is None or sc["bal_acc"] > best[1]["bal_acc"]:
            best = (c, sc)
    c = best[0]
    model = _make(c).fit(X, y)
    sha = hashlib.sha256(Path(args.cells).read_bytes()).hexdigest()[:12]
    joblib.dump(
        {"model": model, "skins": SKINS, "C": c, "cells_npz": str(args.cells), "cells_sha": sha,
         "labels": str(args.labels), "train_photos": photos, "oof": best[1]},
        args.out,
    )
    print(f"chosen C={c}  -> {args.out}")


def load_model(path: Path = MODEL_PATH):
    global _MODEL
    if _MODEL is None:
        import joblib

        _MODEL = joblib.load(path)
    return _MODEL


def predict_cell_skins(image, boxes: list[list[int]], *, device, model_path: Path = MODEL_PATH) -> list[str]:
    """One skin per cell box, from the same DINOv2 embedding as the cell model."""
    from balcony_train.enclosed_detect import _dino

    if not boxes:
        return []
    bundle = load_model(model_path)
    feats = _dino(device, image.convert("RGB"), [list(map(int, b)) for b in boxes])
    return [bundle["skins"][int(v)] for v in bundle["model"].predict(feats.astype(np.float64))]


def cmd_score_test100(args) -> None:
    from PIL import Image

    from balcony_train.depth_cells import cell_box
    from balcony_train.enclosed_grid_store import find_photo, load_grid
    from balcony_train.evaluate_enclosed_grid import load_pred_cells

    labels = _read_jsonl(args.labels)
    train = set(load_model(args.model)["train_photos"])
    leak = sorted(set(labels) & train)
    if leak:
        raise SystemExit(f"test photos in skin training: {leak[:5]}")
    preds = load_pred_cells(args.preds)
    gt, pr, no_cells = {}, {}, []
    for stem, rec in sorted(labels.items()):
        if not rec.get("cells") or rec.get("skin") not in SKINS:
            continue
        cells = sorted(preds.get(stem, set()))
        if not cells:
            no_cells.append(stem)
            continue
        floor_y, bay_x = load_grid(args.dsl_dir / stem / "facade_dsl.json")
        cells = [c for c in cells if c[0] in floor_y and c[1] in bay_x]
        image = Image.open(find_photo(args.in_dir, stem)).convert("RGB")
        skins = predict_cell_skins(image, [cell_box(c, floor_y, bay_x) for c in cells], device=args.device, model_path=args.model)
        gt[stem], pr[stem] = rec["skin"], vote(skins)
        n_cw = sum(s == "curtain_wall" for s in skins)
        mark = "" if gt[stem] == pr[stem] else "  WRONG"
        print(f"{stem}  gt={gt[stem]}  pred={pr[stem]}  curtain cells {n_cw}/{len(skins)}{mark}", flush=True)
    print(f"no predicted cells (not scored): {len(no_cells)}")
    print("test100 photos " + fmt(confusion(gt, pr)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    t.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    t.add_argument("--out", type=Path, default=MODEL_PATH)
    s = sub.add_parser("score-test100")
    s.add_argument("--preds", type=Path, required=True)
    s.add_argument("--labels", type=Path, default=TEST100 / "oriel_labels.jsonl")
    s.add_argument("--in-dir", type=Path, default=TEST100 / "photos")
    s.add_argument("--dsl-dir", type=Path, default=TEST100 / "batch_grid2")
    s.add_argument("--model", type=Path, default=MODEL_PATH)
    s.add_argument("--device", default="cuda")
    args = ap.parse_args()
    {"train": cmd_train, "score-test100": cmd_score_test100}[args.cmd](args)


if __name__ == "__main__":
    main()
