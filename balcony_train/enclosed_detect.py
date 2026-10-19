"""Official enclosed detection: classifier rectangle, four cell rules, gate 0.10.

A photo whose gate-model max is below 0.10 predicts no cells. Every other
photo keeps the four-rule cells. The previous SAM3+DA3 volumes stay available
as ``--enclosed-method sam3``.

The classifier is the same one as ``clf_three_rules.py`` (hand-feature gate,
enclosure-only cell model, DINOv2 PCA 16). It is fit on ``cells_train.npz``
once and cached at ``runs/cell_clf/model_rules_gate.joblib``.
"""

from __future__ import annotations

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

from balcony_train.cell_clf_experiments import Data  # noqa: E402
from balcony_train.cell_clf_search import SPECS, Model  # noqa: E402
from balcony_train.cell_clf_train import DEFAULT_TRAIN_CELLS, DEFAULT_TRAIN_LABELS  # noqa: E402
from balcony_train.cell_clf_v3 import rect_decode  # noqa: E402
from balcony_train.cell_features_build import dino_embed, photo_rows  # noqa: E402
from balcony_train.clf_three_rules import apply_rules  # noqa: E402
from balcony_train.depth_cells import cell_box, mirror_bay  # noqa: E402
from balcony_train.enclosed_symmetry import SYM_GATE, grid_axis_x, window_boxes_from_dsl  # noqa: E402
from balcony_train.evaluate_enclosed_grid import _floor_h, cells_from_boxes  # noqa: E402
from balcony_train.try_da3_enclosure import resize_depth  # noqa: E402
from balcony_train.try_sam3_da3_volumes import (  # noqa: E402
    run_sam3,
    split_seed_records,
    volume_stages,
)
from enclosed import ensure_da3_depth  # noqa: E402
from snap import _bands_from_windows_dsl  # noqa: E402

CELL_T = 0.3
PHOTO_GATE = 0.10
MODEL_PATH = ROOT / "runs" / "cell_clf" / "model_rules_gate.joblib"
SAM3_THRESHOLD = 0.15

_MODEL: dict | None = None
_DINO: tuple | None = None


class PhotoData:
    """One photo laid out like ``cell_clf_experiments.Data``."""

    def __init__(self, stem: str, names: list[str], X: np.ndarray, cells: list[tuple[int, int]], dino: np.ndarray) -> None:
        self.names = list(names)
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.zeros(len(cells), dtype=int)
        self.cells = [tuple(map(int, c)) for c in cells]
        self.dino = np.asarray(dino, dtype=np.float64)
        self.stems = [stem]
        self.gt = {stem: []}
        self.idx = {stem: np.arange(len(self.cells))}
        self.col = {n: i for i, n in enumerate(self.names)}

    def f(self, name: str) -> np.ndarray:
        return self.X[:, self.col[name]]

    def floor_order(self, stem: str) -> dict[int, int]:
        floors = sorted({self.cells[i][0] for i in self.idx[stem]})
        return {f: k for k, f in enumerate(floors)}


def photo_mirror(data: PhotoData, stem: str, bay_x: dict[int, tuple[int, int]]) -> np.ndarray:
    """Mirror row index for each cell, or -1. Same rule as ``mirror_index``."""
    out = np.full(len(data.y), -1, dtype=int)
    ii = data.idx[stem]
    if len(ii) == 0 or float(data.f("window_sym")[ii[0]]) < SYM_GATE:
        return out
    cx = grid_axis_x(bay_x)
    where = {data.cells[i]: int(i) for i in ii}
    for i in ii:
        f, b = data.cells[i]
        mb = mirror_bay(b, bay_x, cx) if cx is not None else None
        if mb is not None and mb != b and (f, mb) in where:
            out[i] = where[(f, mb)]
    return out


def decode_cells(
    data: PhotoData | Data,
    stem: str,
    p_gate: np.ndarray,
    p_cell: np.ndarray,
    floor_y: dict[int, tuple[int, int]],
    bay_x: dict[int, tuple[int, int]],
    *,
    gate: float = PHOTO_GATE,
    t: float = CELL_T,
) -> tuple[set, float, bool]:
    """Return (cells, gate max, gated off)."""
    ii = data.idx[stem]
    if len(ii) == 0:
        return set(), 0.0, True
    gmax = float(np.max(p_gate[ii]))
    mir = photo_mirror(data, stem, bay_x)  # type: ignore[arg-type]
    clf = rect_decode(data, stem, p_cell, t, mir)
    rules, _top, _low, _added, _up = apply_rules(data, stem, clf, p_cell, floor_y)
    if gmax < gate:
        return set(), gmax, True
    return rules, gmax, False


def load_model() -> dict:
    """Fit the cmp_b classifier once and reuse the saved estimators."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import joblib

    if MODEL_PATH.is_file():
        _MODEL = joblib.load(MODEL_PATH)
        return _MODEL
    if not DEFAULT_TRAIN_CELLS.is_file():
        raise FileNotFoundError(
            f"missing {DEFAULT_TRAIN_CELLS}. "
            "Build train features first, or pass --enclosed-method sam3."
        )
    train = Data(DEFAULT_TRAIN_CELLS, DEFAULT_TRAIN_LABELS)
    fitted = Model(train, train.stems, SPECS["base"], None)
    _MODEL = {"parts": fitted.parts, "names": list(train.names)}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_MODEL, MODEL_PATH)
    print(f"cached cell classifier -> {MODEL_PATH}", flush=True)
    return _MODEL


def _predict(model: dict, data: PhotoData) -> tuple[np.ndarray, np.ndarray]:
    if list(data.names) != list(model["names"]):
        raise SystemExit("feature names differ from the cached cell classifier")
    out = []
    for est, pca, cols in model["parts"]:
        x = np.hstack([data.X[:, cols], pca.transform(data.dino)])
        out.append(est.predict_proba(x)[:, 1])
    return out[0], out[1]


def _dino(device, image: Image.Image, boxes: list[list[int]]) -> np.ndarray:
    global _DINO
    import torch

    dev = torch.device(device)
    if _DINO is None or _DINO[0] != str(dev):
        net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
        _DINO = (str(dev), net.to(dev).eval())
    if not boxes:
        return np.zeros((0, 384), dtype=np.float64)
    return dino_embed(_DINO[1], dev, image, boxes)


def detect_enclosed_cells(
    image: Image.Image,
    image_path: Path,
    windows_dsl: dict,
    *,
    device,
    depth_dir: Path,
    cache_path: Path,
    sam3_threshold: float = SAM3_THRESHOLD,
) -> set[tuple[int, int]]:
    """Floor×bay cells for one facade. Empty when the gate is off or inputs are missing."""
    image = image.convert("RGB")
    floor_y, bay_x = _bands_from_windows_dsl(windows_dsl)
    if not floor_y or not bay_x:
        print("warn: no window grid; skip enclosed cells")
        return set()
    stem = Path(image_path).stem
    depth_path = ensure_da3_depth(Path(image_path), Path(depth_dir), device=str(device))
    if depth_path is None:
        return set()
    depth = resize_depth(np.load(depth_path), image.width, image.height)
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    seeds = run_sam3(
        [Path(image_path)],
        device=torch.device(device) if not isinstance(device, torch.device) else device,
        cache_path=Path(cache_path),
        threshold=sam3_threshold,
    ).get(stem, [])
    enclosed, _excluded = split_seed_records(seeds)
    stages: list = []
    if enclosed:
        stages = volume_stages(
            image,
            depth,
            seeds,
            floor_h=_floor_h(floor_y),
            floor_y=floor_y,
            bay_x=bay_x,
            windows=window_boxes_from_dsl(windows_dsl),
            symmetry=True,
        )
    by_id = {sid: boxes for sid, _title, boxes in stages}
    volume = {
        "stages": {sid: [list(map(int, b["box_xyxy"])) for b in boxes] for sid, boxes in by_id.items()},
        "cells_5": sorted(cells_from_boxes(by_id.get("5_clip", []), floor_y, bay_x)),
        "cells_6": sorted(cells_from_boxes(by_id.get("6_sym", by_id.get("5_clip", [])), floor_y, bay_x)),
    }
    windows = window_boxes_from_dsl(windows_dsl)
    names, rows, _ys, cells = photo_rows(
        stem, image, depth, floor_y, bay_x, windows, seeds, volume, set()
    )
    if not cells:
        print(f"enclosed cell_clf  {stem}  no cells")
        return set()
    boxes = [cell_box(c, floor_y, bay_x) for c in cells]
    dino = _dino(device, image, boxes)
    data = PhotoData(stem, names, np.asarray(rows, dtype=np.float64), cells, dino)
    p_gate, p_cell = _predict(load_model(), data)
    chosen, gmax, off = decode_cells(data, stem, p_gate, p_cell, floor_y, bay_x)
    state = "off" if off else "on"
    print(f"enclosed cell_clf  {stem}  gate={gmax:.2f} {state}  cells={len(chosen)}", flush=True)
    return chosen
