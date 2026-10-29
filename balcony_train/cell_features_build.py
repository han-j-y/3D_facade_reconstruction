"""Build per-cell features for every labeled photo (cell classifier input).

One row per floor×bay cell with depth, SAM3, current-method, mirror and
position features plus a DINOv2 embedding of the cell crop (with context).
Output: ``runs/cell_clf/cells_ext_no0227.npz`` (X, y, groups, names, cells, dino).

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/cell_features_build.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.depth_cells import cell_box, cell_features, mirror_bay  # noqa: E402
from balcony_train.enclosed_grid_store import (  # noqa: E402
    DEFAULT_DSL_DIR,
    DEFAULT_EVAL_LABELS_JSONL,
    DEFAULT_EVAL_PHOTO_DIR,
    find_photo,
    load_grid,
    load_labels,
)
from balcony_train.enclosed_symmetry import grid_axis_x, window_boxes_from_dsl, window_symmetry  # noqa: E402
from balcony_train.evaluate_enclosed_grid import DEFAULT_SAM3_CACHE  # noqa: E402
from balcony_train.extract_enclosed import foliage_mask  # noqa: E402
from balcony_train.try_da3_enclosure import resize_depth  # noqa: E402
from balcony_train.try_sam3_da3_volumes import (  # noqa: E402
    DEFAULT_DEPTH_DIR,
    SAM3_PROMPTS,
    front_fill_frac,
)

DEFAULT_VOLUME_CACHE = ROOT / "runs" / "enclosed_grid" / "volume_cells_floor_bays.json"
DEFAULT_OUT = ROOT / "runs" / "cell_clf" / "cells_ext_no0227.npz"
ENCLOSURE_PROMPTS = SAM3_PROMPTS[:3]


def _cover(cell: list[int], box: list[float]) -> float:
    ix = max(0.0, min(cell[2], box[2]) - max(cell[0], box[0]))
    iy = max(0.0, min(cell[3], box[3]) - max(cell[1], box[1]))
    return ix * iy / max(1.0, (cell[2] - cell[0]) * (cell[3] - cell[1]))


def _num(v) -> float:
    return float("nan") if v is None else float(v)


def dino_embed(model, device, image: Image.Image, boxes: list[list[int]], *, ctx: float = 0.25) -> np.ndarray:
    import torch

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    crops = []
    for x0, y0, x1, y1 in boxes:
        w, h = x1 - x0, y1 - y0
        box = (
            max(0, int(x0 - ctx * w)),
            max(0, int(y0 - ctx * h)),
            min(image.width, int(x1 + ctx * w)),
            min(image.height, int(y1 + ctx * h)),
        )
        crop = image.crop(box).resize((224, 224), Image.Resampling.BILINEAR)
        crops.append(np.asarray(crop, dtype=np.float32) / 255.0)
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), 64):
            t = torch.from_numpy(np.stack(crops[i : i + 64])).permute(0, 3, 1, 2).to(device)
            out.append(model((t - mean) / std).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def photo_rows(
    stem: str,
    image: Image.Image,
    depth: np.ndarray,
    floor_y,
    bay_x,
    windows,
    sam3_records,
    volume: dict | None,
    gt: set,
) -> tuple[list[str], list[list[float]], list[int], list[tuple[int, int]]]:
    veg = foliage_mask(np.asarray(image, dtype=np.uint8))
    base = cell_features(depth, veg, floor_y, bay_x, sam3_records)
    cx = grid_axis_x(bay_x)
    wsym = window_symmetry(windows, cx) if cx is not None else 0.0
    floors = sorted(floor_y, key=lambda f: floor_y[f][0])
    bays = sorted(bay_x, key=lambda b: bay_x[b][0])
    frank = {f: i for i, f in enumerate(floors)}
    brank = {b: i for i, b in enumerate(bays)}
    cells_5 = {tuple(c) for c in (volume or {}).get("cells_5", [])}
    cells_6 = {tuple(c) for c in (volume or {}).get("cells_6", [])}
    stages = (volume or {}).get("stages", {})

    col_L = {b: np.nanmedian([_num(base[(f, b)]["L"]) for f in floors]) for b in bays}
    row_L = {f: np.nanmedian([_num(base[(f, b)]["L"]) for b in bays]) for f in floors}
    iw, ih = image.size

    names: list[str] = []
    rows: list[list[float]] = []
    labels: list[int] = []
    cells: list[tuple[int, int]] = []
    for f in floors:
        for b in bays:
            box = cell_box((f, b), floor_y, bay_x)
            feat = base[(f, b)]
            fi, bi = frank[f], brank[b]
            above = base.get((floors[fi - 1], b)) if fi > 0 else None
            below = base.get((floors[fi + 1], b)) if fi + 1 < len(floors) else None
            lower_box = [box[0], box[1] + (box[3] - box[1]) * 2 // 3, box[2], box[3]]
            prompt_scores = {}
            for p in SAM3_PROMPTS:
                best = 0.0
                for r in sam3_records:
                    if r.get("prompt") == p and _cover(box, [float(v) for v in r["box_xyxy"]]) >= 0.25:
                        best = max(best, float(r.get("score") or 0.0))
                prompt_scores[p] = best
            enc_cover = 0.0
            for r in sam3_records:
                if r.get("prompt") in ENCLOSURE_PROMPTS:
                    enc_cover = max(enc_cover, _cover(box, [float(v) for v in r["box_xyxy"]]))
            stage_cover = {}
            for sid in ("2_depth", "3_merge", "4_grow", "5_clip"):
                stage_cover[sid] = max(
                    [_cover(box, [float(v) for v in sb]) for sb in stages.get(sid, [])] or [0.0]
                )
            mb = mirror_bay(b, bay_x, cx) if cx is not None else None
            mfeat = base.get((f, mb)) if mb is not None else None
            x_mid = 0.5 * (box[0] + box[2])
            row = {
                "L": _num(feat["L"]),
                "R": _num(feat["R"]),
                "upper": _num(feat["upper"]),
                "lower": _num(front_fill_frac(depth, lower_box, veg=veg, min_rel=0.015)),
                "L_above": _num(above["L"]) if above else float("nan"),
                "L_below": _num(below["L"]) if below else float("nan"),
                "R_above": _num(above["R"]) if above else float("nan"),
                "R_below": _num(below["R"]) if below else float("nan"),
                "col_L": float(col_L[b]),
                "row_L": float(row_L[f]),
                "L_minus_col": _num(feat["L"]) - float(col_L[b]),
                "L_minus_row": _num(feat["L"]) - float(row_L[f]),
                **{f"sam3_{i}": prompt_scores[p] for i, p in enumerate(SAM3_PROMPTS)},
                "sam3_enc_cover": enc_cover,
                "in_cells_5": float((f, b) in cells_5),
                "in_cells_6": float((f, b) in cells_6),
                **{f"cover_{sid}": v for sid, v in stage_cover.items()},
                "mirror_L": _num(mfeat["L"]) if mfeat else float("nan"),
                "mirror_upper": _num(mfeat["upper"]) if mfeat else float("nan"),
                "mirror_in_6": float(mb is not None and (f, mb) in cells_6),
                "mirror_self": float(mb == b),
                "window_sym": float(wsym),
                "floor_rank": fi / max(1, len(floors) - 1),
                "floor_from_bottom": float(len(floors) - 1 - fi),
                "bay_rank": bi / max(1, len(bays) - 1),
                "edge_bay": float(bi in (0, len(bays) - 1)),
                "axis_dist": abs(x_mid - cx) / max(1.0, iw) if cx is not None else float("nan"),
                "cell_w": (box[2] - box[0]) / iw,
                "cell_h": (box[3] - box[1]) / ih,
                "y_mid": 0.5 * (box[1] + box[3]) / ih,
                "n_floors": float(len(floors)),
                "n_bays": float(len(bays)),
            }
            if not names:
                names = list(row)
            rows.append([row[k] for k in names])
            labels.append(int((f, b) in gt))
            cells.append((int(f), int(b)))
    return names, rows, labels, cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=DEFAULT_EVAL_LABELS_JSONL)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_EVAL_PHOTO_DIR)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--depth-dir", type=Path, default=DEFAULT_DEPTH_DIR)
    ap.add_argument("--sam3-cache", type=Path, default=DEFAULT_SAM3_CACHE)
    ap.add_argument("--volume-cache", type=Path, default=DEFAULT_VOLUME_CACHE)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    import torch

    device = torch.device(args.device)
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True).to(device).eval()
    labels = load_labels(args.labels)
    sam3 = json.loads(args.sam3_cache.read_text(encoding="utf-8"))
    volumes = json.loads(args.volume_cache.read_text(encoding="utf-8")) if args.volume_cache.is_file() else {}

    X, Y, G, C, D = [], [], [], [], []
    names: list[str] = []
    for stem in sorted(labels):
        photo = find_photo(args.in_dir, stem)
        dsl = args.dsl_dir / stem / "facade_dsl.json"
        dpath = args.depth_dir / f"{stem}_depth.npy"
        if photo is None or not dsl.is_file() or not dpath.is_file():
            print(f"skip {stem}")
            continue
        floor_y, bay_x = load_grid(dsl)
        if not floor_y or not bay_x:
            print(f"skip {stem} (no grid)")
            continue
        image = Image.open(photo).convert("RGB")
        depth = resize_depth(np.load(dpath), image.width, image.height)
        windows = window_boxes_from_dsl(json.loads(dsl.read_text(encoding="utf-8")))
        gt = {tuple(c) for c in labels[stem]}
        names, rows, ys, cells = photo_rows(
            stem, image, depth, floor_y, bay_x, windows, sam3.get(stem, []), volumes.get(stem), gt
        )
        emb = dino_embed(dino, device, image, [cell_box(c, floor_y, bay_x) for c in cells])
        X += rows
        Y += ys
        G += [stem] * len(rows)
        C += cells
        D.append(emb)
        print(f"{stem}  cells={len(rows)}  pos={sum(ys)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(Y, dtype=np.int8),
        groups=np.asarray(G),
        names=np.asarray(names),
        cells=np.asarray(C, dtype=np.int32),
        dino=np.concatenate(D, axis=0).astype(np.float32),
    )
    print(f"saved {len(Y)} cells -> {args.out}")


if __name__ == "__main__":
    main()
