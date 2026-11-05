"""Per-cell features for labeled photos, computed like the pipeline does.

Same ``.npz`` layout as ``cell_features_build.py`` (X, y, groups, names,
cells, dino), but the OP1 volumes are recomputed on the grid in ``--dsl-dir``
through ``enclosed_detect.photo_cell_data``, so training, evaluation and
``--enclosed-method cell_clf`` share one feature path. SAM3 seeds and DA3
depth do not depend on the grid; missing ones are computed into the caches.
Grid-failure rows are skipped.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/cell_features_pipeline.py ^
      --labels runs/enclosed_grid/labels_train.jsonl --photo-dir data/base ^
      --dsl-dir runs/batch_grid2 --out runs/cell_clf/cells_train.npz
"""

from __future__ import annotations

import argparse
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

from balcony_train.enclosed_detect import SAM3_THRESHOLD, photo_cell_data  # noqa: E402
from balcony_train.enclosed_grid_store import GRID_FAILURE, find_photo, load_records  # noqa: E402
from balcony_train.evaluate_enclosed_grid import DEFAULT_SAM3_CACHE  # noqa: E402
from balcony_train.try_da3_enclosure import resize_depth  # noqa: E402
from balcony_train.try_sam3_da3_volumes import DEFAULT_DEPTH_DIR, run_sam3  # noqa: E402
from enclosed import ensure_da3_depth  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--photo-dir", type=Path, action="append", required=True, help="repeatable; first match wins")
    ap.add_argument("--dsl-dir", type=Path, required=True)
    ap.add_argument("--depth-dir", type=Path, default=DEFAULT_DEPTH_DIR)
    ap.add_argument("--sam3-cache", type=Path, default=DEFAULT_SAM3_CACHE)
    ap.add_argument("--stems", nargs="+", default=None, help="only these photos")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import json

    import torch

    records = load_records(args.labels)
    photos: dict[str, Path] = {}
    for stem, rec in sorted(records.items()):
        if rec.status == GRID_FAILURE or (args.stems and stem not in args.stems):
            continue
        if not (args.dsl_dir / stem / "facade_dsl.json").is_file():
            print(f"skip {stem}: no grid in {args.dsl_dir}")
            continue
        path = next((p for d in args.photo_dir if (p := find_photo(d, stem)) is not None), None)
        if path is None:
            print(f"skip {stem}: no photo")
            continue
        photos[stem] = path
    seeds = run_sam3(list(photos.values()), device=torch.device(args.device), cache_path=args.sam3_cache, threshold=SAM3_THRESHOLD)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    X, Y, G, C, D = [], [], [], [], []
    names: list[str] = []
    for stem, path in photos.items():
        depth_path = ensure_da3_depth(path, args.depth_dir, device=args.device)
        if depth_path is None:
            print(f"skip {stem}: no depth")
            continue
        image = Image.open(path).convert("RGB")
        depth = resize_depth(np.load(depth_path), image.width, image.height)
        dsl = json.loads((args.dsl_dir / stem / "facade_dsl.json").read_text(encoding="utf-8"))
        gt = {tuple(c) for c in records[stem].cells}
        data = photo_cell_data(stem, image, dsl, depth, seeds.get(stem, []), device=args.device, gt=gt)
        if data is None:
            print(f"skip {stem}: no cells")
            continue
        missing = gt - set(data.cells)
        if missing:
            print(f"warn {stem}: labeled cells not on the grid {sorted(missing)}")
        names = data.names
        X.append(data.X)
        Y.append(data.y)
        G += [stem] * len(data.cells)
        C += data.cells
        D.append(data.dino)
        print(f"{stem}  cells={len(data.cells)}  pos={int(data.y.sum())}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X=np.concatenate(X).astype(np.float32),
        y=np.concatenate(Y).astype(np.int8),
        groups=np.asarray(G),
        names=np.asarray(names),
        cells=np.asarray(C, dtype=np.int32),
        dino=np.concatenate(D).astype(np.float32),
    )
    print(f"wrote {args.out}: {len(photos)} photos, {len(G)} cells, {int(sum(y.sum() for y in Y))} positive")


if __name__ == "__main__":
    main()
