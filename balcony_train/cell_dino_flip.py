"""DINOv2 embeddings of horizontally flipped cell crops, row-aligned with a cells npz.

Used for flip augmentation: the flipped photo has the same depth features and
labels, only ``bay_rank`` and the appearance change.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/cell_dino_flip.py --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.cell_features_build import dino_embed  # noqa: E402
from balcony_train.depth_cells import cell_box  # noqa: E402
from balcony_train.enclosed_grid_store import DEFAULT_DSL_DIR, find_photo, load_grid  # noqa: E402

DEFAULT_CELLS = ROOT / "runs" / "cell_clf" / "cells_train.npz"
DEFAULT_IN_DIR = ROOT / "data" / "base_jpg"
DEFAULT_OUT = ROOT / "runs" / "cell_clf" / "cells_train_dino_flip.npy"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    import torch

    z = np.load(args.cells, allow_pickle=False)
    groups = z["groups"].astype(str)
    cells = [tuple(map(int, c)) for c in z["cells"]]
    out = np.zeros_like(z["dino"], dtype=np.float32)
    device = torch.device(args.device)
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True).to(device).eval()
    for stem in sorted(set(groups)):
        rows = np.where(groups == stem)[0]
        floor_y, bay_x = load_grid(args.dsl_dir / stem / "facade_dsl.json")
        image = Image.open(find_photo(args.in_dir, stem)).convert("RGB")
        w = image.width
        boxes = []
        for i in rows:
            x0, y0, x1, y1 = cell_box(cells[i], floor_y, bay_x)
            boxes.append([w - x1, y0, w - x0, y1])
        out[rows] = dino_embed(dino, device, ImageOps.mirror(image), boxes)
        print(f"{stem}  cells={len(rows)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out)
    print(f"saved {out.shape} -> {args.out}")


if __name__ == "__main__":
    main()
