"""Run the SAM3+DA3 volume detector once per labeled photo and cache its output.

Saves every stage's boxes plus the Stage 5 / Stage 6 (symmetry) cells, so
cell experiments can reuse the current method without SAM3 or DA3.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/cache_volume_cells.py --device cuda
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
    DEFAULT_EVAL_PHOTO_DIR,
    find_photo,
    load_grid,
    load_labels,
)
from balcony_train.enclosed_symmetry import window_boxes_from_dsl  # noqa: E402
from balcony_train.evaluate_enclosed_grid import (  # noqa: E402
    DEFAULT_DEPTH_DIR,
    DEFAULT_SAM3_CACHE,
    _floor_h,
    cells_from_boxes,
)

DEFAULT_OUT = ROOT / "runs" / "enclosed_grid" / "volume_cells_floor_bays.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=DEFAULT_EVAL_LABELS_JSONL)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_EVAL_PHOTO_DIR)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL_DIR)
    ap.add_argument("--depth-dir", type=Path, default=DEFAULT_DEPTH_DIR)
    ap.add_argument("--sam3-cache", type=Path, default=DEFAULT_SAM3_CACHE)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    import torch
    from PIL import Image

    from balcony_pipeline.run_balcony import enclosed_volume_stages

    out: dict[str, dict] = {}
    if args.out.is_file():
        out = json.loads(args.out.read_text(encoding="utf-8"))
    for stem in sorted(load_labels(args.labels)):
        if stem in out:
            continue
        photo = find_photo(args.in_dir, stem)
        dsl = args.dsl_dir / stem / "facade_dsl.json"
        if photo is None or not dsl.is_file():
            continue
        floor_y, bay_x = load_grid(dsl)
        if not floor_y or not bay_x:
            continue
        windows = window_boxes_from_dsl(json.loads(dsl.read_text(encoding="utf-8")))
        image = Image.open(photo).convert("RGB")
        stages = enclosed_volume_stages(
            image,
            photo,
            device=torch.device(args.device),
            depth_dir=args.depth_dir,
            cache_path=args.sam3_cache,
            floor_h=_floor_h(floor_y),
            floor_y=floor_y,
            bay_x=bay_x,
            windows=windows,
            symmetry=True,
        )
        by_id = {sid: boxes for sid, _t, boxes in stages}
        rec = {
            "stages": {
                sid: [list(map(int, b["box_xyxy"])) for b in boxes] for sid, boxes in by_id.items()
            },
            "cells_5": sorted(cells_from_boxes(by_id.get("5_clip", []), floor_y, bay_x)),
            "cells_6": sorted(cells_from_boxes(by_id.get("6_sym", by_id.get("5_clip", [])), floor_y, bay_x)),
        }
        out[stem] = rec
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out) + "\n", encoding="utf-8")
        print(f"{stem}  cells_6={len(rec['cells_6'])}", flush=True)
    print(f"cached {len(out)} -> {args.out}")


if __name__ == "__main__":
    main()
