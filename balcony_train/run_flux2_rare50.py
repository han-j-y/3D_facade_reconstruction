"""Generate 50 images each for the three rare balcony sets, one model load.

Start this on GruVi, confirm the log is moving, then the laptop can shut
down. The process stays on the remote machine.

    module load LANG/PYTHON/3.11.0
    source ~/venvs/flux2/bin/activate
    cd ~/3D_facade_reconstruction
    mkdir -p runs/balcony_clf
    nohup python -u balcony_train/run_flux2_rare50.py > runs/balcony_clf/flux2_rare50.log 2>&1 &
    echo $! | tee runs/balcony_clf/flux2_rare50.pid

Previous ``flux2_triangle_``, ``flux2_half_enclosed_``, and
``flux2_enclosed_`` files in unlabeled are
removed first so the batch is exactly 50 each. Pass ``--keep-existing`` to
leave them and append.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.generate_flux2 import (  # noqa: E402
    DEFAULT_QUANTIZED_MODEL,
    generate_prompt_sets,
)
from balcony_train.labels import ensure_crop_dirs  # noqa: E402
from balcony_train.paths import DEFAULT_CROPS_DIR  # noqa: E402

RARE_SETS: tuple[str, ...] = (
    "triangle",
    "half_enclosed",
    "enclosed",
)
CLEAR_PREFIXES: tuple[str, ...] = (
    "flux2_triangle_",
    "flux2_half_enclosed_",
    "flux2_enclosed_",
)


def clear_previous(crops_dir: Path) -> int:
    unlabeled = Path(crops_dir) / "unlabeled"
    if not unlabeled.is_dir():
        return 0
    removed = 0
    for path in unlabeled.iterdir():
        if path.is_file() and path.name.startswith(CLEAR_PREFIXES):
            path.unlink()
            removed += 1
    return removed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS_DIR)
    ap.add_argument("-n", "--count", type=int, default=50)
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--keep-existing", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance-scale", type=float, default=4.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=384)
    args = ap.parse_args()

    ensure_crop_dirs(args.crops_dir)
    if args.keep_existing:
        print("keep existing unlabeled files", flush=True)
    else:
        removed = clear_previous(args.crops_dir)
        print(f"removed {removed} previous rare crops from unlabeled", flush=True)
    print(
        f"generating {args.count} each of {', '.join(RARE_SETS)}",
        flush=True,
    )
    generate_prompt_sets(
        args.crops_dir,
        list(RARE_SETS),
        count=max(1, int(args.count)),
        seed=int(args.seed),
        model_id=DEFAULT_QUANTIZED_MODEL,
        width=int(args.width),
        height=int(args.height),
        steps=int(args.steps),
        guidance_scale=float(args.guidance_scale),
        device=str(args.device),
        cpu_offload=True,
    )
    print("batch done", flush=True)


if __name__ == "__main__":
    main()
