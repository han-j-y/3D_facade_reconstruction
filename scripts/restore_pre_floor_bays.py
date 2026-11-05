"""Put back the window grids and cell labels from before the floor-bay update.

The snapshot lives in ``runs/enclosed_grid/backup_pre_floor_bays``. Running this
script copies those files over the current ones. It does not delete the snapshot.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe scripts/restore_pre_floor_bays.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "runs" / "enclosed_grid" / "backup_pre_floor_bays"


def main() -> None:
    if not BACKUP.is_dir():
        raise SystemExit(f"no snapshot: {BACKUP}")
    dsl_root = BACKUP / "dsl"
    stems = sorted(p.name for p in dsl_root.iterdir() if (p / "facade_dsl.json").is_file())
    if not stems:
        raise SystemExit(f"snapshot has no grids: {dsl_root}")
    for stem in stems:
        src = dsl_root / stem / "facade_dsl.json"
        dest = ROOT / "runs" / "batch" / stem / "facade_dsl.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    restored_labels = []
    for name in ("labels_ext_all.jsonl", "labels_eval.jsonl"):
        src = BACKUP / name
        if not src.is_file():
            continue
        dest = ROOT / "runs" / "enclosed_grid" / name
        shutil.copy2(src, dest)
        restored_labels.append(name)
    print(f"restored {len(stems)} grids")
    print("restored labels: " + ", ".join(restored_labels))
    print(f"snapshot kept at {BACKUP}")


if __name__ == "__main__":
    sys.exit(main())
