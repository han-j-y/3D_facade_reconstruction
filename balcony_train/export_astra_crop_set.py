"""Export the test100 classifier test crops as anonymized packages for an external labeler (Astra).

Same crops the classifier table is scored on (default ``runs/test100/crops_final_gt``,
606 crops in ``<kind>/`` folders). The kind folders and photo names would leak the
answer, so each crop is copied flat as ``crops/cNNN.png`` in a shuffled order
(fixed seed) and split into packages of ``--chunk`` crops:

    <out_root>/part01/PROMPT.md, crops/c001.png ...
    <out_root>/part02/...

The id → file mapping is written to ``<work_dir>/crop_set_mapping.json``, outside
the packages.

Usage:
  python balcony_train/export_astra_crop_set.py <out_root> <work_dir> --prompt PROMPT.md
      [--dir runs/test100/crops_final_gt] [--chunk 100] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def find_crop(base: Path, row: dict) -> Path:
    for sub in (row.get("kind") or "", "open_work", "surface_panel", "solid", ""):
        p = base / sub / row["path"]
        if p.is_file():
            return p
    raise FileNotFoundError(row["path"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_root", type=Path)
    ap.add_argument("work_dir", type=Path)
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--dir", type=Path, default=ROOT / "runs" / "test100" / "crops_final_gt")
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = args.dir.resolve()
    rows = [json.loads(l) for l in (base / "labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out_root = args.out_root.resolve()
    work = args.work_dir.resolve()
    if work == out_root or out_root in work.parents or work in out_root.parents:
        raise SystemExit("work_dir must not overlap out_root (mapping must stay hidden)")
    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(f"out_root not empty: {out_root}")
    work.mkdir(parents=True, exist_ok=True)

    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    width = max(3, len(str(len(rows))))
    mapping = []
    for k, ri in enumerate(order):
        part = k // args.chunk + 1
        pdir = out_root / f"part{part:02d}"
        (pdir / "crops").mkdir(parents=True, exist_ok=True)
        if not (pdir / "PROMPT.md").exists():
            shutil.copyfile(args.prompt, pdir / "PROMPT.md")
        cid = f"c{k + 1:0{width}d}"
        src = find_crop(base, rows[ri])
        shutil.copyfile(src, pdir / "crops" / f"{cid}.png")
        mapping.append({"id": cid, "part": part, "path": rows[ri]["path"],
                        "src": str(src.relative_to(ROOT)) if ROOT in src.parents else str(src)})
    (work / "crop_set_mapping.json").write_text(
        json.dumps({"dir": str(base), "seed": args.seed, "chunk": args.chunk,
                    "out_root": str(out_root), "crops": mapping}, indent=2) + "\n",
        encoding="utf-8",
    )
    n_parts = (len(rows) + args.chunk - 1) // args.chunk
    print(f"{len(rows)} crops in {n_parts} packages -> {out_root}")
    print(f"mapping -> {work / 'crop_set_mapping.json'}")


if __name__ == "__main__":
    main()
