"""Build an isolated labeling package for an external labeler (Astra) from one balcony run.

Reads ``<balcony_dir>/balcony_units.json`` (written by ``run_balcony.py``) and the
unit crops it lists, the same PNGs the classifier reads in step 4. Writes:

- ``<package_dir>/crops/bNN.png``: crops renamed and shuffled (fixed seed), so type
  folders, unit order and the photo name are hidden.
- ``<package_dir>/PROMPT.md``: copied from ``--prompt`` when given.
- ``<work_dir>/answer_template.json``: one empty entry per id (not given to the labeler;
  the prompt defines the format).
- ``<work_dir>/mapping.json``: id → unit_index, box, type (kept OUT of the package).
- ``<work_dir>/classifier_labels.json``: our classifier's ``predict_full`` on the same
  PNGs, in ``--unit-labels`` format (for the control run).

Usage:
  python balcony_train/export_astra_package.py <balcony_dir> <package_dir> <work_dir>
      [--prompt PROMPT.md] [--railing-ckpt CKPT] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("balcony_dir", type=Path)
    ap.add_argument("package_dir", type=Path)
    ap.add_argument("work_dir", type=Path)
    ap.add_argument("--prompt", type=Path, default=None)
    ap.add_argument("--railing-ckpt", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-classifier", action="store_true")
    args = ap.parse_args()

    bdir = args.balcony_dir.resolve()
    info = json.loads((bdir / "balcony_units.json").read_text(encoding="utf-8"))
    units = info["units"]
    if not units:
        raise SystemExit("no open balcony units in this run")

    pkg = args.package_dir.resolve()
    work = args.work_dir.resolve()
    if work == pkg or pkg in work.parents or work in pkg.parents:
        raise SystemExit("work_dir must not overlap package_dir (mapping must stay hidden)")
    if pkg.exists() and any(pkg.iterdir()):
        raise SystemExit(f"package_dir not empty: {pkg}")
    (pkg / "crops").mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    order = list(range(len(units)))
    random.Random(args.seed).shuffle(order)
    width = max(2, len(str(len(units))))
    mapping = []
    for k, ui in enumerate(order, start=1):
        u = units[ui]
        cid = f"b{k:0{width}d}"
        shutil.copyfile(bdir / u["asset"], pkg / "crops" / f"{cid}.png")
        mapping.append(
            {
                "id": cid,
                "unit_index": int(u["unit_index"]),
                "box_xyxy": u["box_xyxy"],
                "type_id": u["type_id"],
                "asset": u["asset"],
            }
        )
    mapping.sort(key=lambda m: m["id"])

    template = {
        "answers": [
            {"id": m["id"], "kind": None, "material": None, "structure": None, "floor_shape": None}
            for m in mapping
        ]
    }
    (work / "answer_template.json").write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    if args.prompt is not None:
        shutil.copyfile(args.prompt, pkg / "PROMPT.md")

    (work / "mapping.json").write_text(
        json.dumps(
            {"stem": info.get("stem"), "balcony_dir": str(bdir), "package_dir": str(pkg),
             "seed": args.seed, "crops": mapping},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{len(mapping)} crops -> {pkg / 'crops'}")
    print(f"mapping -> {work / 'mapping.json'}")

    if args.no_classifier:
        return
    import torch
    from PIL import Image

    from balcony_train.model import try_load_predictor
    from balcony_train.paths import DEFAULT_STRUCTURE_CKPT

    pred = try_load_predictor(
        args.railing_ckpt, torch.device(args.device), structure_ckpt=DEFAULT_STRUCTURE_CKPT
    )
    if pred is None:
        raise SystemExit("railing classifier not found")
    rows = []
    for u in units:
        p = pred.predict_full(Image.open(bdir / u["asset"]).convert("RGB"))
        rows.append(
            {
                "unit_index": int(u["unit_index"]),
                "box_xyxy": u["box_xyxy"],
                "kind": p["kind"],
                "material": p["material"],
                "floor_shape": p["floor_shape"],
                "structure": p["structure"],
            }
        )
    (work / "classifier_labels.json").write_text(
        json.dumps({"source_ckpt": str(pred.ckpt_path), "units": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"classifier labels ({pred.ckpt_path.name}) -> {work / 'classifier_labels.json'}")


if __name__ == "__main__":
    main()
