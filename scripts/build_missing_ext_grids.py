"""Write window facade_dsl.json for Extended photos that do not have one.

Uses CMP XML window boxes (x = row, y = column) and the same floor/bay
geometry as the recovery pipeline. Skips stems that already have
runs/batch/<stem>/facade_dsl.json. Does not run SAM, DINO, or structure.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe scripts/build_missing_ext_grids.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from facade_recovery.cmp_windows import load_cmp_xml_windows  # noqa: E402
from facade_recovery.column_layout import (  # noqa: E402
    assign_floors,
    infer_window_bay_columns,
)
from run_pipeline import build_facade_dsl  # noqa: E402

PHOTO_DIR = ROOT / "data" / "extended"
BATCH_ROOT = ROOT / "runs" / "batch"
ROW_TOL = 0.055


def missing_photos() -> list[Path]:
    photos = sorted(PHOTO_DIR.glob("*.jpg"))
    return [
        photo
        for photo in photos
        if not (BATCH_ROOT / photo.stem / "facade_dsl.json").is_file()
    ]


def grid_for_photo(photo: Path) -> dict:
    xml_path = photo.with_suffix(".xml")
    if not xml_path.is_file():
        raise FileNotFoundError(f"missing CMP XML: {xml_path}")
    with Image.open(photo) as im:
        image_size = im.size
    raw = load_cmp_xml_windows(xml_path, image_size, stem=photo.stem)
    if not raw:
        raise ValueError("no XML windows")
    boxes = [[int(v) for v in w["box_xyxy"]] for w in raw]
    w, h = image_size
    cy = np.array([((b[1] + b[3]) / 2.0) / h for b in boxes], dtype=np.float64)
    floors = assign_floors(cy, ROW_TOL)
    units = [
        {
            "id": f"{photo.stem}_w{i}",
            "box_xyxy": box,
            "floor": int(floors[i]),
            "bay": 0,
            "colspan": 1,
            "type_id": 0,
            "kind": "window",
        }
        for i, box in enumerate(boxes)
    ]
    columns = infer_window_bay_columns(units, w)
    types = [
        {
            "type_id": 0,
            "name": "win_type_00",
            "n_instances": len(units),
            "exemplar_asset": "",
        }
    ]
    dsl = build_facade_dsl(
        facade_id=photo.stem,
        image_path=str(photo),
        image_size=image_size,
        units=units,
        types=types,
        columns=columns,
    )
    dsl["meta"]["windows"] = "cmp_xml"
    dsl["meta"]["grid"] = "xml_geometry"
    return dsl


def main() -> None:
    photos = missing_photos()
    print(f"missing grids: {len(photos)}", flush=True)
    failures: list[tuple[str, str]] = []
    for i, photo in enumerate(photos, start=1):
        out_dir = BATCH_ROOT / photo.stem
        try:
            dsl = grid_for_photo(photo)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "facade_dsl.json").write_text(
                json.dumps(dsl, indent=2),
                encoding="utf-8",
            )
            n_f = len(dsl["layout"]["floors"])
            n_b = len(dsl["layout"]["bays"])
            print(
                f"[{i}/{len(photos)}] {photo.stem}  windows={dsl['meta']['n_units']}  "
                f"floors={n_f} bays={n_b}",
                flush=True,
            )
        except (OSError, ValueError, FileNotFoundError) as exc:
            failures.append((photo.name, str(exc)))
            print(f"[{i}/{len(photos)}] skip {photo.name}: {exc}", flush=True)
    ok = len(photos) - len(failures)
    print(f"done: {ok}/{len(photos)} new grids", flush=True)
    if failures:
        print("failures:", flush=True)
        for name, err in failures:
            print(f"  {name}: {err}", flush=True)


if __name__ == "__main__":
    main()
