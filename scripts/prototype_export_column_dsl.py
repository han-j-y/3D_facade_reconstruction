#!/usr/bin/env python3
"""Export recovery DSL with structural 5-column layout + colspan, then render."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from facade_recovery.column_layout import (
    assign_box_to_columns,
    assign_floors,
    infer_bay_column_bounds_from_units,
    infer_structural_columns,
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


base = _load(ROOT / "scripts" / "overlay_facade_asset_clusters.py", "exp_base")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--recovery",
        type=Path,
        required=True,
        help="source facade_dsl.json (types + structure IR preserved)",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--row-tol", type=float, default=0.055)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--blender", default=None)
    return ap.parse_args()


def type_name(type_id: int, kind: str = "window") -> str:
    if kind == "door":
        return f"door_arch_{type_id:02d}"
    return f"win_type_{type_id:02d}"


def build_column_dsl(recovery: dict[str, Any], columns: list[tuple[float, float]]) -> dict[str, Any]:
    iw, ih = recovery["meta"]["image_size"]
    instances = list(recovery.get("instances") or [])
    window_instances = [u for u in instances if u.get("kind", "window") != "door"]
    door_instances = [u for u in instances if u.get("kind") == "door"]

    boxes = [list(u["box_xyxy"]) for u in window_instances]
    cy = np.array([0.5 * (b[1] + b[3]) / ih for b in boxes], dtype=np.float64)
    floors = assign_floors(cy, 0.055)

    floor_ids = sorted(set(int(f) for f in floors.tolist()))
    bay_ids = list(range(len(columns)))
    floor_i = {f: i for i, f in enumerate(floor_ids)}
    n_floors = len(floor_ids)
    n_bays = len(bay_ids)

    placement: list[list[str | None]] = [[None for _ in bay_ids] for _ in floor_ids]
    placement_spans: list[list[int]] = [[1 for _ in bay_ids] for _ in floor_ids]

    updated_units: list[dict[str, Any]] = []
    for ui, u in enumerate(window_instances):
        box = list(u["box_xyxy"])
        bay, span = assign_box_to_columns(box, columns)
        floor = int(floors[ui])
        r, c = floor_i[floor], bay
        tid = type_name(int(u["type_id"]))
        placement[r][c] = tid
        placement_spans[r][c] = span
        for j in range(1, span):
            if c + j < n_bays:
                placement[r][c + j] = None
                placement_spans[r][c + j] = 0

        nu = copy.deepcopy(u)
        nu["floor"] = floor
        nu["bay"] = bay
        nu["colspan"] = span
        updated_units.append(nu)

    # Doors kept in instances only (ground parking not in window grid for render).
    for du in door_instances:
        nd = copy.deepcopy(du)
        box = list(du["box_xyxy"])
        bay, span = assign_box_to_columns(box, columns)
        nd["bay"] = bay
        nd["colspan"] = span
        updated_units.append(nd)

    row_heights = []
    for f in floor_ids:
        ys = [u["box_xyxy"] for u in updated_units if int(u["floor"]) == f]
        if not ys:
            row_heights.append(1.0)
            continue
        h = float(np.mean([b[3] - b[1] for b in ys])) / ih
        row_heights.append(max(0.05, h))

    col_widths = []
    refined_columns = infer_bay_column_bounds_from_units(
        updated_units,
        len(columns),
        iw=iw,
        structural_columns=columns,
    )
    for j, (xl, xr) in enumerate(refined_columns):
        col_widths.append(max(0.05, (xr - xl) / iw))

    out = copy.deepcopy(recovery)
    out["layout"] = {
        "floors": [
            {"name": f"F{f}", "id": int(f), "h_norm": float(row_heights[i])}
            for i, f in enumerate(floor_ids)
        ],
        "bays": [
            {"name": f"B{b}", "id": int(b), "w_norm": float(col_widths[i])}
            for i, b in enumerate(bay_ids)
        ],
        "placement": placement,
        "placement_spans": placement_spans,
    }
    out["instances"] = updated_units
    notes = (out.get("meta") or {}).get("notes", "")
    out.setdefault("meta", {})["notes"] = (
        notes + " | column_layout: 5 structural cols (L + 3 middle + R), colspan on wide bays."
    ).strip()
    return out


def draw_grid_overlay(
    facade: Image.Image,
    columns: list[tuple[float, float]],
    units: list[dict[str, Any]],
    *,
    title: str,
    out_path: Path,
) -> None:
    ih = facade.height
    im = facade.copy()
    draw = ImageDraw.Draw(im, "RGBA")
    try:
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font_sm = ImageFont.load_default()

    for j, (xl, xr) in enumerate(columns):
        x = int(round(xl))
        draw.line([(x, 0), (x, ih)], fill=(255, 220, 60, 200), width=2)
        draw.text((x + 2, 4), f"C{j}", fill=(255, 220, 60), font=font_sm)
    if columns:
        xr = int(round(columns[-1][1]))
        draw.line([(xr, 0), (xr, ih)], fill=(255, 220, 60, 200), width=2)

    palette = base.cluster_palette(8)
    for u in units:
        if u.get("kind") == "door":
            continue
        x0, y0, x1, y1 = [int(v) for v in u["box_xyxy"]]
        tid = int(u["type_id"])
        color = palette[tid % len(palette)]
        draw.rectangle([x0, y0, x1, y1], outline=color + (255,), width=2)
        tag = f"F{u['floor']} C{u['bay']}×{u.get('colspan', 1)} T{tid}"
        draw.rectangle([x0, max(0, y0 - 16), x0 + 7 * len(tag) + 6, y0], fill=color + (210,))
        draw.text((x0 + 2, max(0, y0 - 15)), tag, fill=(0, 0, 0), font=font_sm)

    bar = Image.new("RGB", (im.width, im.height + 28), (24, 24, 24))
    bar.paste(im, (0, 28))
    d = ImageDraw.Draw(bar)
    d.text((8, 6), title, fill=(230, 230, 230), font=font_sm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bar.save(out_path)


def print_placement_table(dsl: dict[str, Any]) -> None:
    placement = dsl["layout"]["placement"]
    spans = dsl["layout"].get("placement_spans") or []
    print("placement (row=floor F0=top):")
    for r, row in enumerate(placement):
        span_row = spans[r] if r < len(spans) else [1] * len(row)
        cells = []
        for c, tok in enumerate(row):
            if tok is None:
                cells.append(".")
            else:
                sp = span_row[c] if c < len(span_row) else 1
                cells.append(f"{tok}×{sp}" if sp > 1 else str(tok))
        print(f"  F{r}: " + " | ".join(cells))


def main() -> None:
    args = parse_args()
    recovery_path = args.recovery.expanduser().resolve()
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    facade_id = recovery.get("meta", {}).get("facade_id", recovery_path.stem)
    out_dir = (args.out_dir or ROOT / "runs" / "prototype_column_bays" / facade_id / "column_dsl").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    instances = recovery.get("instances") or []
    window_boxes = [list(u["box_xyxy"]) for u in instances if u.get("kind", "window") != "door"]
    iw, ih = recovery["meta"]["image_size"]
    cy = np.array([0.5 * (b[1] + b[3]) / ih for b in window_boxes], dtype=np.float64)
    floors = assign_floors(cy, args.row_tol)
    columns = infer_structural_columns(window_boxes, floors, iw=iw)

    dsl = build_column_dsl(recovery, columns)
    dsl_path = out_dir / "facade_dsl_column.json"
    dsl_path.write_text(json.dumps(dsl, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dsl_path}  grid={len(dsl['layout']['floors'])}×{len(dsl['layout']['bays'])}")
    print_placement_table(dsl)

    img_path = recovery.get("meta", {}).get("image")
    if img_path and Path(img_path).is_file():
        draw_grid_overlay(
            Image.open(img_path).convert("RGB"),
            columns,
            dsl["instances"],
            title=f"{facade_id} 5-column layout",
            out_path=out_dir / "grid_overlay.png",
        )

    meta = {
        "facade_id": facade_id,
        "source": str(recovery_path),
        "columns_xy": [[round(a, 1), round(b, 1)] for a, b in columns],
        "dsl": str(dsl_path),
    }
    (out_dir / "export_meta.json").write_text(json.dumps(meta, indent=2))

    if args.render:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "render_facade.py"),
            "--recovery",
            str(dsl_path),
            "--out-dir",
            str(out_dir / "blender"),
            "--render",
        ]
        if args.blender:
            cmd.extend(["--blender", args.blender])
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

    print(f"done → {out_dir}")


if __name__ == "__main__":
    main()
