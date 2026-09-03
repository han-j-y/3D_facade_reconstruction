#!/usr/bin/env python3
"""Prototype: edge-based column inference + cross-floor alignment (Option A).

Does NOT touch run_pipeline.py. Compares greedy centroid bays vs column-overlap bays.

Example:
  python scripts/prototype_column_bays.py --facade-ids cmp_b0010,cmp_b0100
  python scripts/prototype_column_bays.py --index runs/.../index_raw.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


base = _load(ROOT / "scripts" / "overlay_facade_asset_clusters.py", "proto_base")
lay = _load(ROOT / "scripts" / "overlay_facade_layout_first.py", "proto_lay")
merge_mod = _load(ROOT / "scripts" / "overlay_facade_merge_boxes.py", "proto_merge")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--facade-ids",
        type=str,
        default="cmp_b0010,cmp_b0100",
        help="comma-separated facade stems under data/facades/base/",
    )
    ap.add_argument("--index", type=Path, default=None, help="reuse index_raw.json boxes")
    ap.add_argument(
        "--facade-dir",
        type=Path,
        default=EXP / "data" / "facades" / "base",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "runs" / "prototype_column_bays",
    )
    ap.add_argument("--col-tol", type=float, default=0.04, help="old centroid bay tol (norm x)")
    ap.add_argument("--row-tol", type=float, default=0.055, help="floor tol (norm y)")
    ap.add_argument(
        "--boundary-tol",
        type=float,
        default=0.035,
        help="merge nearby column boundaries (norm x)",
    )
    ap.add_argument(
        "--floor-gap",
        type=float,
        default=0.55,
        help="merge box intervals on same floor when gap <= frac * med_w",
    )
    ap.add_argument(
        "--min-col-support",
        type=float,
        default=0.22,
        help="min overlap fraction to keep a global column",
    )
    return ap.parse_args()


def merge_intervals(
    intervals: list[tuple[float, float]],
    gap: float,
) -> list[tuple[float, float]]:
    """Merge [x0,x1] intervals separated by gap <= gap (pixels)."""
    if not intervals:
        return []
    iv = sorted(intervals, key=lambda t: t[0])
    out: list[tuple[float, float]] = []
    cur_l, cur_r = iv[0]
    for l, r in iv[1:]:
        if l <= cur_r + gap:
            cur_r = max(cur_r, r)
        else:
            out.append((cur_l, cur_r))
            cur_l, cur_r = l, r
    out.append((cur_l, cur_r))
    return out


def cluster_boundaries(xs: list[float], tol: float) -> list[float]:
    """1D greedy cluster of boundary positions → cluster centers."""
    if not xs:
        return []
    order = sorted(xs)
    centers: list[float] = []
    acc = [order[0]]
    for x in order[1:]:
        if x - acc[-1] <= tol:
            acc.append(x)
        else:
            centers.append(float(np.mean(acc)))
            acc = [x]
    centers.append(float(np.mean(acc)))
    return centers


def x_overlap_frac(box: list[int], col: tuple[float, float]) -> float:
    x0, x1 = float(box[0]), float(box[2])
    cl, cr = col
    inter = max(0.0, min(x1, cr) - max(x0, cl))
    bw = max(1.0, x1 - x0)
    return inter / bw


def infer_global_columns(
    boxes: list[list[int]],
    floors: np.ndarray,
    *,
    iw: int,
    boundary_tol: float,
    floor_gap_frac: float,
    min_col_support: float,
) -> list[tuple[float, float]]:
    """Per-floor interval merge → boundary voting → global column spans."""
    med_w = float(np.median([max(1.0, b[2] - b[0]) for b in boxes]))
    floor_gap = floor_gap_frac * med_w
    b_tol = boundary_tol * iw

    boundaries: list[float] = []
    per_floor_cols: dict[int, list[tuple[float, float]]] = {}

    for f in sorted(set(int(v) for v in floors.tolist())):
        idxs = [i for i in range(len(boxes)) if int(floors[i]) == f]
        ivs = [(float(boxes[i][0]), float(boxes[i][2])) for i in idxs]
        cols = merge_intervals(ivs, floor_gap)
        per_floor_cols[f] = cols
        for l, r in cols:
            boundaries.append(l)
            boundaries.append(r)

    if not boundaries:
        return []

    merged_b = cluster_boundaries(boundaries, b_tol)
    if len(merged_b) < 2:
        return []

    cols: list[tuple[float, float]] = []
    for i in range(len(merged_b) - 1):
        l, r = merged_b[i], merged_b[i + 1]
        if r - l < 0.02 * iw:
            continue
        cols.append((l, r))

    # drop columns with weak support (no box overlaps enough on any floor)
    kept: list[tuple[float, float]] = []
    for col in cols:
        best = 0.0
        for i, box in enumerate(boxes):
            best = max(best, x_overlap_frac(box, col))
        if best >= min_col_support:
            kept.append(col)
    return kept


def assign_columns_by_overlap(
    boxes: list[list[int]],
    columns: list[tuple[float, float]],
) -> np.ndarray:
    bay = np.full(len(boxes), -1, dtype=np.int32)
    if not columns:
        return bay
    for i, box in enumerate(boxes):
        scores = [x_overlap_frac(box, col) for col in columns]
        j = int(np.argmax(scores))
        if scores[j] > 0.05:
            bay[i] = j
    return bay


def assign_centroid_bays(cx: np.ndarray, tol: float) -> np.ndarray:
    return lay.assign_bays(cx, tol)


def load_boxes_and_image(
    facade_id: str,
    facade_dir: Path,
    index_path: Path | None,
) -> tuple[Image.Image, list[list[int]], str]:
    if index_path is not None:
        data = json.loads(index_path.read_text())
        img_path = Path(data.get("facade_path") or data.get("image") or "")
        if not img_path.is_file():
            stem = data.get("facade_id", facade_id)
            img_path = facade_dir / f"{stem}.jpg"
        boxes = [list(w["box_xyxy"]) for w in data.get("windows") or data.get("boxes") or []]
        return Image.open(img_path).convert("RGB"), boxes, str(img_path)

    img_path = facade_dir / f"{facade_id}.jpg"
    if not img_path.is_file():
        img_path = facade_dir / f"{facade_id}.png"
    idx = facade_dir / f"{facade_id}_index.json"
    if idx.is_file():
        data = json.loads(idx.read_text())
        boxes = [list(w["box_xyxy"]) for w in data.get("windows") or []]
        return Image.open(img_path).convert("RGB"), boxes, str(img_path)

    raise FileNotFoundError(f"no boxes for {facade_id}: pass --index")


def draw_column_compare(
    facade: Image.Image,
    boxes: list[list[int]],
    floors: np.ndarray,
    bay_old: np.ndarray,
    bay_new: np.ndarray,
    columns: list[tuple[float, float]],
    *,
    title: str,
) -> Image.Image:
    iw, ih = facade.size
    bar_h = 40
    gap = 8
    panel_w = iw
    panel_h = ih + bar_h
    canvas = Image.new("RGB", (panel_w * 2 + gap, panel_h), (24, 24, 24))

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = font_sm = ImageFont.load_default()

    def draw_panel(ix: int, labels: np.ndarray, label_title: str, show_cols: bool) -> None:
        im = facade.copy()
        draw = ImageDraw.Draw(im, "RGBA")
        if show_cols:
            for l, r in columns:
                x = int(round(l))
                draw.line([(x, 0), (x, ih)], fill=(255, 220, 60, 180), width=2)
            if columns:
                draw.line([(int(round(columns[-1][1])), 0), (int(round(columns[-1][1])), ih)],
                          fill=(255, 220, 60, 180), width=2)

        uniq = sorted(set(int(v) for v in labels.tolist() if int(v) >= 0))
        palette = base.cluster_palette(max(1, len(uniq)))
        remap = {u: i for i, u in enumerate(uniq)}

        for i, box in enumerate(boxes):
            lab = int(labels[i])
            if lab < 0:
                continue
            x0, y0, x1, y1 = [int(v) for v in box]
            cid = remap.get(lab, 0)
            color = palette[cid % len(palette)]
            draw.rectangle([x0, y0, x1, y1], outline=color + (255,), width=2)
            draw.rectangle([x0, y0, x1, y1], fill=color + (45,))
            tag = f"B{lab} F{int(floors[i])}"
            draw.rectangle([x0, max(0, y0 - 16), x0 + 8 * len(tag) + 6, y0], fill=color + (210,))
            draw.text((x0 + 2, max(0, y0 - 15)), tag, fill=(0, 0, 0), font=font_sm)

        canvas.paste(im, (ix * (panel_w + gap), bar_h))
        d = ImageDraw.Draw(canvas)
        d.text((ix * (panel_w + gap) + 10, 10), label_title, fill=(235, 235, 235), font=font)

    draw_panel(0, bay_old, "centroid bays (current)", False)
    draw_panel(1, bay_new, "column overlap (prototype)", True)

    d = ImageDraw.Draw(canvas)
    d.text((10, panel_h - 22), title, fill=(200, 200, 200), font=font_sm)
    return canvas


def process_one(
    facade_id: str,
    facade_dir: Path,
    out_dir: Path,
    index_path: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    facade, raw_boxes, img_path = load_boxes_and_image(facade_id, facade_dir, index_path)
    iw, ih = facade.size
    if len(raw_boxes) < 2:
        raise ValueError(f"{facade_id}: need >= 2 boxes")

    cx = np.array([0.5 * (b[0] + b[2]) / iw for b in raw_boxes], dtype=np.float64)
    cy = np.array([0.5 * (b[1] + b[3]) / ih for b in raw_boxes], dtype=np.float64)
    floors_raw = lay.assign_bays(cy, args.row_tol)

    # merged units (same as pipeline unitize step)
    merged_boxes, members, bay_raw, floor_raw = merge_mod.merge_adjacent_boxes(
        raw_boxes,
        cx,
        cy,
        row_tol=args.row_tol,
        adj_gap=1.0,
        merge_bays=False,
        col_tol=args.col_tol,
    )
    mcx = np.array([0.5 * (b[0] + b[2]) / iw for b in merged_boxes], dtype=np.float64)
    mcy = np.array([0.5 * (b[1] + b[3]) / ih for b in merged_boxes], dtype=np.float64)
    floors_m = merge_mod.lay.assign_bays(mcy, args.row_tol)

    bay_old_raw = assign_centroid_bays(cx, args.col_tol)
    columns = infer_global_columns(
        raw_boxes,
        floors_raw,
        iw=iw,
        boundary_tol=args.boundary_tol,
        floor_gap_frac=args.floor_gap,
        min_col_support=args.min_col_support,
    )
    bay_new_raw = assign_columns_by_overlap(raw_boxes, columns)

    bay_old_m = assign_centroid_bays(mcx, args.col_tol)
    bay_new_m = assign_columns_by_overlap(merged_boxes, columns)

    def n_bays(lab: np.ndarray) -> int:
        vals = [int(v) for v in lab.tolist() if int(v) >= 0]
        return len(set(vals))

    stem_dir = out_dir / facade_id
    stem_dir.mkdir(parents=True, exist_ok=True)

    overlay_raw = draw_column_compare(
        facade,
        raw_boxes,
        floors_raw,
        bay_old_raw,
        bay_new_raw,
        columns,
        title=f"{facade_id} raw n={len(raw_boxes)} cols={len(columns)}",
    )
    overlay_raw.save(stem_dir / "compare_raw.png")

    overlay_merged = draw_column_compare(
        facade,
        merged_boxes,
        floors_m,
        bay_old_m,
        bay_new_m,
        columns,
        title=f"{facade_id} merged n={len(merged_boxes)} cols={len(columns)}",
    )
    overlay_merged.save(stem_dir / "compare_merged.png")

    report = {
        "facade_id": facade_id,
        "image": img_path,
        "image_size": [iw, ih],
        "n_raw": len(raw_boxes),
        "n_merged": len(merged_boxes),
        "n_columns": len(columns),
        "columns_xy": [[round(l, 1), round(r, 1)] for l, r in columns],
        "bays_centroid_raw": n_bays(bay_old_raw),
        "bays_column_raw": n_bays(bay_new_raw),
        "bays_centroid_merged": n_bays(bay_old_m),
        "bays_column_merged": n_bays(bay_new_m),
        "raw_assignments": {
            "centroid": [int(v) for v in bay_old_raw.tolist()],
            "column": [int(v) for v in bay_new_raw.tolist()],
            "floors": [int(v) for v in floors_raw.tolist()],
        },
        "merged_assignments": {
            "centroid": [int(v) for v in bay_old_m.tolist()],
            "column": [int(v) for v in bay_new_m.tolist()],
            "floors": [int(v) for v in floors_m.tolist()],
            "members": members,
        },
        "params": {
            "col_tol": args.col_tol,
            "row_tol": args.row_tol,
            "boundary_tol": args.boundary_tol,
            "floor_gap": args.floor_gap,
            "min_col_support": args.min_col_support,
        },
    }
    (stem_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(
        f"{facade_id}: raw {report['bays_centroid_raw']}→{report['bays_column_raw']} bays, "
        f"merged {report['bays_centroid_merged']}→{report['bays_column_merged']} bays, "
        f"cols={len(columns)}"
    )
    return report


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.index is not None:
        facade_id = json.loads(args.index.read_text()).get("facade_id", "custom")
        reports = [process_one(facade_id, args.facade_dir, args.out_dir, args.index, args)]
    else:
        reports = []
        for fid in [s.strip() for s in args.facade_ids.split(",") if s.strip()]:
            idx = (
                ROOT / "runs" / f"facade_e2e_base_{fid}" / "index_raw.json"
            )
            index_path = idx if idx.is_file() else None
            if index_path is None:
                alt = EXP / "window_ast_predictor" / "runs" / f"facade_e2e_base_{fid}" / "index_raw.json"
                index_path = alt if alt.is_file() else None
            reports.append(process_one(fid, args.facade_dir, args.out_dir, index_path, args))

    summary = {"facades": reports, "out_dir": str(args.out_dir)}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done → {args.out_dir}")


if __name__ == "__main__":
    main()
