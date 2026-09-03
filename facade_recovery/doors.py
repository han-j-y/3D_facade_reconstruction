"""Door SAM detections → layout cells and door-only type clustering."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from facade_recovery.box_cluster import cluster_box_geometry_pipeline
from facade_recovery.door_types import door_type_token

EXP_ROOT = Path(__file__).resolve().parents[2]


def box_iou(a: list[int], b: list[int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    return float(inter) / float(area_a + area_b - inter)


def filter_door_boxes(
    door_boxes: list[dict[str, Any]],
    window_boxes: list[list[int]],
    *,
    image_size: tuple[int, int],
    min_y_frac: float,
    iou_nms: float,
) -> list[dict[str, Any]]:
    """Keep ground-biased door detections that do not overlap windows."""
    iw, ih = image_size
    kept: list[dict[str, Any]] = []
    for d in door_boxes:
        x0, y0, x1, y1 = d["box_xyxy"]
        cy = 0.5 * (y0 + y1) / max(ih, 1)
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        if cy < float(min_y_frac):
            continue
        if bh / bw < 1.05:
            continue
        if any(box_iou(d["box_xyxy"], wb) >= iou_nms for wb in window_boxes):
            continue
        kept.append(d)
    kept.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    final: list[dict[str, Any]] = []
    for d in kept:
        if any(box_iou(d["box_xyxy"], e["box_xyxy"]) >= 0.45 for e in final):
            continue
        final.append(d)
    return final


def assign_doors_to_layout(
    door_boxes: list[dict[str, Any]],
    window_units: list[dict[str, Any]],
    *,
    image_size: tuple[int, int],
    below_floor_frac: float = 0.55,
) -> list[dict[str, Any]]:
    """Map doors to floor×bay; shopfront doors get a new ground floor below windows."""
    if not door_boxes:
        return []
    iw, ih = image_size
    floor_ys: dict[int, list[float]] = defaultdict(list)
    bay_xs: dict[int, list[float]] = defaultdict(list)
    for u in window_units:
        x0, y0, x1, y1 = u["box_xyxy"]
        floor_ys[int(u["floor"])].append(0.5 * (y0 + y1) / ih)
        bay_xs[int(u["bay"])].append(0.5 * (x0 + x1) / iw)
    if not floor_ys or not bay_xs:
        return []
    floor_c = {f: float(np.mean(vs)) for f, vs in floor_ys.items()}
    bay_c = {b: float(np.mean(vs)) for b, vs in bay_xs.items()}
    ground = max(floor_c.items(), key=lambda kv: kv[1])[0]
    sorted_floors = sorted(floor_c.items(), key=lambda kv: kv[1])
    if len(sorted_floors) >= 2:
        spacings = [
            sorted_floors[i + 1][1] - sorted_floors[i][1]
            for i in range(len(sorted_floors) - 1)
        ]
        floor_pitch = float(np.median(spacings))
    else:
        floor_pitch = 0.12
    floor_pitch = max(0.06, floor_pitch)
    ground_extra = max(floor_c.keys()) + 1
    next_ground = ground_extra

    out: list[dict[str, Any]] = []
    for i, d in enumerate(door_boxes):
        x0, y0, x1, y1 = d["box_xyxy"]
        cx = 0.5 * (x0 + x1) / iw
        cy = 0.5 * (y0 + y1) / ih
        bay = min(bay_c.keys(), key=lambda b: abs(bay_c[b] - cx))
        gap_below = cy - floor_c[ground]
        if gap_below >= below_floor_frac * floor_pitch:
            floor = next_ground
            floor_c.setdefault(floor, cy)
        else:
            floor = min(floor_c.keys(), key=lambda f: abs(floor_c[f] - cy))
            if abs(cy - floor_c[ground]) <= abs(cy - floor_c[floor]) + 0.08:
                floor = ground
        print(
            f"  door[{i}] floor={floor} bay={bay} "
            f"gap_below={gap_below:.3f} pitch={floor_pitch:.3f}"
        )
        out.append(
            {
                "unit_id": f"door_{i}",
                "kind": "door",
                "box_xyxy": d["box_xyxy"],
                "floor": int(floor),
                "bay": int(bay),
                "colspan": 1,
                "score": float(d.get("score", 0.0)),
                "is_exemplar": False,
            }
        )
    return out


def classify_door_shape(
    door_unit: dict[str, Any],
    facade: Image.Image,
    *,
    rise_thr: float = 0.08,
) -> dict[str, Any]:
    """Rectangle vs arch from crop (mask-free fallback)."""
    metrics: dict[str, Any] = {
        "shape": "rectangle",
        "rise_ratio": 0.0,
        "source": "default",
    }
    try:
        pred_root = str(EXP_ROOT / "window_ast_predictor")
        if pred_root not in sys.path:
            sys.path.insert(0, pred_root)
        from window_ast.door_shape import classify_door_opening

        crop = facade.crop(tuple(door_unit["box_xyxy"]))
        metrics = classify_door_opening(
            mask=door_unit.get("mask"),
            crop=crop,
            rise_thr=rise_thr,
        )
    except Exception:
        pass
    return metrics


def cluster_door_units(
    door_units: list[dict[str, Any]],
    facade: Image.Image,
    *,
    image_size: tuple[int, int],
    base: Any,
    seed: int = 42,
    k_max: int = 4,
    box_method: str = "gmm_diag",
    rise_thr: float = 0.08,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cluster doors among themselves (box geometry); separate from window types."""
    if not door_units:
        return [], []

    for du in door_units:
        metrics = classify_door_shape(du, facade, rise_thr=rise_thr)
        du["shape"] = str(metrics.get("shape") or "rectangle")
        du["arch_rise_ratio"] = float(metrics.get("rise_ratio") or 0.0)
        du["shape_metrics"] = {
            k: metrics[k]
            for k in ("rise_px", "rise_ratio", "width_px", "source", "ok")
            if k in metrics
        }

    boxes = [du["box_xyxy"] for du in door_units]
    n = len(boxes)
    if n == 1:
        labels = np.array([0], dtype=np.int32)
    else:
        labels, _ = cluster_box_geometry_pipeline(
            boxes,
            image_size,
            base,
            method=box_method,
            k_max=min(k_max, n - 1),
            seed=seed,
            conservative_k=True,
            conservative_k_wh_tol=0.012,
            merge_near=True,
            snap_wh=True,
        )

    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i, du in enumerate(door_units):
        tid = int(labels[i])
        du["type_id"] = tid
        du["name"] = door_type_token(tid)
        by_label[tid].append(du)
        print(
            f"  door[{du['unit_id']}] → {door_type_token(tid)} "
            f"shape={du.get('shape')} rise={du.get('arch_rise_ratio', 0):.3f}"
        )

    door_types: list[dict[str, Any]] = []
    for tid in sorted(by_label):
        group = by_label[tid]
        shape = str(group[0].get("shape") or "rectangle")
        rises = [float(g.get("arch_rise_ratio") or 0.0) for g in group]
        exemplar = group[0]
        for g in group:
            if g.get("is_exemplar"):
                exemplar = g
                break
        door_types.append(
            {
                "type_id": int(tid),
                "name": door_type_token(tid),
                "kind": "door",
                "shape": shape,
                "primitive": shape,
                "arch_rise_ratio": float(np.mean(rises)) if rises else 0.0,
                "n_instances": len(group),
                "exemplar_asset": exemplar.get("asset"),
            }
        )
    return door_units, door_types
