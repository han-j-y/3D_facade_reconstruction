"""Heuristic balcony IR (same *role* as window structure_best.pt, not the same model).

Each crop gets a BDSL JSON IR. Majority vote uses balcony_view (discrete only).
Axes match the balcony catalog: structure, enclosure, floor.shape, railing.kind.
Wall opening and floor×bay come from the window FDSL grid, not this IR.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def _rail_kind(raw: Any) -> str:
    k = str(raw or "baluster").strip().lower()
    return "baluster" if k == "metal" else k


def balcony_view(ir: dict[str, Any]) -> dict[str, Any]:
    """Discrete fingerprint — analogue of window_ast.structure.structure_view."""
    floor = ir.get("floor") or {}
    railing = ir.get("railing") or {}
    supports = ir.get("supports") or {}
    enclosure = ir.get("enclosure")
    view = {
        "structure": ir.get("structure"),
        "enclosure": enclosure,
        "floor_shape": floor.get("shape"),
        "supports_count": int(supports.get("count") or 0),
    }
    if enclosure != "enclosed":
        view["railing_kind"] = _rail_kind(railing.get("kind"))
    return view


def ir_to_tokens(ir: dict[str, Any]) -> list[str]:
    floor = ir.get("floor") or {}
    railing = ir.get("railing") or {}
    toks = [
        "BALCONY",
        f"shape={floor.get('shape', 'rectangle')}",
        f"structure={ir.get('structure')}",
        f"enclosure={ir.get('enclosure')}",
    ]
    if ir.get("enclosure") != "enclosed":
        toks.append(f"railing={_rail_kind(railing.get('kind'))}")
    return toks


def _to_np(crop: Image.Image) -> np.ndarray:
    return np.asarray(crop.convert("RGB"), dtype=np.float32)


def infer_balcony_ir(crop: Image.Image, *, box: list[int], image_size: tuple[int, int]) -> dict[str, Any]:
    """Cheap appearance cues → BDSL IR. Floats filled; vote ignores them."""
    arr = _to_np(crop)
    h, w = arr.shape[:2]
    iw, ih = image_size
    x0, y0, x1, y1 = box
    bw = max(1.0, float(x1 - x0))
    bh = max(1.0, float(y1 - y0))
    aspect = bw / bh

    gray = arr.mean(axis=2)
    # inner third: glazed enclosure tends to be brighter / smoother
    y0i, y1i = h // 4, 3 * h // 4
    x0i, x1i = w // 6, 5 * w // 6
    inner = gray[y0i:y1i, x0i:x1i]
    inner_mean = float(inner.mean()) if inner.size else float(gray.mean())
    inner_std = float(inner.std()) if inner.size else float(gray.std())
    top = gray[: max(1, h // 3)]
    top_edges = float(np.abs(np.diff(top, axis=1)).mean()) if top.size > 1 else 0.0

    enclosure = "enclosed" if inner_mean > 140 and inner_std < 45 else "open"

    if aspect > 3.2:
        structure = "projecting"
    elif aspect < 0.85 and bh / ih > 0.18:
        structure = "free_standing"
    elif inner_std < 28 and enclosure == "open":
        structure = "inset"
    else:
        structure = "projecting"

    if aspect < 0.75:
        floor_shape = "triangle"
    elif 0.85 <= aspect <= 1.15 and inner_std < 35:
        floor_shape = "circle"
    else:
        floor_shape = "rectangle"

    if enclosure == "enclosed":
        rail_kind = "baluster"
    elif inner_mean > 160:
        rail_kind = "glass"
    elif top_edges < 8:
        rail_kind = "solid"
    else:
        rail_kind = "baluster"

    depth = max(0.4, min(1.6, 0.8 * (bh / max(bw, 1.0))))
    width = max(0.8, bw / max(iw, 1) * 12.0)

    ir: dict[str, Any] = {
        "type": "balcony",
        "debug": False,
        "structure": structure,
        "enclosure": enclosure,
        "floor": {
            "shape": floor_shape,
            "params": {"width": round(width, 3), "depth": round(depth, 3)},
        },
        "supports": {"count": 4 if structure == "free_standing" else 0},
        "glazing": None,
        "output": {
            "slab_thickness": 0.20 if enclosure == "enclosed" else 0.12,
            "railing_thickness": 0.20 if rail_kind == "solid" else (0.01 if rail_kind == "glass" else 0.04),
        },
    }
    if enclosure != "enclosed":
        ir["railing"] = {"kind": rail_kind, "height": 1.1}
    return ir
