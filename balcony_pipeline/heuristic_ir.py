"""Heuristic balcony IR (same *role* as window structure_best.pt, not the same model).

Each crop gets a BDSL JSON IR. Majority vote uses balcony_view (discrete only).
Which axes are inferred/voted is controlled by ``recovery_profile`` (default
``railing_only``). Railing recovery classes are ``baluster`` | ``solid`` only
(vertical members vs opaque panel/parapet); ``glass`` is never chosen.
Disabled axes use fixed defaults and are omitted from the fingerprint.
Wall opening and floor×bay come from the window FDSL grid, not this IR.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from recovery_profile import FIXED_DEFAULTS, resolve_profile


def _rail_kind(raw: Any) -> str:
    """Normalize railing kind for recovery (baluster | solid only)."""
    k = str(raw or "baluster").strip().lower()
    if k in ("metal", "baluster"):
        return "baluster"
    if k in ("solid", "panel", "parapet", "concrete"):
        return "solid"
    # glass and unknowns: not a recovery class — treat as solid (opaque face)
    if k == "glass":
        return "solid"
    return "baluster"


def _infer_rail_kind(*, inner_mean: float, top_edges: float) -> str:
    """Baluster if vertical-member look; solid for panel / parapet (no glass)."""
    # Low edge energy along the top band → continuous opaque face
    if top_edges < 8:
        return "solid"
    # Bright smooth crops used to map to glass; fold into solid
    if inner_mean > 160 and top_edges < 15:
        return "solid"
    return "baluster"


def balcony_view(
    ir: dict[str, Any],
    *,
    profile: dict[str, bool] | None = None,
    profile_name: str | None = None,
) -> dict[str, Any]:
    """Discrete fingerprint — only axes enabled in the recovery profile."""
    p = profile or resolve_profile(profile_name)
    floor = ir.get("floor") or {}
    railing = ir.get("railing") or {}
    supports = ir.get("supports") or {}
    view: dict[str, Any] = {}
    if p.get("structure"):
        view["structure"] = ir.get("structure")
    if p.get("enclosure"):
        view["enclosure"] = ir.get("enclosure")
    if p.get("floor_shape"):
        view["floor_shape"] = floor.get("shape")
    if p.get("railing"):
        view["railing_kind"] = _rail_kind(railing.get("kind"))
    if p.get("supports"):
        view["supports_count"] = int(supports.get("count") or 0)
    return view


def ir_to_tokens(
    ir: dict[str, Any],
    *,
    profile: dict[str, bool] | None = None,
    profile_name: str | None = None,
) -> list[str]:
    p = profile or resolve_profile(profile_name)
    floor = ir.get("floor") or {}
    railing = ir.get("railing") or {}
    toks = ["BALCONY"]
    if p.get("floor_shape"):
        toks.append(f"shape={floor.get('shape', 'rectangle')}")
    if p.get("structure"):
        toks.append(f"structure={ir.get('structure')}")
    if p.get("enclosure"):
        toks.append(f"enclosure={ir.get('enclosure')}")
    if p.get("railing"):
        toks.append(f"railing={_rail_kind(railing.get('kind'))}")
    if p.get("supports"):
        toks.append(f"supports={int((ir.get('supports') or {}).get('count') or 0)}")
    return toks


def _to_np(crop: Image.Image) -> np.ndarray:
    return np.asarray(crop.convert("RGB"), dtype=np.float32)


def infer_balcony_ir(
    crop: Image.Image,
    *,
    box: list[int],
    image_size: tuple[int, int],
    profile_name: str | None = None,
) -> dict[str, Any]:
    """Cheap appearance cues → BDSL IR. Disabled axes use FIXED_DEFAULTS."""
    p = resolve_profile(profile_name)
    arr = _to_np(crop)
    h, w = arr.shape[:2]
    iw, ih = image_size
    x0, y0, x1, y1 = box
    bw = max(1.0, float(x1 - x0))
    bh = max(1.0, float(y1 - y0))
    aspect = bw / bh

    gray = arr.mean(axis=2)
    y0i, y1i = h // 4, 3 * h // 4
    x0i, x1i = w // 6, 5 * w // 6
    inner = gray[y0i:y1i, x0i:x1i]
    inner_mean = float(inner.mean()) if inner.size else float(gray.mean())
    inner_std = float(inner.std()) if inner.size else float(gray.std())
    top = gray[: max(1, h // 3)]
    top_edges = float(np.abs(np.diff(top, axis=1)).mean()) if top.size > 1 else 0.0

    if p.get("enclosure"):
        enclosure = "enclosed" if inner_mean > 140 and inner_std < 45 else "open"
    else:
        enclosure = str(FIXED_DEFAULTS["enclosure"])

    if p.get("structure"):
        if aspect > 3.2:
            structure = "projecting"
        elif aspect < 0.85 and bh / ih > 0.18:
            structure = "free_standing"
        elif inner_std < 28 and enclosure == "open":
            structure = "inset"
        else:
            structure = "projecting"
    else:
        structure = str(FIXED_DEFAULTS["structure"])

    if p.get("floor_shape"):
        if aspect < 0.75:
            floor_shape = "triangle"
        elif 0.85 <= aspect <= 1.15 and inner_std < 35:
            floor_shape = "circle"
        else:
            floor_shape = "rectangle"
    else:
        floor_shape = str(FIXED_DEFAULTS["floor_shape"])

    if p.get("railing"):
        rail_kind = _infer_rail_kind(inner_mean=inner_mean, top_edges=top_edges)
    else:
        rail_kind = "baluster"

    if p.get("supports"):
        supports_count = 4 if structure == "free_standing" else 0
    else:
        supports_count = int(FIXED_DEFAULTS["supports_count"])

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
        "supports": {"count": supports_count},
        "glazing": None,
        "output": {
            "slab_thickness": 0.20 if enclosure == "enclosed" else 0.12,
            "railing_thickness": 0.20 if rail_kind == "solid" else 0.04,
        },
    }
    if enclosure != "enclosed" or p.get("railing"):
        ir["railing"] = {"kind": rail_kind, "height": 1.1}
    return ir
