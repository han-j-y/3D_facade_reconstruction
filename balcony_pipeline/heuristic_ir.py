"""Heuristic balcony IR (same *role* as window structure_best.pt, not the same model).

Each crop gets a BDSL JSON IR. Majority vote uses balcony_view (discrete only).
Which axes are inferred/voted is controlled by ``recovery_profile`` (default
``railing_only``). Railing recovery classes are ``baluster`` | ``solid`` only
(vertical members vs opaque panel/parapet); ``glass`` is never chosen.
Disabled axes use fixed defaults and are omitted from the fingerprint.
Railing kind (``baluster`` | ``solid``) comes from a trained classifier when
``rail_kind_override`` is set (see ``balcony_train`` + ``checkpoints/railing_best.pt``).
Otherwise the top 50% crop band heuristic: horizontal opaque run, vertical
thickness, uniform luminance (max−min ≤ ``RAILING_SOLID_UNIFORM_RANGE``).
Previous edge/CV rules are off.
Wall opening and floor×bay come from the window FDSL grid, not this IR.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from recovery_profile import FIXED_DEFAULTS, resolve_profile


def railing_kind_from_ir(ir: dict[str, Any]) -> str:
    """Recovery railing class from a BDSL IR dict (baluster | solid)."""
    return _rail_kind((ir.get("railing") or {}).get("kind"))


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


# --- Active: thick horizontal opaque face (top-half band) ---
# Opaque = deviates from band median (brighter or darker than flat gap/sky).
RAILING_OPAQUE_MEDIAN_DEV = 20.0
# Min horizontal opaque run on a row to qualify a face candidate.
RAILING_MAX_OPAQUE_RUN = 8
# Min vertical extent of opaque pixels spanning that run (filters thin handrails).
RAILING_MIN_OPAQUE_VERTICAL_THICKNESS = 3
# Face patch max-min <= this → solid (panel / parapet).
RAILING_SOLID_UNIFORM_RANGE = 40.0

# --- Callout: previous vertical-edge / column-CV rules (temporarily off) ---
# RAILING_VERTICAL_EDGE_THRESHOLD = 10.0
# RAILING_COLUMN_CV_THRESHOLD = 0.12


def _railing_band(gray: np.ndarray) -> np.ndarray:
    h = gray.shape[0]
    return gray[: max(1, h // 2)]


def _max_horizontal_run(mask_row: np.ndarray) -> int:
    best = 0
    cur = 0
    for v in mask_row.astype(bool):
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _iter_horizontal_runs(row: np.ndarray, min_len: int):
    """Yield inclusive (x0, x1) for opaque runs with length >= min_len."""
    x = 0
    n = int(row.shape[0])
    while x < n:
        if not row[x]:
            x += 1
            continue
        x0 = x
        while x < n and row[x]:
            x += 1
        if x - x0 >= int(min_len):
            yield x0, x - 1


def _vertical_opaque_span(
    opaque: np.ndarray,
    y: int,
    x0: int,
    x1: int,
) -> tuple[int, int]:
    """Inclusive row range covering opaque pixels in [x0, x1] connected through y."""
    y0, y1 = y, y
    for yy in range(y - 1, -1, -1):
        if opaque[yy, x0 : x1 + 1].any():
            y0 = yy
        else:
            break
    for yy in range(y + 1, opaque.shape[0]):
        if opaque[yy, x0 : x1 + 1].any():
            y1 = yy
        else:
            break
    return y0, y1


def _vertical_thickness(opaque: np.ndarray, y: int, x0: int, x1: int) -> int:
    y0, y1 = _vertical_opaque_span(opaque, y, x0, x1)
    return y1 - y0 + 1


def _railing_opaque_mask(band: np.ndarray) -> np.ndarray:
    med = float(np.median(band))
    return np.abs(band - med) > float(RAILING_OPAQUE_MEDIAN_DEV)


def _railing_max_opaque_run(band: np.ndarray) -> int:
    """Longest horizontal run of opaque pixels in any row of the band."""
    if band.size == 0 or band.shape[1] < 1:
        return 0
    opaque = _railing_opaque_mask(band)
    return max(_max_horizontal_run(opaque[y]) for y in range(opaque.shape[0]))


def _railing_max_opaque_run_score(gray: np.ndarray) -> int:
    return _railing_max_opaque_run(_railing_band(gray))


def _railing_has_solid_patch(
    band: np.ndarray,
    *,
    run_threshold: int = RAILING_MAX_OPAQUE_RUN,
    min_vertical_thickness: int = RAILING_MIN_OPAQUE_VERTICAL_THICKNESS,
    uniform_range: float = RAILING_SOLID_UNIFORM_RANGE,
) -> bool:
    """True when a thick horizontal opaque face is uniformly bright/dark."""
    if band.size == 0 or band.shape[1] < 1:
        return False
    opaque = _railing_opaque_mask(band)
    for y in range(opaque.shape[0]):
        for x0, x1 in _iter_horizontal_runs(opaque[y], run_threshold):
            if _vertical_thickness(opaque, y, x0, x1) < int(min_vertical_thickness):
                continue
            y0, y1 = _vertical_opaque_span(opaque, y, x0, x1)
            patch = band[y0 : y1 + 1, x0 : x1 + 1]
            if patch.size == 0:
                continue
            if float(patch.max() - patch.min()) <= float(uniform_range):
                return True
    return False


def _infer_rail_kind(*, has_qualifying_solid_patch: bool) -> str:
    """Solid when a thick, uniform horizontal opaque face exists in the band."""
    return "solid" if has_qualifying_solid_patch else "baluster"


# --- Callout: previous edge / CV helpers (temporarily unused) ---
#
# def _railing_vertical_edge_score(gray: np.ndarray) -> float:
#     band = _railing_band(gray)
#     if band.size <= 1 or band.shape[1] < 2:
#         return 0.0
#     return float(np.abs(np.diff(band, axis=1)).mean())
#
#
# def _railing_column_cv_score(gray: np.ndarray) -> float:
#     band = _railing_band(gray)
#     if band.size == 0 or band.shape[1] < 2:
#         return 0.0
#     col_means = band.mean(axis=0)
#     mu = float(col_means.mean())
#     if mu <= 1e-6:
#         return 0.0
#     return float(col_means.std() / mu)
#
#
# def _infer_rail_kind_edge_cv(
#     *,
#     vertical_edge: float,
#     column_cv: float,
#     edge_threshold: float = RAILING_VERTICAL_EDGE_THRESHOLD,
#     cv_threshold: float = RAILING_COLUMN_CV_THRESHOLD,
# ) -> str:
#     if vertical_edge >= float(edge_threshold) or column_cv >= float(cv_threshold):
#         return "baluster"
#     return "solid"


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
    rail_kind_override: str | None = None,
) -> dict[str, Any]:
    """Appearance cues → BDSL IR. Disabled axes use FIXED_DEFAULTS.

    ``rail_kind_override`` skips the opaque-run heuristic (classifier or tests).
    """
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
        if rail_kind_override is not None:
            rail_kind = _rail_kind(rail_kind_override)
        else:
            band = _railing_band(gray)
            rail_kind = _infer_rail_kind(
                has_qualifying_solid_patch=_railing_has_solid_patch(band),
            )
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
