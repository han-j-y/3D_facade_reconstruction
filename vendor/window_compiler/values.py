"""Resolve grammar numeric parameters (prior/range, relative_to)."""

from __future__ import annotations

from typing import Any

from geom2d import Region


def resolve_scalar(value: Any, region: Region | None = None) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, dict):
        raise TypeError(f"expected number or param dict, got {type(value)}")

    raw = value.get("prior", value.get("value"))
    if raw is None:
        raise ValueError(f"param dict missing prior/value: {value}")

    rel = value.get("relative_to", "absolute")
    x = float(raw)

    if rel in ("absolute", None):
        return x
    if region is None:
        raise ValueError(f"relative_to={rel!r} requires a target region")

    if rel == "min_side":
        return x * region.min_side
    if rel == "width":
        return x * region.width
    if rel == "height":
        return x * region.height
    if rel == "max_side":
        return x * max(region.width, region.height)
    raise ValueError(f"unknown relative_to: {rel!r}")


def resolve_normalized_coord(value: Any, *, axis: str, region: Region) -> float:
    """Resolve x/y in [0,1] relative to region, or absolute local coordinate."""
    if isinstance(value, (int, float)):
        v = float(value)
        if 0.0 <= v <= 1.0:
            if axis == "x":
                return region.min_x + v * region.width
            return region.min_y + v * region.height
        return v

    if isinstance(value, dict):
        if "prior" in value or "value" in value:
            rel = value.get("relative_to", "normalized")
            raw = float(value.get("prior", value.get("value")))
            if rel == "normalized":
                if axis == "x":
                    return region.min_x + raw * region.width
                return region.min_y + raw * region.height
            return resolve_scalar(value, region)
    raise TypeError(f"cannot resolve coordinate: {value!r}")
