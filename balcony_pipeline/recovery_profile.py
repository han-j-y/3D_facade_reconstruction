"""Photo-recovery axis profile for balcony IR.

Authoring BDSL / catalog / compiler keep the full axis set. Photo recovery
only *infers and votes* axes marked True below; disabled axes get fixed
defaults and are omitted from ``balcony_view``.

Switch profile (or flip flags) to bring axes back later.
"""

from __future__ import annotations

from typing import Any

# Defaults when an axis is disabled for recovery.
FIXED_DEFAULTS: dict[str, Any] = {
    "structure": "projecting",
    "enclosure": "open",
    "floor_shape": "rectangle",
    "supports_count": 0,
}

PROFILES: dict[str, dict[str, bool]] = {
    # Current default: vote railing only (Richard / SAM3-realism).
    "railing_only": {
        "structure": False,
        "enclosure": False,
        "floor_shape": False,
        "railing": True,
        "supports": False,
    },
    # Re-enable all recovery axes (except opening — still FDSL-only).
    "full": {
        "structure": True,
        "enclosure": True,
        "floor_shape": True,
        "railing": True,
        "supports": True,
    },
}

DEFAULT_PROFILE = "railing_only"


def resolve_profile(name: str | None = None) -> dict[str, bool]:
    key = (name or DEFAULT_PROFILE).strip().lower()
    if key not in PROFILES:
        raise ValueError(
            f"unknown recovery profile {name!r}; expected one of {sorted(PROFILES)}"
        )
    return dict(PROFILES[key])


def active_axes(profile: dict[str, bool] | None = None) -> list[str]:
    p = profile or resolve_profile()
    return [k for k, on in p.items() if on]
