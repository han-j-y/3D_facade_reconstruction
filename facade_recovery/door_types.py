"""Door type naming tokens (d00, d01, …) — separate from window win_type_XX."""

from __future__ import annotations


def door_type_token(type_id: int) -> str:
    """Short placement / DSL name for a door cluster."""
    return f"d{int(type_id):02d}"


def parse_door_type_token(token: str) -> int | None:
    """Parse ``d00`` → 0; returns None if not a door token."""
    s = str(token).strip().lower()
    if not s.startswith("d") or len(s) < 2:
        return None
    try:
        return int(s[1:])
    except ValueError:
        return None


def is_door_token(token: str) -> bool:
    return parse_door_type_token(token) is not None
