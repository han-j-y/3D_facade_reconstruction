"""Path helpers for vendored window_ast (standalone facade_e2e package)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve_blender(explicit: str | Path | None = None) -> str | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("BLENDER")
    if env:
        candidates.append(Path(env).expanduser())
    which = shutil.which("blender")
    if which:
        candidates.append(Path(which))
    for path in candidates:
        if path.is_file():
            return str(path.resolve())
    return None


def default_structure_ckpt() -> Path:
    return ROOT / "checkpoints" / "structure_best.pt"
