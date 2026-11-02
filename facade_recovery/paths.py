"""Path helpers for the standalone facade_e2e package."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# Package root: .../facade_e2e/
PKG_ROOT = Path(__file__).resolve().parents[1]
def resolve_blender(explicit: str | Path | None = None) -> str | None:
    """Resolve Blender binary, or None if unavailable (render is optional)."""
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


def resolve_compiler_root(explicit: str | Path | None = None) -> Path | None:
    """Optional Blender window-compiler package (must contain main.py)."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("FACADE_COMPILER_ROOT")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend([PKG_ROOT / "vendor" / "window_compiler", PKG_ROOT / "window_compiler"])
    for path in candidates:
        if (path / "main.py").is_file():
            return path.resolve()
    return None


def default_structure_ckpt() -> Path:
    return PKG_ROOT / "checkpoints" / "structure_best.pt"


def default_vocab() -> Path:
    preferred = (PKG_ROOT / "checkpoints" / "vocab.json", PKG_ROOT / "data" / "vocab_structure.json")
    for path in preferred:
        if path.is_file() or (path.is_symlink() and path.exists()):
            return path.resolve()
    return preferred[1]


def default_train_up() -> Path:
    return PKG_ROOT / "data" / "facades" / "train_up"
