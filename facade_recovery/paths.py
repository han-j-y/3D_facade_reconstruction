"""Path helpers for the standalone facade_e2e package."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# Package root: .../facade_e2e/
PKG_ROOT = Path(__file__).resolve().parents[1]
# Sibling experiments/ (for optional train_up / facade_dsl8 fallbacks)
EXP_ROOT = PKG_ROOT.parent


def resolve_blender(explicit: str | Path | None = None) -> str | None:
    """Resolve Blender binary, or return None if not found (Blender is optional)."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("BLENDER")
    if env:
        candidates.append(Path(env).expanduser())
    which = shutil.which("blender")
    if which:
        candidates.append(Path(which))
    candidates.extend(
        [
            Path.home() / "local_install/usr/share/blender-5.1.0-linux-x64/blender",
            Path("/usr/bin/blender"),
        ]
    )
    for path in candidates:
        if path.is_file():
            return str(path.resolve())
    return None


def default_structure_ckpt() -> Path:
    preferred = (
        PKG_ROOT / "checkpoints" / "structure_best.pt",
        EXP_ROOT / "window_ast_predictor" / "runs" / "exp2_structure_dino_real_canny_ft2" / "best.pt",
    )
    for path in preferred:
        if path.is_file() or path.is_symlink():
            return path.resolve() if path.exists() else path
    return preferred[0]


def default_vocab() -> Path:
    preferred = (
        PKG_ROOT / "checkpoints" / "vocab.json",
        PKG_ROOT / "data" / "vocab_structure.json",
        EXP_ROOT / "window_ast_predictor" / "data" / "vocab_structure.json",
    )
    for path in preferred:
        if path.is_file() or (path.is_symlink() and path.exists()):
            return path.resolve()
    return preferred[1]


def facade_dsl8_root() -> Path | None:
    candidates = (
        PKG_ROOT / "vendor" / "facade_dsl8",
        EXP_ROOT / "facade_dsl8",
    )
    for path in candidates:
        if (path / "main.py").is_file():
            return path.resolve()
    return None


def default_train_up() -> Path:
    return EXP_ROOT / "data" / "facades" / "train_up"
