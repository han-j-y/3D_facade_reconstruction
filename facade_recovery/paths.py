"""Path helpers for the standalone facade_e2e package (no facade_dsl8 / Blender)."""

from __future__ import annotations

from pathlib import Path

# Package root: .../facade_e2e/
PKG_ROOT = Path(__file__).resolve().parents[1]
# Sibling experiments/ (optional train_up images)
EXP_ROOT = PKG_ROOT.parent


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


def default_train_up() -> Path:
    return EXP_ROOT / "data" / "facades" / "train_up"
