"""Path helpers for vendored window_ast (standalone facade_e2e package)."""

from __future__ import annotations

from pathlib import Path

# .../facade_e2e/window_ast -> package root is parent
ROOT = Path(__file__).resolve().parents[1]


def default_structure_ckpt() -> Path:
    preferred = (
        ROOT / "checkpoints" / "structure_best.pt",
        ROOT.parent / "window_ast_predictor" / "runs" / "exp2_structure_dino_real_canny_ft2" / "best.pt",
    )
    for path in preferred:
        if path.exists():
            return path.resolve()
    return preferred[0]
