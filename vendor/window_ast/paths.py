"""Path helpers for vendored window_ast (standalone facade_e2e package)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# .../facade_e2e/window_ast -> package root is parent
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
        ROOT / "checkpoints" / "structure_best.pt",
        ROOT.parent / "window_ast_predictor" / "runs" / "exp2_structure_dino_real_canny_ft2" / "best.pt",
    )
    for path in preferred:
        if path.exists():
            return path.resolve()
    return preferred[0]
