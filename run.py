#!/usr/bin/env python3
"""Facade photo → window types → structure IR → recovery DSL.

Blender rendering is optional (``--blender-render``).

Optional ``--with-balconies`` runs the balcony track (``balcony_pipeline/run_balcony.py``, unchanged from
hiroo/balcony-pipeline) after the window DSL is written. It reads ``<out-dir>/facade_dsl.json`` and writes
``<out-dir>/facade_dsl_with_balconies.json`` (its stage images go to ``<out-dir>/balcony/``). The window stages and
``facade_dsl.json`` are not changed, so window evaluation is the same with or without it.

Examples
--------
  python run.py --image /path/to/facade.png --device cuda
  python run.py --image /path/to/facade.png --out-dir runs/x --device cuda --with-balconies
  python run.py --facade-id 8 --device cuda
  python run.py --facade-id 8 --blender-render --device cuda
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import run_pipeline

ROOT = Path(__file__).resolve().parent


def _arg(argv: list[str], name: str) -> str | None:
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def run_balconies(image: Path, out_dir: Path, device: str | None) -> Path | None:
    """Balcony track on a finished window DSL; returns the merged DSL path (None if it produced none)."""
    cmd = [sys.executable, str(ROOT / "balcony_pipeline" / "run_balcony.py"),
           "--image", str(image), "--windows-dsl", str(out_dir / "facade_dsl.json"),
           "--out-dir", str(out_dir / "balcony")]
    if device:
        cmd += ["--device", device]
    # The default enclosed detector (cell_clf) needs hiroo's trained classifier / labelled cell features, which are
    # not in the repo; without them use the branch's SAM3+DA3 volume detector.
    clf = ROOT / "runs" / "cell_clf"
    if not ((clf / "model_rules_gate.joblib").is_file() or (clf / "cells_train.npz").is_file()):
        cmd += ["--enclosed-method", "sam3"]
    print("=== balcony track (balcony_pipeline/run_balcony.py) ===", flush=True)
    subprocess.run(cmd, check=True)
    merged = out_dir / "balcony" / "facade_dsl_with_balconies.json"
    if not merged.is_file():
        print("warn: balcony track wrote no merged DSL")
        return None
    dst = out_dir / "facade_dsl_with_balconies.json"
    shutil.copy(merged, dst)
    print(f"balconies → {dst}")
    return dst


if __name__ == "__main__":
    with_balconies = "--with-balconies" in sys.argv
    if with_balconies:
        sys.argv.remove("--with-balconies")
    run_pipeline.main()
    if with_balconies:
        image, out_dir = _arg(sys.argv, "--image"), _arg(sys.argv, "--out-dir")
        if not image or not out_dir:
            print("warn: --with-balconies needs --image and --out-dir; skipped")
        else:
            run_balconies(Path(image), Path(out_dir), _arg(sys.argv, "--device"))
