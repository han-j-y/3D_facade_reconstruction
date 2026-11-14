#!/usr/bin/env python3
"""Facade photo → window types → structure IR → recovery DSL.

Optional ``--with-balconies`` runs balcony_pipeline after the window DSL
(does not change window stages). Blender rendering is optional (``--blender-render``).

Optional ``--with-balconies`` runs the balcony track (``balcony_pipeline/run_balcony.py``, unchanged from
hiroo/balcony-pipeline) after the window DSL is written. It reads ``<out-dir>/facade_dsl.json`` and writes
``<out-dir>/facade_dsl_with_balconies.json`` (its stage images go to ``<out-dir>/balcony/``). The window stages and
``facade_dsl.json`` are not changed, so window evaluation is the same with or without it. ``--depth-dir`` and
``--enclosed-sam3-cache`` are passed to the balcony track (DA3 depth and oriel SAM3 seed caches; e.g. an empty
directory to time a cold run).

Examples
--------
  python run.py --image /path/to/facade.png --device cuda
  python run.py --image /path/to/facade.png --out-dir runs/x --device cuda --with-balconies
  python run.py --facade-id 8 --device cuda
  python run.py --facade-id 8 --blender-render --device cuda
"""

from __future__ import annotations

import shutil
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


def is_real_lfs_file(path: Path) -> bool:
    """True when the file exists and is not an unfetched Git LFS pointer."""
    return path.is_file() and not path.open("rb").read(40).startswith(b"version https://git-lfs")


def _balcony_module():
    """balcony_pipeline/run_balcony.py as a module (it is a script, not a package member)."""
    import importlib.util

    name = "run_balcony"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "balcony_pipeline" / "run_balcony.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def _pop_arg(argv: list[str], name: str) -> str | None:
    """Remove ``name VALUE`` / ``name=VALUE`` from ``argv`` and return the value."""
    value = _arg(argv, name)
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            del argv[i:i + 2]
            break
        if a.startswith(name + "="):
            del argv[i]
            break
    return value


def run_balconies(image: Path, out_dir: Path, device: str | None, side_windows: bool = False,
                  extra: list[str] | None = None) -> Path | None:
    """Balcony track on a finished window DSL; returns the merged DSL path (None if it produced none)."""
    cmd = [sys.executable, str(ROOT / "balcony_pipeline" / "run_balcony.py"),
           "--image", str(image), "--windows-dsl", str(out_dir / "facade_dsl.json"),
           "--out-dir", str(out_dir / "balcony")]
    if device:
        cmd += ["--device", device]
    if side_windows:
        cmd.append("--side-windows")
    cmd += extra or []
    # The default enclosed detector (cell_clf) needs the trained model (Git LFS); without it use the SAM3+DA3 detector.
    if not is_real_lfs_file(ROOT / "checkpoints" / "oriel_cell_gate.joblib"):
        print("warn: oriel_cell_gate.joblib missing, enclosed detection falls back to sam3")
        cmd += ["--enclosed-method", "sam3"]
    print("=== balcony track (balcony_pipeline/run_balcony.py) ===", flush=True)
    # In this process (was a subprocess): the balcony track reuses the models the window track already loaded
    # (facade_recovery.model_cache) and skips a second interpreter start. Same arguments, same outputs.
    rb = _balcony_module()
    argv = sys.argv
    sys.argv = [str(cmd[1])] + cmd[2:]
    try:
        args = rb.parse_args()
    finally:
        sys.argv = argv
    rb.run(args)
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
    side_windows = "--side-windows" in sys.argv
    if side_windows:
        sys.argv.remove("--side-windows")
    balcony_extra = []
    for opt in ("--depth-dir", "--enclosed-sam3-cache"):
        value = _pop_arg(sys.argv, opt)
        if value is not None:
            balcony_extra += [opt, value]
    run_pipeline.main()
    if with_balconies:
        image, out_dir = _arg(sys.argv, "--image"), _arg(sys.argv, "--out-dir")
        if not image or not out_dir:
            print("warn: --with-balconies needs --image and --out-dir; skipped")
        else:
            run_balconies(Path(image), Path(out_dir), _arg(sys.argv, "--device"), side_windows, balcony_extra)
