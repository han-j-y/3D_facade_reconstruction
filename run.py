#!/usr/bin/env python3
"""Facade photo → window types → structure IR → recovery DSL.

Blender rendering is optional (``--blender-render``).

Examples
--------
  python run.py --image /path/to/facade.png --device cuda
  python run.py --facade-id 8 --device cuda
  python run.py --facade-id 8 --blender-render --device cuda
"""

from __future__ import annotations

import run_pipeline

if __name__ == "__main__":
    run_pipeline.main()
