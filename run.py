#!/usr/bin/env python3
"""Facade photo → window types → structure IR → recovery DSL.

No facade_dsl8 / Blender dependency.

Examples
--------
  python run.py --image /path/to/facade.png --device cuda
  python run.py --facade-id 8 --device cuda
"""

from __future__ import annotations

import run_pipeline

if __name__ == "__main__":
    run_pipeline.main()
