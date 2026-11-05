"""Label open balconies cut from CMP XML.

Kind, material, floor, and structure (P projecting / J juliet). Press E to
send an enclosed balcony to ``label_enclosed_ui.py``.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/label_open_ui.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.xml_label_ui import main_open  # noqa: E402


if __name__ == "__main__":
    main_open()
