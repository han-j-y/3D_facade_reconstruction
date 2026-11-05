"""Label enclosed balconies cut from CMP XML.

Run the open UI first and press E on each enclosed crop. This UI then asks
for floor shape and skin. A marks half_enclosed. O sends a crop back.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/label_enclosed_ui.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.xml_label_ui import main_enclosed  # noqa: E402


if __name__ == "__main__":
    main_enclosed()
