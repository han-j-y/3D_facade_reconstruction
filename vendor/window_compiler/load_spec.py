"""Load window spec from JSON or WDSL file."""

from __future__ import annotations

import json
from pathlib import Path

from parse_wdsl import parse_wdsl_file


def load_spec(path: str | Path) -> dict:
    p = Path(path)
    if p.suffix.lower() == ".wdsl":
        return parse_wdsl_file(p)
    return json.loads(p.read_text(encoding="utf-8-sig"))
