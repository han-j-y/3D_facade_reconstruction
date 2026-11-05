"""Print the Extended XML-crop label distribution.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/summarize_xml_labels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.label_store import merge_label_sources  # noqa: E402
from balcony_train.paths import EXT_XML_CROPS_DIR, EXT_XML_LABELS, EXT_XML_NONE  # noqa: E402
from balcony_train.xml_balcony_labels import (  # noqa: E402
    distribution_counts,
    enclosed_queue,
    load_none_count,
    open_queue,
)


def format_distribution(crops_dir: Path, labels_path: Path, none_path: Path) -> str:
    labels = merge_label_sources(crops_dir, labels_path)
    counts = distribution_counts(labels)
    lines = [
        f"photos with no balcony box: {load_none_count(none_path)}",
        f"open queue remaining: {len(open_queue(crops_dir, labels))}",
        f"enclosed queue remaining: {len(enclosed_queue(crops_dir, labels))}",
        "finished instance counts:",
    ]
    for group, rows in counts.items():
        body = "  ".join(f"{name}={n}" for name, n in rows.items())
        lines.append(f"  {group}: {body}")
    return "\n".join(lines)


def main() -> None:
    print(format_distribution(EXT_XML_CROPS_DIR, EXT_XML_LABELS, EXT_XML_NONE))


if __name__ == "__main__":
    main()
