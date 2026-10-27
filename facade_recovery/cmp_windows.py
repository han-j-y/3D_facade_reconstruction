"""CMP XML window boxes (same source as window_ast_predictor clustering)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def parse_cmp_windows(xml_path: Path) -> list[tuple[float, float, float, float]]:
    """Normalized (u0, v0, u1, v1); CMP ``<x>``=row, ``<y>``=col."""
    text = xml_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text.startswith("<objects"):
        text = f"<objects>\n{text}\n</objects>"
    root = ET.fromstring(text)
    boxes: list[tuple[float, float, float, float]] = []
    for obj in root.iter("object"):
        name = (obj.findtext("labelname") or "").strip().lower()
        if name != "window":
            continue
        pts = obj.find("points")
        if pts is None:
            continue
        rows = [float(x.text) for x in pts.findall("x") if x.text]
        cols = [float(y.text) for y in pts.findall("y") if y.text]
        if len(rows) < 2 or len(cols) < 2:
            continue
        u0, u1 = min(cols), max(cols)
        v0, v1 = min(rows), max(rows)
        if u1 - u0 < 1e-4 or v1 - v0 < 1e-4:
            continue
        boxes.append((u0, v0, u1, v1))
    return boxes


def load_cmp_xml_windows(
    xml_path: Path,
    image_size: tuple[int, int],
    *,
    stem: str = "xml",
) -> list[dict[str, Any]]:
    w, h = image_size
    raw: list[dict[str, Any]] = []
    for i, (u0, v0, u1, v1) in enumerate(parse_cmp_windows(xml_path)):
        box = [
            int(round(u0 * w)),
            int(round(v0 * h)),
            int(round(u1 * w)),
            int(round(v1 * h)),
        ]
        raw.append({"id": f"{stem}_w{i}", "box_xyxy": box, "score": 1.0, "source": "cmp_xml"})
    return raw


CMP_WINDOW_CLASS = 3


def load_cmp_labelmap_windows(
    png_path: Path,
    *,
    stem: str = "labelmap",
    min_side: int = 6,
    aspect: tuple[float, float] = (0.15, 6.0),
) -> list[dict[str, Any]]:
    """Window boxes from the CMP label map (``<stem>.png``, class 3 = window), one per connected component.

    The CMP XML boxes are not aligned with the base jpgs; the label map is."""
    import numpy as np
    from PIL import Image
    from scipy import ndimage

    lab, _ = ndimage.label(np.asarray(Image.open(png_path)) == CMP_WINDOW_CLASS)
    raw: list[dict[str, Any]] = []
    for sl in ndimage.find_objects(lab):
        y, x = sl
        w, h = x.stop - x.start, y.stop - y.start
        if min(w, h) < min_side or not aspect[0] <= w / h <= aspect[1]:
            continue
        raw.append({"id": f"{stem}_w{len(raw)}", "box_xyxy": [int(x.start), int(y.start), int(x.stop), int(y.stop)],
                    "score": 1.0, "source": "cmp_labelmap"})
    return raw
