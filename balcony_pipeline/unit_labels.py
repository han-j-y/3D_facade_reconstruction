"""External per-unit railing labels (e.g. from another labeler such as Astra).

Replaces only the step-4 classifier call in ``run_balcony.py`` (``infer_unit_ir``).
Detection, split, snap, clustering, vote and merge stay the pipeline's. Each entry
carries the unit box it was made for; a different box means the run is not the
same as the one the labels were made on, and the lookup raises.

File format::

    {"units": [{"unit_index": 0, "box_xyxy": [x0, y0, x1, y1],
                "kind": "open_work", "material": "metal",
                "floor_shape": "rectangle", "structure": "juliet"}, ...]}
"""

from __future__ import annotations

import json
from pathlib import Path

from balcony_train.labels import CLASSES, FLOOR_SHAPES, LABEL_STRUCTURES, MATERIALS

FIELDS = ("kind", "material", "floor_shape", "structure")


ASTRA_FLOOR_SHAPES = ("rectangle", "circle")


def normalize_label(row: dict, *, strict: bool = False) -> dict:
    """Check the vocabulary; material is null unless kind is open_work (as ``predict_full``).

    Default (replay): floor_shape in FLOOR_SHAPES, or null on a juliet unit (the type
    floor is voted over projecting units only, so a juliet floor never reaches the DSL).
    ``strict`` (external labeler answers): floor_shape must be null on juliet and
    rectangle / circle on projecting (no triangle, as the test labels).
    """
    kind = row.get("kind")
    if kind not in CLASSES:
        raise ValueError(f"kind {kind!r} not in {CLASSES}")
    material = row.get("material")
    if kind == "open_work":
        if material not in MATERIALS:
            raise ValueError(f"open_work needs material in {MATERIALS}, got {material!r}")
    else:
        material = None
    structure = row.get("structure")
    # predict_full returns structure None when its enclosure head says non-open;
    # replay keeps that, an external labeler must answer.
    if structure not in LABEL_STRUCTURES and (strict or structure is not None):
        raise ValueError(f"structure {structure!r} not in {LABEL_STRUCTURES}")
    floor = row.get("floor_shape")
    if strict:
        if structure == "juliet" and floor is not None:
            raise ValueError(f"juliet needs floor_shape null, got {floor!r}")
        if structure == "projecting" and floor not in ASTRA_FLOOR_SHAPES:
            raise ValueError(f"floor_shape {floor!r} not in {ASTRA_FLOOR_SHAPES}")
    elif not (floor in FLOOR_SHAPES or (floor is None and structure == "juliet")):
        raise ValueError(f"floor_shape {floor!r} not in {FLOOR_SHAPES} (null only for juliet)")
    return {"kind": kind, "material": material, "floor_shape": floor, "structure": structure}


class ExternalUnitLabels:
    """``predict_for(unit_index, box)`` → predict_full-style dict."""

    def __init__(self, rows: list[dict], source: str = "external") -> None:
        self.source = source
        self.by_index: dict[int, dict] = {}
        for r in rows:
            i = int(r["unit_index"])
            if i in self.by_index:
                raise ValueError(f"duplicate unit_index {i}")
            self.by_index[i] = {
                "box_xyxy": [int(v) for v in r["box_xyxy"]],
                **normalize_label(r),
            }

    @classmethod
    def load(cls, path: Path) -> "ExternalUnitLabels":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(list(data["units"]), source=f"external:{path.name}")

    def check_units(self, boxes: list[list[int]]) -> None:
        """Every unit has a label, no extra labels, every box equal."""
        if set(self.by_index) != set(range(len(boxes))):
            raise ValueError(
                f"labels cover units {sorted(self.by_index)}, run has {len(boxes)} units"
            )
        for i, b in enumerate(boxes):
            self.predict_for(i, b)

    def predict_for(self, unit_index: int, box: list[int]) -> dict:
        if unit_index not in self.by_index:
            raise KeyError(f"no external label for unit {unit_index}")
        row = self.by_index[unit_index]
        if [int(v) for v in box] != row["box_xyxy"]:
            raise ValueError(
                f"unit {unit_index} box {list(box)} != labeled box {row['box_xyxy']}"
            )
        return {
            "kind": row["kind"],
            "material": row["material"],
            "floor_shape": row["floor_shape"],
            "enclosure": None,
            "skin": None,
            "structure": row["structure"],
        }


class UnitLabelPredictor:
    """Wraps ``ExternalUnitLabels`` for one unit so ``infer_unit_ir`` can call ``predict_full``."""

    def __init__(self, labels: ExternalUnitLabels, unit_index: int, box: list[int]) -> None:
        self._pred = labels.predict_for(unit_index, box)

    def predict_full(self, crop) -> dict:  # noqa: ARG002 - crop unused by design
        return dict(self._pred)
