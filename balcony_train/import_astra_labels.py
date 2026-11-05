"""Turn an external labeler's answers.json into ``run_balcony --unit-labels`` format.

Usage:
  python balcony_train/import_astra_labels.py <answers.json> <mapping.json> <out unit_labels.json>

Strict: every crop id exactly once, no unknown ids, labels from the classifier
vocabulary (``balcony_train/labels.py``). Material is cleared unless kind is
open_work, as ``RailingPredictor.predict_full`` does. floor_shape must be null on
juliet and rectangle / circle on projecting (``normalize_label(strict=True)``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "balcony_pipeline")):
    if p not in sys.path:
        sys.path.insert(0, p)

from unit_labels import normalize_label  # noqa: E402


def convert(answers: dict | list, mapping: dict) -> dict:
    rows = answers["answers"] if isinstance(answers, dict) else answers
    by_id: dict[str, dict] = {}
    for r in rows:
        cid = str(r.get("id"))
        if cid in by_id:
            raise ValueError(f"duplicate id {cid}")
        by_id[cid] = r
    want = {m["id"] for m in mapping["crops"]}
    missing = sorted(want - set(by_id))
    extra = sorted(set(by_id) - want)
    if missing or extra:
        raise ValueError(f"missing ids {missing}, unknown ids {extra}")
    units = []
    for m in mapping["crops"]:
        try:
            lab = normalize_label(by_id[m["id"]], strict=True)
        except ValueError as e:
            raise ValueError(f"{m['id']}: {e}") from None
        units.append({"unit_index": m["unit_index"], "box_xyxy": m["box_xyxy"], "id": m["id"], **lab})
    units.sort(key=lambda u: u["unit_index"])
    return {"units": units}


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    answers = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    mapping = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    out = convert(answers, mapping)
    out["source"] = str(Path(sys.argv[1]).resolve())
    Path(sys.argv[3]).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"{len(out['units'])} units -> {sys.argv[3]}")


if __name__ == "__main__":
    main()
