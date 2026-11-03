r"""Save the test100 crops an external labeler (Astra) got wrong, as ``export_test100_errors.py`` does.

Same heads and same scored crops as ``score_astra_crops.py`` (a null answer is not an
error and is not saved). Writes, per head:

* ``<out>/<head>/<true>_as_<pred>/<crop>.jpg``: the crop (left) beside the photo with its
  red box (right), header true / Astra;
* ``<out>/<head>/<true>_as_<pred>/sheet_NN.jpg``: 24 crops per page;
* ``<out>/errors.csv``: one row per error (with the Astra id).

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_train/export_astra_errors.py runs/astra_balcony/answers.json runs/astra_balcony/test100_crops/crop_set_mapping.json --out runs/astra_balcony/errors
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.export_test100_errors import panel, sheet  # noqa: E402
from balcony_train.score_astra_crops import FIELD, HEADS, truth_of  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers", type=Path)
    ap.add_argument("mapping", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--who", default="Astra")
    args = ap.parse_args()

    mp = json.loads(args.mapping.read_text(encoding="utf-8"))
    base = Path(mp["dir"])
    labels = {}
    for line in (base / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            labels[r["path"]] = r
    data = json.loads(args.answers.read_text(encoding="utf-8"))
    answers = {str(r["id"]): r for r in (data["answers"] if isinstance(data, dict) else data)}

    if args.out.exists():
        shutil.rmtree(args.out)
    rows = []
    for head, names in HEADS.items():
        for m in mp["crops"]:
            t = truth_of(head, labels[m["path"]])
            if t is None:
                continue
            p = answers.get(m["id"], {}).get(FIELD[head])
            if head == "floor" and p == "triangle":
                p = "rectangle"
            if p not in names or p == t:
                continue
            src = Path(m["src"])
            src = src if src.is_absolute() else ROOT / src
            dest = args.out / head / f"{t}_as_{p}"
            dest.mkdir(parents=True, exist_ok=True)
            panel(src, f"{head}: true {t}  {args.who} {p}   [{m['id']}]").save(dest / f"{src.stem}.jpg", quality=88)
            rows.append({"head": head, "true": t, "pred": p, "id": m["id"], "crop": m["path"],
                         "source": "cmp" if m["path"].startswith("cmp_") else "lsaa"})
    for d in sorted(args.out.glob("*/*")):
        sheet(sorted(d.glob("*.jpg")), d)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "errors.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["head", "true", "pred", "id", "crop", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} errors -> {args.out}")
    for d in sorted(args.out.glob("*/*")):
        print(f"  {d.relative_to(args.out)}: {sum(1 for p in d.glob('*.jpg') if not p.name.startswith('sheet_'))}")


if __name__ == "__main__":
    main()
