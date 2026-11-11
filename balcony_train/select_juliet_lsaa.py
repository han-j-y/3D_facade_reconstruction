"""Pick LSAA facades for a juliet training round (structure labels on pipeline crops).

Pool (``balcony_facades.csv``): downloaded, ``|view_angle| <= --max-view-angle``,
``num_balconies_filtered >= 1``, not a building of the 100-photo test set
(``data/test100/photos``), not one of the facades already annotated in
``crops_lsaa``. One facade per building.

Per city quota (``--quota``, e.g. ``Vienna=26``); inside a city, facades with
more LSAA decoration (``deco``) and more balconies come first, then by name.
The chosen photos are copied to ``--photos-dir``; cut their pipeline crops with
``export_test100_crops.py --photos-dir … --out-dir … --cache …``.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/select_juliet_lsaa.py
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

from balcony_train.select_test_lsaa import DEFAULT_CROPS_DIR, DEFAULT_FACADE_DIR, building_of  # noqa: E402

DEFAULT_TEST_PHOTOS = ROOT / "data" / "test100" / "photos"
DEFAULT_OUT = ROOT / "runs" / "lsaa_juliet"
DEFAULT_QUOTA = ("Vienna=26", "Paris=18", "Rome=11", "Brussels=3", "Bucharest=2")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--facade-dir", type=Path, default=DEFAULT_FACADE_DIR)
    ap.add_argument("--test-photos", type=Path, default=DEFAULT_TEST_PHOTOS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--quota", nargs="*", default=list(DEFAULT_QUOTA), help="City=N")
    ap.add_argument("--max-view-angle", type=float, default=15.0)
    args = ap.parse_args(argv)

    test = {building_of(p.name)[1] for p in args.test_photos.glob("*.jpg") if not p.name.startswith("cmp_b")}
    annotated = {json.loads(l)["facade_name"]
                 for l in (DEFAULT_CROPS_DIR / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    with (args.facade_dir / "balcony_facades.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["downloaded"] == "True" and abs(float(r["view_angle"])) <= args.max_view_angle
                and int(r["num_balconies_filtered"]) >= 1 and r["building"] not in test
                and r["name"] not in annotated and (args.facade_dir / r["name"]).is_file()]
    rows.sort(key=lambda r: (-float(r["deco"]), -int(r["num_balconies_filtered"]), r["name"]))

    chosen: list[dict] = []
    for q in args.quota:
        city, n = q.split("=")
        used: set[str] = set()
        for r in rows:
            if len(used) >= int(n):
                break
            if r["city"] == city and r["building"] not in used:
                used.add(r["building"])
                chosen.append(r)
        print(f"  {city:10s} {len(used)} / {n}")

    photos = args.out_dir / "photos"
    photos.mkdir(parents=True, exist_ok=True)
    for r in chosen:
        shutil.copy2(args.facade_dir / r["name"], photos / r["name"])
    keep = ("name", "city", "building", "view_angle", "deco", "num_balconies_filtered")
    (args.out_dir / "facades.json").write_text(
        json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                    "facades": [{k: r[k] for k in keep} for r in chosen]}, indent=2),
        encoding="utf-8")
    print(f"{len(chosen)} facades -> {photos}")


if __name__ == "__main__":
    main()
