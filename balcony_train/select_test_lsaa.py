"""Pick the LSAA facades for the Base + LSAA 100-photo test set.

Pool: facades with at least one tagged crop in ``crops_lsaa`` (the FiftyOne
annotation exported by lsaa-dataset ``export_balcony_labels.py``). Class counts
use the same rules as ``select_test_base.crop_classes``. LSAA has no structure
or oriel tags.

At most one facade per building. The building is the OSM id in the name
(``Austria_Vienna_relation2445885_Fid…`` → ``relation2445885``). Greedy
selection: gain in sum over classes of log(1 + count) per LSAA training crop of
the building, so classes stay balanced and the hold-out stays small. Ties go to
facades whose tagged crops cover LSAA's filtered balcony count, then to
buildings with fewer training crops, then by name.

Outputs in ``--out-dir``:
- ``lsaa_selected.json``: per-facade counts and totals;
- ``lsaa_facades.txt``: chosen facade image names;
- ``lsaa_heldout_prefixes.txt``: one ``lsaa_<country>_<city>_<building>_``
  prefix per building, for ``train.py --exclude-prefix``; it covers every
  LSAA crop of that building, not only the chosen facade;
- ``board_lsaa.jpg``.

``--keep`` takes hand-checked facades first; ``--max-view-angle`` limits the
other picks to near-frontal photos (LSAA ``view_angle``). ``--photos-dir`` also copies the chosen facade images there.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.label_store import load_labels  # noqa: E402
from balcony_train.select_test_base import CLASSES, crop_classes  # noqa: E402

LSAA_ROOT = Path(r"C:\Users\hiroo\Desktop\temp\lsaa-dataset")
DEFAULT_CROPS_DIR = ROOT / "runs" / "balcony_clf" / "crops_lsaa"
DEFAULT_FACADE_DIR = LSAA_ROOT / "data" / "Facades_balcony"
DEFAULT_OUT_DIR = ROOT / "runs" / "test100"
BUILDING_RE = re.compile(r"^(.*?_((?:relation|way|node)\d+))_Fid")
# LSAA has no structure labels; these never count.
LSAA_CLASSES = tuple(c for c in CLASSES if c not in ("projecting", "juliet"))


def building_of(name: str) -> tuple[str, str]:
    """(``country_city_building`` prefix, OSM building id) of a facade or crop name."""
    m = BUILDING_RE.match(name)
    if not m:
        raise ValueError(f"no building id in {name!r}")
    return m.group(1), m.group(2)


def facade_counts(manifest: list[dict], labels: dict[str, dict]) -> dict[str, dict]:
    """Per facade name: class counts of its tagged crops."""
    out: dict[str, dict] = {}
    for m in manifest:
        f = out.setdefault(m["facade_name"], {"counts": Counter(), "tagged": 0})
        f["tagged"] += 1
        f["counts"].update(crop_classes(labels[Path(m["crop"]).stem]))
    return out


def select(facades: dict[str, dict], *, n: int, filtered: dict[str, int],
           building_crops: Counter, keep: list[str] = (), allowed=None) -> list[str]:
    """Greedy on the gain in sum over classes of log(1 + count), per training crop
    the building takes out of LSAA training. The log favors the classes that are
    furthest behind; the division keeps the hold-out small. ``keep`` facades
    are taken first; ``allowed`` (if given) limits the rest of the pool."""
    chosen: list[str] = []
    used: set[str] = set()
    tot: Counter = Counter()
    for f in keep:
        chosen.append(f)
        used.add(building_of(f)[1])
        tot.update(facades[f]["counts"])

    def key(name: str):
        c = facades[name]["counts"]
        bid = building_of(name)[1]
        gain = sum(math.log1p(tot[k] + c[k]) - math.log1p(tot[k]) for k in LSAA_CLASSES)
        gain = round(gain / max(1, building_crops[bid]), 9)
        complete = facades[name]["tagged"] >= filtered.get(name, 10**9)
        return (-gain, not complete, building_crops[bid], name)

    while len(chosen) < n:
        pool = [f for f in facades if building_of(f)[1] not in used
                and (allowed is None or f in allowed)]
        if not pool:
            break
        best = min(pool, key=key)
        chosen.append(best)
        used.add(building_of(best)[1])
        tot.update(facades[best]["counts"])
    return chosen


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS_DIR)
    ap.add_argument("--facade-dir", type=Path, default=DEFAULT_FACADE_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--exclude", nargs="*", default=[], help="facade names or building ids never selected")
    ap.add_argument("--keep", nargs="*", default=[], help="facade names or building ids always selected (checked by hand)")
    ap.add_argument("--max-view-angle", type=float, help="new picks need |view_angle| <= this (frontal photos)")
    ap.add_argument("--photos-dir", type=Path, help="copy the chosen facade images here")
    args = ap.parse_args(argv)

    manifest = [json.loads(l) for l in (args.crops_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    labels = load_labels(args.crops_dir / "labels.jsonl")
    facades = facade_counts(manifest, labels)
    for x in args.exclude:
        for f in [f for f in facades if f == x or building_of(f)[1] == x]:
            facades.pop(f)
    building_crops = Counter(building_of(m["facade_name"])[1] for m in manifest)
    filtered: dict[str, int] = {}
    view_angle: dict[str, float] = {}
    with (args.facade_dir / "balcony_facades.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            filtered[row["name"]] = int(row["num_balconies_filtered"])
            view_angle[row["name"]] = float(row["view_angle"])
    keep = [f for f in sorted(facades) if any(f == x or building_of(f)[1] == x for x in args.keep)]
    ambiguous = sorted({building_of(f)[1] for f in keep if sum(building_of(g)[1] == building_of(f)[1] for g in keep) > 1})
    if ambiguous:
        raise SystemExit(f"--keep matches several facades of {ambiguous}; pass the facade name instead")
    allowed = None
    if args.max_view_angle is not None:
        allowed = {f for f in facades if abs(view_angle.get(f, 180.0)) <= args.max_view_angle}

    chosen = select(facades, n=args.n, filtered=filtered, building_crops=building_crops,
                    keep=keep, allowed=allowed)
    tot: Counter = Counter()
    for f in chosen:
        tot.update(facades[f]["counts"])
    print(f"{len(chosen)} facades from {len({building_of(f)[1] for f in chosen})} buildings")
    for c in LSAA_CLASSES:
        print(f"  {c:14s} {tot[c]:4d}")
    complete = sum(facades[f]["tagged"] >= filtered.get(f, 10**9) for f in chosen)
    print(f"  fully tagged facades {complete}, held-out training crops "
          f"{sum(building_crops[building_of(f)[1]] for f in chosen)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "n_facades": len(chosen),
        "totals": {c: tot[c] for c in LSAA_CLASSES},
        "facades": [
            {"name": f, "building": building_of(f)[1], "kept": f in keep,
             "view_angle": view_angle.get(f), "tagged": facades[f]["tagged"],
             "lsaa_filtered_balconies": filtered.get(f),
             "building_crops": building_crops[building_of(f)[1]],
             "counts": {c: facades[f]["counts"][c] for c in LSAA_CLASSES if facades[f]["counts"][c]}}
            for f in chosen
        ],
    }
    (args.out_dir / "lsaa_selected.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for fname, rows in (("lsaa_facades.txt", chosen),
                        ("lsaa_heldout_prefixes.txt", sorted(f"lsaa_{building_of(f)[0]}_" for f in chosen))):
        with (args.out_dir / fname).open("w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(rows) + "\n")
    draw_board(chosen, facades, args.facade_dir, args.out_dir / "board_lsaa.jpg")
    if args.photos_dir:
        args.photos_dir.mkdir(parents=True, exist_ok=True)
        for f in chosen:
            shutil.copy2(args.facade_dir / f, args.photos_dir / f)
    print(f"wrote {args.out_dir}")


def draw_board(names: list[str], facades: dict[str, dict], facade_dir: Path, out: Path,
               *, thumb: int = 220, cols: int = 10) -> None:
    from PIL import Image, ImageDraw

    short = {"open_work": "ow", "surface_panel": "sp", "solid": "so", "metal": "me",
             "masonry": "ma", "rectangle": "re", "circle": "ci"}
    rows = (len(names) + cols - 1) // cols
    board = Image.new("RGB", (cols * thumb, rows * (thumb + 34)), (255, 255, 255))
    draw = ImageDraw.Draw(board)
    for i, f in enumerate(names):
        x, y = (i % cols) * thumb, (i // cols) * (thumb + 34)
        path = facade_dir / f
        if path.is_file():
            im = Image.open(path).convert("RGB")
            im.thumbnail((thumb - 4, thumb - 4))
            board.paste(im, (x + 2, y + 2))
        c = facades[f]["counts"]
        draw.text((x + 4, y + thumb + 2), f"{i:02d} {building_of(f)[1]}", fill=(0, 0, 0))
        draw.text((x + 4, y + thumb + 17), " ".join(f"{short[k]}{c[k]}" for k in LSAA_CLASSES if c[k]),
                  fill=(0, 0, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    board.save(out, quality=88)


if __name__ == "__main__":
    main()
