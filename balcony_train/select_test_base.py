"""Pick the cmp_b (Base) photos for the Base + LSAA 100-photo test set.

Counts open-balcony pipeline crops per photo from ``labels.jsonl`` (crop names
start with the photo stem, e.g. ``cmp_b0001_pbal_000``):

- kind: open_work / surface_panel / solid;
- material on open_work only: metal / masonry;
- floor: rectangle (triangle counts as rectangle) / circle; null is not counted;
- structure: projecting / juliet; null is not counted.

Selection: ``--n-oriel`` photos from the oriel photos (enclosure labels with
cells), then greedy cover over the other photos until every class reaches
``--target``, then pruning of non-oriel photos that are not needed. Rare
classes weigh more (1 / Base total). Ties break by stem, so runs repeat.

``--stems FILE`` skips selection and uses those photos (hand-picked lists, or
a re-check after labeling); the outputs are rewritten for them.

Outputs in ``--out-dir``: ``base_selected.json``, ``heldout_stems.txt`` and
``board.jpg`` (contact sheet).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.enclosed_grid_store import load_labels as load_cell_labels  # noqa: E402
from balcony_train.label_store import load_labels  # noqa: E402
from balcony_train.paths import DEFAULT_LABELS_JSONL  # noqa: E402

CLASSES: tuple[str, ...] = (
    "open_work", "surface_panel", "solid",
    "metal", "masonry",
    "rectangle", "circle",
    "projecting", "juliet",
)
STEM_RE = re.compile(r"^(cmp_b\d{4})_")
DEFAULT_ORIEL_LABELS = ROOT / "runs" / "enclosed_grid" / "labels_train.jsonl"
DEFAULT_PHOTO_DIR = ROOT / "data" / "base_jpg"
DEFAULT_OUT_DIR = ROOT / "runs" / "test100"


def crop_classes(rec: dict) -> list[str]:
    """Class values one crop counts toward (open balconies only)."""
    if rec.get("enclosure") not in (None, "open") or not rec.get("kind"):
        return []
    out = [rec["kind"]]
    if rec["kind"] == "open_work" and rec.get("material"):
        out.append(rec["material"])
    floor = rec.get("floor_shape")
    if floor:
        out.append("rectangle" if floor == "triangle" else floor)
    if rec.get("structure"):
        out.append(rec["structure"])
    return out


def photo_counts(records: dict[str, dict]) -> dict[str, dict]:
    """Per photo stem: class counts and unlabeled floor/structure counts."""
    photos: dict[str, dict] = {}
    for rec in records.values():
        m = STEM_RE.match(Path(rec["path"]).name)
        if not m:
            continue
        p = photos.setdefault(m.group(1), {"counts": Counter(), "crops": 0,
                                           "no_floor": 0, "no_structure": 0})
        cls = crop_classes(rec)
        if not cls:
            continue
        p["crops"] += 1
        p["counts"].update(cls)
        if rec.get("structure") != "juliet" and not rec.get("floor_shape"):
            p["no_floor"] += 1
        if not rec.get("structure"):
            p["no_structure"] += 1
    return photos


def _total(photos: dict[str, dict], stems) -> Counter:
    tot: Counter = Counter()
    for s in stems:
        tot.update(photos[s]["counts"])
    return tot


def _gain(counts: Counter, tot: Counter, target: int, weight: dict[str, float]) -> float:
    return sum(weight[c] * min(counts[c], max(0, target - tot[c])) for c in CLASSES)


def select(photos: dict[str, dict], oriel: set[str], *, n_oriel: int, target: int) -> list[str]:
    """Oriel photos first, then greedy cover of the rest, then pruning."""
    base_tot = _total(photos, photos)
    weight = {c: 1.0 / max(1, base_tot[c]) for c in CLASSES}
    chosen: list[str] = []

    def pick(pool: list[str]) -> str | None:
        tot = _total(photos, chosen)
        best = max(pool, key=lambda s: (_gain(photos[s]["counts"], tot, target, weight),
                                        photos[s]["crops"], _neg(s)), default=None)
        return best

    oriel_pool = sorted(s for s in oriel if s in photos)
    if len(oriel_pool) < n_oriel:
        raise ValueError(f"only {len(oriel_pool)} oriel photos with crops, need {n_oriel}")
    for _ in range(n_oriel):
        s = pick([x for x in oriel_pool if x not in chosen])
        chosen.append(s)

    rest = sorted(s for s in photos if s not in oriel)
    while any(_total(photos, chosen)[c] < target for c in CLASSES):
        pool = [s for s in rest if s not in chosen]
        s = pick(pool)
        if s is None or _gain(photos[s]["counts"], _total(photos, chosen), target, weight) == 0:
            break
        chosen.append(s)

    fill = [s for s in chosen if s not in oriel]
    for s in sorted(fill, key=lambda x: (photos[x]["crops"], x)):
        trial = [x for x in chosen if x != s]
        tot = _total(photos, trial)
        if all(tot[c] >= target for c in CLASSES):
            chosen = trial
    return sorted(chosen)


def _neg(stem: str) -> tuple[int, ...]:
    # max() keeps the first best; negate so the smaller stem wins ties.
    return tuple(-ord(ch) for ch in stem)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draw_board(stems: list[str], photos: dict[str, dict], oriel: set[str],
               photo_dir: Path, out: Path, *, thumb: int = 260, cols: int = 6) -> None:
    from PIL import Image, ImageDraw

    rows = (len(stems) + cols - 1) // cols
    board = Image.new("RGB", (cols * thumb, rows * (thumb + 40)), (255, 255, 255))
    draw = ImageDraw.Draw(board)
    short = {"open_work": "ow", "surface_panel": "sp", "solid": "so", "metal": "me",
             "masonry": "ma", "rectangle": "re", "circle": "ci", "projecting": "pr",
             "juliet": "ju"}
    for i, s in enumerate(stems):
        x, y = (i % cols) * thumb, (i // cols) * (thumb + 40)
        path = photo_dir / f"{s}.jpg"
        if path.is_file():
            im = Image.open(path).convert("RGB")
            im.thumbnail((thumb - 4, thumb - 4))
            board.paste(im, (x + 2, y + 2))
        c = photos[s]["counts"]
        head = s + (" ORIEL" if s in oriel else "")
        body = " ".join(f"{short[k]}{c[k]}" for k in CLASSES if c[k])
        draw.text((x + 4, y + thumb + 2), head, fill=(200, 0, 0) if s in oriel else (0, 0, 0))
        draw.text((x + 4, y + thumb + 18), body, fill=(0, 0, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    board.save(out, quality=90)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS_JSONL)
    ap.add_argument("--oriel-labels", type=Path, default=DEFAULT_ORIEL_LABELS)
    ap.add_argument("--photo-dir", type=Path, default=DEFAULT_PHOTO_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--n-oriel", type=int, default=10)
    ap.add_argument("--target", type=int, default=12, help="crops per class (10 + margin)")
    ap.add_argument("--stems", type=Path, help="report counts for these stems, no selection")
    ap.add_argument("--exclude", nargs="*", default=[], help="stems never selected")
    args = ap.parse_args(argv)

    photos = photo_counts(load_labels(args.labels))
    for s in args.exclude:
        photos.pop(s, None)
    oriel = {s for s, cells in load_cell_labels(args.oriel_labels).items() if cells}
    if args.stems:
        stems = sorted(l.strip() for l in args.stems.read_text(encoding="utf-8").split() if l.strip())
        missing = [s for s in stems if s not in photos]
        if missing:
            print(f"no labeled crops: {missing}")
        stems = [s for s in stems if s in photos]
    else:
        stems = select(photos, oriel, n_oriel=args.n_oriel, target=args.target)

    tot = _total(photos, stems)
    print(f"{len(stems)} photos, {sum(s in oriel for s in stems)} with oriels")
    for c in CLASSES:
        flag = "" if tot[c] >= args.target else "  < target"
        print(f"  {c:14s} {tot[c]:4d}{flag}")
    print(f"  unlabeled floor {sum(photos[s]['no_floor'] for s in stems)}, "
          f"structure {sum(photos[s]['no_structure'] for s in stems)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "inputs_sha256": {"labels": _sha256(args.labels), "oriel_labels": _sha256(args.oriel_labels)},
        "n_photos": len(stems),
        "n_oriel": sum(s in oriel for s in stems),
        "totals": {c: tot[c] for c in CLASSES},
        "photos": [
            {"stem": s, "oriel": s in oriel, "crops": photos[s]["crops"],
             "counts": {c: photos[s]["counts"][c] for c in CLASSES if photos[s]["counts"][c]},
             "no_floor": photos[s]["no_floor"], "no_structure": photos[s]["no_structure"]}
            for s in stems
        ],
    }
    (args.out_dir / "base_selected.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.out_dir / "heldout_stems.txt").open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(stems) + "\n")
    draw_board(stems, photos, oriel, args.photo_dir, args.out_dir / "board.jpg")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
