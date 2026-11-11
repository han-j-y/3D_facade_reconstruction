r"""Every test100 photo: input | balcony types with per-unit labels | same types with voted labels.

Reads ``photos`` and ``rows`` of ``score_test100_types_final.py`` (``eval_test100_final_facade_v4.json``):
every open-balcony unit the pipeline types (after the oriel removal), not only the scored crops.

- Box color = type (cluster); thick box = the type's medoid (exemplar).
- Label ``T<type>`` then kind / material / floor / structure (``OW|SP|SO``, ``me|ma|-``, ``R|C``, ``P|J``).
- Scored unit (hand label + GT match): a wrong field is red as ``pred/true``.
- Unscored unit (no hand label or no GT match, still votes): label starts with ``?`` on a grey tag.
- Oriel cells: orange dashed (their open boxes were removed before typing).

Files are numbered by labels changed by the vote (all units), most first; ``ranking.csv`` lists them.

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_pipeline/draw_test100_clusters.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from draw_test100_vote_before_after import ABBR, HEADS, font

ROOT = Path(__file__).resolve().parents[1]
T100 = ROOT / "runs" / "test100"
PANEL_H = 1000
TYPE_COLORS = [(0, 170, 255), (255, 0, 200), (255, 220, 0), (0, 230, 120), (150, 90, 255), (255, 120, 0),
               (0, 255, 255), (255, 80, 80), (170, 255, 0), (120, 160, 255)]
ORANGE, WHITE, RED, BLACK, GREY = (255, 150, 0), (255, 255, 255), (255, 60, 60), (0, 0, 0), (90, 90, 90)


def dashed(d: ImageDraw.ImageDraw, box, color, width=3, dash=10) -> None:
    x0, y0, x1, y1 = box
    for a, b, horiz, c in ((x0, x1, True, y0), (x0, x1, True, y1), (y0, y1, False, x0), (y0, y1, False, x1)):
        for t in range(int(a), int(b), dash * 2):
            e = min(t + dash, b)
            d.line((t, c, e, c) if horiz else (c, t, c, e), fill=color, width=width)


def unit_fields(u: dict, truth: dict | None, stage: str) -> list[tuple[str, bool]]:
    out = []
    for h in HEADS:
        p = u[stage][h]
        txt = ABBR[h].get(p, str(p))
        t = truth.get(h) if truth else None
        if t is None or p == t:
            out.append((txt, True))
        else:
            out.append((f"{txt}/{ABBR[h].get(t, t)}", False))
    return out


def panel(photo: Image.Image, data: dict, truth_of: dict, stage: str | None, title: str) -> Image.Image:
    s = PANEL_H / photo.height
    im = photo.resize((max(1, round(photo.width * s)), PANEL_H), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    f = font(max(13, round(PANEL_H / 58)))
    if stage:
        for c in data.get("oriel_cells", []):
            dashed(d, [v * s for v in c], ORANGE)
        for u in data["units"]:
            x0, y0, x1, y1 = (round(v * s) for v in u["box"])
            col = TYPE_COLORS[u["type_id"] % len(TYPE_COLORS)]
            d.rectangle((x0, y0, x1, y1), outline=col, width=6 if u["medoid"] else 3)
            truth = truth_of.get(u["unit_id"])
            parts = [(("" if truth else "? ") + f"T{u['type_id']}", True)] + unit_fields(u, truth, stage)
            w = sum(d.textlength(t + " ", font=f) for t, _ in parts)
            x, y = x0, max(0, y0 - f.size - 6)
            d.rectangle((x, y, x + w + 4, y + f.size + 5), fill=BLACK if truth else GREY, outline=col, width=2)
            for t, ok in parts:
                d.text((x + 2, y + 2), t, font=f, fill=WHITE if ok else RED)
                x += d.textlength(t + " ", font=f)
    bar = Image.new("RGB", (im.width, 46), BLACK)
    ImageDraw.Draw(bar).text((10, 10), title, font=font(24), fill=WHITE)
    out = Image.new("RGB", (im.width, im.height + bar.height), BLACK)
    out.paste(bar, (0, 0))
    out.paste(im, (0, bar.height))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, default=T100 / "eval_test100_final_facade_v4.json")
    ap.add_argument("--photos", type=Path, default=ROOT / "data" / "test100" / "photos")
    ap.add_argument("--out-dir", type=Path, default=T100 / "vote_clusters_all")
    args = ap.parse_args()

    blob = json.loads(args.rows.read_text(encoding="utf-8"))
    photos = blob["photos"]
    truth_by: dict[str, dict[int, dict]] = defaultdict(dict)
    wrong = defaultdict(lambda: [0, 0])
    for r in blob["rows"]:
        truth_by[r["stem"]].setdefault(r["unit_id"], r["truth"])
        for h in HEADS:
            if r["truth"][h] is not None:
                wrong[r["stem"]][0] += r["unit"][h] != r["truth"][h]
                wrong[r["stem"]][1] += r["final"][h] != r["truth"][h]

    rank = []
    for s, data in photos.items():
        units = data["units"]
        rank.append({
            "stem": s, "units": len(units), "scored_units": len(truth_by[s]),
            "types": len({u["type_id"] for u in units}),
            "changed_labels": sum(1 for u in units for h in HEADS if u["unit"][h] != u["final"][h]),
            "wrong_before": wrong[s][0], "wrong_after": wrong[s][1], "oriel_cells": len(data["oriel_cells"]),
        })
    rank.sort(key=lambda x: (-x["changed_labels"], -x["units"], x["stem"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "ranking.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rank[0]))
        w.writeheader()
        w.writerows(rank)
    for i, e in enumerate(rank, 1):
        s, data = e["stem"], photos[e["stem"]]
        photo = Image.open(args.photos / f"{s}.jpg").convert("RGB")
        tr = truth_by[s]
        ps = [
            panel(photo, data, tr, None, "Input"),
            panel(photo, data, tr, "unit", f"Types + per-unit labels (before vote)  wrong {e['wrong_before']}"),
            panel(photo, data, tr, "final", f"Types + voted labels (after vote)  wrong {e['wrong_after']}"),
        ]
        gap = 12
        sheet = Image.new("RGB", (sum(p.width for p in ps) + gap * 2, ps[0].height + 40), (30, 30, 30))
        x = 0
        for p in ps:
            sheet.paste(p, (x, 40))
            x += p.width + gap
        ImageDraw.Draw(sheet).text(
            (10, 8),
            f"#{i}  {s}   units {e['units']} (scored {e['scored_units']})  types {e['types']}  "
            f"changed {e['changed_labels']}  wrong {e['wrong_before']} -> {e['wrong_after']}   "
            f"[color = type, thick = medoid, ? = unscored, red = pred/true, orange dashed = oriel]",
            font=font(22), fill=WHITE)
        sheet.save(args.out_dir / f"{i:03d}_{s}.jpg", quality=88)
    print(f"{len(rank)} sheets -> {args.out_dir}")


if __name__ == "__main__":
    main()
