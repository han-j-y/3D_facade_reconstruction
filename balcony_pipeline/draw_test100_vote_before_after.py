r"""Photos where the type vote changes the balcony labels most: input | before vote | after vote.

Reads the rows of ``score_test100_types_final.py`` (``eval_test100_final_facade_v4.json``).
Before = classifier on each unit (``unit``), after = voted type label (``final``). Only the
scored crops (hand label, GT match, not oriel-removed) are drawn.

Each box: ``T<type>`` (after panel only), then kind / material / floor / structure as
``OW|SP|SO``, ``me|ma|-``, ``R|C``, ``P|J``. A wrong field is red with the truth after a
slash (``J/P`` = predicted J, true P). Box outline green when all four fields are right,
red otherwise. Photos are ranked by changed (crop, head) labels; ``ranking.csv`` lists all.

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_pipeline/draw_test100_vote_before_after.py --top 15
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
T100 = ROOT / "runs" / "test100"
HEADS = ("kind", "material", "floor", "structure")
ABBR = {
    "kind": {"open_work": "OW", "surface_panel": "SP", "solid": "SO"},
    "material": {"metal": "me", "masonry": "ma", None: "-"},
    "floor": {"rectangle": "R", "circle": "C"},
    "structure": {"projecting": "P", "juliet": "J"},
}
GREEN, RED, WHITE, BLACK = (40, 200, 70), (235, 40, 40), (255, 255, 255), (0, 0, 0)
PANEL_H = 1000


def font(size: int):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fields(r: dict, stage: str) -> list[tuple[str, bool]]:
    out = []
    for h in HEADS:
        p, t = r[stage][h], r["truth"][h]
        txt = ABBR[h].get(p, str(p))
        if t is None or (h == "material" and r["truth"]["kind"] != "open_work"):
            out.append((txt, True))
        elif p == t:
            out.append((txt, True))
        else:
            out.append((f"{txt}/{ABBR[h].get(t, t)}", False))
    return out


def n_wrong(r: dict, stage: str) -> int:
    return sum(1 for h in HEADS if r["truth"][h] is not None and r[stage][h] != r["truth"][h])


def panel(photo: Image.Image, rows: list[dict], stage: str | None, title: str) -> Image.Image:
    s = PANEL_H / photo.height
    im = photo.resize((max(1, round(photo.width * s)), PANEL_H), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    f = font(max(13, round(PANEL_H / 55)))
    if stage:
        for r in rows:
            x0, y0, x1, y1 = (round(v * s) for v in r["box"])
            fl = fields(r, stage)
            ok = all(o for _, o in fl)
            d.rectangle((x0, y0, x1, y1), outline=GREEN if ok else RED, width=3)
            parts = ([(f"T{r['type_id']}", True)] if stage == "final" else []) + fl
            x, y = x0, max(0, y0 - f.size - 6)
            w = sum(d.textlength(t + " ", font=f) for t, _ in parts)
            d.rectangle((x, y, x + w + 4, y + f.size + 5), fill=BLACK)
            for t, o in parts:
                d.text((x + 2, y + 2), t, font=f, fill=WHITE if o else RED)
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
    ap.add_argument("--photos", type=Path, default=T100 / "photos")
    ap.add_argument("--out-dir", type=Path, default=T100 / "vote_before_after")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    by_stem: dict[str, list[dict]] = defaultdict(list)
    for r in json.loads(args.rows.read_text(encoding="utf-8"))["rows"]:
        by_stem[r["stem"]].append(r)
    rank = []
    for s, rows in by_stem.items():
        changed = sum(1 for r in rows for h in HEADS if r["unit"][h] != r["final"][h])
        wb = sum(n_wrong(r, "unit") for r in rows)
        wa = sum(n_wrong(r, "final") for r in rows)
        rank.append({"stem": s, "crops": len(rows), "types": len({r["type_id"] for r in rows}),
                     "changed_labels": changed, "wrong_before": wb, "wrong_after": wa, "net": wb - wa})
    rank.sort(key=lambda x: (-x["changed_labels"], x["stem"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "ranking.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rank[0]))
        w.writeheader()
        w.writerows(rank)
    for i, e in enumerate(rank[: args.top], 1):
        if e["changed_labels"] == 0:
            break
        s, rows = e["stem"], by_stem[e["stem"]]
        photo = Image.open(args.photos / f"{s}.jpg").convert("RGB")
        ps = [
            panel(photo, rows, None, "Input"),
            panel(photo, rows, "unit", f"Before vote (per unit)  wrong labels {e['wrong_before']}"),
            panel(photo, rows, "final", f"After vote ({e['types']} types)  wrong labels {e['wrong_after']}"),
        ]
        gap = 12
        sheet = Image.new("RGB", (sum(p.width for p in ps) + gap * 2, ps[0].height + 40), (30, 30, 30))
        x = 0
        for p in ps:
            sheet.paste(p, (x, 40))
            x += p.width + gap
        ImageDraw.Draw(sheet).text(
            (10, 8), f"#{i}  {s}   changed {e['changed_labels']}   wrong {e['wrong_before']} -> {e['wrong_after']}"
                     f"   (kind/material/floor/structure; red = wrong, pred/true)", font=font(22), fill=WHITE)
        name = f"{i:02d}_{s}.jpg"
        sheet.save(args.out_dir / name, quality=90)
        print(f"{name}  changed {e['changed_labels']}  wrong {e['wrong_before']} -> {e['wrong_after']}")
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
