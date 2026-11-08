r"""val30: crops whose label the type vote broke (own prediction right, voted label wrong), per head and setup.

Reads the ``--dump`` JSON of ``score_val30_types.py`` (per setup: truth / unit / final labels, photo, box, type,
GT group). Conditional scoring as the scorer: material only where truth and prediction are open work, floor
only where both are projecting. For each setup and head:

* ``<setup>_<head>_broken.jpg``: one tile per broken crop (context crop, box in red), truth / own / voted label,
  the type id, the type's own-label votes and whether the type mixes GT groups;
* ``<setup>_<head>_<id>.jpg``: the photo with every unit of the affected types; box color = GT group,
  red tag = broken, tag = own label -> voted label.

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_pipeline/score_val30_types.py --setup NEW= \
        --setup "LINKS=--cluster-links" --dump runs/val30/_vote_broken/rows.json
    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_pipeline/draw_val30_vote_broken.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from draw_test100_vote_before_after import font  # noqa: E402
from score_test100_types_final import conditional_skip  # noqa: E402

AB = {"open_work": "OW", "surface_panel": "SP", "solid": "SO", "metal": "me", "masonry": "ma", "rectangle": "R",
      "circle": "C", "triangle": "T", "projecting": "P", "juliet": "J", None: "-"}
GROUP_COLORS = [(0, 170, 255), (255, 0, 200), (255, 220, 0), (0, 230, 120), (150, 90, 255), (255, 120, 0),
                (0, 255, 255), (255, 80, 80), (170, 255, 0), (120, 160, 255)]
TW, TH, CAP = 340, 270, 66


def ab(v):
    return AB.get(v, str(v))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, default=ROOT / "runs" / "val30" / "_vote_broken" / "rows.json")
    ap.add_argument("--photos", type=Path, default=ROOT / "data" / "val30")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "val30" / "_vote_broken")
    ap.add_argument("--heads", default="kind,material,floor,structure")
    ap.add_argument("--setups", default="", help="comma list (default: all but RUN)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.rows.read_text(encoding="utf-8"))
    setups = [s for s in (args.setups.split(",") if args.setups else data) if s != "RUN"]
    cache: dict[str, Image.Image] = {}

    def photo(stem):
        if stem not in cache:
            cache.clear()
            p = next(q for q in (args.photos / f"{stem}.jpg", args.photos / f"{stem}.png") if q.is_file())
            cache[stem] = Image.open(p).convert("RGB")
        return cache[stem]

    for setup in setups:
        rows = data[setup]
        by_type = defaultdict(list)
        for r in rows:
            by_type[(r["stem"], r["type_id"])].append(r)
        for head in args.heads.split(","):
            broken = []
            for r in rows:
                t = r["truth"][head]
                if t is None or conditional_skip(r, "unit", head) or conditional_skip(r, "final", head):
                    continue
                if r["unit"][head] == t and r["final"][head] != t:
                    broken.append(r)
            broken.sort(key=lambda r: (r["id"], r["type_id"], r["unit_id"]))
            tiles = []
            for r in broken:
                im = photo(r["stem"])
                x0, y0, x1, y1 = r["box"]
                m = max(x1 - x0, y1 - y0) * 0.6
                cx0, cy0 = max(0, int(x0 - m)), max(0, int(y0 - m))
                cx1, cy1 = min(im.width, int(x1 + m)), min(im.height, int(y1 + m))
                c = im.crop((cx0, cy0, cx1, cy1))
                sc = min(TW / c.width, (TH - CAP) / c.height)
                c = c.resize((max(1, int(c.width * sc)), max(1, int(c.height * sc))))
                ImageDraw.Draw(c).rectangle(((x0 - cx0) * sc, (y0 - cy0) * sc, (x1 - cx0) * sc, (y1 - cy0) * sc),
                                            outline=(255, 40, 40), width=3)
                tile = Image.new("RGB", (TW, TH), (30, 30, 30))
                tile.paste(c, ((TW - c.width) // 2, CAP + (TH - CAP - c.height) // 2))
                d = ImageDraw.Draw(tile)
                f = font(15)
                mem = by_type[(r["stem"], r["type_id"])]
                votes = Counter(ab(x["unit"][head]) for x in mem
                                if not (head == "floor" and x["unit"]["structure"] == "juliet"))
                ngt = len({x["gt_group"] for x in mem})
                d.text((5, 2), f"{r['id']} u{r['unit_id']} T{r['type_id']}  GT groups in type: {ngt}", font=f,
                       fill=(255, 120, 120))
                d.text((5, 22), f"truth {ab(r['truth'][head])}  own {ab(r['unit'][head])}  voted {ab(r['final'][head])}",
                       font=f, fill=(235, 235, 235))
                d.text((5, 42), f"type {len(mem)} units, own votes {dict(votes)}", font=f, fill=(200, 200, 200))
                tiles.append(tile)
            cols = 6
            nrow = max(1, (len(tiles) + cols - 1) // cols)
            sh = Image.new("RGB", (cols * (TW + 6) + 6, 46 + nrow * (TH + 6)), (15, 15, 15))
            ImageDraw.Draw(sh).text(
                (10, 10), f"val30 {setup} {head}: own label right, voted label wrong ({len(broken)} crops, "
                          f"{len({r['id'] for r in broken})} photos)", font=font(22), fill=(255, 255, 255))
            for k, t in enumerate(tiles):
                sh.paste(t, (6 + (k % cols) * (TW + 6), 46 + (k // cols) * (TH + 6)))
            sh.save(args.out_dir / f"{setup}_{head}_broken.jpg", quality=88)
            print(f"{setup} {head}: {len(broken)} broken in {len({r['id'] for r in broken})} photos")
            for vid in sorted({r["id"] for r in broken}):
                rs = [r for r in broken if r["id"] == vid]
                stem = rs[0]["stem"]
                types = {r["type_id"] for r in rs}
                bad = {r["unit_id"] for r in rs}
                im = photo(stem).copy()
                d = ImageDraw.Draw(im)
                f = font(max(14, im.height // 55))
                for x in rows:
                    if x["stem"] != stem or x["type_id"] not in types:
                        continue
                    col = GROUP_COLORS[(int(x["gt_group"]) - 1) % len(GROUP_COLORS)]
                    bx0, by0, bx1, by1 = x["box"]
                    d.rectangle((bx0, by0, bx1, by1), outline=(255, 40, 40) if x["unit_id"] in bad else col,
                                width=5 if x["unit_id"] in bad else 3)
                    txt = f"T{x['type_id']} gt-g{x['gt_group']} {ab(x['truth'][head])}: {ab(x['unit'][head])}->{ab(x['final'][head])}"
                    tw = d.textlength(txt, font=f)
                    d.rectangle((bx0, by0 - f.size - 6, bx0 + tw + 6, by0), fill=(0, 0, 0))
                    d.text((bx0 + 3, by0 - f.size - 4), txt, font=f, fill=(255, 80, 80) if x["unit_id"] in bad else col)
                bar = Image.new("RGB", (im.width, 44), (0, 0, 0))
                ImageDraw.Draw(bar).text((8, 8), f"{vid} {setup} {head}: box color = GT group, red = broken by the vote, "
                                                 f"tag = truth: own -> voted", font=font(20), fill=(255, 255, 255))
                out = Image.new("RGB", (im.width, im.height + 44))
                out.paste(bar, (0, 0))
                out.paste(im, (0, 44))
                out.save(args.out_dir / f"{setup}_{head}_{vid}.jpg", quality=88)
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
