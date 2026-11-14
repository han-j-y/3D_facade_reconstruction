#!/usr/bin/env python3
"""Sheets for layout_floorwise.py (--norm grid): per façade, for Astra and ours, the render and the photo with
photo windows green = found / orange = missed / grey = unusable (don't care), and render windows (placed on the photo)
cyan = on a photo window / red = extra / white dashed = on an unusable window. Writes eval_layout_viz/<tid>.jpg and
pages of 6."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent)); import paths  # noqa: E402
HERE = paths.ASTRA_DIR
import eval_testset as ev  # noqa: E402
import layout_floorwise as lf  # noqa: E402
from eval_astra import init_alignment, refine_alignment  # noqa: E402
from eval_astra_protocol import grid_map  # noqa: E402

F = lambda n: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", n)  # noqa: E731
S = paths.OUT_DIR / "renders"  # rendered front views (only used by the sheet helpers)
RENDERS = {"astra": lambda stem: next(p for d in ("light_labels", "light_labels2") for p in (HERE / d / "photos").glob(f"{stem}.*")),
           "ours_full": lambda stem: S / "ours_full" / "photos" / f"{stem}.png",
           "ours_full_4702b22": lambda stem: S / "ours_full47" / "photos" / f"{stem}.png",
           "ours_cur": lambda stem: S / "ours_cur46" / "photos" / f"{stem}.png"}
NAMES = {"astra": "Astra", "ours_full": "Ours (full output)", "ours_full_4702b22": "Ours (full output, 4702b22)", "ours_cur": "Ours (full output, 2026-11-11)"}


def classify(photo, render):
    use = [not w["unusable"] for w in photo]
    pb = [w["box"] for w in photo]
    gb = [b for b, u in zip(pb, use) if u]
    sc, sh = init_alignment(render, gb)
    sc, sh = refine_alignment(render, gb, sc, sh)
    fm = grid_map(render, gb, sc, sh)
    rb = [fm(b) for b in render]
    pf, rf = lf.floors(pb), lf.floors(rb)
    pc, rc = [lf.ycentre(pb, f) for f in pf], [lf.ycentre(rb, f) for f in rf]
    pitch = float(np.median(np.diff(pc))) if len(pc) > 1 else 1.0
    pairs = lf.match_floors(pc, rc, 0.5 * pitch)
    pstat = ["unusable" if not u else "missed" for u in use]
    rstat = ["extra"] * len(rb)
    for i, j in pairs:
        rint = [(rb[k][0], rb[k][2]) for k in rf[j]]
        pall = [(pb[k][0], pb[k][2]) for k in pf[i]]
        puse = [(pb[q][0], pb[q][2]) for q in pf[i] if use[q]]
        punu = [(pb[q][0], pb[q][2]) for q in pf[i] if not use[q]]
        for k in pf[i]:
            if use[k] and lf.cover_1d((pb[k][0], pb[k][2]), rint) >= 0.5:
                pstat[k] = "found"
        for k in rf[j]:
            iv = (rb[k][0], rb[k][2])
            if punu and lf.cover_1d(iv, punu) >= 0.5 and lf.cover_1d(iv, puse) < 0.5:
                rstat[k] = "dontcare"
            elif lf.cover_1d(iv, pall) >= 0.5:
                rstat[k] = "ok"
    return pb, pstat, rb, rstat, len(pf), len(rf)


def panel(t, G, m, H=430):
    render = [x["box_xyxy"] for x in json.loads((HERE / "eval_layout_dets" / m / f"{G['stem']}.json").read_text())["windows"]]
    pb, pstat, rb, rstat, nfp, nfr = classify(G["windows"], render)
    im = Image.open(ev.TESTSET / "photos" / f"{G['stem']}.jpg").convert("RGB")
    d = ImageDraw.Draw(im)
    lw = max(3, im.width // 280)
    pc = {"found": (0, 210, 80), "missed": (255, 140, 0), "unusable": (190, 190, 190)}
    for b, s in zip(pb, pstat):
        d.rectangle(b, outline=pc[s], width=lw + 1)
    rc = {"ok": (110, 240, 255), "extra": (255, 30, 30), "dontcare": (255, 255, 255)}
    for b, s in zip(rb, rstat):
        d.rectangle(b, outline=rc[s], width=lw - 1 if s != "extra" else lw + 1)
    rimg = Image.open(RENDERS[m](G["stem"])).convert("RGB")
    a = rimg.resize((max(1, int(rimg.width * H / rimg.height)), H))
    b = im.resize((max(1, int(im.width * H / im.height)), H))
    nu = sum(s != "unusable" for s in pstat)
    found = sum(s == "found" for s in pstat)
    counted = sum(s in ("ok", "extra") for s in rstat)
    ok = sum(s == "ok" for s in rstat)
    pan = Image.new("RGB", (a.width + b.width + 8, H + 52), "white")
    pan.paste(a, (0, 52)); pan.paste(b, (a.width + 8, 52))
    pd = ImageDraw.Draw(pan)
    pd.text((4, 4), NAMES[m], fill="black", font=F(17))
    pd.text((4, 27), f"recall {found}/{nu} = {found / max(1, nu):.2f}   precision {ok}/{counted} = {ok / max(1, counted):.2f}   floors photo {nfp} / render {nfr}",
            fill=(60, 60, 60), font=F(14))
    return pan


def main():
    gt = ev.load_gt(ev.TESTSET / "labels")
    tids = json.loads((HERE / "eval_protocol_grid_iou01" / "eval_astra_protocol.json").read_text())["tids"]
    out = HERE / "eval_layout_viz"
    out.mkdir(exist_ok=True)
    legend = ("photo windows: green found, orange missed, grey unusable (don't care)  |  "
              "render windows on the photo: cyan on a photo window, red extra, white on an unusable window")
    sheets = []
    for t in tids:
        ps = [panel(t, gt[t], m) for m in ("astra", "ours_full")]
        W = ps[0].width + ps[1].width + 30
        sh = Image.new("RGB", (W, max(p.height for p in ps) + 34), "white")
        ImageDraw.Draw(sh).text((4, 6), f"{t}   {legend}", fill="black", font=F(14))
        sh.paste(ps[0], (0, 34)); sh.paste(ps[1], (ps[0].width + 30, 34))
        sh.save(out / f"{t}.jpg", quality=85)
        sheets.append(sh)
    for k in range(0, len(sheets), 6):
        grp = sheets[k:k + 6]
        W = max(s.width for s in grp)
        pg = Image.new("RGB", (W, sum(s.height + 16 for s in grp)), "white")
        y = 0
        for s in grp:
            pg.paste(s, (0, y)); y += s.height + 16
        pg.save(out / f"page_{k // 6 + 1}.jpg", quality=80)
    print(out, len(sheets))


if __name__ == "__main__":
    main()
