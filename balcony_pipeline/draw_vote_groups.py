r"""Per photo: how the open balconies are grouped before and after the type vote.

Reads a finished ``run.py --with-balconies`` batch (``<run-dir>/<stem>/balcony/balcony_units.json``:
unit boxes, cluster type, crop). Each unit crop goes through the pipeline's railing classifier
(``infer_unit_ir``, same crops as ``run_balcony.py`` step 4), then the step-4 vote is redone with
``score_test100_types_final.vote_types``. Panel 4 shows the run's own voted labels
(``balcony_types.json``, what the DSL holds); the recomputed vote is only a check, and a type
where it differs (a tie: the run's medoid preference compares an index with ``unit_id``) is
named in the sheet header.

Four panels on the same photo, left to right:

1. Input.
2. Classification only (before vote): box color = the unit's predicted label set
   (kind / material / floor / structure). Same color = same labels.
3. Clustering: box color = type (cluster); thick box = medoid. Tag = ``T<type>`` + the unit's
   own labels; a field the vote will change is yellow.
4. After vote (DSL): box color = voted label set, in the same colors as panel 2, so a color
   change between 2 and 4 is a unit whose labels the vote changed.
5. Blender perspective render of the same run (``--render-dir``, default ``<run-dir>/oblique``),
   when the file exists.

Label codes: kind ``OW|SP|SO`` (open work, surface panel, solid), material ``me|ma|-``,
floor ``R|C``, structure ``P|J`` (projecting, juliet).

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_pipeline/draw_vote_groups.py \
        --run-dir runs/val30 --photos data/val30 --out-dir runs/val30/_vote_groups
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from draw_test100_vote_before_after import ABBR, HEADS, font  # noqa: E402
from recovery_profile import DEFAULT_PROFILE  # noqa: E402
from run_balcony import infer_unit_ir  # noqa: E402
from score_test100_types_final import labels_from_ir, vote_types  # noqa: E402

from balcony_train.model import try_load_predictor  # noqa: E402
from balcony_train.paths import DEFAULT_STRUCTURE_CKPT  # noqa: E402

PANEL_H = 900
TYPE_COLORS = [(0, 170, 255), (255, 0, 200), (255, 220, 0), (0, 230, 120), (150, 90, 255), (255, 120, 0),
               (0, 255, 255), (255, 80, 80), (170, 255, 0), (120, 160, 255)]
# label-set colors: distinct from each other, fixed order of first appearance per photo
LABEL_COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
                (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40), (255, 250, 200)]
WHITE, BLACK, YELLOW, GREY = (255, 255, 255), (0, 0, 0), (255, 230, 0), (40, 40, 40)


def combo_key(lab: dict) -> tuple:
    return tuple(lab[h] for h in HEADS)


def combo_text(lab: dict) -> str:
    return " ".join(ABBR[h].get(lab[h], str(lab[h])) for h in HEADS)


def tag(d: ImageDraw.ImageDraw, x: int, y: int, parts: list[tuple[str, tuple]], f, edge) -> None:
    w = sum(d.textlength(t + " ", font=f) for t, _ in parts)
    y = max(0, y - f.size - 6)
    d.rectangle((x, y, x + w + 4, y + f.size + 5), fill=BLACK, outline=edge, width=2)
    for t, col in parts:
        d.text((x + 2, y + 2), t, font=f, fill=col)
        x += d.textlength(t + " ", font=f)


def panel(photo: Image.Image, units: list[dict], mode: str, color_of: dict, title: str) -> Image.Image:
    s = PANEL_H / photo.height
    im = photo.resize((max(1, round(photo.width * s)), PANEL_H), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    f = font(max(12, round(PANEL_H / 60)))
    if mode != "input":
        for u in units:
            x0, y0, x1, y1 = (round(v * s) for v in u["box_xyxy"])
            if mode == "before":
                col, parts, wd = color_of[combo_key(u["unit"])], [(combo_text(u["unit"]), WHITE)], 4
            elif mode == "after":
                col, parts, wd = color_of[combo_key(u["final"])], [(combo_text(u["final"]), WHITE)], 4
                if combo_key(u["unit"]) != combo_key(u["final"]):
                    parts = [("*", YELLOW)] + [(combo_text(u["final"]), YELLOW)]
            else:  # cluster
                col = TYPE_COLORS[u["type_id"] % len(TYPE_COLORS)]
                wd = 7 if u["medoid"] else 3
                parts = [(f"T{u['type_id']}", col)] + [
                    (ABBR[h].get(u["unit"][h], str(u["unit"][h])), YELLOW if u["unit"][h] != u["final"][h] else WHITE)
                    for h in HEADS
                ]
            d.rectangle((x0, y0, x1, y1), outline=col, width=wd)
            tag(d, x0, y0, parts, f, col)
    bar = Image.new("RGB", (im.width, 40), BLACK)
    ImageDraw.Draw(bar).text((8, 8), title, font=font(21), fill=WHITE)
    out = Image.new("RGB", (im.width, im.height + bar.height), BLACK)
    out.paste(bar, (0, 0))
    out.paste(im, (0, bar.height))
    return out


def render_panel(render: Image.Image, title: str) -> Image.Image:
    s = PANEL_H / render.height
    im = render.resize((max(1, round(render.width * s)), PANEL_H), Image.LANCZOS)
    bar = Image.new("RGB", (im.width, 40), BLACK)
    ImageDraw.Draw(bar).text((8, 8), title, font=font(21), fill=WHITE)
    out = Image.new("RGB", (im.width, im.height + bar.height), BLACK)
    out.paste(bar, (0, 0))
    out.paste(im, (0, bar.height))
    return out


def legend(width: int, units: list[dict], color_of: dict) -> Image.Image:
    f = font(18)
    rows: list[list[tuple[str, tuple]]] = []
    # label-set legend with counts before / after
    items = []
    for key, col in color_of.items():
        nb = sum(combo_key(u["unit"]) == key for u in units)
        na = sum(combo_key(u["final"]) == key for u in units)
        lab = dict(zip(HEADS, key))
        items.append((col, f"{combo_text(lab)}   before {nb} -> after {na}"))
    types = sorted({u["type_id"] for u in units})
    titems = []
    for t in types:
        mem = [u for u in units if u["type_id"] == t]
        n_unique = len({combo_key(u["unit"]) for u in mem})
        titems.append((TYPE_COLORS[t % len(TYPE_COLORS)],
                       f"T{t}: {len(mem)} units, {n_unique} label sets before vote -> {combo_text(mem[0]['final'])}"))
    line_h = 26
    n_lines = 2 + len(items) + len(titems)
    im = Image.new("RGB", (width, 16 + line_h * n_lines), GREY)
    d = ImageDraw.Draw(im)
    y = 8
    d.text((10, y), "Label sets (panels 2 and 4: same color = same labels)", font=f, fill=WHITE)
    y += line_h
    for col, txt in items:
        d.rectangle((14, y + 3, 44, y + 19), fill=col)
        d.text((54, y), txt, font=f, fill=WHITE)
        y += line_h
    d.text((10, y), "Types / clusters (panel 3: color = type, thick = medoid, yellow field = changed by the vote)",
           font=f, fill=WHITE)
    y += line_h
    for col, txt in titems:
        d.rectangle((14, y + 3, 44, y + 19), outline=col, width=4)
        d.text((54, y), txt, font=f, fill=WHITE)
        y += line_h
    return im


def unit_bays(units: list[dict], bdir: Path) -> list[int]:
    """Bay per unit: ``balcony_units.json`` ``bay`` (runs after 2026-11-07), else the DSL balcony with the same box,
    else the DSL column holding the box centre."""
    if all("bay" in u for u in units):
        return [int(u["bay"]) for u in units]
    dsl = json.loads((bdir / "facade_dsl_with_balconies.json").read_text(encoding="utf-8"))
    by_box = {tuple(b["box_xyxy"]): b for b in dsl["layout"].get("balconies", []) if "box_xyxy" in b}
    cols = (dsl.get("meta") or {}).get("columns_xy") or []
    out = []
    for u in units:
        b = by_box.get(tuple(u["box_xyxy"]))
        if b is not None and "bay" in b:
            out.append(int(b["bay"]))
            continue
        cx = 0.5 * (u["box_xyxy"][0] + u["box_xyxy"][2])
        out.append(min(range(len(cols)), key=lambda i: abs(0.5 * (cols[i][0] + cols[i][1]) - cx)) if cols else 0)
    return out


def recluster(photo: Image.Image, units: list[dict], bays: list[int], device: torch.device,
              extra_args: list[str] | None = None, windows: list[dict] | None = None,
              label_probs: list[dict] | None = None) -> tuple[np.ndarray, dict]:
    """Labels and medoids with the ``run_balcony`` clustering defaults plus ``extra_args``."""
    import run_pipeline as rp
    from run_balcony import cluster_boxes, parse_args as rb_parse_args

    old_argv = sys.argv
    sys.argv = ["run_balcony", "--image", "x.jpg", "--windows-dsl", "x.json", "--out-dir", "x"] + list(extra_args or [])
    try:
        cargs = rb_parse_args()
    finally:
        sys.argv = old_argv
    clustered = cluster_boxes(photo, [u["box_xyxy"] for u in units], [int(u["floor"]) for u in units], bays,
                              cargs, device, windows=windows, label_probs=label_probs)
    labels = np.asarray(clustered["labels"], dtype=np.int32)
    return labels, rp.pick_medoids(clustered["feats"], labels)


def records(units, irs, labels, medoids, finals) -> list[dict]:
    return [{"box_xyxy": u["box_xyxy"], "type_id": int(labels[i]), "medoid": medoids.get(int(labels[i])) == i,
             "unit": labels_from_ir(irs[i]), "final": finals[i]} for i, u in enumerate(units)]


def stats(rec: list[dict]) -> dict:
    return {"types": len({r["type_id"] for r in rec}),
            "before": len({combo_key(r["unit"]) for r in rec}),
            "after": len({combo_key(r["final"]) for r in rec}),
            "changed": sum(combo_key(r["unit"]) != combo_key(r["final"]) for r in rec)}


def row(photo, rec, color_of, tag_: str, render: Image.Image | None) -> list[Image.Image]:
    s = stats(rec)
    pre = f"{tag_} " if tag_ else ""
    ps = [
        panel(photo, rec, "input", color_of, f"{pre}1 Input"),
        panel(photo, rec, "before", color_of, f"{pre}2 Classification only: {s['before']} label sets"),
        panel(photo, rec, "cluster", color_of, f"{pre}3 Clustering: {s['types']} types (own labels)"),
        panel(photo, rec, "after", color_of,
              f"{pre}4 After vote: {s['after']} label sets, {s['changed']} units changed (*)"),
    ]
    if render is not None:
        ps.append(render_panel(render, f"{pre}5 Blender model (perspective)"))
    return ps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "val30")
    ap.add_argument("--photos", type=Path, default=ROOT / "data" / "val30")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <run-dir>/_vote_groups (or _vote_groups_compare with --compare-new)")
    ap.add_argument("--ids", type=Path, default=None, help="optional {id: stem} json (default: <photos>/val30_ids.json)")
    ap.add_argument("--render-dir", type=Path, default=None,
                    help="Blender perspective renders <stem>.png for a 5th panel (default: <run-dir>/oblique if present)")
    ap.add_argument("--compare-new", action="store_true",
                    help="second row: the same units re-clustered with the current run_balcony clustering defaults "
                    "and re-voted (no Blender render); top row = the run as saved")
    ap.add_argument("--variant", action="append", default=[], metavar="NAME=ARGS",
                    help="with --compare-new: one more row per variant, re-clustered with the defaults plus these "
                    "run_balcony args, e.g. --variant \"LINKS=--cluster-links\"")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out_dir = args.out_dir or args.run_dir / ("_vote_groups_compare" if args.compare_new else "_vote_groups")
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_path = args.ids or args.photos / "val30_ids.json"
    short = {v: k for k, v in json.loads(ids_path.read_text(encoding="utf-8")).items()} if ids_path.is_file() else {}
    render_dir = args.render_dir or args.run_dir / "oblique"
    device = torch.device(args.device)

    pred = try_load_predictor(None, device, structure_ckpt=DEFAULT_STRUCTURE_CKPT)
    print(f"railing classifier {pred.ckpt_path}")

    summary = []
    for uj in sorted(args.run_dir.glob("*/balcony/balcony_units.json")):
        bdir = uj.parent
        blob = json.loads(uj.read_text(encoding="utf-8"))
        stem = blob["stem"]
        units = blob["units"]
        if not units:
            continue
        photo_p = next((p for p in (args.photos / f"{stem}.jpg", args.photos / f"{stem}.png") if p.is_file()), None)
        if photo_p is None:
            print(f"skip {stem}: no photo")
            continue
        photo = Image.open(photo_p).convert("RGB")
        types_json = json.loads((bdir / "balcony_types.json").read_text(encoding="utf-8"))
        open_types = {int(t["type_id"]): t for t in types_json if str(t.get("name", "")).startswith("balc_type_")}
        medoids = {tid: int(t["exemplar_unit"]) for tid, t in open_types.items()}

        vunits, irs, probs = [], [], []
        for u in units:
            crop = Image.open(bdir / u["asset"]).convert("RGB")
            probs.append(pred.predict_probs(crop))
            vu = {"unit_id": int(u["unit_index"]), "box_xyxy": u["box_xyxy"], "floor": u["floor"]}
            irs.append(infer_unit_ir(crop, vu, image_size=photo.size, profile_name=DEFAULT_PROFILE, predictor=pred))
            vunits.append(vu)
        labels = np.asarray([int(u["type_id"]) for u in units], dtype=np.int32)
        for tid in set(labels.tolist()) - set(medoids):  # should not happen; vote still needs a medoid
            medoids[tid] = int(np.flatnonzero(labels == tid)[0])
        voted = vote_types(vunits, labels, medoids, irs)
        finals, mismatch = [], set()
        for i in range(len(units)):
            f = labels_from_ir(voted[i])
            tid = int(labels[i])
            if tid in open_types:
                # panel 4 shows what the DSL holds; the recomputed vote is only a check
                run_final = labels_from_ir(open_types[tid]["structure_ir"])
                if run_final != f:
                    mismatch.add(tid)
                f = run_final
            finals.append(f)
        rows = [("CURRENT" if args.compare_new else "", records(units, irs, labels, medoids, finals))]

        if args.compare_new:
            bays = unit_bays(units, bdir)
            wdsl = bdir.parent / "facade_dsl.json"
            windows = json.loads(wdsl.read_text(encoding="utf-8")).get("instances") if wdsl.is_file() else None
            variants = [("NEW", [])] + [(v.split("=", 1)[0], v.split("=", 1)[1].split()) for v in args.variant]
            for name, extra in variants:
                nlabels, nmed = recluster(photo, units, bays, device, extra, windows, probs)
                jp = [p.get("structure")[1] if p.get("structure") else None for p in probs]
                fp = [dict(zip(("rectangle", "triangle", "circle"), p["floor"])) if p.get("floor") else None
                      for p in probs]
                nvoted = vote_types(vunits, nlabels, nmed, irs, jp, fp)
                rows.append((name, records(units, irs, nlabels, nmed, [labels_from_ir(v) for v in nvoted])))

        color_of: dict[tuple, tuple] = {}
        for _, rr in rows:
            for r in rr:
                for k in (combo_key(r["unit"]), combo_key(r["final"])):
                    if k not in color_of:
                        color_of[k] = LABEL_COLORS[len(color_of) % len(LABEL_COLORS)]

        render_p = render_dir / f"{stem}.png"
        render = Image.open(render_p).convert("RGB") if render_p.is_file() else None
        panel_rows = [row(photo, rr, color_of, t, render if i == 0 else None) for i, (t, rr) in enumerate(rows)]
        gap = 10
        width = max(sum(p.width for p in ps) + gap * (len(ps) - 1) for ps in panel_rows)
        leg_w = (width - 12 * (len(rows) - 1)) // len(rows)
        legs = []
        for t, rr in rows:
            lg = legend(leg_w, rr, color_of)
            if t:
                ImageDraw.Draw(lg).text((leg_w - 140, 8), t, font=font(22), fill=YELLOW)
            legs.append(lg)
        leg_h = max(lg.height for lg in legs)
        head_h, row_gap = 40, 16
        rows_h = sum(ps[0].height for ps in panel_rows) + row_gap * (len(panel_rows) - 1)
        sheet = Image.new("RGB", (width, head_h + rows_h + leg_h), (25, 25, 25))
        d = ImageDraw.Draw(sheet)
        sid = short.get(stem, "")
        s0 = stats(rows[0][1])
        head = f"{sid} {stem}   units {len(units)}   label sets before vote {s0['before']}"
        for t, rr in rows:
            s = stats(rr)
            head += f"   |  {t or 'run'}: types {s['types']}, after vote {s['after']} sets, changed {s['changed']}"
        if mismatch:
            head += f"   (CURRENT type {sorted(mismatch)}: recomputed vote differs = tie, shown as run)"
        d.text((10, 8), head, font=font(22), fill=WHITE)
        y = head_h
        for ps in panel_rows:
            x = 0
            for p in ps:
                sheet.paste(p, (x, y))
                x += p.width + gap
            y += ps[0].height + row_gap
        y -= row_gap
        x = 0
        for lg in legs:
            sheet.paste(lg, (x, y))
            x += lg.width + 12
        name = f"{sid}_{stem}.jpg" if sid else f"{stem}.jpg"
        sheet.save(out_dir / name, quality=88)

        rec_s = {"id": sid, "stem": stem, "units": len(units), "label_sets_before": s0["before"]}
        for t, rr in rows:
            s = stats(rr)
            p = (t or "run").lower()
            rec_s.update({f"{p}_types": s["types"], f"{p}_sets_after": s["after"], f"{p}_changed": s["changed"]})
        if len(rows) > 1:
            for t, rr in rows[1:]:
                rec_s[f"{t.lower()}_units_other_than_current"] = sum(
                    a["final"] != b["final"] for a, b in zip(rows[0][1], rr))
        rec_s["vote_mismatch_types"] = " ".join(map(str, sorted(mismatch)))
        summary.append(rec_s)
        print("  ".join(f"{k}={v}" for k, v in rec_s.items() if k != "stem"))

    summary.sort(key=lambda r: r["id"] or r["stem"])
    if summary:
        with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
    print(f"{len(summary)} sheets -> {out_dir}")


if __name__ == "__main__":
    main()


