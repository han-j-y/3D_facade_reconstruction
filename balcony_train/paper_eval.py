"""Paper evaluation on the cmp_x eval photos: every stage plus the compiled facade.

Ground truth is the CMP annotation next to each photo (``<stem>.xml`` boxes and
the ``<stem>.png`` label map; 3 window, 4 door, 7 balcony) and
``runs/enclosed_grid/labels_eval.jsonl`` for enclosure cells.

Predictions per photo (``--run`` dirs):

* raw      SAM3 window boxes (``runs/batch/<stem>/index_raw.json``)
* units    DSL window / door instances (``facade_dsl.json``)
* compiled windows and doors the compiler places (``planned_windows`` on
           ``facade_compile.json``), projected back onto the photo
* balcony  ``layout.balconies`` of ``facade_dsl_with_balconies.json``

All counts are pooled over the photos (TP / FP / FN summed, then P, R, F1,
IoU). 95% intervals come from a photo-level bootstrap.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/paper_eval.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts", ROOT / "balcony_pipeline"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from render_facade import planned_windows, total_grid_size  # noqa: E402
from snap import _bands_from_windows_dsl  # noqa: E402

# CMP xml and label maps live next to these photos. The 133-photo test does not have them.
CMP_PHOTO_DIR = ROOT / "runs" / "balcony_clf" / "crops_ext_eval" / "Before Crop"
LABELS = ROOT / "runs" / "enclosed_grid" / "labels_eval.jsonl"
BATCH = ROOT / "runs" / "batch"
PAPER = ROOT / "runs" / "paper_eval"
OUT = PAPER / "results"
WINDOW, DOOR, BALCONY = 3, 4, 7
N_BOOT = 2000

Box = tuple[float, float, float, float]


# ---------------------------------------------------------------- ground truth


def load_labels() -> dict[str, dict[str, Any]]:
    out = {}
    for line in LABELS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["stem"]] = rec
    return out


def xml_boxes(stem: str, size: tuple[int, int]) -> dict[int, list[Box]]:
    """CMP objects as pixel boxes, by label id."""
    text = (CMP_PHOTO_DIR / f"{stem}.xml").read_bytes().decode("utf-8", "ignore")
    w, h = size
    out: dict[int, list[Box]] = {}
    for obj in re.findall(r"<object>(.*?)</object>", text, flags=re.S):
        xs = [float(v) for v in re.findall(r"<x>\s*([-\d.eE]+)\s*</x>", obj)]
        ys = [float(v) for v in re.findall(r"<y>\s*([-\d.eE]+)\s*</y>", obj)]
        lab = re.search(r"<label>\s*(\d+)\s*</label>", obj)
        if len(xs) < 2 or len(ys) < 2 or not lab:
            continue
        # CMP stores rows under <x> and columns under <y> (checked against the label map).
        box = (min(ys) * w, min(xs) * h, max(ys) * w, max(xs) * h)
        out.setdefault(int(lab.group(1)), []).append(box)
    return out


def label_map(stem: str) -> np.ndarray:
    return np.array(Image.open(CMP_PHOTO_DIR / f"{stem}.png"))


# ---------------------------------------------------------------- predictions


def raw_windows(stem: str) -> list[Box]:
    d = json.loads((BATCH / stem / "index_raw.json").read_text(encoding="utf-8"))
    return [tuple(map(float, w["box_xyxy"])) for w in d.get("windows", [])]


def unit_boxes(dsl: dict) -> dict[str, list[Box]]:
    out: dict[str, list[Box]] = {"window": [], "door": []}
    for inst in dsl.get("instances", []):
        kind = "door" if inst.get("kind") == "door" else "window"
        out[kind].append(tuple(map(float, inst["box_xyxy"])))
    return out


def compiled_boxes(compile_spec: dict) -> dict[str, list[Box]]:
    """Placed windows / doors in world XZ, mapped onto the photo through ``image_extent``."""
    total_w, total_h = total_grid_size(compile_spec["grid"])
    ext = compile_spec["image_extent"]
    sx = (ext["x1"] - ext["x0"]) / total_w
    sy = (ext["y1"] - ext["y0"]) / total_h
    out: dict[str, list[Box]] = {"window": [], "door": []}
    for w in planned_windows(compile_spec):
        # The compiler mirrors world +X onto the photo's left.
        u0 = ext["x0"] + (total_w / 2.0 - w["x1"]) * sx
        u1 = ext["x0"] + (total_w / 2.0 - w["x0"]) * sx
        v0 = ext["y1"] - w["z1"] * sy
        v1 = ext["y1"] - w["z0"] * sy
        out["door" if w.get("kind") == "door" else "window"].append((u0, v0, u1, v1))
    return out


def balcony_units(dsl: dict) -> tuple[list[Box], list[Box], set[tuple[int, int]]]:
    """Open balcony boxes, enclosed unit boxes, enclosed floor x bay cells."""
    opened, enclosed, cells = [], [], set()
    for b in dsl.get("layout", {}).get("balconies", []):
        box = tuple(map(float, b["box_xyxy"]))
        if b.get("type") == "balc_enclosed":
            enclosed.append(box)
            for bay in b.get("bays") or [b.get("bay")]:
                cells.add((int(b["floor"]), int(bay)))
        else:
            opened.append(box)
    return opened, enclosed, cells


# ---------------------------------------------------------------- metrics


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gt: list[Box], pred: list[Box], thr: float) -> dict[str, float]:
    """Greedy one-to-one matching by IoU."""
    pairs = sorted(
        ((iou(g, p), i, j) for i, g in enumerate(gt) for j, p in enumerate(pred)),
        reverse=True,
    )
    used_g, used_p, ious = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        ious.append(v)
    return {"tp": len(ious), "fp": len(pred) - len(ious), "fn": len(gt) - len(ious), "iou_sum": sum(ious)}


def rates(c: dict[str, float]) -> dict[str, float]:
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    out = {"P": p, "R": r, "F1": 2 * p * r / max(p + r, 1e-12), "IoU": tp / max(tp + fp + fn, 1)}
    if "iou_sum" in c:
        out["mIoU_matched"] = c["iou_sum"] / max(tp, 1)
    return out


def pooled(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys() if rows else ()
    return {k: float(sum(r[k] for r in rows)) for k in keys}


def boot_ci(rows: list[dict[str, float]], key: str, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(rows)
    vals = []
    for _ in range(N_BOOT):
        pick = [rows[i] for i in rng.integers(0, n, n)]
        vals.append(rates(pooled(pick))[key])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def raster(boxes: list[Box], shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    h, w = shape
    for x0, y0, x1, y1 in boxes:
        a, b = int(max(0, round(x0))), int(min(w, round(x1)))
        c, d = int(max(0, round(y0))), int(min(h, round(y1)))
        if b > a and d > c:
            m[c:d, a:b] = True
    return m


def pixel_counts(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "tp": float(np.sum(gt & pred)),
        "fp": float(np.sum(~gt & pred)),
        "fn": float(np.sum(gt & ~pred)),
    }


def occupancy(dsl: dict, gt_open: list[Box]) -> dict[str, float]:
    """Floor x bay cells holding an opening: CMP window / door centers vs DSL placement."""
    floor_y, bay_x = _bands_from_windows_dsl(dsl)
    floors = sorted(floor_y)
    bays = sorted(bay_x)
    gt_cells = set()
    for x0, y0, x1, y1 in gt_open:
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        f = next((k for k in floors if floor_y[k][0] <= cy < floor_y[k][1]), None)
        b = next((k for k in bays if bay_x[k][0] <= cx < bay_x[k][1]), None)
        if f is not None and b is not None:
            gt_cells.add((f, b))
    layout = dsl.get("layout", {})
    pred_cells = set()
    spans = layout.get("placement_spans") or []
    for r, row in enumerate(layout.get("placement") or []):
        for c, tok in enumerate(row):
            if tok is None:
                continue
            span = 1
            if r < len(spans) and c < len(spans[r]) and spans[r][c]:
                span = max(1, int(spans[r][c]))
            for k in range(span):
                pred_cells.add((r, c + k))
    n = len(floors) * len(bays)
    tp = len(gt_cells & pred_cells)
    return {"tp": tp, "fp": len(pred_cells - gt_cells), "fn": len(gt_cells - pred_cells), "cells": n}


def count_bands(spans: list[tuple[float, float]]) -> int:
    """Groups of boxes whose intervals overlap along one axis (chains count as one band)."""
    if not spans:
        return 0
    spans = sorted(spans)
    n, hi = 1, spans[0][1]
    for lo, up in spans[1:]:
        if lo > hi:
            n += 1
            hi = up
        else:
            hi = max(hi, up)
    return n


def vote_consistency(dsl: dict) -> tuple[int, int]:
    """Window instances whose own structure differs from their type's voted structure."""
    diff = total = 0
    for t in dsl.get("window_types", []):
        vote = t.get("structure_vote") or {}
        winner = vote.get("winner_key")
        for m in t.get("structure_members", []):
            total += 1
            diff += int(m.get("structure_key") != winner)
    return diff, total


def enclosure_ablation() -> list[tuple[str, dict[str, list[dict]]]]:
    """Per-photo cell counts for each enclosure method on the eval set (cached eval probabilities)."""
    from balcony_train.cell_clf_experiments import DEFAULT_CELLS
    from balcony_train.threshold_f1_search import DEFAULT_OUT as TF_OUT, Params, load_or_build
    from balcony_train.enclosed_grid_store import DEFAULT_DSL_DIR

    cache = TF_OUT / "probs.npz"
    if cache.is_file():
        cached = int(np.load(cache)["ev_c"].shape[0])
        live = int(np.load(DEFAULT_CELLS)["X"].shape[0])
        if cached != live:
            print(f"skip enclosure ablation: {cache} has {cached} cells, test features have {live}")
            return []
    _sets, _folds, ev = load_or_build(TF_OUT, DEFAULT_DSL_DIR)
    data = ev.data

    def op1(name: str, gate: float | None = None) -> dict[str, set]:
        col = data.f(name)
        out = {}
        for s in ev.stems:
            if gate is not None and ev.gmax[s] < gate:
                out[s] = set()
                continue
            out[s] = {data.cells[i] for i in data.idx[s] if col[i] > 0}
        return out

    methods = [
        ("OP1: SAM3 seeds + depth (stage 5)", op1("in_cells_5")),
        ("OP1 + symmetry repair (stage 6)", op1("in_cells_6")),
        ("Gate >= 0.2, then OP1 cells", op1("in_cells_6", gate=0.2)),
        ("OP3 classifier rectangle", ev.preds(Params(gate=0.0, rules=False))),
        ("OP3 + 4 cell rules", ev.preds(Params(gate=0.0))),
        ("OP3 + 4 rules + gate 0.10 (ours)", ev.preds(Params())),
    ]
    out = []
    for name, preds in methods:
        rows = []
        for s in ev.stems:
            gt, pr = ev.gts[s], preds[s]
            rows.append({"tp": len(gt & pr), "fp": len(pr - gt), "fn": len(gt - pr), "exact": int(gt == pr)})
        out.append((name, rows))
    return out


# ---------------------------------------------------------------- run


def table(rows: list[tuple[str, dict, tuple[float, float] | None]], counts: bool = True) -> list[str]:
    head = "| Method | P | R | F1 | IoU | 95% CI (F1) |" + (" TP | FP | FN |" if counts else "")
    sep = "|---" * (6 + (3 if counts else 0)) + "|"
    lines = [head, sep]
    for name, c, ci in rows:
        r = rates(c)
        ci_s = f"{ci[0]:.3f}–{ci[1]:.3f}" if ci else "-"
        line = f"| {name} | {r['P']:.3f} | {r['R']:.3f} | {r['F1']:.3f} | {r['IoU']:.3f} | {ci_s} |"
        if counts:
            line += f" {c['tp']:.0f} | {c['fp']:.0f} | {c['fn']:.0f} |"
        lines.append(line)
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper-dir", type=Path, default=PAPER)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument(
        "--ablation",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="window-track variant: DIR/<stem>/facade_dsl.json and DIR/<stem>/blender/facade_compile.json",
    )
    args = ap.parse_args()
    labels = load_labels()
    stems = sorted(labels)
    variants = {"full": (BATCH, args.paper_dir)}
    for spec in args.ablation:
        name, d = spec.split("=", 1)
        variants[name] = (Path(d), Path(d))

    per: dict[str, list[dict]] = {}
    missing: list[str] = []
    extra: dict[str, Any] = {"floors_ok": 0, "bays_ok": 0, "floors_err": [], "bays_err": [], "vote_diff": 0,
                             "vote_total": 0, "n_win_inst": 0, "n_win_types": 0, "n_bal_inst": 0, "n_bal_types": 0,
                             "dsl_bytes": [], "n_photos": 0}

    def add(key: str, row: dict) -> None:
        per.setdefault(key, []).append(row)

    for s in stems:
        photo = Image.open(CMP_PHOTO_DIR / f"{s}.jpg")
        size = photo.size
        gt = xml_boxes(s, size)
        lm = label_map(s)
        gwin, gdoor, gbal = gt.get(WINDOW, []), gt.get(DOOR, []), gt.get(BALCONY, [])
        dsl = json.loads((BATCH / s / "facade_dsl.json").read_text(encoding="utf-8"))
        comp_path = args.paper_dir / s / "blender" / "facade_compile.json"
        bal_path = args.paper_dir / s / "balcony" / "facade_dsl_with_balconies.json"
        if not comp_path.is_file() or not bal_path.is_file():
            missing.append(s)
            continue
        comp = json.loads(comp_path.read_text(encoding="utf-8"))
        full = json.loads(bal_path.read_text(encoding="utf-8"))
        extra["n_photos"] += 1

        raw = raw_windows(s)
        units = unit_boxes(dsl)
        compiled = compiled_boxes(comp)
        for thr in (0.5, 0.3):
            add(f"win@{thr}/raw", match(gwin, raw, thr))
            add(f"win@{thr}/units", match(gwin, units["window"], thr))
            add(f"win@{thr}/compiled", match(gwin, compiled["window"], thr))
            add(f"door@{thr}/units", match(gdoor, units["door"], thr))
            add(f"door@{thr}/compiled", match(gdoor, compiled["door"], thr))

        shape = lm.shape[:2]
        gt_win_px = lm == WINDOW
        gt_door_px = lm == DOOR
        gt_bal_px = lm == BALCONY
        add("px_win/raw", pixel_counts(gt_win_px, raster(raw, shape)))
        add("px_win/units", pixel_counts(gt_win_px, raster(units["window"], shape)))
        add("px_win/compiled", pixel_counts(gt_win_px, raster(compiled["window"], shape)))
        add("px_door/units", pixel_counts(gt_door_px, raster(units["door"], shape)))
        add("px_door/compiled", pixel_counts(gt_door_px, raster(compiled["door"], shape)))

        add("cells", occupancy(dsl, gwin + gdoor))
        gt_floors = count_bands([(b[1], b[3]) for b in gwin])
        gt_bays = count_bands([(b[0], b[2]) for b in gwin])
        rows = [r for r in dsl["layout"]["placement"] if any(t is not None for t in r)]
        cols = {c for r in dsl["layout"]["placement"] for c, t in enumerate(r) if t is not None}
        extra["floors_ok"] += int(len(rows) == gt_floors)
        extra["bays_ok"] += int(len(cols) == gt_bays)
        extra["floors_err"].append(abs(len(rows) - gt_floors))
        extra["bays_err"].append(abs(len(cols) - gt_bays))

        opened, enclosed, enc_cells = balcony_units(full)
        for thr in (0.5, 0.3):
            add(f"bal@{thr}/all", match(gbal, opened + enclosed, thr))
            add(f"bal@{thr}/open", match(gbal, opened, thr))
        add("px_bal/all", pixel_counts(gt_bal_px, raster(opened + enclosed, shape)))
        gt_cells = {tuple(c) for c in labels[s].get("cells", [])}
        add("enc_cells", {"tp": len(gt_cells & enc_cells), "fp": len(enc_cells - gt_cells),
                          "fn": len(gt_cells - enc_cells)})

        add("types/full", {"tp": len(dsl.get("window_types", [])), "fp": 0, "fn": 0,
                           "inst": sum(i.get("kind") != "door" for i in dsl.get("instances", []))})
        d, t = vote_consistency(dsl)
        extra["vote_diff"] += d
        extra["vote_total"] += t
        extra["n_win_inst"] += len(dsl.get("instances", []))
        extra["n_win_types"] += len(dsl.get("window_types", [])) + len(dsl.get("door_types", []))
        bal_types = [b for b in full.get("balcony_types", [])]
        extra["n_bal_inst"] += sum(int(b.get("n_instances", 0)) for b in bal_types)
        extra["n_bal_types"] += len(bal_types)
        lib = {k: full.get(k) for k in ("window_types", "door_types", "balcony_types")}
        for key in ("window_types", "balcony_types"):
            lib[key] = [{"name": x.get("name"), "structure_ir": x.get("structure_ir")} for x in lib[key] or []]
        program = {"layout": {k: full["layout"].get(k) for k in ("floors", "bays", "placement", "balconies")}, **lib}
        extra["dsl_bytes"].append(len(json.dumps(program, separators=(",", ":"))))

        for name, (dsl_dir, comp_dir) in variants.items():
            if name == "full":
                continue
            vd = dsl_dir / s / "facade_dsl.json"
            vc = comp_dir / s / "blender" / "facade_compile.json"
            if not vd.is_file() or not vc.is_file():
                continue
            vdsl = json.loads(vd.read_text(encoding="utf-8"))
            vcomp = compiled_boxes(json.loads(vc.read_text(encoding="utf-8")))
            add(f"abl/{name}/win@0.5", match(gwin, vcomp["window"], 0.5))
            add(f"abl/{name}/px_win", pixel_counts(gt_win_px, raster(vcomp["window"], shape)))
            add(f"abl/{name}/cells", occupancy(vdsl, gwin + gdoor))
            vd_, vt_ = vote_consistency(vdsl)
            add(f"abl/{name}/types", {"tp": len(vdsl.get("window_types", [])), "fp": 0, "fn": 0,
                                       "inst": sum(i.get("kind") != "door" for i in vdsl.get("instances", []))})

    args.out.mkdir(parents=True, exist_ok=True)
    n = extra["n_photos"]
    L: list[str] = [f"# Quantitative results (cmp_x eval, {n} photos)", ""]
    if missing:
        L += [f"Missing pipeline outputs, left out: {', '.join(missing)}", ""]

    def section(title: str, keys: list[tuple[str, str]], counts: bool = True) -> None:
        rows = [(lbl, pooled(per[k]), boot_ci(per[k], "F1")) for lbl, k in keys if k in per]
        L.extend([f"## {title}", ""] + table(rows, counts) + [""])

    section("Window detection vs CMP windows (IoU >= 0.5)", [
        ("SAM3 raw boxes", "win@0.5/raw"), ("DSL units (merged)", "win@0.5/units"),
        ("Compiled 3D layout (projected)", "win@0.5/compiled")])
    section("Window detection (IoU >= 0.3)", [
        ("SAM3 raw boxes", "win@0.3/raw"), ("DSL units", "win@0.3/units"), ("Compiled 3D layout", "win@0.3/compiled")])
    section("Door detection (IoU >= 0.5 / 0.3)", [
        ("DSL units @0.5", "door@0.5/units"), ("Compiled @0.5", "door@0.5/compiled"),
        ("DSL units @0.3", "door@0.3/units"), ("Compiled @0.3", "door@0.3/compiled")])
    section("Pixel-level vs CMP label map", [
        ("window: SAM3 raw", "px_win/raw"), ("window: DSL units", "px_win/units"),
        ("window: compiled 3D", "px_win/compiled"), ("door: DSL units", "px_door/units"),
        ("door: compiled 3D", "px_door/compiled"), ("balcony: DSL (open+enclosed)", "px_bal/all")], counts=False)
    section("Floor x bay grid: cells holding a window or door", [("DSL placement", "cells")])
    L += [
        f"Floor count exact: {extra['floors_ok']}/{n} (MAE {np.mean(extra['floors_err']):.2f}); "
        f"bay count exact: {extra['bays_ok']}/{n} (MAE {np.mean(extra['bays_err']):.2f}). "
        "CMP counts: CMP window boxes grouped by overlapping extents along each axis.",
        "",
    ]
    section("Balcony detection vs CMP balconies", [
        ("open + enclosed @0.5", "bal@0.5/all"), ("open only @0.5", "bal@0.5/open"),
        ("open + enclosed @0.3", "bal@0.3/all"), ("open only @0.3", "bal@0.3/open")])
    section("Enclosure cells (pipeline output)", [("cell_clf + 4 rules + gate 0.10", "enc_cells")])

    ablation = enclosure_ablation()
    if not ablation:
        L += ["## Enclosure ablation (eval cells, pooled)", "", "Skipped: the probability cache does not match the test features.", ""]
    else:
        rows, exact = [], []
        for name, prs in ablation:
            rows.append((name, pooled(prs), boot_ci(prs, "F1")))
            exact.append(f"{name}: exact {sum(r['exact'] for r in prs)}/{len(prs)}")
        L += ["## Enclosure ablation (eval cells, pooled)", ""] + table(rows) + [""] + exact + [""]

    abl = sorted({k.split("/")[1] for k in per if k.startswith("abl/")})
    if abl:
        rows = [("full (runs/batch + paper_eval)", pooled(per["win@0.5/compiled"]), boot_ci(per["win@0.5/compiled"], "F1"))]
        rows += [(f"{a}", pooled(per[f"abl/{a}/win@0.5"]), boot_ci(per[f"abl/{a}/win@0.5"], "F1")) for a in abl]
        L += ["## Ablation: compiled windows @0.5", ""] + table(rows) + [""]
        rows = [("full", pooled(per["px_win/compiled"]), None)]
        rows += [(a, pooled(per[f"abl/{a}/px_win"]), None) for a in abl]
        L += ["## Ablation: window pixels (compiled)", ""] + table(rows, counts=False) + [""]
        rows = [("full", pooled(per["cells"]), boot_ci(per["cells"], "F1"))]
        rows += [(a, pooled(per[f"abl/{a}/cells"]), boot_ci(per[f"abl/{a}/cells"], "F1")) for a in abl]
        L += ["## Ablation: grid cells", ""] + table(rows) + [""]
        t = pooled(per["types/full"])
        L.append(f"full: {t['inst']:.0f} window instances in {t['tp']:.0f} window types")
        for a in abl:
            t = pooled(per[f"abl/{a}/types"])
            L.append(f"{a}: {t['inst']:.0f} window instances in {t['tp']:.0f} window types")
        L.append("")

    vd, vt = extra["vote_diff"], extra["vote_total"]
    L += [
        "## Repetition and compactness",
        "",
        f"- Window instances {extra['n_win_inst']} in {extra['n_win_types']} window/door types "
        f"({extra['n_win_inst'] / max(extra['n_win_types'], 1):.2f} instances per type).",
        f"- Balcony instances {extra['n_bal_inst']} in {extra['n_bal_types']} balcony types "
        f"({extra['n_bal_inst'] / max(extra['n_bal_types'], 1):.2f} per type).",
        f"- Before the type vote, {vd}/{vt} window instances ({100 * vd / max(vt, 1):.1f}%) had a structure "
        "different from their type's winner; the vote makes every repeated window identical.",
        f"- DSL program (layout + type library) median {np.median(extra['dsl_bytes']) / 1024:.1f} KB per facade.",
        "",
    ]
    text = "\n".join(L)
    (args.out / "quantitative.md").write_text(text + "\n", encoding="utf-8")
    raw = {k: {**pooled(v), **rates(pooled(v))} for k, v in per.items()}
    (args.out / "quantitative.json").write_text(json.dumps({"metrics": raw, "extra": extra}, indent=1), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
