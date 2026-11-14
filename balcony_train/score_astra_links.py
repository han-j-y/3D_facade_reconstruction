"""Element-level balcony attributes of Astra (macro F1, the paper's rule), from hand links.

Input: ``gt_links.jsonl`` written by ``astra_link_ui.py`` (one row per facade, pairs
[test100 GT crop, Astra crop]). Truth = the test100 hand label of the GT crop
(``runs/test100/crops`` + ``add5_crops``), prediction = the hand label of the Astra
crop (``test_Astra Result/Cropped Balcony/labels.jsonl``), i.e. what Astra's model shows.

Scoring is ``head_metrics(..., conditional=True)`` of
``balcony_pipeline/score_test100_types_final.py`` (the paper's Element-level Balcony
columns): kind and structure on every scored crop, material only where truth and
prediction are both open_work, floor only where both are projecting; macro F1 over the
classes of each head, plus accuracy and n.

One row per GT crop, as for ours (each crop scored once). A GT crop linked to several
Astra crops (Astra split one balcony) takes the label of the Astra crop with the largest
box area; ``--multi all`` scores every pair instead.

``--ours-json`` (output of ``score_test100_types_final.py``; default the v5 run behind the
paper's numbers) also scores ours on the same GT crops: the ``final`` label of the row with
the same crop name (split pieces: the largest), else the final label of our unit with IoU >=
0.5 with the GT crop box; GT crops with neither are left out for ours. The report then has
three blocks: Astra on its linked crops, ours on the crops among them where ours has a
balcony, and Astra on that same set.

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/score_astra_links.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "balcony_train"))

from astra_link_ui import ASTRA, GT_CROP_DIRS, load_crops  # noqa: E402
from balcony_pipeline.score_test100_types_final import (  # noqa: E402
    HEADS, TITLES, head_metrics, iou, labels_from_pred, truth_from_record)


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def summary(title: str, rows: list[dict], stage: str) -> tuple[list[str], dict]:
    m = {h: head_metrics(rows, stage, h, conditional=True) for h in HEADS}
    out = ["", f"## {title} ({len(rows)} crops)", "", "| head | macro F1 | accuracy | n |", "|---|---|---|---|"]
    for h in HEADS:
        out.append(f"| {TITLES[h]} | {m[h]['macro_f1']:.3f} | {m[h]['acc']:.3f} | {m[h]['n']} |")
    out.append(f"| **mean of 4** | {np.mean([m[h]['macro_f1'] for h in HEADS]):.3f} | "
               f"{np.mean([m[h]['acc'] for h in HEADS]):.3f} | |")
    return out, m


def confusion(title: str, m: dict) -> list[str]:
    out = ["", f"# Confusion: {title}"]
    for h in HEADS:
        cols = m[h]["cols"]
        pad = " | " if h == "material" else ""
        out += ["", f"## {TITLES[h]}", "", "| truth \\ pred | " + " | ".join(cols) + " | recall |",
                "|---" * (len(cols) + 2) + "|"]
        for name, row, pc in zip(HEADS[h], m[h]["confusion"], m[h]["per_class"]):
            out.append(f"| {name} | " + " | ".join(map(str, row)) + f" | {pc['recall']:.3f} |")
        out.append("| precision | " + " | ".join(f"{pc['precision']:.3f}" for pc in m[h]["per_class"]) + pad + " | |")
        out.append("| F1 | " + " | ".join(f"{pc['f1']:.3f}" for pc in m[h]["per_class"]) + pad + " | |")
    return out


def attach_ours(rows: list[dict], gt: dict, ours_json: Path, unit_fallback: bool = True) -> dict:
    od = json.loads(ours_json.read_text(encoding="utf-8"))
    by_name = defaultdict(list)
    for r in od["rows"]:
        by_name[re.sub(r"_s\d+$", "", r["crop_name"])].append(r)
    src = {"name": 0, "unit_iou": 0, "none": 0}
    for r in rows:
        stem = Path(r["gt"]).stem
        cands = by_name.get(stem)
        if cands:
            r["ours"] = max(cands, key=lambda x: area(x["box"]))["final"]
            src["name"] += 1
            continue
        if not unit_fallback:
            src["none"] += 1
            continue
        box = gt[r["gt"]]["box"]
        units = od["photos"].get(re.sub(r"_pbal_\d+$", "", stem), {}).get("units", [])
        best = max(units, key=lambda u: iou(u["box"], box), default=None)
        if best is not None and iou(best["box"], box) >= 0.5:
            r["ours"] = best["final"]
            src["unit_iou"] += 1
        else:
            src["none"] += 1
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", type=Path, default=ASTRA / "Cropped Balcony/gt_links.jsonl")
    ap.add_argument("--multi", choices=("largest", "all"), default="largest")
    ap.add_argument("--ours-json", type=Path,
                    default=ROOT / "runs/test100/eval_test100_final_facade_x0005model_20261110.json")
    ap.add_argument("--no-ours", action="store_true")
    ap.add_argument("--gt-iou", type=float, default=0.0,
                    help="keep only GT crops with IoU >= this with a hand GT box (balcony_gt_boxes.jsonl), and "
                         "ours only by crop name (rows already need IoU >= 0.5 with a GT box); 0 = off")
    ap.add_argument("--out", type=Path, default=ASTRA / "Cropped Balcony/astra_attr_score.md")
    ap.add_argument("--snapshot", type=Path, default=None,
                    help="read every input from a frozen copy (eval_astra_compare/balcony_attributes/data) "
                         "instead of runs/ and test_Astra Result/; overrides --links and --ours-json")
    a = ap.parse_args()

    gt_dirs, astra_dir = GT_CROP_DIRS, ASTRA / "Cropped Balcony"
    gt_boxes_path = ROOT / "runs/test100/balcony_gt_boxes.jsonl"
    if a.snapshot:
        d = a.snapshot
        gt_dirs, astra_dir, gt_boxes_path = [d / "gt_crops", d / "gt_add5_crops"], d / "astra_crops", d / "balcony_gt_boxes.jsonl"
        a.links, a.ours_json = d / "gt_links.jsonl", d / "ours_final_v5_20261110.json"

    gt = {c["id"]: c for v in load_crops(gt_dirs).values() for c in v}
    ast = {c["id"]: c for v in load_crops([astra_dir]).values() for c in v}
    links = [json.loads(l) for l in a.links.read_text(encoding="utf-8").splitlines() if l.strip()]

    by_gt = defaultdict(list)
    n_pairs = n_bad = 0
    facades = set()
    for r in links:
        for g, p in r["links"]:
            if g not in gt or p not in ast:
                n_bad += 1
                continue
            n_pairs += 1
            by_gt[g].append(p)
            facades.add(r["facade"])

    gt_boxes = {json.loads(l)["stem"]: json.loads(l)["boxes"]
                for l in gt_boxes_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    n_low = 0
    rows = []
    for g, ps in by_gt.items():
        if a.gt_iou > 0:
            stem = re.sub(r"_pbal_\d+$", "", Path(g).stem)
            if max((iou(gt[g]["box"], b) for b in gt_boxes.get(stem, [])), default=0.0) < a.gt_iou:
                n_low += 1
                continue
        if a.multi == "largest":
            ps = [max(ps, key=lambda p: area(ast[p]["box"]))]
        for p in ps:
            rows.append({"truth": truth_from_record(gt[g]["label"]),
                         "astra": labels_from_pred(ast[p]["label"]), "gt": g, "pred": p})

    multi = sum(len(v) > 1 for v in by_gt.values())
    out = ["# Balcony attributes vs test100 hand labels (conditional macro F1)", "",
           f"Links: `{a.links}`; facades with links {len(facades)}, pairs {n_pairs} (dropped {n_bad} with unknown "
           f"crop ids), GT crops linked {len(by_gt)} ({multi} linked to several Astra crops, rule `{a.multi}`)."]
    blocks = [("Astra, all linked GT crops", rows, "astra")]
    if not a.no_ours:
        if a.gt_iou > 0:
            out.append(f"GT crops with IoU < {a.gt_iou} with every hand GT box left out: {n_low}; ours by crop name only.")
        src = attach_ours(rows, gt, a.ours_json, unit_fallback=a.gt_iou <= 0)
        both = [r for r in rows if "ours" in r]
        out.append(f"Ours: `{a.ours_json.name}` (`final` = voted type label); by crop name {src['name']}, "
                   f"by unit IoU >= 0.5 {src['unit_iou']}, no ours balcony there {src['none']}.")
        blocks += [("Ours, same GT crops", both, "ours"), ("Astra, same crops as ours", both, "astra")]

    mets = []
    for title, rs, stage in blocks:
        lines, m = summary(title, rs, stage)
        out += lines
        mets.append((title, m))
    for title, m in mets:
        out += confusion(title, m)

    txt = "\n".join(out) + "\n"
    a.out.write_text(txt, encoding="utf-8")
    print(txt)
    a.out.with_suffix(".rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
