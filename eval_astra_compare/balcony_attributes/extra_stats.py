"""SAM3 balcony counts (photos vs Astra front views) and the IoU-threshold check, from data/ only.

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe eval_astra_compare/balcony_attributes/extra_stats.py
writes results/extra_stats.txt.
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "balcony_train")]
from astra_link_ui import facade_id, load_crops  # noqa: E402
from balcony_pipeline.score_test100_types_final import HEADS, head_metrics, iou  # noqa: E402

D = HERE / "data"
jl = lambda p: [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
out = []

# 1. SAM3 balcony boxes, same prompt and filter on both sides
idx = list(csv.DictReader(open(D / "astra_index.csv", encoding="utf-8-sig")))
f2s = {facade_id(r["index"]): r["stem"] for r in idx}
ast_s3 = json.loads((D / "astra_sam3_stage_boxes.json").read_text())
pho_s3 = json.loads((D / "photo_sam3_stage_boxes.json").read_text())
gtb = {r["stem"]: r["boxes"] for r in jl(D / "balcony_gt_boxes.jsonl")}
out.append(f"## SAM3 balcony boxes on the {len(f2s)} test facades (photo | Astra front view)")
for st in ("raw", "sized", "filtered"):
    out.append(f"{st:9s} {sum(len(pho_s3[s][st]) for s in f2s.values()):4d} | {sum(len(ast_s3[f][st]) for f in f2s):4d}")
out.append(f"facades with >= 1 filtered box: {sum(len(pho_s3[s]['filtered']) > 0 for s in f2s.values())} | "
           f"{sum(len(ast_s3[f]['filtered']) > 0 for f in f2s)}")
out.append(f"labelled crops: {sum(len(v) for v in load_crops([D / 'gt_crops', D / 'gt_add5_crops']).values())} | "
           f"{sum(len(v) for v in load_crops([D / 'astra_crops']).values())}")
out.append(f"hand GT boxes: {sum(len(gtb.get(s, [])) for s in f2s.values())} on "
           f"{sum(len(gtb.get(s, [])) > 0 for s in f2s.values())} photos")

# 2. IoU of the scored GT crops with the hand GT boxes, and the score per threshold
gtc = {c["id"]: (s, c) for s, v in load_crops([D / "gt_crops", D / "gt_add5_crops"]).items() for c in v}
rows = json.loads((HERE / "results/score_all.rows.json").read_text())
od = json.loads((D / "ours_final_v5_20261110.json").read_text())
byname = {}
for r in od["rows"]:
    byname.setdefault(re.sub(r"_s\d+$", "", r["crop_name"]), []).append(r)
for r in rows:
    s, c = gtc[r["gt"]]
    r["gt_iou"] = max((iou(c["box"], b) for b in gtb.get(s, [])), default=0)
    cands = byname.get(Path(r["gt"]).stem)
    r["ours_iou"] = None
    if cands:
        ob = max(cands, key=lambda x: (x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1]))["box"]
        r["ours_iou"] = max((iou(ob, b) for b in gtb.get(s, [])), default=0)
both = [r for r in rows if "ours" in r]
v = np.array([r["gt_iou"] for r in both])
out += ["", f"## IoU of the {len(both)} crops scored for both with the hand GT boxes",
        f"median {np.median(v):.2f}, p10 {np.percentile(v, 10):.2f}; >= 0.7 {(v >= .7).sum()}, >= 0.9 {(v >= .9).sum()}",
        "", "## Mean macro F1 of 4 heads per threshold (n | ours | Astra)"]


def mean(rs, st):
    return float(np.mean([head_metrics(rs, st, h, True)["macro_f1"] for h in HEADS]))


for t in (0.0, 0.3, 0.5, 0.7, 0.8):
    rs = [r for r in both if r["gt_iou"] >= t and (r["ours_iou"] is None or r["ours_iou"] >= t)]
    out.append(f"IoU >= {t}: {len(rs)} | {mean(rs, 'ours'):.3f} | {mean(rs, 'astra'):.3f}")

txt = "\n".join(out) + "\n"
(HERE / "results/extra_stats.txt").write_text(txt, encoding="utf-8")
print(txt)
