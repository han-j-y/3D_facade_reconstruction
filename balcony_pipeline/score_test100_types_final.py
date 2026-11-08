r"""Balcony type labels on test100 as the final facade holds them (after clustering + type vote).

``balcony_train/evaluate_test100.py`` scores the railing classifier on each crop alone. The
pipeline does not write those labels: ``run_balcony.py`` clusters the units into types (DINOv2
appearance) and votes one label per type. This script reruns the pipeline's own functions on
the cached SAM3 boxes, in the pipeline's order, and scores the labels the DSL ends up with:

1. ``snap_open_units`` (filter → ``side_wall`` → ``masonry_juliet_split`` with the live railing
   classifier → merge → snap), as ``score_test100_pipeline.py``.
2. Oriel cells of the 457 model (``preds_official.json``) → ``drop_open_under_enclosed``.
3. ``cluster_boxes`` (default ``--cluster-mode appearance``) → ``pick_medoids``.
4. Per type: ``infer_unit_ir`` per unit → ``vote_cluster_ir`` (kind / material) →
   ``vote_structure`` → ``apply_floor_majority`` (juliet members do not vote), copied from
   ``run_balcony.run`` step 4 (keep the two in sync).

Scored set: every box after the split that has a hand label (``runs/test100/crops`` or
``crops_split``, same box) and IoU >= ``--gt-iou`` with a GT box, as
``balcony_train/build_test100_final_set.py``. Each crop gets the label of the unit it ends up
in. Crops whose unit the oriel step removes are counted, not scored (the facade has no open
balcony there). Split pieces of the live v4 split that were never labeled are counted too.

Three stages on the same crops:

- ``crop``: classifier on the crop's own box (per-crop, like ``evaluate_test100.py``),
- ``unit``: classifier on the snapped unit box, no vote (= ``--no-railing-vote``),
- ``final``: the voted type label written to ``facade_dsl_with_balconies.json``.

All stages use the DSL meaning: material exists only when the predicted kind is ``open_work``
(else column ``none``, counted wrong), a missing structure is ``projecting``, triangle counts
as rectangle. So ``crop`` can differ slightly from ``evaluate_test100.py``, which reads the
material and structure heads directly.

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_pipeline/score_test100_types_final.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_pipeline as rp  # noqa: E402
from enclosed import apply_floor_majority, drop_open_under_enclosed, place_enclosed_cells, vote_structure  # noqa: E402
from heuristic_ir import (  # noqa: E402
    floor_shape_from_ir,
    ir_to_tokens,
    railing_kind_from_ir,
    railing_material_from_ir,
    structure_from_ir,
)
from recovery_profile import DEFAULT_PROFILE  # noqa: E402
from run_balcony import cluster_boxes, floor_probs, infer_unit_ir, juliet_prob, snap_open_units  # noqa: E402
from structure_depth import structure_from_depth  # noqa: E402
from vote import soft_type_labels, vote_cluster_ir  # noqa: E402
from balcony_train.labels import CLASSES, LABEL_STRUCTURES, MATERIALS  # noqa: E402
from balcony_train.model import try_load_predictor  # noqa: E402
from balcony_train.paths import DEFAULT_STRUCTURE_CKPT  # noqa: E402
from balcony_train.try_da3_enclosure import resize_depth  # noqa: E402

T100 = ROOT / "runs" / "test100"
STAGES = ("crop", "unit", "final")
HEADS = {
    "kind": CLASSES,
    "material": MATERIALS,
    "floor": ("rectangle", "circle"),
    "structure": LABEL_STRUCTURES,
}
TITLES = {"kind": "Kind", "material": "Material (true open_work)", "floor": "Floor", "structure": "Structure"}


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.is_file() else []


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0.0


def _floor(s: str | None) -> str | None:
    if s is None:
        return None
    return "rectangle" if s in ("rectangle", "triangle") else s


def labels_from_pred(pred: dict) -> dict:
    """``predict_full`` output in DSL meaning."""
    kind = pred.get("kind")
    return {
        "kind": kind,
        "material": pred.get("material") if kind == "open_work" else None,
        "floor": _floor(pred.get("floor_shape") or "rectangle"),
        "structure": "juliet" if pred.get("structure") == "juliet" else "projecting",
    }


def labels_from_ir(ir: dict) -> dict:
    return {
        "kind": railing_kind_from_ir(ir),
        "material": railing_material_from_ir(ir),
        "floor": _floor(floor_shape_from_ir(ir)),
        "structure": "juliet" if structure_from_ir(ir) == "juliet" else "projecting",
    }


def truth_from_record(rec: dict) -> dict:
    kind = rec.get("kind")
    return {
        "kind": kind if kind in CLASSES else None,
        "material": rec.get("material") if kind == "open_work" and rec.get("material") in MATERIALS else None,
        "floor": _floor(rec.get("floor_shape")),
        "structure": rec.get("structure") if rec.get("structure") in LABEL_STRUCTURES else None,
    }


def soft_labels(probs: list[dict], voters_floor: list[int]) -> dict:
    """``vote.soft_type_labels`` (the pipeline's soft vote); floor rectangle without voters."""
    out = soft_type_labels(probs, voters_floor)
    return {**out, "floor": out["floor"] or "rectangle"}


def vote_types(units: list[dict], labels: np.ndarray, medoids: dict[int, int], unit_irs: list[dict],
               juliet_probs: list[float | None] | None = None,
               floor_probs: list[dict | None] | None = None,
               soft_probs: list[dict] | None = None) -> list[dict]:
    """``soft_probs`` (per unit ``RailingPredictor.predict_probs``): trial soft vote for kind / material /
    floor (``soft_labels``); structure keeps the majority + tie rule."""
    """Voted IR per unit, as ``run_balcony.run`` step 4 (vote on, floor majority on)."""
    profile = DEFAULT_PROFILE
    voted_ir: list[dict | None] = [None] * len(units)
    for tid, med_i in sorted(medoids.items()):
        idx = [i for i in range(len(units)) if int(labels[i]) == int(tid)]
        member_preds = [
            {"unit_id": int(units[i]["unit_id"]), "ir": unit_irs[i], "tokens": ir_to_tokens(unit_irs[i], profile_name=profile)}
            for i in idx
        ]
        voted = vote_cluster_ir(member_preds, prefer_unit_id=int(units[med_i]["unit_id"]), profile_name=profile)
        structure, _ = vote_structure([structure_from_ir(unit_irs[i]) for i in idx],
                                      [juliet_probs[i] for i in idx] if juliet_probs is not None else None)
        ir = voted["structure_ir"]
        if isinstance(ir, dict):
            ir = copy.deepcopy(ir)
            ir["structure"] = structure
        voters = [i for i in idx if isinstance(unit_irs[i], dict) and structure_from_ir(unit_irs[i]) != "juliet"]
        ir, _shape, _counts = apply_floor_majority(
            ir,
            [floor_shape_from_ir(unit_irs[i]) for i in voters],
            [floor_probs[i] for i in voters] if floor_probs is not None else None,
        )
        if soft_probs is not None and isinstance(ir, dict):
            sl = soft_labels([soft_probs[i] for i in idx], [k for k, i in enumerate(idx) if i in set(voters)])
            ir = copy.deepcopy(ir)
            rail = dict(ir.get("railing") or {})
            rail["kind"] = sl["kind"]
            rail["material"] = sl["material"]
            ir["railing"] = rail
            fl = dict(ir.get("floor") or {})
            fl["shape"] = sl["floor"]
            ir["floor"] = fl
        for i in idx:
            voted_ir[i] = copy.deepcopy(ir)
    return voted_ir


def conditional_skip(r: dict, stage: str, head: str) -> bool:
    """Conditional scoring (user, 2026-11-07): material only where truth and prediction are both open work,
    floor only where truth and prediction are both projecting. A wrong kind / structure is counted once, in
    its own head, and leaves the dependent head out of the denominator."""
    if head == "material":
        return r[stage]["kind"] != "open_work"
    if head == "floor":
        return r[stage]["structure"] != "projecting"
    return False


def head_metrics(rows: list[dict], stage: str, head: str, conditional: bool = True) -> dict:
    names = HEADS[head]
    cols = names + ("none",) if head == "material" else names
    c = np.zeros((len(names), len(cols)), dtype=np.int64)
    for r in rows:
        t = r["truth"][head]
        if t is None:
            continue
        if conditional and conditional_skip(r, stage, head):
            continue
        p = r[stage][head]
        c[names.index(t), cols.index(p if p in cols else "none")] += 1
    per, ps, rs, fs = [], [], [], []
    for i, nm in enumerate(names):
        tp, col, row = int(c[i, i]), int(c[:, i].sum()), int(c[i].sum())
        pr = tp / col if col else 0.0
        rc = tp / row if row else 0.0
        f = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per.append({"class": nm, "tp": tp, "pred": col, "true": row, "precision": pr, "recall": rc, "f1": f})
        ps.append(pr); rs.append(rc); fs.append(f)
    n = int(c.sum())
    k = len(names)
    return {
        "cols": list(cols), "confusion": c.tolist(), "n": n, "correct": int(np.trace(c[:, :k])),
        "acc": float(np.trace(c[:, :k]) / n) if n else 0.0,
        "per_class": per, "macro_p": sum(ps) / k, "macro_r": sum(rs) / k, "macro_f1": sum(fs) / k,
    }


def report(rows: list[dict], header: list[str], conditional: bool = True) -> tuple[str, dict]:
    out = header[:]
    metrics = {s: {h: head_metrics(rows, s, h, conditional) for h in HEADS} for s in STAGES}
    out += ["## Summary (macro F1 / accuracy, n per stage)", "",
            "| head | crop | unit (no vote) | final (voted) |", "|---|---|---|---|"]
    for h in HEADS:
        cells = [f"{metrics[s][h]['macro_f1']:.3f} / {metrics[s][h]['acc']:.3f} ({metrics[s][h]['n']})" for s in STAGES]
        out.append(f"| {TITLES[h]} | " + " | ".join(cells) + " |")
    means = [f"{np.mean([metrics[s][h]['macro_f1'] for h in HEADS]):.3f} / "
             f"{np.mean([metrics[s][h]['acc'] for h in HEADS]):.3f}" for s in STAGES]
    out.append("| **mean of 4** | " + " | ".join(means) + " |")
    out += ["", "## Vote effect (unit → final, same crops)", "", "| head | fixed by vote | broken by vote | net |", "|---|---|---|---|"]
    for h in HEADS:
        fixed = broken = 0
        for r in rows:
            t = r["truth"][h]
            if t is None:
                continue
            if conditional and (conditional_skip(r, "unit", h) or conditional_skip(r, "final", h)):
                continue
            u, f = r["unit"][h] == t, r["final"][h] == t
            fixed += (not u) and f
            broken += u and not f
        out.append(f"| {TITLES[h]} | {fixed} | {broken} | {fixed - broken:+d} |")
    out.append("")
    for s in STAGES:
        out += [f"## Stage `{s}`", "", "```"]
        for h in HEADS:
            m = metrics[s][h]
            out.append(f"{h} n={m['n']}  acc={m['acc']:.3f}   rows=true cols=pred  " + "  ".join(m["cols"]))
            for i, nm in enumerate(HEADS[h]):
                out.append(f"  {nm:14s} " + " ".join(f"{v:5d}" for v in m["confusion"][i]))
        out += ["```", ""]
        for h in HEADS:
            m = metrics[s][h]
            out += [f"**{TITLES[h]}**", "", "| class | precision | recall | F1 |", "|---|---|---|---|"]
            for pc in m["per_class"]:
                out.append(f"| {pc['class']} | {pc['tp']}/{pc['pred']} = {pc['precision']:.3f} | "
                           f"{pc['tp']}/{pc['true']} = {pc['recall']:.3f} | {pc['f1']:.3f} |")
            out += [f"| **macro** | **{m['macro_p']:.3f}** | **{m['macro_r']:.3f}** | **{m['macro_f1']:.3f}** |", ""]
    return "\n".join(out) + "\n", metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=T100 / "balcony_gt_boxes.jsonl")
    ap.add_argument("--gt-iou", type=float, default=0.5)
    ap.add_argument("--cache", type=Path, default=ROOT / "runs" / "batch" / "_rows" / "test100_sam3_stage_boxes.json")
    ap.add_argument("--oriel-preds", type=Path,
                    default=ROOT / "runs" / "oriel_grid_compare" / "final_457_test100" / "preds_official.json")
    ap.add_argument("--dsl-dir", type=Path, default=T100 / "batch_grid2")
    ap.add_argument("--depth-dir", type=Path, default=T100 / "da3_depth")
    ap.add_argument("--photos", type=Path, default=T100 / "photos")
    ap.add_argument("--railing-ckpt", type=Path, default=None, help="default: pipeline default (paths.py)")
    ap.add_argument("--cluster-mode", choices=("appearance", "box"), default="appearance")
    ap.add_argument("--cluster-k1", action=argparse.BooleanOptionalAction, default=True,
                    help="as run_balcony: allow k = 1 (--no-cluster-k1 = clustering before 2026-11-07)")
    ap.add_argument("--k1-sil-min", type=float, default=0.25)
    ap.add_argument("--k1-pair-sim-min", type=float, default=0.80)
    ap.add_argument("--cluster-width-ratio", type=float, default=2.0, help="as run_balcony; 0 = off")
    ap.add_argument("--majority-vote", action="store_true",
                    help="old type vote for kind / material / floor (default: soft, as run_balcony --vote soft)")
    ap.add_argument("--unconditional", action="store_true",
                    help="old scoring: material with a wrong kind and floor with a wrong structure count as wrong "
                    "(default: conditional, each attribute only where its prerequisite is right in truth and prediction)")
    ap.add_argument("--structure-depth", type=float, default=0.10,
                    help="as run_balcony --structure-depth (metres; default 0.10 as the pipeline; 0 = classifier)")
    ap.add_argument("--cluster-label-split", type=float, default=0.0,
                    help="as run_balcony --cluster-label-split (0 = off)")
    ap.add_argument("--cluster-links", action=argparse.BooleanOptionalAction, default=True,
                    help="as run_balcony --cluster-links (default on, as the pipeline; --no-cluster-links = off)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=T100 / "eval_test100_final_facade_v4.md")
    args = ap.parse_args()

    # run_balcony.parse_args defaults for the clustering step
    cargs = argparse.Namespace(
        cluster_mode=args.cluster_mode, dino="dinov2_vits14", facade_max_side=896, pca_dim=32, seed=42,
        k_max=8, k_max_box=6, twin_mirror_min=0.35, col_tol=0.04, row_tol=0.055,
        spatial_strength=1.8, unary_weight=0.9, cluster_k1=args.cluster_k1, k1_sil_min=args.k1_sil_min,
        k1_pair_sim_min=args.k1_pair_sim_min, cluster_width_ratio=args.cluster_width_ratio, cluster_links=args.cluster_links, cluster_label_split=args.cluster_label_split,
    )
    device = torch.device(args.device)
    gt = {r["stem"]: r["boxes"] for r in jl(args.gt)}
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    oriel = json.loads(args.oriel_preds.read_text(encoding="utf-8"))
    crop_of = {(Path(m["photo"]).stem, tuple(m["box_xyxy"])): Path(m["crop"]).stem for m in jl(T100 / "crops/manifest.jsonl")}
    split_of = {(Path(m["photo"]).stem, tuple(m["box_xyxy"])): Path(m["crop"]).stem
                for m in jl(T100 / "crops_split/manifest.jsonl")}
    lab_main = {Path(r["path"]).stem: r for r in jl(T100 / "crops/labels.jsonl")}
    lab_split = {Path(r["path"]).stem: r for r in jl(T100 / "crops_split/labels.jsonl")}
    pred = try_load_predictor(args.railing_ckpt, device, structure_ckpt=DEFAULT_STRUCTURE_CKPT)
    ckpt = Path(pred.ckpt_path)
    print(f"railing classifier {ckpt}")

    rows: list[dict] = []
    photos: dict[str, dict] = {}
    counts = Counter()
    stems = sorted(s for s in gt if (args.photos / f"{s}.jpg").is_file())
    for k, s in enumerate(stems, 1):
        facade = Image.open(args.photos / f"{s}.jpg").convert("RGB")
        dsl = json.loads((args.dsl_dir / s / "facade_dsl.json").read_text(encoding="utf-8"))
        dp = args.depth_dir / f"{s}_depth.npy"
        depth = resize_depth(np.load(dp), facade.width, facade.height) if dp.is_file() else None
        units, split_boxes, _ = snap_open_units(
            cache[s]["sized"], facade, dsl, row_tol=cargs.row_tol, col_tol=cargs.col_tol,
            side_wall=True, depth=depth, split_predictor=pred,
        )
        enc = place_enclosed_cells([tuple(c) for c in oriel.get(s, [])], dsl, unit_id0=len(units))
        kept, _drop = drop_open_under_enclosed(units, enc) if units and enc else (units, [])
        kept_ids = {int(u["unit_id"]) for u in kept}
        unit_of_box = {j: u for u in units for j in u["member_raw_idxs"]}

        final_lab: dict[int, dict] = {}
        unit_lab: dict[int, dict] = {}
        type_of: dict[int, int] = {}
        medoids: dict[int, int] = {}
        if kept:
            label_probs = ([pred.predict_probs(facade.crop(tuple(u["box_xyxy"]))) for u in kept]
                           if args.cluster_label_split > 0 else None)
            clustered = cluster_boxes(facade, [u["box_xyxy"] for u in kept], [int(u["floor"]) for u in kept],
                                      [int(u["bay"]) for u in kept], cargs, device,
                                      windows=dsl.get("instances"), label_probs=label_probs)
            labels = clustered["labels"]
            medoids = rp.pick_medoids(clustered["feats"], labels)
            unit_irs = [infer_unit_ir(facade.crop(tuple(u["box_xyxy"])), u, image_size=facade.size,
                                      profile_name=DEFAULT_PROFILE, predictor=pred) for u in kept]
            if args.structure_depth > 0:
                for ir, u in zip(unit_irs, kept):
                    st_d = structure_from_depth(depth, u["box_xyxy"], args.structure_depth)
                    if st_d is not None:
                        ir["structure"] = st_d
            jprobs = [juliet_prob(pred, facade.crop(tuple(u["box_xyxy"]))) for u in kept]
            fprobs = [floor_probs(pred, facade.crop(tuple(u["box_xyxy"]))) for u in kept]
            sprobs = ([pred.predict_probs(facade.crop(tuple(u["box_xyxy"]))) for u in kept]
                      if not args.majority_vote else None)
            voted = vote_types(kept, labels, medoids, unit_irs, jprobs, fprobs, sprobs)
            for i, u in enumerate(kept):
                uid = int(u["unit_id"])
                unit_lab[uid] = labels_from_ir(unit_irs[i])
                final_lab[uid] = labels_from_ir(voted[i])
                type_of[uid] = int(labels[i])
        photos[s] = {
            "units": [{"unit_id": int(u["unit_id"]), "box": [int(v) for v in u["box_xyxy"]],
                       "type_id": type_of[int(u["unit_id"])], "medoid": medoids.get(type_of[int(u["unit_id"])]) == i,
                       "unit": unit_lab[int(u["unit_id"])], "final": final_lab[int(u["unit_id"])]}
                      for i, u in enumerate(kept)],
            "oriel_dropped": [[int(v) for v in u["box_xyxy"]] for u in _drop],
            "oriel_cells": [[int(v) for v in u["box_xyxy"]] for u in enc],
        }

        for j, b in enumerate(split_boxes):
            key = (s, tuple(int(v) for v in b))
            name = split_of.get(key) or crop_of.get(key)
            rec = (lab_split.get(name) if key in split_of else lab_main.get(name)) if name else None
            if rec is None:
                counts["no_label"] += 1
                continue
            best = max((iou(b, g) for g in gt.get(s, [])), default=0.0)
            if args.gt_iou > 0 and best < args.gt_iou:
                counts["not_in_gt"] += 1
                continue
            u = unit_of_box.get(j)
            if u is None:
                counts["no_unit"] += 1
                continue
            uid = int(u["unit_id"])
            if uid not in kept_ids:
                counts["oriel_dropped"] += 1
                continue
            rows.append({
                "stem": s, "crop_name": name, "box": [int(v) for v in b], "unit_id": uid, "type_id": type_of[uid],
                "truth": truth_from_record(rec),
                "crop": labels_from_pred(pred.predict_full(facade.crop(tuple(b)))),
                "unit": unit_lab[uid], "final": final_lab[uid],
            })
            counts["scored"] += 1
        print(f"[{k}/{len(stems)}] {s}  split {len(split_boxes)}  units {len(units)}  kept {len(kept)}  "
              f"types {len(set(type_of.values()))}", flush=True)

    header = [
        "# test100 balcony types: per crop vs final facade (after clustering + type vote)", "",
        f"- checkpoint `{ckpt.name}` sha256 {hashlib.sha256(ckpt.read_bytes()).hexdigest()[:12]}; "
        f"cluster mode `{args.cluster_mode}` (k=1 {'on' if args.cluster_k1 else 'off'}, "
        f"width ratio {args.cluster_width_ratio or 'off'}, links {'on' if args.cluster_links else 'off'}, label split {args.cluster_label_split or 'off'}, structure depth {args.structure_depth or 'off'}, vote {'majority' if args.majority_vote else 'soft'}); oriel cells `{args.oriel_preds.parent.name}/{args.oriel_preds.name}`",
        f"- boxes after the live split: scored {counts['scored']}, no hand label {counts['no_label']}, "
        f"IoU < {args.gt_iou} with every GT box {counts['not_in_gt']}, removed by the oriel step "
        f"{counts['oriel_dropped']}, no unit {counts['no_unit']}",
        "- `crop`: classifier on the crop box; `unit`: on the snapped unit box, no vote; "
        "`final`: voted type label in the DSL. Material `none` = predicted kind is not open_work.",
        ("- scoring: unconditional (material with a wrong kind and floor with a wrong structure count as wrong)"
         if args.unconditional else
         "- scoring: conditional (default since 2026-11-07): material only where truth and prediction are both "
         "open work, floor only where both are projecting; n differs per stage. Vote effect: crops that pass the "
         "condition at both stages."), "",
    ]
    text, metrics = report(rows, header, conditional=not args.unconditional)
    args.out.write_text(text, encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps({"counts": dict(counts), "metrics": metrics, "rows": rows, "photos": photos}, indent=1),
        encoding="utf-8")
    print(text)
    print(f"-> {args.out}\n-> {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
