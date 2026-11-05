r"""Rule 2 (solid + juliet sill removal) with and without a confidence condition, per checkpoint.

Base: the pipeline boxes (test100 SAM3 ``filtered`` cache) with rule 1
(``side_wall``), as ``run_balcony.py`` does now. For each checkpoint the
railing classifier is run on every box crop and its softmax probabilities are
cached (``runs/test100/railing_probs_<ckpt stem>.json``).

Variants per checkpoint:

* ``argmax`` - drop a box whose top kind is ``solid`` and top structure is
  ``juliet`` (enclosure head says ``open``), the rule 2 tested before.
* ``conf`` (option D) - the same, and P(solid) >= ``--conf`` and
  P(juliet) >= ``--conf``. ``--conf`` 0.8 was fixed before scoring.

Scored against ``runs/test100/balcony_gt_boxes.jsonl`` without ignore regions
(group rule, IoU 0.5 and 0.3). Images (last checkpoint): GT | pipeline |
+ rule 2 argmax | + rule 2 conf; removed boxes blue dashed with P(solid) / P(juliet).

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_pipeline/score_test100_rule2_conf.py --ckpt checkpoints/railing_test100_v2.pt --ckpt checkpoints/railing_test100_v3.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from postfilter import apply_rules  # noqa: E402
from score_sam3_label_iou import box_iou  # noqa: E402
from score_test100_gt_boxes import prf, score_photo  # noqa: E402
from score_test100_user_rules import dashed  # noqa: E402

T100 = ROOT / "runs" / "test100"


def class_probs(ckpt: Path, cache: dict, stems: list[str], out: Path, device: str) -> dict[str, list[dict]]:
    """Per box: P(solid), P(juliet), top kind / structure / enclosure."""
    if out.is_file():
        got = json.loads(out.read_text(encoding="utf-8"))
        if all(s in got and len(got[s]) == len(cache[s]["filtered"]) for s in stems):
            return got
    import torch

    from balcony_train.labels import CLASSES, ENCLOSURES, LABEL_STRUCTURES
    from balcony_train.model import try_load_predictor

    pred = try_load_predictor(ckpt, torch.device(device))
    if pred is None:
        sys.exit(f"cannot load {ckpt}")
    m = pred.model
    res: dict[str, list[dict]] = {}
    with torch.no_grad():
        for s in stems:
            img = Image.open(T100 / "photos" / f"{s}.jpg").convert("RGB")
            rows = []
            for b in cache[s]["filtered"]:
                x = pred.tfm(img.crop(tuple(b))).unsqueeze(0).to(pred.device)
                kind, _mat, _floor, enc, _skin, struct = m(x)
                pk = torch.softmax(kind, 1)[0].cpu().numpy()
                ps = torch.softmax(struct, 1)[0].cpu().numpy()
                enc_top = ENCLOSURES[int(enc.argmax(1))] if getattr(m, "_enclosure_trained", False) else None
                rows.append({
                    "kind": CLASSES[int(pk.argmax())],
                    "structure": LABEL_STRUCTURES[int(ps.argmax())] if getattr(m, "_structure_trained", False) else None,
                    "enclosure": enc_top,
                    "p_solid": round(float(pk[CLASSES.index("solid")]), 4),
                    "p_juliet": round(float(ps[LABEL_STRUCTURES.index("juliet")]), 4),
                })
            res[s] = rows
    out.write_text(json.dumps(res, indent=0), encoding="utf-8")
    return res


def is_sill(r: dict, conf: float | None) -> bool:
    if r["kind"] != "solid" or r["structure"] != "juliet" or r["enclosure"] not in (None, "open"):
        return False
    return conf is None or (r["p_solid"] >= conf and r["p_juliet"] >= conf)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, action="append", required=True)
    ap.add_argument("--conf", type=float, default=0.8)
    ap.add_argument("--gt", type=Path, default=T100 / "balcony_gt_boxes.jsonl")
    ap.add_argument("--cache", type=Path, default=ROOT / "runs" / "batch" / "_rows" / "test100_sam3_stage_boxes.json")
    ap.add_argument("--dsl-dir", type=Path, default=T100 / "batch_grid2")
    ap.add_argument("--depth-dir", type=Path, default=T100 / "da3_depth")
    ap.add_argument("--img-dir", type=Path, default=T100 / "compare_rule2_conf")
    ap.add_argument("--out", type=Path, default=T100 / "balcony_rule2_conf_score.json")
    ap.add_argument("--panel-height", type=int, default=760)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    gt = {json.loads(l)["stem"]: json.loads(l) for l in args.gt.read_text(encoding="utf-8").splitlines() if l.strip()}
    stems = sorted(gt)
    cache = json.loads(args.cache.read_text(encoding="utf-8"))

    # rule 1 base, as indices into the filtered boxes
    base_idx: dict[str, list[int]] = {}
    for s in stems:
        dsl = json.loads((args.dsl_dir / s / "facade_dsl.json").read_text(encoding="utf-8"))
        dp = args.depth_dir / f"{s}_depth.npy"
        depth = np.load(dp) if dp.is_file() else None
        boxes = cache[s]["filtered"]
        kept, _ = apply_rules(boxes, tuple(cache[s]["size"]), dsl, depth=depth, rules=("side_wall",))
        left = list(range(len(boxes)))
        idx = []
        for b in kept:
            j = next(j for j in left if boxes[j] == b)
            left.remove(j)
            idx.append(j)
        base_idx[s] = idx

    variants: dict[str, dict[str, list[int]]] = {"pipeline": base_idx}
    removed: dict[str, dict[str, list[int]]] = {}
    probs_by: dict[str, dict] = {}
    for ck in args.ckpt:
        probs = class_probs(ck, cache, stems, T100 / f"railing_probs_{ck.stem}.json", args.device)
        probs_by[ck.stem] = probs
        for mode, conf in (("argmax", None), (f"conf{args.conf:g}", args.conf)):
            name = f"{ck.stem.replace('railing_test100_', '')} {mode}"
            variants[name] = {s: [i for i in base_idx[s] if not is_sill(probs[s][i], conf)] for s in stems}
            removed[name] = {s: [i for i in base_idx[s] if is_sill(probs[s][i], conf)] for s in stems}

    subsets = {"all": lambda s: True, "cmp": lambda s: s.startswith("cmp_"), "lsaa": lambda s: not s.startswith("cmp_")}
    results: dict = {}
    per_photo: dict = {s: {} for s in stems}
    print(f"{'variant':16s} {'subset':5s} IoU   pred   TP   FP   FN     P      R     F1   removed  real")
    for thr in (0.5, 0.3):
        for name, idx in variants.items():
            for sub, f in subsets.items():
                rows = []
                for s in stems:
                    if f(s):
                        sc = score_photo(gt[s]["boxes"], [], [cache[s]["filtered"][i] for i in idx[s]], thr, True)
                        rows.append(sc)
                        if thr == 0.5:
                            per_photo[s][name] = sc
                t = prf(rows)
                rem = [(s, i) for s in stems if f(s) for i in removed.get(name, {}).get(s, [])]
                real = sum(any(box_iou(cache[s]["filtered"][i], g) >= 0.5 for g in gt[s]["boxes"]) for s, i in rem)
                results[f"{name}|{sub}|{thr}"] = {**t, "removed": len(rem), "removed_real": real}
                print(f"{name:16s} {sub:5s} {thr:<4}  {t['pred']:4d} {t['tp']:4d} {t['fp']:4d} {t['fn']:4d}  "
                      f"{t['precision']:.3f}  {t['recall']:.3f}  {t['f1']:.3f}   {len(rem):5d} {real:5d}")
    args.out.write_text(json.dumps({"conf": args.conf, "ckpts": [str(c) for c in args.ckpt], "results": results,
                                    "photos": per_photo}, indent=0), encoding="utf-8")
    print(f"wrote {args.out}")

    # images for the last checkpoint
    last = args.ckpt[-1].stem
    tag = last.replace("railing_test100_", "")
    probs = probs_by[last]
    shown = [f"{tag} argmax", f"{tag} conf{args.conf:g}"]
    args.img_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("arial.ttf", 22)
    small = ImageFont.truetype("arial.ttf", 18)
    log = {}
    for s in stems:
        base = Image.open(T100 / "photos" / f"{s}.jpg").convert("RGB")
        W, H = base.size
        lw = max(3, W // 200)
        scale = args.panel_height / H
        panels = []
        g = base.copy()
        dg = ImageDraw.Draw(g)
        for b in gt[s]["boxes"]:
            dg.rectangle(b, outline=(255, 0, 0), width=lw)
        panels.append((g, [f"Ground truth: {len(gt[s]['boxes'])}", ""], []))
        im = base.copy()
        d = ImageDraw.Draw(im)
        for i in base_idx[s]:
            d.rectangle(cache[s]["filtered"][i], outline=(255, 0, 0), width=lw)
        sc = per_photo[s]["pipeline"]
        panels.append((im, ["Pipeline (rule 1)", f"TP {sc['tp']} FP {sc['fp']} FN {sc['fn']}"], []))
        for name in shown:
            im = base.copy()
            d = ImageDraw.Draw(im)
            for i in variants[name][s]:
                d.rectangle(cache[s]["filtered"][i], outline=(255, 0, 0), width=lw)
            notes = []
            for i in removed[name][s]:
                b = cache[s]["filtered"][i]
                dashed(d, b, (0, 140, 255), lw)
                notes.append((b, f"{probs[s][i]['p_solid']:.2f}/{probs[s][i]['p_juliet']:.2f}"))
            sc = per_photo[s][name]
            title = f"+ Rule 2, {tag}" + ("" if "argmax" in name else f", P >= {args.conf:g}")
            panels.append((im, [title, f"TP {sc['tp']} FP {sc['fp']} FN {sc['fn']} | removed {len(removed[name][s])}"],
                           notes))
            if removed[name][s]:
                log.setdefault(s, {})[name] = [
                    {"box": cache[s]["filtered"][i], **probs[s][i],
                     "real_balcony": any(box_iou(cache[s]["filtered"][i], gg) >= 0.5 for gg in gt[s]["boxes"])}
                    for i in removed[name][s]]
        pw = max(1, round(W * scale))
        head = 64
        out = Image.new("RGB", (len(panels) * (pw + 12), args.panel_height + head), "white")
        do = ImageDraw.Draw(out)
        for k, (pim, lines, notes) in enumerate(panels):
            x = k * (pw + 12)
            out.paste(pim.resize((pw, args.panel_height), Image.BILINEAR), (x, head))
            for j, line in enumerate(lines):
                do.text((x + 6, 4 + 28 * j), line, fill="black", font=font)
            for b, txt in notes:  # P(solid)/P(juliet) above each removed box
                tx, ty = x + b[0] * scale, head + max(0, b[1] * scale - 20)
                do.rectangle([tx, ty, tx + 9 * len(txt), ty + 19], fill="white")
                do.text((tx + 1, ty), txt, fill=(0, 90, 200), font=small)
        out.save(args.img_dir / f"{s}.jpg", quality=86)
    (args.img_dir / "_removed.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"images -> {args.img_dir}")


if __name__ == "__main__":
    main()
