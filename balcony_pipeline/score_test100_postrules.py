"""Score the open-balcony post-rules (``postfilter.py``) on test100.

Boxes: the pipeline ``filtered`` boxes from the test100 SAM3 cache. Window grids:
``runs/test100/batch_grid2``. Oriel cells: held-out cell model predictions
(``runs/oriel_grid_compare/grid2_test100/preds_heldout.json``; 89 photos, the
other 11 get no oriel rule). Classifier labels: ``railing_test100_v2.pt`` (test
buildings held out), cached in ``--label-cache``.

Rows: baseline, each rule alone, rules added in order, all rules. Matching as in
``score_test100_gt_boxes.py`` (group rule), with and without ignore regions.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_pipeline/score_test100_postrules.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from postfilter import RULES, apply_rules  # noqa: E402
from score_test100_gt_boxes import prf, score_photo  # noqa: E402

T100 = ROOT / "runs" / "test100"
DEFAULT_GT = T100 / "balcony_gt_boxes.jsonl"
DEFAULT_CACHE = ROOT / "runs" / "batch" / "_rows" / "test100_sam3_stage_boxes.json"
DEFAULT_DSL = T100 / "batch_grid2"
DEFAULT_ORIEL = ROOT / "runs" / "oriel_grid_compare" / "grid2_test100" / "preds_heldout.json"
DEFAULT_CKPT = ROOT / "checkpoints" / "railing_test100_v2.pt"
DEFAULT_LABELS = T100 / "railing_preds_v2.json"
DEFAULT_OUT = T100 / "balcony_postrules_score.json"


def classify(cache: dict, stems: list[str], ckpt: Path, out: Path, device: str) -> dict[str, list[dict]]:
    if out.is_file():
        got = json.loads(out.read_text(encoding="utf-8"))
        if all(s in got and len(got[s]) == len(cache[s]["filtered"]) for s in stems):
            return got
    import torch
    from balcony_train.model import try_load_predictor

    pred = try_load_predictor(ckpt, torch.device(device))
    if pred is None:
        sys.exit(f"cannot load {ckpt}")
    print(f"classifier {pred.ckpt_path} structure from {pred.structure_ckpt or 'same checkpoint'}", flush=True)
    labels: dict[str, list[dict]] = {}
    for s in stems:
        img = Image.open(T100 / "photos" / f"{s}.jpg").convert("RGB")
        labels[s] = [pred.predict_full(img.crop(tuple(b))) for b in cache[s]["filtered"]]
    out.write_text(json.dumps(labels, indent=0), encoding="utf-8")
    return labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--dsl-dir", type=Path, default=DEFAULT_DSL)
    ap.add_argument("--oriel", type=Path, default=DEFAULT_ORIEL)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--label-cache", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    gt = {json.loads(l)["stem"]: json.loads(l) for l in args.gt.read_text(encoding="utf-8").splitlines() if l.strip()}
    stems = sorted(gt)
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    oriel = json.loads(args.oriel.read_text(encoding="utf-8"))
    dsl = {s: json.loads((args.dsl_dir / s / "facade_dsl.json").read_text(encoding="utf-8")) for s in stems}
    labels = classify(cache, stems, args.ckpt, args.label_cache, args.device)

    variants: list[tuple[str, tuple[str, ...]]] = [("baseline", ())]
    variants += [(f"only {r}", (r,)) for r in RULES]
    variants += [(f"+ {r}", RULES[: k + 1]) for k, r in enumerate(RULES[:-1])]
    variants += [("all rules", RULES)]

    subsets = {"all": lambda s: True, "cmp": lambda s: s.startswith("cmp_"), "lsaa": lambda s: not s.startswith("cmp_")}
    results: dict = {}
    boxes_out: dict = {}
    hdr = f"{'variant':24s} {'subset':5s} {'ign':3s} {'IoU':4s}  pred   TP   FP   FN     P      R     F1"
    print(hdr)
    for name, rules in variants:
        kept: dict[str, list] = {}
        logs: dict[str, list] = {}
        for s in stems:
            size = tuple(cache[s]["size"])
            kept[s], logs[s] = apply_rules(cache[s]["filtered"], size, dsl[s], labels=labels[s],
                                           oriel_cells=[tuple(c) for c in oriel.get(s, [])], rules=rules)
        boxes_out[name] = {"boxes": kept, "log": logs}
        for thr in (0.5, 0.3):
            for use_ign in (False, True):
                for sub, f in subsets.items():
                    rows = [score_photo(gt[s]["boxes"], gt[s]["ignore"] if use_ign else [], kept[s], thr, True)
                            for s in stems if f(s)]
                    t = prf(rows)
                    results[f"{name}|{sub}|ignore={use_ign}|{thr}"] = t
                    if sub == "all" or (thr == 0.5 and not use_ign):
                        print(f"{name:24s} {sub:5s} {'on' if use_ign else 'off':3s} {thr:<4}  {t['pred']:4d} {t['tp']:4d} "
                              f"{t['fp']:4d} {t['fn']:4d}  {t['precision']:.3f}  {t['recall']:.3f}  {t['f1']:.3f}")
        n = {r: sum(1 for s in stems for e in logs[s] if e["rule"] == r) for r in RULES}
        print(f"{'':24s} rule hits: {n}")
    args.out.write_text(json.dumps({"results": results, "variants": boxes_out}, indent=0), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
