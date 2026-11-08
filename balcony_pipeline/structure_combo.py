r"""Open-balcony structure from DA3 depth + the railing classifier, fitted on val30, scored per unit.

Features per unit: protrusion in metres against the wall beside the box (``structure_depth.protrusion``) and the
classifier's logit P(juliet) on the unit crop. A logistic regression (projecting = 1) is fitted on the val30
ground truth (``data/val30/type_gt.jsonl``) only; units with no wall strip fall back to the classifier.

* val30: leave-one-photo-out cross-validation (fit on 19 photos, predict the 20th);
* test100: the model fitted on all of val30, applied once to the scored crops' units of a
  ``score_test100_types_final.py`` JSON (labels and unit boxes from it, DA3 from ``--t100/da3_depth``).

Compared per unit with the classifier alone and the depth rule alone (> 0.10 m). Accuracy, balanced accuracy
and macro F1 (projecting / juliet).

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_pipeline/structure_combo.py \
        --t100 C:\Users\hiroo\3D_facade_reconstruction\runs\test100 \
        --t100-json C:\Users\hiroo\3D_facade_reconstruction\runs\test100\eval_test100_final_facade_v4_k1w2_depth010_20261107.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_balcony import juliet_prob  # noqa: E402
from structure_depth import protrusion  # noqa: E402

from balcony_train.model import try_load_predictor  # noqa: E402
from balcony_train.paths import DEFAULT_STRUCTURE_CKPT  # noqa: E402
from balcony_train.try_da3_enclosure import resize_depth  # noqa: E402

DEPTH_T = 0.10


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


def unit_record(pred, photo: Image.Image, depth: np.ndarray, box, crop: Image.Image | None = None) -> dict:
    pj = juliet_prob(pred, crop if crop is not None else photo.crop(tuple(int(v) for v in box)))
    m = protrusion(depth, box)
    return {"p_juliet": pj, "prot_m": m["prot_m"]}


def metrics(pred: list[str], truth: list[str]) -> str:
    acc = np.mean([p == t for p, t in zip(pred, truth)])
    recs, f1s = [], []
    for c in ("projecting", "juliet"):
        tp = sum(p == c and t == c for p, t in zip(pred, truth))
        pp, tt = pred.count(c), truth.count(c)
        pr, rc = (tp / pp if pp else 0), (tp / tt if tt else 0)
        recs.append(rc)
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    jp = sum(p == "projecting" and t == "juliet" for p, t in zip(pred, truth))
    pj = sum(p == "juliet" and t == "projecting" for p, t in zip(pred, truth))
    return (f"macro F1 {np.mean(f1s):.3f}  acc {acc:.3f}  balanced {np.mean(recs):.3f}   "
            f"(juliet->proj {jp}, proj->juliet {pj})")


def predict_all(model, recs: list[dict]) -> dict[str, list[str]]:
    out = {"classifier": [], "depth > 0.10 m": [], "depth + classifier": []}
    for r in recs:
        clf = "juliet" if r["p_juliet"] > 0.5 else "projecting"
        out["classifier"].append(clf)
        if r["prot_m"] is None:
            out["depth > 0.10 m"].append(clf)
            out["depth + classifier"].append(clf)
            continue
        out["depth > 0.10 m"].append("projecting" if r["prot_m"] > DEPTH_T else "juliet")
        x = np.array([[r["prot_m"], logit(r["p_juliet"])]])
        out["depth + classifier"].append("projecting" if model.predict_proba(x)[0, 1] > 0.5 else "juliet")
    return out


def fit(recs: list[dict]):
    from sklearn.linear_model import LogisticRegression

    ok = [r for r in recs if r["prot_m"] is not None]
    X = np.array([[r["prot_m"], logit(r["p_juliet"])] for r in ok])
    y = np.array([r["truth"] == "projecting" for r in ok], dtype=int)
    return LogisticRegression(class_weight="balanced", C=1.0).fit(X, y)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=ROOT / "data" / "val30" / "type_gt.jsonl")
    ap.add_argument("--photos", type=Path, default=ROOT / "data" / "val30")
    ap.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "val30")
    ap.add_argument("--depth-dir", type=Path, default=ROOT / "runs" / "da3_enclosure")
    ap.add_argument("--t100", type=Path, default=None, help="test100 folder (photos/, da3_depth/)")
    ap.add_argument("--t100-json", type=Path, default=None, help="score_test100_types_final.py JSON (labels, units)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    pred = try_load_predictor(None, torch.device(args.device), structure_ckpt=DEFAULT_STRUCTURE_CKPT)

    # ---- val30 ----
    val = []
    for r in (json.loads(l) for l in args.gt.read_text(encoding="utf-8").splitlines() if l.strip()):
        photo = Image.open(next(p for p in (args.photos / f"{r['stem']}.jpg", args.photos / f"{r['stem']}.png")
                                if p.is_file())).convert("RGB")
        depth = resize_depth(np.load(args.depth_dir / f"{r['stem']}_depth.npy"), photo.width, photo.height)
        bdir = args.run_dir / r["stem"] / "balcony"
        units = json.loads((bdir / "balcony_units.json").read_text(encoding="utf-8"))["units"]
        for u, g in zip(units, r["units"]):
            if g["not_balcony"] or g["structure"] is None:
                continue
            rec = unit_record(pred, photo, depth, g["box"], Image.open(bdir / u["asset"]).convert("RGB"))
            val.append({**rec, "truth": g["structure"], "photo": r["id"]})
    truth = [r["truth"] for r in val]
    print(f"val30: {len(val)} units (projecting {truth.count('projecting')}, juliet {truth.count('juliet')}), "
          f"{sum(r['prot_m'] is None for r in val)} without a wall strip")
    cv = {k: [None] * len(val) for k in ("classifier", "depth > 0.10 m", "depth + classifier")}
    for ph in sorted({r["photo"] for r in val}):
        tr = [r for r in val if r["photo"] != ph]
        te_idx = [i for i, r in enumerate(val) if r["photo"] == ph]
        model = fit(tr)
        out = predict_all(model, [val[i] for i in te_idx])
        for k, v in out.items():
            for i, p in zip(te_idx, v):
                cv[k][i] = p
    print("  val30, leave-one-photo-out:")
    for k, v in cv.items():
        print(f"    {k:20s} {metrics(v, truth)}")
    model = fit(val)
    coef = model.coef_[0]
    print(f"  model on all val30: logit(projecting) = {model.intercept_[0]:+.2f} {coef[0]:+.2f} * protrusion_m "
          f"{coef[1]:+.2f} * logit(P_juliet)")

    # ---- test100 ----
    if args.t100 and args.t100_json:
        d = json.loads(args.t100_json.read_text(encoding="utf-8"))
        boxes = {(s, u["unit_id"]): u["box"] for s, ph in d["photos"].items() for u in ph["units"]}
        test, cache = [], {}
        for r in d["rows"]:
            if r["truth"]["structure"] is None:
                continue
            s = r["stem"]
            if s not in cache:
                photo = Image.open(args.t100 / "photos" / f"{s}.jpg").convert("RGB")
                dp = args.t100 / "da3_depth" / f"{s}_depth.npy"
                cache = {s: (photo, resize_depth(np.load(dp), photo.width, photo.height) if dp.is_file() else None)}
            photo, depth = cache[s]
            box = boxes[(s, r["unit_id"])]
            if depth is None:
                rec = {"p_juliet": juliet_prob(pred, photo.crop(tuple(int(v) for v in box))), "prot_m": None}
            else:
                rec = unit_record(pred, photo, depth, box)
            test.append({**rec, "truth": r["truth"]["structure"]})
        tt = [r["truth"] for r in test]
        print(f"\ntest100: {len(test)} crops (projecting {tt.count('projecting')}, juliet {tt.count('juliet')}), "
              f"{sum(r['prot_m'] is None for r in test)} without a wall strip; model fitted on val30 only")
        for k, v in predict_all(model, test).items():
            print(f"    {k:20s} {metrics(v, tt)}")


if __name__ == "__main__":
    main()
