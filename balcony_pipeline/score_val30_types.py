r"""Score open-balcony type clustering against the val30 type ground truth (``data/val30/type_gt.jsonl``).

For each setup (``NAME=run_balcony args``) the units of every labeled photo are re-clustered with the
``run_balcony`` clustering defaults plus those args, on the cached units of the val30 run (no SAM3).
Units marked "not a balcony" and photos not complete are left out. Pooled over photos:

* pairs: same-photo unit pairs; precision = same type that are the same GT group, recall = same GT group
  that are the same type, F1;
* ARI per photo (mean over photos with 2+ units);
* exact: photos whose types equal the GT groups (up to renaming);
* labels: the railing classifier per unit and the run_balcony type vote (incl. the structure tie rule)
  per setup, scored like test100 (macro F1 / acc per head; material only on true open work, floor only
  on true projecting). The ``RUN`` setup is re-voted with the current rules on the run's types.

The val30 run itself (``balcony_units.json`` type ids) is scored as ``RUN``.

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_pipeline/score_val30_types.py \
        --setup "NEW=" --setup "OLD=--no-cluster-k1 --cluster-width-ratio 0" --setup "LINKS=--cluster-links"
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from draw_vote_groups import recluster, unit_bays  # noqa: E402
from recovery_profile import DEFAULT_PROFILE  # noqa: E402
from run_balcony import infer_unit_ir, juliet_prob  # noqa: E402
from score_test100_types_final import conditional_skip, head_metrics, labels_from_ir, vote_types  # noqa: E402
from structure_depth import structure_from_depth  # noqa: E402

from balcony_train.try_da3_enclosure import resize_depth  # noqa: E402

from balcony_train.model import try_load_predictor  # noqa: E402
from balcony_train.paths import DEFAULT_STRUCTURE_CKPT  # noqa: E402


def ari(a: list[int], b: list[int]) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(a, b))


def same_partition(a: list[int], b: list[int]) -> bool:
    return len(set(a)) == len(set(b)) == len(set(zip(a, b)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=ROOT / "data" / "val30" / "type_gt.jsonl")
    ap.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "val30")
    ap.add_argument("--photos", type=Path, default=ROOT / "data" / "val30")
    ap.add_argument("--setup", action="append", default=[], metavar="NAME=ARGS")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-labels", action="store_true", help="grouping scores only (no classifier, no vote)")
    ap.add_argument("--structure-depth", type=float, default=0.10,
                    help="as run_balcony: structure from DA3 depth (metres; 0 = classifier)")
    ap.add_argument("--depth-dir", type=Path, default=ROOT / "runs" / "da3_enclosure")
    ap.add_argument("--majority-vote", action="store_true",
                    help="old type vote for kind / material / floor (default: soft, as run_balcony --vote soft)")
    ap.add_argument("--dump", type=Path, default=None, help="write the per-unit label rows of every setup (json)")
    ap.add_argument("--railing-ckpt", type=Path, default=None,
                    help="railing classifier for every setup (default: pipeline default, paths.py); "
                         "--setup args do not change the classifier")
    args = ap.parse_args()

    if not args.gt.is_file():
        raise SystemExit(f"no labels yet: {args.gt} (label with balcony_train/type_gt_ui.py)")
    rows = [json.loads(l) for l in args.gt.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("complete")]
    if not rows:
        raise SystemExit(f"no complete photos in {args.gt}")
    device = torch.device(args.device)
    setups = [("RUN", None)] + [(s.split("=", 1)[0], s.split("=", 1)[1].split()) for s in args.setup]
    need_probs = any(a and "--cluster-label-split" in a for _, a in setups)
    do_labels = not args.no_labels
    pred = (try_load_predictor(args.railing_ckpt, device, structure_ckpt=DEFAULT_STRUCTURE_CKPT)
            if need_probs or do_labels else None)
    label_rows: dict[str, list[dict]] = {name: [] for name, _ in setups}

    stats = {name: {"tp": 0, "fp": 0, "fn": 0, "ari": [], "exact": 0, "types": 0} for name, _ in setups}
    n_units = n_gt_groups = 0
    for r in rows:
        bdir = args.run_dir / r["stem"] / "balcony"
        units = json.loads((bdir / "balcony_units.json").read_text(encoding="utf-8"))["units"]
        if [[int(v) for v in u["box_xyxy"]] for u in units] != [u["box"] for u in r["units"]]:
            print(f"skip {r['id']}: run boxes differ from the labeled boxes")
            continue
        keep = [i for i, u in enumerate(r["units"]) if not u["not_balcony"]]
        gt = [int(r["units"][i]["group"]) for i in keep]
        n_units += len(keep)
        n_gt_groups += len(set(gt))
        photo = Image.open(next(p for p in (args.photos / f"{r['stem']}.jpg", args.photos / f"{r['stem']}.png")
                                if p.is_file())).convert("RGB")
        bays = unit_bays(units, bdir)
        wdsl = bdir.parent / "facade_dsl.json"
        windows = json.loads(wdsl.read_text(encoding="utf-8")).get("instances") if wdsl.is_file() else None
        crops = [Image.open(bdir / u["asset"]).convert("RGB") for u in units]
        probs = [pred.predict_probs(c) for c in crops] if pred else None
        if do_labels:
            vunits = [{"unit_id": i, "box_xyxy": u["box_xyxy"], "floor": u["floor"]} for i, u in enumerate(units)]
            irs = [infer_unit_ir(c, vu, image_size=photo.size, profile_name=DEFAULT_PROFILE, predictor=pred)
                   for c, vu in zip(crops, vunits)]
            jprobs = [juliet_prob(pred, c) for c in crops]
            dp = args.depth_dir / f"{r['stem']}_depth.npy"
            if args.structure_depth > 0 and dp.is_file():
                depth = resize_depth(np.load(dp), photo.width, photo.height)
                for ir, u in zip(irs, units):
                    st_d = structure_from_depth(depth, u["box_xyxy"], args.structure_depth)
                    if st_d is not None:
                        ir["structure"] = st_d
            truth = []
            for i in keep:
                g = r["units"][i]
                truth.append({"kind": g["kind"], "material": g["material"] if g["kind"] == "open_work" else None,
                              "floor": g["floor_shape"] if g["structure"] == "projecting" else None,
                              "structure": g["structure"]})
        for name, extra in setups:
            if extra is None:
                lab_all = [int(u["type_id"]) for u in units]
                types_json = json.loads((bdir / "balcony_types.json").read_text(encoding="utf-8"))
                med = {int(t["type_id"]): int(t["exemplar_unit"]) for t in types_json
                       if str(t.get("name", "")).startswith("balc_type_")}
            else:
                lab_all, med = recluster(photo, units, bays, device, extra, windows, probs)
                lab_all = [int(x) for x in lab_all]
            if do_labels:
                la = np.asarray(lab_all, dtype=np.int32)
                for t in set(lab_all) - set(med):
                    med[t] = int(np.flatnonzero(la == t)[0])
                fprobs = [dict(zip(("rectangle", "triangle", "circle"), q["floor"])) if q and q.get("floor") else None
                          for q in probs]
                voted = vote_types(vunits, la, med, irs, jprobs, fprobs, None if args.majority_vote else probs)
                for j, i in enumerate(keep):
                    label_rows[name].append({"truth": truth[j], "unit": labels_from_ir(irs[i]),
                                             "final": labels_from_ir(voted[i]), "id": r["id"], "stem": r["stem"],
                                             "unit_id": int(i), "box": r["units"][i]["box"],
                                             "type_id": int(lab_all[i]), "gt_group": r["units"][i]["group"]})
            lab = [lab_all[i] for i in keep]
            st = stats[name]
            for a, b in combinations(range(len(keep)), 2):
                same_t, same_g = lab[a] == lab[b], gt[a] == gt[b]
                st["tp"] += same_t and same_g
                st["fp"] += same_t and not same_g
                st["fn"] += same_g and not same_t
            if len(keep) >= 2:
                st["ari"].append(ari(gt, lab))
            st["exact"] += same_partition(gt, lab)
            st["types"] += len(set(lab))
    print(f"\n{len(rows)} complete photos, {n_units} balcony units, {n_gt_groups} GT groups")
    print(f"{'setup':10s} {'types':>6s} {'pairP':>7s} {'pairR':>7s} {'pairF1':>7s} {'ARI':>6s} {'exact':>7s}")
    for name, _ in setups:
        st = stats[name]
        p = st["tp"] / (st["tp"] + st["fp"]) if st["tp"] + st["fp"] else 0.0
        rc = st["tp"] / (st["tp"] + st["fn"]) if st["tp"] + st["fn"] else 0.0
        f = 2 * p * rc / (p + rc) if p + rc else 0.0
        print(f"{name:10s} {st['types']:6d} {p:7.3f} {rc:7.3f} {f:7.3f} {np.mean(st['ari']) if st['ari'] else 0:6.3f}"
              f" {st['exact']:3d}/{len(rows)}")
    if do_labels and args.dump:
        args.dump.write_text(json.dumps(label_rows), encoding="utf-8")
    if do_labels:
        heads = ("kind", "material", "floor", "structure")
        first = label_rows[setups[0][0]]
        print(chr(10) + "labels after the type vote, macro F1 / acc (n per head: "
              + ", ".join(f"{h} {sum(r['truth'][h] is not None for r in first)}" for h in heads) + ")")
        print(f"{'setup':10s} " + " ".join(f"{h:>15s}" for h in heads) + f" {'mean F1':>8s}")
        for name, stage in [(setups[0][0] + " novote", "unit")] + [(n, "final") for n, _ in setups]:
            rows_ = label_rows[setups[0][0]] if stage == "unit" else label_rows[name]
            ms = [head_metrics(rows_, stage, h) for h in heads]
            print(f"{name[:10]:10s} " + " ".join(f"{m['macro_f1']:7.3f} / {m['acc']:.3f}" for m in ms)
                  + f" {np.mean([m['macro_f1'] for m in ms]):8.3f}  mean acc {np.mean([m['acc'] for m in ms]):.3f}"
                  + "  n " + "/".join(str(m["n"]) for m in ms))
        print(chr(10) + "vote effect (fixed / broken; crops passing the condition at both stages)")
        for name, _ in setups:
            cells = []
            for h in heads:
                fx = br = 0
                for r in label_rows[name]:
                    t = r["truth"][h]
                    if t is None or conditional_skip(r, "unit", h) or conditional_skip(r, "final", h):
                        continue
                    u, f = r["unit"][h] == t, r["final"][h] == t
                    fx += (not u) and f
                    br += u and not f
                cells.append(f"{h} {fx}/{br}")
            print(f"  {name:10s} " + "   ".join(cells))


if __name__ == "__main__":
    main()
