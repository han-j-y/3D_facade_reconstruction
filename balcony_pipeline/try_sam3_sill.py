r"""Option C: drop open-balcony boxes that SAM3 also finds as a "window sill".

Pipeline boxes: the test100 SAM3 ``filtered`` cache with rule 1 (``side_wall``)
applied, as ``run_balcony.py`` now does. SAM3 is run once more per photo with
the prompt ``window sill`` (score >= 0.30 cached with scores; mask clamp and
16 px minimum side as for balconies). Rule, fixed before looking at results: a
pipeline box is dropped when a sill box with score >= ``--sill-thr`` (0.45, the
balcony threshold) has IoU >= 0.5 with it, or covers at least 70% of its area.

Scored against ``runs/test100/balcony_gt_boxes.jsonl`` without ignore regions
(group rule, IoU 0.5 and 0.3). Images: ``--img-dir/<stem>.jpg``, left the
pipeline, right pipeline + C (removed boxes blue dashed, sill detections thin
yellow).

Example::

    C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_pipeline/try_sam3_sill.py --device cuda
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
PROMPT = "window sill"
CACHE_THR = 0.30
IOU_DROP = 0.5
COVER_DROP = 0.7


def detect_sills(stems: list[str], cache: Path, device: str) -> dict[str, list[dict]]:
    done: dict[str, list[dict]] = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else {}
    todo = [s for s in stems if s not in done]
    if not todo:
        return done
    import torch
    from transformers import Sam3Model, Sam3Processor

    import run_pipeline as rp

    dev = torch.device(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    sam = Sam3Model.from_pretrained("facebook/sam3").to(dev).eval()
    for k, s in enumerate(todo, 1):
        photo = Image.open(ROOT / "data" / "test100" / "photos" / f"{s}.jpg").convert("RGB")
        with torch.no_grad():
            recs = rp.detect_windows(photo, prompt=PROMPT, threshold=CACHE_THR, min_side=16, max_side_frac=1.0,
                                     device=dev, sam=sam, processor=processor)
        done[s] = [{"box": r["box_xyxy"], "score": round(r["score"], 4)} for r in recs]
        print(f"[{k}/{len(todo)}] {s} sills={len(recs)}", flush=True)
        cache.write_text(json.dumps(done), encoding="utf-8")
    return done


def cover(b, s) -> float:
    iw = max(0, min(b[2], s[2]) - max(b[0], s[0]))
    ih = max(0, min(b[3], s[3]) - max(b[1], s[1]))
    return iw * ih / max(1, (b[2] - b[0]) * (b[3] - b[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=T100 / "balcony_gt_boxes.jsonl")
    ap.add_argument("--cache", type=Path, default=ROOT / "runs" / "batch" / "_rows" / "test100_sam3_stage_boxes.json")
    ap.add_argument("--sill-cache", type=Path, default=T100 / "sam3_window_sill_boxes.json")
    ap.add_argument("--dsl-dir", type=Path, default=T100 / "batch_grid2")
    ap.add_argument("--depth-dir", type=Path, default=T100 / "da3_depth")
    ap.add_argument("--sill-thr", type=float, default=0.45)
    ap.add_argument("--img-dir", type=Path, default=T100 / "compare_sill")
    ap.add_argument("--out", type=Path, default=T100 / "balcony_sill_score.json")
    ap.add_argument("--panel-height", type=int, default=900)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    gt = {json.loads(l)["stem"]: json.loads(l) for l in args.gt.read_text(encoding="utf-8").splitlines() if l.strip()}
    stems = sorted(gt)
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    sills_all = detect_sills(stems, args.sill_cache, args.device)

    base: dict[str, list] = {}
    kept: dict[str, list] = {}
    removed: dict[str, list] = {}
    sills: dict[str, list] = {}
    for s in stems:
        dsl = json.loads((args.dsl_dir / s / "facade_dsl.json").read_text(encoding="utf-8"))
        dp = args.depth_dir / f"{s}_depth.npy"
        depth = np.load(dp) if dp.is_file() else None
        base[s], _ = apply_rules(cache[s]["filtered"], tuple(cache[s]["size"]), dsl, depth=depth, rules=("side_wall",))
        sills[s] = [d["box"] for d in sills_all[s] if d["score"] >= args.sill_thr]
        kept[s], removed[s] = [], []
        for b in base[s]:
            hit = any(box_iou(b, x) >= IOU_DROP or cover(b, x) >= COVER_DROP for x in sills[s])
            (removed if hit else kept)[s].append(b)

    subsets = {"all": lambda s: True, "cmp": lambda s: s.startswith("cmp_"), "lsaa": lambda s: not s.startswith("cmp_")}
    results: dict = {}
    per_photo: dict = {s: {} for s in stems}
    print(f"{'variant':10s} {'subset':5s} IoU   pred   TP   FP   FN     P      R     F1")
    for thr in (0.5, 0.3):
        for name, boxes in (("pipeline", base), ("+C", kept)):
            for sub, f in subsets.items():
                rows = []
                for s in stems:
                    if f(s):
                        sc = score_photo(gt[s]["boxes"], [], boxes[s], thr, True)
                        rows.append(sc)
                        if thr == 0.5:
                            per_photo[s][name] = sc
                t = prf(rows)
                results[f"{name}|{sub}|{thr}"] = t
                print(f"{name:10s} {sub:5s} {thr:<4}  {t['pred']:4d} {t['tp']:4d} {t['fp']:4d} {t['fn']:4d}  "
                      f"{t['precision']:.3f}  {t['recall']:.3f}  {t['f1']:.3f}")

    # which removals were real balconies (matched a GT box in the pipeline at IoU 0.5)
    real = 0
    rem_log = {}
    for s in stems:
        rows = []
        for b in removed[s]:
            r = any(box_iou(b, g) >= 0.5 for g in gt[s]["boxes"])
            real += r
            rows.append({"box": b, "real_balcony": r})
        if rows:
            rem_log[s] = rows
    nrem = sum(len(v) for v in removed.values())
    print(f"sill boxes >= {args.sill_thr}: {sum(len(v) for v in sills.values())}; removed {nrem} in {len(rem_log)} photos, "
          f"{real} real balconies")
    args.out.write_text(json.dumps({"sill_thr": args.sill_thr, "results": results, "photos": per_photo,
                                    "removed": rem_log}, indent=0), encoding="utf-8")

    args.img_dir.mkdir(parents=True, exist_ok=True)
    (args.img_dir / "_removed.json").write_text(json.dumps(rem_log, indent=1), encoding="utf-8")
    font = ImageFont.truetype("arial.ttf", 24)
    for s in stems:
        im0 = Image.open(ROOT / "data" / "test100" / "photos" / f"{s}.jpg").convert("RGB")
        W, H = im0.size
        lw = max(3, W // 200)
        left, right = im0.copy(), im0.copy()
        dl, dr = ImageDraw.Draw(left), ImageDraw.Draw(right)
        for b in base[s]:
            dl.rectangle(b, outline=(255, 0, 0), width=lw)
        for x in sills[s]:
            dr.rectangle(x, outline=(255, 220, 0), width=max(2, lw // 2))
        for b in kept[s]:
            dr.rectangle(b, outline=(255, 0, 0), width=lw)
        for b in removed[s]:
            dashed(dr, b, (0, 140, 255), lw)
        a, c = per_photo[s]["pipeline"], per_photo[s]["+C"]
        heads = [["Pipeline", f"TP {a['tp']} FP {a['fp']} FN {a['fn']}"],
                 ["Pipeline + C (SAM3 'window sill')",
                  f"TP {c['tp']} FP {c['fp']} FN {c['fn']} | removed {len(removed[s])} | sills {len(sills[s])}"]]
        scale = args.panel_height / H
        pw = max(1, round(W * scale))
        head = 64
        out = Image.new("RGB", (2 * pw + 12, args.panel_height + head), "white")
        do = ImageDraw.Draw(out)
        for k, (im, lines) in enumerate(zip((left, right), heads)):
            x = k * (pw + 12)
            out.paste(im.resize((pw, args.panel_height), Image.BILINEAR), (x, head))
            for j, line in enumerate(lines):
                do.text((x + 6, 4 + 28 * j), line, fill="black", font=font)
        out.save(args.img_dir / f"{s}.jpg", quality=86)
    print(f"images -> {args.img_dir}")


if __name__ == "__main__":
    main()
