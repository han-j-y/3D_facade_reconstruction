#!/usr/bin/env python3
"""Cache SAM3 window detections (low threshold) and object masks per facade, for offline parameter sweeps.

Per facade: ``<stem>.json`` with the ``detect_windows`` records of every --prompts entry at --threshold (same mask
clamp / size filter as the pipeline), and ``<stem>_masks.npz`` with the union mask of every --mask-prompts entry at
--mask-threshold (as ``detect_object_mask``).

    python scripts/cache_sam3_detections.py --out ../window_ast_predictor/runs/clean_eval/testset/sam3_cache/samoff
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import transformers  # noqa: E402

for cls in (transformers.Sam3Model, transformers.Sam3Processor):
    cls.from_pretrained = functools.lru_cache(maxsize=4)(cls.from_pretrained)

import run_pipeline as rp  # noqa: E402

FACADES = "b0001 b0005 b0010 b0020 b0031 b0050 b0071 b0083 b0125 b0150 b0170 b0198 b0215".split()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sam-model", default="facebook/sam3")
    ap.add_argument("--prompts", nargs="+", default=["window"])
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--min-side", type=int, default=13)
    ap.add_argument("--max-side-frac", type=float, default=0.55)
    ap.add_argument("--mask-prompts", nargs="+", default=["car", "storefront", "shop window", "shop"])
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--facades", nargs="+", default=FACADES)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    dirs = [ROOT.parent / "data" / "facades" / "base", ROOT.parent / "beyond-segmentation" / "data" / "x"]
    for fac in args.facades:
        fac = fac.removeprefix("cmp_")
        path = next(p for p in (d / f"cmp_{fac}.jpg" for d in dirs) if p.is_file())
        img = Image.open(path).convert("RGB")
        dets = {p: rp.detect_windows(img, prompt=p, threshold=args.threshold, min_side=args.min_side,
                                     max_side_frac=args.max_side_frac, device=dev, model_id=args.sam_model)
                for p in args.prompts}
        (args.out / f"cmp_{fac}.json").write_text(json.dumps(
            {"image": str(path), "image_size": list(img.size), "threshold": args.threshold, "detections": dets}))
        masks = {p: rp.detect_object_mask(img, prompts=[p], threshold=args.mask_threshold, device=dev)
                 for p in args.mask_prompts}
        np.savez_compressed(args.out / f"cmp_{fac}_masks.npz", **{p: np.packbits(m) for p, m in masks.items()},
                            shape=np.array(img.size[::-1]))
        print(fac, {p: len(d) for p, d in dets.items()})


if __name__ == "__main__":
    main()
