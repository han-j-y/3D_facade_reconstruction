#!/usr/bin/env python3
"""End-to-end facade recovery: photo → window types → structure IR → DSL.

Pipeline (defaults)
  1. Detect   — facebook/sam3 ``window``, two-tier: score >= 0.7, plus [0.5, 0.7) boxes overlapping none of
                those; boxes on ``car`` masks dropped (or reuse ``--from-index``)
  2. Unitize  — ``--merge-mode ast`` (containment / IoU merges)
  3. Layout   — structural columns from box geometry (bay + colspan); legacy ``--layout-mode centroid``
  4. Cluster  — ``--cluster-mode gate``: z-scored box geometry (GMM), then the symmetry axis
                (``scripts/symmetry_axis_ghost_continuous.py``, ``open balcony`` detections break ties), then
                pane-tree structure splits inside each box group that keep mirror pairs together
  5. Doors    — facebook/sam3 ``door`` >= 0.45; side facades (edge columns off the main
                floors) and the doors beside them dropped; then (gate) local symmetries let windows join a
                mirror partner's type (``facade_recovery/local_sym.py``)
  6. Assetize — per-type crops; medoid exemplar
  7. Structure — predict window structure IR per unit; majority-vote within type
  8. DSL       — per-floor bays (``facade_recovery/floor_bays.py``) + type library (``facade_dsl.json``)
  9. (opt)     — Blender façade render via ``scripts/render_facade.py``

Majority vote uses a discrete ``structure_view`` fingerprint (shape + pane
topology + program ops). Continuous floats are ignored. Ties prefer the
type medoid.

Blender rendering is optional (``--blender-render``).

Examples::

  python run.py --facade-id 8 --device cuda
  python run.py --image photo.png --out-dir runs/demo --device cuda
  python run.py --facade-id 8 --blender-render --device cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import types
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from facade_recovery.box_cluster import (  # noqa: E402
    cluster_box_geometry_pipeline,
    cluster_within_groups,
)
from facade_recovery.cmp_windows import load_cmp_labelmap_windows, load_cmp_xml_windows  # noqa: E402
from facade_recovery.merge_guard import (  # noqa: E402
    filter_indices,
    spanning_box_indices,
    split_member_groups_by_floor,
)
from facade_recovery.column_layout import (  # noqa: E402
    assign_box_to_columns,
    assign_floors,
    infer_bay_column_bounds_from_units,
    infer_floor_row_bounds_from_units,
    infer_gap_norms,
    infer_inter_bay_gaps_robust,
    infer_inter_floor_gaps_robust,
    infer_proportional_floor_heights,
    infer_structural_columns,
    infer_window_bay_columns,
)
from facade_recovery.door_types import door_type_token
from facade_recovery.doors import (  # noqa: E402
    assign_doors_to_layout,
    cluster_door_units,
    filter_door_boxes,
)
from facade_recovery.floor_bays import apply_floor_bays, drop_side_facades  # noqa: E402
from facade_recovery.local_sym import OCCLUDER_PROMPTS, local_sym_vote  # noqa: E402
from facade_recovery.sym_gate import gate_cluster, mirror_partners, refine_axis, two_tier  # noqa: E402
from facade_recovery.paths import (  # noqa: E402
    default_structure_ckpt,
    default_train_up,
    resolve_blender,
    resolve_compiler_root,
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


base = _load(ROOT / "scripts" / "overlay_facade_asset_clusters.py", "e2e_base")
patch = _load(ROOT / "scripts" / "overlay_facade_patch_layout.py", "e2e_patch")
merge_mod = _load(ROOT / "scripts" / "overlay_facade_merge_boxes.py", "e2e_merge")
reemb = _load(ROOT / "scripts" / "overlay_facade_merge_reembed.py", "e2e_reemb")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--facade-id", type=str, default=None)
    ap.add_argument("--image", type=Path, default=None, help="facade image (else train_up/{id}.png)")
    ap.add_argument(
        "--train-up",
        type=Path,
        default=None,
        help="directory of {id}.png façades (default: data/facades/train_up)",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--from-index", type=Path, default=None, help="reuse SAM index.json")
    ap.add_argument(
        "--windows",
        choices=("sam", "xml", "labelmap"),
        default="sam",
        help="window boxes: SAM detect (default), CMP XML, or the CMP label map (<stem>.png, class 3) "
        "next to the image",
    )
    ap.add_argument(
        "--merge-mode",
        choices=("structural", "ast"),
        default="ast",
        help="unitize merge: e2e structural columns, or ast "
        "(scripts/overlay_facade_merge_boxes_ast.py: centroid bays + vertical same-stack merge)",
    )
    ap.add_argument(
        "--merge-adjacent",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also merge horizontally nearby same-floor panes (default: off; "
        "containment/IoU only)",
    )
    ap.add_argument(
        "--symmetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zhang-style L–R symmetry repair on type labels after clustering "
        "(default on; --no-symmetry to skip; not used by --cluster-mode gate, which splits under the axis)",
    )
    ap.add_argument(
        "--twin-mirror-min",
        type=float,
        default=0.35,
        help="min cluster mirror score to attempt twin MERGE in symmetry repair",
    )
    ap.add_argument(
        "--vert-gap",
        type=float,
        default=0.55,
        help="vertical merge gap (× median height) when --merge-mode ast",
    )
    ap.add_argument("--prompt", type=str, default="window")
    ap.add_argument(
        "--exclude-prompts",
        type=str,
        default="car,truck,van",
        help="comma SAM3 prompts for non-façade objects; windows/doors on them are dropped ('' = off)",
    )
    ap.add_argument("--exclude-threshold", type=float, default=0.5)
    ap.add_argument(
        "--exclude-min-cover",
        type=float,
        default=0.5,
        help="drop a window/door box when this fraction of it lies on an excluded-object mask",
    )
    ap.add_argument("--threshold", type=float, default=0.5, help="window detection score threshold (fill tier)")
    ap.add_argument(
        "--anchor-score",
        type=float,
        default=0.7,
        help="two-tier detection: keep windows scoring >= this, then add those in [--threshold, this) that overlap "
        "no kept box by --fill-max-overlap of the smaller box (0.5 alone over-merges two-pane windows, 0.6 alone "
        "drops windows). 0 = single threshold",
    )
    ap.add_argument("--fill-max-overlap", type=float, default=0.2)
    ap.add_argument(
        "--sam-model",
        default="facebook/sam3",
        help="SAM3 model (Hugging Face ID or directory) for window detection; default: facebook/sam3. "
        "Doors: --door-sam-model; --exclude-prompts also use facebook/sam3",
    )
    ap.add_argument(
        "--min-side",
        type=int,
        default=13,
        help="drop SAM detections with a side below this (px); 8 keeps slivers of neighbouring facades' windows at "
        "the image edge, which then take over real windows' floor/bay cells; 12 keeps the side pane of b0010's "
        "bay window",
    )
    ap.add_argument(
        "--min-rel-width",
        type=float,
        default=0.0,
        help="drop SAM window detections narrower than this x the facade's median window width (side panes of "
        "bay windows, slivers). 0.35 = 1st percentile of GT windows on the cmp_windows train split. 0 = off",
    )
    ap.add_argument("--max-side-frac", type=float, default=0.55)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dino", default="dinov2_vits14")
    ap.add_argument("--facade-max-side", type=int, default=896)
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-max", type=int, default=8, help="max k for appearance / final clustering")
    ap.add_argument(
        "--cluster-mode",
        choices=("gate", "shape_then_feat", "shape_then_struct", "struct", "box", "appearance"),
        default="gate",
        help="type clustering: gate (default) = z-scored box GMM, then the symmetry axis "
        "(scripts/symmetry_axis_ghost_continuous.py), then structure splits inside each box group that keep "
        "mirror pairs together (facade_recovery/sym_gate.py; needs a pane-tree --structure-ckpt); "
        "shape_then_feat = box GMM then DINO splits; shape_then_struct = box GMM then splits by the pane-tree "
        "root embedding; struct = the root embedding alone over the whole facade; box-only; or legacy "
        "appearance+Potts",
    )
    ap.add_argument(
        "--gate-eps",
        type=float,
        default=0.1,
        help="gate: refuse a split that separates more than this fraction of the node's mirror pairs",
    )
    ap.add_argument(
        "--gate-feature",
        choices=("z", "heads"),
        default="heads",
        help="gate: pane-tree descriptor that is cut at --struct-split-thr",
    )
    ap.add_argument(
        "--axis-refine",
        type=float,
        default=0.02,
        help="gate: shift the symmetry axis within +-this x image width to pair the most windows",
    )
    ap.add_argument(
        "--balcony-prompt",
        default="open balcony",
        help="gate: SAM3 prompt whose detections break ties between window-symmetry peaks ('' = off)",
    )
    ap.add_argument("--balcony-min-score", type=float, default=0.5)
    ap.add_argument(
        "--grid",
        choices=("floor_bays", "columns"),
        default="floor_bays",
        help="DSL layout: per-floor bays merged where lines match across floors (facade_recovery/floor_bays.py, "
        "default) or one column grid for the whole facade (--layout-mode)",
    )
    ap.add_argument(
        "--side-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="drop window columns at the image edges that sit off the main facade's floors (neighbouring "
        "buildings, side walls), and doors outside the kept windows",
    )
    ap.add_argument(
        "--local-sym",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="gate: after the side filter, let windows join a mirror partner's type under local symmetries "
        "(facade_recovery/local_sym.py; joins only, occluder-aware)",
    )
    ap.add_argument(
        "--local-sym-occluders",
        default=",".join(OCCLUDER_PROMPTS),
        help="local symmetry: comma SAM3 prompts whose masks hide window sides ('' = border only)",
    )
    ap.add_argument("--local-sym-occluder-threshold", type=float, default=0.4)
    ap.add_argument(
        "--struct-split-thr",
        type=float,
        default=0.16,
        help="shape_then_struct / struct: average-linkage cosine-distance cut on the root embedding within a box "
        "group / over the facade (0.16 separates same / different labelled structures best within a facade; "
        "0.2 for struct on the e2e test windows)",
    )
    ap.add_argument(
        "--struct-feature",
        choices=("z", "heads"),
        default="z",
        help="shape_then_struct / struct descriptor: pane-tree root embedding (z) or its predicted structure "
        "probabilities (heads; less sensitive to appearance, use with --struct-split-thr 0.2)",
    )
    ap.add_argument("--k-max-box", type=int, default=6, help="max k for stage-1 box geometry clustering")
    ap.add_argument(
        "--k-max-within",
        type=int,
        default=4,
        help="max k per box group when splitting on appearance (shape_then_feat)",
    )
    ap.add_argument(
        "--box-method",
        default="gmm_diag",
        choices=("gmm_diag", "spectral_rbf", "agglo_ward", "spectral_nn", "kmeans"),
        help="stage-1 clustering method on z-scored (w, h, aspect)",
    )
    ap.add_argument(
        "--box-conservative-k",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="lower silhouette k when box cluster medians are within jitter tol",
    )
    ap.add_argument("--box-conservative-k-wh-tol", type=float, default=0.008)
    ap.add_argument(
        "--box-merge-near",
        action="store_true",
        help="merge box clusters with near-identical median (w, h) after GMM",
    )
    ap.add_argument(
        "--box-snap-wh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reassign units to nearest cluster by raw (w, h) medians",
    )
    ap.add_argument(
        "--blender-render",
        action="store_true",
        help="OPTIONAL: Blender façade render after DSL (needs Blender + FACADE_COMPILER_ROOT)",
    )
    ap.add_argument(
        "--blender",
        default=None,
        help="Blender binary (optional; default: $BLENDER or blender on PATH)",
    )
    ap.add_argument(
        "--compiler-root",
        type=Path,
        default=None,
        help="optional window compiler package with main.py (or set FACADE_COMPILER_ROOT)",
    )
    ap.add_argument("--col-tol", type=float, default=0.045)
    ap.add_argument("--row-tol", type=float, default=0.055)
    ap.add_argument(
        "--layout-mode",
        choices=("structural", "centroid"),
        default="structural",
        help="bay layout: structural columns from box geometry (default) or legacy centroid",
    )
    ap.add_argument("--spatial-strength", type=float, default=1.8)
    ap.add_argument("--unary-weight", type=float, default=0.9)
    ap.add_argument(
        "--structure-ckpt",
        type=Path,
        default=None,
        help="window structure AST checkpoint (default: best available under runs/)",
    )
    ap.add_argument("--structure-only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--structure-adapter",
        type=Path,
        default=None,
        help="optional real-to-synthetic feature adapter for a pane-tree --structure-ckpt",
    )
    ap.add_argument(
        "--tree-root",
        type=Path,
        default=ROOT,
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--skip-structure", action="store_true")
    ap.add_argument(
        "--detect-doors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SAM-detect ground doors and add to façade DSL (default on)",
    )
    ap.add_argument("--door-prompt", type=str, default="door")
    ap.add_argument(
        "--door-sam-model",
        default="facebook/sam3",
        help="SAM3 model (Hugging Face ID or directory) for door detection; default: facebook/sam3",
    )
    ap.add_argument("--door-threshold", type=float, default=0.45)
    ap.add_argument(
        "--shop-as-window",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also SAM-detect --shop-prompt with --shop-sam-model and add the boxes as windows (ground floor only, "
        "not overlapping a window or a door detection)",
    )
    ap.add_argument("--shop-prompt", type=str, default="shop")
    ap.add_argument(
        "--shop-sam-model",
        default=None,
        help="SAM3 model for --shop-as-window (default: --sam-model); needs a model trained on CMP 'shop'",
    )
    ap.add_argument(
        "--shop-min-y-frac",
        type=float,
        default=0.55,
        help="keep shop boxes whose center is below this image-height fraction",
    )
    ap.add_argument(
        "--door-min-y-frac",
        type=float,
        default=0.55,
        help="keep doors whose center is below this image-height fraction",
    )
    ap.add_argument(
        "--door-iou-nms",
        type=float,
        default=0.35,
        help="drop door if IoU with any window unit exceeds this",
    )
    ap.add_argument(
        "--structure-vote",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="predict IR on all units in a type and majority-vote (default on)",
    )
    ap.add_argument(
        "--structure-pad",
        type=float,
        default=0.0,
        help="margin per side (fraction of the window box) added around each unit before structure "
        "prediction; assets stay tight",
    )
    ap.add_argument(
        "--structure-glass-margin",
        type=str,
        default="",
        help="pane-tree only: 'l,t,r,b' margins (fractions of the window box) of the context crop; the "
        "unit box is passed as the glass box, as in training on base-photo crops (e.g. 0.15,0.075,0.15,0.10)",
    )
    ap.add_argument(
        "--structure-joint",
        choices=["off", "uniform", "area"],
        default="off",
        help="pane-tree only: decode one structure per type with node logits averaged over all members "
        "(uniform or box-area weights) instead of a majority vote; per-member predictions are still "
        "stored and compared against it",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Detect
# ---------------------------------------------------------------------------


@torch.no_grad()
def detect_windows(
    image: Image.Image,
    *,
    prompt: str,
    threshold: float,
    min_side: int,
    max_side_frac: float,
    device: torch.device,
    model_id: str = "facebook/sam3",
) -> list[dict[str, Any]]:
    from transformers import Sam3Model, Sam3Processor

    iw, ih = image.size
    print(f"loading SAM3 ({model_id})…")
    processor = Sam3Processor.from_pretrained(model_id)
    sam = Sam3Model.from_pretrained(model_id).to(device)
    sam.eval()
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    # The original SAM3 always appends the geometry encoder's CLS token to the text prompt; HF skips the geometry
    # encoder without boxes. A single padding box (label -10) runs it with only the CLS token valid. Fine-tuned
    # checkpoints depend on that token; the released weights also detect differently with it (~5% more windows
    # on the facade_e2e test facades).
    inputs["input_boxes"] = torch.tensor([[[0.5, 0.5, 0.1, 0.1]]], device=device)
    inputs["input_boxes_labels"] = torch.tensor([[-10]], device=device)
    outputs = sam(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=0.5,
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]
    boxes = results["boxes"].detach().float().cpu().numpy()
    scores = results["scores"].detach().float().cpu().numpy()
    masks = results.get("masks")
    if masks is not None:
        if hasattr(masks, "detach"):
            masks = masks.detach().float().cpu().numpy()
        else:
            masks = np.asarray(masks)
    print(f"SAM3 raw detections: {len(boxes)}  masks={'yes' if masks is not None else 'no'}")
    del sam, outputs, inputs
    torch.cuda.empty_cache()

    records = []
    n_mask_box = 0
    n_clamped = 0
    for i, (box, sc) in enumerate(zip(boxes, scores)):
        det = [float(v) for v in box]
        x0, y0, x1, y1 = det
        used_mask = False
        fill = None
        mask_xyxy = None
        if masks is not None and i < len(masks):
            m = masks[i]
            if m.ndim == 3:
                m = m[0]
            ys, xs = np.where(m > 0.5)
            if xs.size >= 16:
                mx0, mx1 = float(xs.min()), float(xs.max() + 1)
                my0, my1 = float(ys.min()), float(ys.max() + 1)
                fill = float(xs.size) / max(1.0, (mx1 - mx0) * (my1 - my0))
                mask_xyxy = [mx0, my0, mx1, my1]
                # Clamp mask AABB to the detector box so mask bleed cannot inflate it.
                cx0, cy0 = max(det[0], mx0), max(det[1], my0)
                cx1, cy1 = min(det[2], mx1), min(det[3], my1)
                if cx1 - cx0 >= 8 and cy1 - cy0 >= 8:
                    if (cx0, cy0, cx1, cy1) != (mx0, my0, mx1, my1):
                        n_clamped += 1
                    x0, y0, x1, y1 = cx0, cy0, cx1, cy1
                    used_mask = True
                    n_mask_box += 1
                else:
                    x0, y0, x1, y1 = det
        bw, bh = x1 - x0, y1 - y0
        if bw < min_side or bh < min_side:
            continue
        if bw > max_side_frac * iw or bh > max_side_frac * ih:
            continue
        rec: dict[str, Any] = {
            "idx": len(records),
            "score": float(sc),
            "box_xyxy": [int(x0), int(y0), int(x1), int(y1)],
            "box_det_xyxy": [int(det[0]), int(det[1]), int(det[2]), int(det[3])],
            "from_mask": used_mask,
        }
        if mask_xyxy is not None:
            rec["box_mask_xyxy"] = [int(v) for v in mask_xyxy]
        if fill is not None:
            rec["mask_fill"] = round(fill, 4)
        records.append(rec)
    print(
        f"after size filter: {len(records)}  "
        f"mask∩det={n_mask_box}  clamped_bleed={n_clamped}"
    )
    return records


def _overlap(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Intersection as a fraction of a, of b, and IoU."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    aa = max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))
    ab = max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / aa, inter / ab, inter / (aa + ab - inter)


def shop_windows(
    shops: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    doors: list[dict[str, Any]],
    *,
    image_height: int,
    min_y_frac: float,
    max_cover: float = 0.3,
    max_door_iou: float = 0.3,
    aspect: tuple[float, float] = (0.25, 4.0),
) -> list[dict[str, Any]]:
    """Shop detections to add as windows: ground floor, glazing-like aspect (h/w), not covering / covered by a window
    (> max_cover of either box), not a door (IoU > max_door_iou); greedy by score among themselves."""
    kept: list[dict[str, Any]] = []
    for s in sorted(shops, key=lambda r: -r["score"]):
        b = s["box_xyxy"]
        w, h = b[2] - b[0], b[3] - b[1]
        if (b[1] + b[3]) / 2 < min_y_frac * image_height or not aspect[0] <= h / max(w, 1) <= aspect[1]:
            continue
        if any(max(_overlap(b, o["box_xyxy"])[:2]) > max_cover for o in windows + kept):
            continue
        if any(_overlap(b, d["box_xyxy"])[2] > max_door_iou for d in doors):
            continue
        kept.append({**s, "from_prompt": "shop"})
    return kept


@torch.no_grad()
def detect_object_mask(
    image: Image.Image,
    *,
    prompts: list[str],
    threshold: float,
    device: torch.device,
) -> np.ndarray | None:
    """Union of SAM3 instance masks for non-façade objects (e.g. ``car``)."""
    if not prompts:
        return None
    from transformers import Sam3Model, Sam3Processor

    iw, ih = image.size
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    sam = Sam3Model.from_pretrained("facebook/sam3").to(device)
    sam.eval()
    union = np.zeros((ih, iw), dtype=bool)
    for prompt in prompts:
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        outputs = sam(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]
        masks = results.get("masks")
        n = 0
        if masks is not None:
            if hasattr(masks, "detach"):
                masks = masks.detach().float().cpu().numpy()
            for m in np.asarray(masks):
                if m.ndim == 3:
                    m = m[0]
                union |= m > 0.5
                n += 1
        print(f"SAM3 exclude '{prompt}': {n} instances")
    del sam
    torch.cuda.empty_cache()
    return union


def drop_boxes_on_mask(
    records: list[dict[str, Any]],
    mask: np.ndarray | None,
    *,
    min_cover: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into (kept, dropped) by the fraction of each box on ``mask``."""
    if mask is None or not mask.any():
        return list(records), []
    kept, dropped = [], []
    for r in records:
        x0, y0, x1, y1 = [int(v) for v in r["box_xyxy"]]
        patch = mask[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
        cover = float(patch.mean()) if patch.size else 0.0
        if cover >= min_cover:
            dropped.append({**r, "exclude_cover": round(cover, 3)})
        else:
            kept.append(r)
    for i, r in enumerate(kept):
        r["idx"] = i
    return kept, dropped


def narrow_box_keep(boxes: list[list[float]], rel_width: float) -> list[bool]:
    """False for boxes narrower than ``rel_width`` x the median box width (bay-window sides, cut-off slivers)."""
    if rel_width <= 0 or len(boxes) < 3:
        return [True] * len(boxes)
    med = float(np.median([b[2] - b[0] for b in boxes]))
    return [(b[2] - b[0]) >= rel_width * med for b in boxes]


def drop_narrow_boxes(
    records: list[dict[str, Any]], rel_width: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keep = narrow_box_keep([r["box_xyxy"] for r in records], rel_width)
    kept = [r for r, k in zip(records, keep) if k]
    dropped = [r for r, k in zip(records, keep) if not k]
    for i, r in enumerate(kept):
        r["idx"] = i
    return kept, dropped


def draw_excluded(
    image: Image.Image,
    mask: np.ndarray,
    dropped: list[dict[str, Any]],
    kept: list[dict[str, Any]],
) -> Image.Image:
    """Tint excluded-object mask red; dropped boxes red, kept boxes green."""
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    rgb[mask] = 0.5 * rgb[mask] + 0.5 * np.array([220.0, 40.0, 40.0])
    out = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(out)
    for r in kept:
        d.rectangle(r["box_xyxy"], outline=(40, 220, 90), width=2)
    for r in dropped:
        d.rectangle(r["box_xyxy"], outline=(255, 30, 30), width=4)
    return out


def load_from_index(index_path: Path) -> tuple[Image.Image, list[dict[str, Any]], dict]:
    index = json.loads(index_path.read_text())
    facade = Image.open(index["facade_path"]).convert("RGB")
    windows = []
    for i, w in enumerate(index["windows"]):
        box = [int(v) for v in w["box_xyxy"]]
        det = w.get("box_det_xyxy")
        mask_box = w.get("box_mask_xyxy")
        # Re-apply mask∩det clamp when older indices stored the inflated mask AABB.
        if det is not None:
            d = [int(v) for v in det]
            src = [int(v) for v in (mask_box or box)]
            x0, y0 = max(d[0], src[0]), max(d[1], src[1])
            x1, y1 = min(d[2], src[2]), min(d[3], src[3])
            if x1 - x0 >= 8 and y1 - y0 >= 8:
                box = [x0, y0, x1, y1]
            else:
                box = d
        raw_id = w.get("idx", w.get("id", i))
        try:
            wid = int(raw_id)
        except (TypeError, ValueError):
            wid = int(i)
        windows.append(
            {
                "idx": wid,
                "score": float(w.get("score", 1.0)),
                "box_xyxy": box,
                "box_det_xyxy": [int(v) for v in det] if det is not None else box,
                "from_mask": bool(w.get("from_mask", False)),
                "mask_fill": w.get("mask_fill"),
            }
        )
    return facade, windows, index


# ---------------------------------------------------------------------------
# Cluster + assetize
# ---------------------------------------------------------------------------


def cluster_units(
    facade: Image.Image,
    boxes: list[list[int]],
    *,
    model,
    device: torch.device,
    facade_max_side: int,
    pca_dim: int,
    seed: int,
    k_max: int,
    col_tol: float,
    row_tol: float,
    spatial_strength: float,
    unary_weight: float,
    layout_mode: str = "structural",
    merge_mode: str = "structural",
    merge_adjacent: bool = False,
    vert_gap: float = 0.55,
    cluster_mode: str = "shape_then_feat",
    k_max_box: int = 6,
    k_max_within: int = 4,
    box_method: str = "gmm_diag",
    box_conservative_k: bool = True,
    box_conservative_k_wh_tol: float = 0.008,
    box_merge_near: bool = False,
    box_snap_wh: bool = True,
    struct_describe=None,
    struct_split_thr: float = 0.16,
) -> dict[str, Any]:
    iw, ih = facade.size
    work_boxes = [list(b) for b in boxes]
    cx = np.array([0.5 * (b[0] + b[2]) / iw for b in work_boxes], dtype=np.float64)
    cy = np.array([0.5 * (b[1] + b[3]) / ih for b in work_boxes], dtype=np.float64)

    floors_raw = assign_floors(cy, row_tol)
    spanning = spanning_box_indices(work_boxes, floors_raw)
    if spanning:
        keep = [i for i in range(len(work_boxes)) if i not in set(spanning)]
        print(f"drop {len(spanning)} box(es) that span multiple floors")
        work_boxes = filter_indices(work_boxes, keep)
        cx = cx[keep]
        cy = cy[keep]
        floors_raw = assign_floors(cy, row_tol)

    columns: list[tuple[float, float]] | None = None
    structural_col: np.ndarray | None = None
    bay_raw: np.ndarray
    if merge_mode == "ast":
        ast_merge = _load(
            ROOT / "scripts" / "overlay_facade_merge_boxes_ast.py",
            "ast_merge",
        )
        merged_boxes, members, bay_raw, _ = ast_merge.merge_adjacent_boxes(
            work_boxes,
            cx,
            cy,
            row_tol=row_tol,
            adj_gap=1.0,
            merge_bays=False,
            col_tol=col_tol,
            merge_vertical=False,
            adjacent=merge_adjacent,
        )
    else:
        if layout_mode == "structural":
            columns = infer_structural_columns(work_boxes, floors_raw, iw=iw)
            structural_col = np.array(
                [assign_box_to_columns(b, columns)[0] for b in work_boxes],
                dtype=np.int32,
            )
        merged_boxes, members, bay_raw, _ = merge_mod.merge_adjacent_boxes(
            work_boxes,
            cx,
            cy,
            row_tol=row_tol,
            adj_gap=1.0,
            merge_bays=False,
            col_tol=col_tol,
            structural_col=structural_col,
            adjacent=merge_adjacent,
        )
    merged_boxes, members = split_member_groups_by_floor(
        work_boxes, merged_boxes, members, floors_raw
    )
    n_m = len(merged_boxes)
    mcy = np.array([0.5 * (b[1] + b[3]) / ih for b in merged_boxes], dtype=np.float64)
    floor_m = assign_floors(mcy, row_tol)

    bay_u: list[int] = []
    colspan_u: list[int] = []
    if layout_mode == "structural":
        temp_units = [
            {
                "floor": int(floor_m[i]),
                "box_xyxy": merged_boxes[i],
                "kind": "window",
                "colspan": 1,
            }
            for i in range(n_m)
        ]
        columns = infer_window_bay_columns(temp_units, iw=iw)
        for u in temp_units:
            bay_u.append(int(u["bay"]))
            colspan_u.append(int(u.get("colspan", 1)))
    else:
        columns = None
        for mem in members:
            labs = [int(bay_raw[i]) for i in mem]
            bay_u.append(max(set(labs), key=labs.count))
        colspan_u = [1] * n_m

    spatial, meta = patch.facade_patch_spatial(
        model, facade, device=device, max_side=facade_max_side
    )
    feats_raw = patch.roi_pool_patches(spatial, merged_boxes, meta)
    feats_pca = base.apply_pca(feats_raw, min(pca_dim, max(2, n_m - 1)), seed)
    labels_box: np.ndarray | None = None
    box_groups: list[dict[str, Any]] | None = None

    if n_m < 2:
        labels = np.zeros(n_m, dtype=np.int32)
        labels_box = labels.copy()
    elif cluster_mode == "appearance":
        k = base.select_k(feats_pca, "spectral_rbf", min(k_max, n_m - 1), seed)
        k = max(2, min(k, n_m - 1))
        labels0 = base.cluster_features(feats_pca, "spectral_rbf", k, seed)
        W = reemb.spatial_affinity_units(
            merged_boxes,
            bay_u,
            floor_m,
            iw=iw,
            ih=ih,
            col_tol=col_tol,
            row_tol=row_tol,
        )
        labels = reemb.neighbor_consistency_labels(
            labels0,
            W,
            feats_pca,
            strength=spatial_strength,
            iters=12,
            unary_weight=unary_weight,
        )
    elif cluster_mode == "box":
        labels_box, _ = cluster_box_geometry_pipeline(
            merged_boxes,
            (iw, ih),
            base,
            method=box_method,
            k_max=k_max_box,
            seed=seed,
            conservative_k=box_conservative_k,
            conservative_k_wh_tol=box_conservative_k_wh_tol,
            merge_near=box_merge_near,
            snap_wh=box_snap_wh,
        )
        labels = labels_box
    elif cluster_mode == "shape_then_struct":
        if struct_describe is None:
            raise ValueError("cluster_mode shape_then_struct needs struct_describe")
        labels_box, _ = cluster_box_geometry_pipeline(
            merged_boxes,
            (iw, ih),
            base,
            method=box_method,
            k_max=k_max_box,
            seed=seed,
            conservative_k=box_conservative_k,
            conservative_k_wh_tol=box_conservative_k_wh_tol,
            merge_near=box_merge_near,
            snap_wh=box_snap_wh,
        )
        labels, box_groups = split_groups_by_structure(
            struct_describe(merged_boxes), labels_box, thr=struct_split_thr
        )
    elif cluster_mode == "struct":
        if struct_describe is None:
            raise ValueError("cluster_mode struct needs struct_describe")
        labels, box_groups = split_groups_by_structure(
            struct_describe(merged_boxes), np.zeros(n_m, dtype=np.int32), thr=struct_split_thr
        )
    else:
        labels_box, _ = cluster_box_geometry_pipeline(
            merged_boxes,
            (iw, ih),
            base,
            method=box_method,
            k_max=k_max_box,
            seed=seed,
            conservative_k=box_conservative_k,
            conservative_k_wh_tol=box_conservative_k_wh_tol,
            merge_near=box_merge_near,
            snap_wh=box_snap_wh,
        )
        labels, box_groups = cluster_within_groups(
            feats_pca,
            labels_box,
            base,
            k_max_within=k_max_within,
            seed=seed,
        )

    return {
        "merged_boxes": merged_boxes,
        "members": members,
        "bay": bay_u,
        "colspan": colspan_u,
        "floor": [int(f) for f in floor_m.tolist()],
        "labels": labels,
        "labels_box": labels_box,
        "box_groups": box_groups,
        "feats": feats_pca,
        "feats_raw": feats_raw,
        "spatial": spatial,
        "patch_meta": meta,
        "columns": columns,
        "layout_mode": layout_mode,
        "merge_mode": merge_mode,
        "merge_adjacent": merge_adjacent,
        "cluster_mode": cluster_mode,
    }


def split_groups_by_structure(
    emb: np.ndarray, box_labels: np.ndarray, *, thr: float
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Split each box-geometry group by average-linkage clustering of structure embeddings (cosine)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    labels = np.zeros(len(emb), dtype=np.int32)
    info: list[dict[str, Any]] = []
    nxt = 0
    for g in sorted(set(int(x) for x in box_labels.tolist())):
        idx = np.where(box_labels == g)[0]
        if len(idx) == 1:
            sub = np.zeros(1, dtype=np.int64)
        else:
            d = pdist(emb[idx], metric="cosine")
            sub = fcluster(linkage(d, method="average"), t=thr, criterion="distance") - 1
        remap = {s: nxt + k for k, s in enumerate(sorted(set(sub.tolist())))}
        nxt += len(remap)
        labels[idx] = [remap[s] for s in sub.tolist()]
        info.append({"box_cluster": g, "n": int(len(idx)), "k_within": len(remap),
                     "split_reason": f"structure embedding, average linkage, cosine cut {thr}",
                     "indices": [int(i) for i in idx], "sub_labels": [int(s) for s in sub.tolist()]})
    return labels, info


def pick_medoids(feats: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    """Return type_id → unit index of cosine medoid."""
    medoids = {}
    for tid in sorted(set(int(x) for x in labels.tolist())):
        idxs = [i for i, lab in enumerate(labels.tolist()) if int(lab) == tid]
        if len(idxs) == 1:
            medoids[tid] = idxs[0]
            continue
        sub = feats[idxs]
        sub = sub / np.clip(np.linalg.norm(sub, axis=1, keepdims=True), 1e-8, None)
        sim = sub @ sub.T
        # medoid = max average similarity to others
        scores = sim.mean(axis=1)
        medoids[tid] = idxs[int(np.argmax(scores))]
    return medoids


def apply_symmetry_repair(
    facade: Image.Image,
    boxes: list[list[int]],
    labels: np.ndarray,
    feats: np.ndarray,
    *,
    labels_box: np.ndarray | None = None,
    twin_mirror_min: float = 0.35,
    match_iou: float = 0.15,
    size_mismatch_sym_gain: float = 0.004,
) -> tuple[np.ndarray, list[str]]:
    """L–R facade symmetry repair (twin MERGE + leftover STEAL/JOIN).

    When ``labels_box`` is set, flips that would mix distinct box-geometry
    groups are reverted (same gate as ``cluster_hierarchy.symmetry_repair_box_gated``).
    """
    if len(boxes) < 2:
        return labels.astype(np.int32), ["symmetry: skipped (<2 units)"]

    walk = _load(
        ROOT / "scripts" / "walkthrough_symmetry_repair.py",
        "e2e_sym_walk",
    )
    boxes_f = [[float(v) for v in b] for b in boxes]
    labels_before = labels.astype(np.int32).copy()
    cx, cy, facade_extent = walk.facade_axis_center(
        boxes_f, extent_mode="boxes", image_size=facade.size
    )
    windows = [
        {"id": f"u{i:03d}", "box_xyxy": boxes_f[i], "cluster": int(labels_before[i])}
        for i in range(len(boxes_f))
    ]
    labels_after, log, _panels = walk.repair_all_clusters(
        boxes=boxes_f,
        labels=[int(x) for x in labels_before.tolist()],
        feats=feats,
        windows=windows,
        facade=facade,
        axis="v",
        cx=cx,
        cy=cy,
        facade_extent=facade_extent,
        img_size=facade.size,
        extent_mode="boxes",
        twin_mirror_min=twin_mirror_min,
        match_iou=match_iou,
        size_mismatch_sym_gain=size_mismatch_sym_gain,
    )
    out = np.asarray(labels_after, dtype=np.int32)

    if labels_box is not None and len(labels_box) == len(out):
        cluster_box_mode: dict[int, int] = {}
        for lab in sorted(set(int(x) for x in out.tolist())):
            idxs = [i for i, x in enumerate(out.tolist()) if int(x) == lab]
            modes = [int(labels_box[i]) for i in idxs]
            cluster_box_mode[lab] = max(set(modes), key=modes.count)
        reverted = 0
        for i in range(len(out)):
            if int(out[i]) == int(labels_before[i]):
                continue
            new_c = int(out[i])
            if cluster_box_mode.get(new_c) != int(labels_box[i]):
                out[i] = labels_before[i]
                reverted += 1
        if reverted:
            log.append(f"box-gate: reverted {reverted} symmetry flips across box groups")

    uniq = sorted(set(int(x) for x in out.tolist()))
    remap = {u: i for i, u in enumerate(uniq)}
    out = np.asarray([remap[int(x)] for x in out.tolist()], dtype=np.int32)
    n_before = len(set(int(x) for x in labels_before.tolist()))
    n_after = len(uniq)
    changed = sum(
        1
        for i in range(len(labels_before))
        if int(labels_after[i]) != int(labels_before[i])
    )
    log.append(f"symmetry: types {n_before}→{n_after}  label_moves={changed}")
    return out, log


class StructurePredictor:
    """Load window-AST once; predict structure IR for crops."""

    def __init__(
        self,
        ckpt_path: Path,
        device: torch.device,
        *,
        structure_only: bool = True,
    ) -> None:
        from window_ast.dataset import make_eval_transform
        from window_ast.model import WindowAstModel, resolve_backbone
        from window_ast.schema import Vocab
        from window_ast.structure import structure_tokens_to_ir

        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        from facade_recovery.paths import default_vocab

        vocab_path = ckpt_path.parent / "vocab.json"
        if not vocab_path.is_file():
            vocab_path = default_vocab()
        if not vocab_path.is_file():
            raise FileNotFoundError(f"no vocab for {ckpt_path}")
        self.vocab = Vocab.load(vocab_path)
        cfg = ckpt.get("args") or {}
        self.structure_only = structure_only or bool(cfg.get("structure_only"))
        if not self.structure_only:
            try:
                from window_ast.polygon import tokens_to_ir as parse_fn
            except ModuleNotFoundError:
                print("warn: window_ast.polygon missing; forcing structure_only parse")
                self.structure_only = True
                parse_fn = structure_tokens_to_ir
        else:
            parse_fn = structure_tokens_to_ir
        self.parse_fn = parse_fn
        self.device = device
        backbone = resolve_backbone(cfg, ckpt.get("model"))
        self.model = WindowAstModel(
            len(self.vocab),
            d_model=int(cfg.get("d_model", 256)),
            nhead=int(cfg.get("nhead", 8)),
            num_layers=int(cfg.get("num_layers", 4)),
            dim_feedforward=int(cfg.get("dim_feedforward", 512)),
            dropout=float(cfg.get("dropout", 0.1)),
            max_len=int(cfg.get("max_len", 160)),
            pad_id=self.vocab.pad_id,
            pretrained_encoder=False,
            backbone=backbone,
            freeze_backbone=True,
        ).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.tfm = make_eval_transform(int(cfg.get("image_size", 224)))
        print(f"structure backbone={backbone}")

    @torch.no_grad()
    def predict(self, crop_path: Path) -> dict[str, Any]:
        image = self.tfm(Image.open(crop_path).convert("RGB")).unsqueeze(0).to(self.device)
        ids = self.model.generate(
            image, bos_id=self.vocab.bos_id, eos_id=self.vocab.eos_id
        )[0].tolist()
        tokens = self.vocab.decode(ids, skip_special=True)
        try:
            ir = self.parse_fn(tokens)
        except Exception as exc:
            return {"tokens": tokens, "parse_error": str(exc)}
        return {"tokens": tokens, "ir": ir}


class _ForeignPackages:
    """Swap ``window_ast`` / ``scripts`` in ``sys.modules`` for another project's packages
    (both projects use those top-level names); the foreign modules stay cached between uses."""

    NAMES = ("window_ast", "scripts")

    def __init__(self, root: Path) -> None:
        self.root = root
        self.cache: dict[str, Any] = {}

    def _take(self) -> dict[str, Any]:
        return {k: sys.modules.pop(k) for k in list(sys.modules) if k.split(".")[0] in self.NAMES}

    def __enter__(self) -> "_ForeignPackages":
        self.saved = self._take()
        sys.modules.update(self.cache)
        sys.path.insert(0, str(self.root))
        return self

    def __exit__(self, *exc: Any) -> None:
        sys.path.remove(str(self.root))
        self.cache = self._take()
        sys.modules.update(self.saved)


def _binarize_panes(panes: dict[str, Any]) -> dict[str, Any]:
    """Rewrite n-ary splits as right-nested binary splits (the vendored compiler takes 2 children)."""
    rules = panes["rules"]
    out: dict[str, Any] = {}

    def go(nid: str) -> None:
        r = rules[nid]
        kids = list(r.get("children") or [])
        if r.get("op") == "split" and len(kids) > 2:
            at = r.get("at")
            cuts = list(at) if isinstance(at, (list, tuple)) and len(at) == len(kids) - 1 else \
                [i / len(kids) for i in range(1, len(kids))]
            cur, lo = nid, 0.0
            for i in range(len(kids) - 1):
                right = kids[-1] if i == len(kids) - 2 else f"{nid}_n{i + 1}"
                rel = (cuts[i] - lo) / max(1.0 - lo, 1e-6)
                out[cur] = {"op": "split", "axis": r["axis"], "children": [kids[i], right],
                            "at": round(min(0.999, max(0.001, rel)), 4)}
                cur, lo = right, cuts[i]
        else:
            out[nid] = dict(r)
        for c in kids:
            go(c)

    go(panes.get("root", "root"))
    return {**panes, "rules": out}


class TreeStructurePredictor:
    """Bundled pane-tree model with an optional feature adapter.

    Crops are letterboxed onto the render background as in adapter training. Pane cut
    positions from the line-profile head are kept as ``at`` in the IR."""

    def __init__(self, ckpt_path: Path, device: torch.device, *, adapter_path: Path | None = None,
                 tree_root: Path, place_scale: float = 0.68) -> None:
        from window_ast.dataset import make_eval_transform
        from window_ast.pane_tree import NODE_CLASSES
        from window_ast.pane_tree_model import PaneTreeModel
        from window_ast.structure import ir_to_structure_tokens, structure_tokens_to_ir

        self._tokens_of = ir_to_structure_tokens
        self._parse = structure_tokens_to_ir
        self.pkgs = nullcontext()
        self.device = device
        self.place_scale = place_scale
        self._classes = NODE_CLASSES
        self._child_id = lambda parent, i: ("" if parent == "root" else parent) + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]
        self._canonical = lambda rules, region: {"region": region, "root": "root", "rules": rules}

        def place(img: Image.Image, glass: list[float], scale: float) -> tuple[Image.Image, list[float]]:
            canvas_size = 512
            width, height = img.size
            factor = scale * canvas_size / max(width, height)
            new_width, new_height = max(1, round(width * factor)), max(1, round(height * factor))
            x0, y0 = (canvas_size - new_width) / 2, (canvas_size - new_height) / 2
            canvas = Image.new("RGB", (canvas_size, canvas_size), (186, 186, 186))
            canvas.paste(img.resize((new_width, new_height), Image.BICUBIC), (round(x0), round(y0)))
            mapped = [
                (x0 + glass[0] * new_width) / canvas_size,
                (y0 + glass[1] * new_height) / canvas_size,
                (x0 + glass[2] * new_width) / canvas_size,
                (y0 + glass[3] * new_height) / canvas_size,
            ]
            return canvas, mapped

        self.place = place
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_args = ck["args"]
        self.model = PaneTreeModel(
            ck["prefixes"],
            d_model=model_args["d_model"],
            roi=model_args["roi"],
            pretrained=False,
            global_feat=model_args.get("global_feat", "mean"),
        ).to(device)
        self.model.load_state_dict(ck["model"])
        if adapter_path is not None:
            channels = self.model.encoder.feat_dim
            adapter = torch.nn.Sequential(
                torch.nn.Conv2d(channels, 256, 1),
                torch.nn.GELU(),
                torch.nn.Conv2d(256, 256, 3, padding=1),
                torch.nn.GELU(),
                torch.nn.Conv2d(256, 256, 3, padding=1),
                torch.nn.GELU(),
                torch.nn.Conv2d(256, channels, 1),
            ).to(device)
            adapter_state = torch.load(adapter_path, map_location=device, weights_only=False)["adapter"]
            adapter.load_state_dict({key.removeprefix("net."): value for key, value in adapter_state.items()})

            def encode_with_adapter(model, images):
                spatial = model.encoder.forward_spatial(images)
                spatial = spatial + adapter(spatial)
                global_features = model.glob(model._global_input(spatial, getattr(model.encoder, "last_cls", None)))
                output = {
                    "prefix": model.prefix_head(global_features),
                    "root_box": torch.sigmoid(model.root_box_head(global_features)),
                }
                return model.reduce(spatial), global_features, output

            self.model.encode = types.MethodType(encode_with_adapter, self.model)
        self.model.eval()
        self.tfm = make_eval_transform(int(ck["args"].get("image_size", 336)))
        print(f"structure: pane tree {ckpt_path}  adapter={adapter_path}")

    def _panes(self, region: str, nodes: list[dict[str, Any]]) -> dict[str, Any] | None:
        rules: dict[str, Any] = {}

        def build(i: int, nid: str) -> None:
            n = nodes[i]
            c = self._classes[n["cls"]]
            if c == "eps":
                rules[nid] = {"op": "eps"}
            elif c == "grid":
                rules[nid] = {"op": "grid", "vertical": int(n["V"]), "horizontal": int(n["H"])}
            else:
                kids = list(reversed(n["children"]))
                ids = [self._child_id(nid, j) for j in range(len(kids))]
                at = [round(1.0 - x, 4) for x in reversed(n["cuts"])]
                rules[nid] = {"op": "split", "axis": c[0], "children": ids, "at": at[0] if len(at) == 1 else at}
                for cid, ci in zip(ids, kids):
                    build(ci, cid)

        build(0, "root")
        return self._canonical(rules, region)

    def _input(self, crop_path: Path, glass: list[float] | None) -> torch.Tensor:
        canvas, _ = self.place(Image.open(crop_path).convert("RGB"), glass or [0, 0, 1, 1], self.place_scale)
        return self.tfm(canvas)

    def _result(self, toks: list[str], key: str, tree: list[dict[str, Any]]) -> dict[str, Any]:
        from window_ast.pane_tree import parse_prefix_key

        prefix, region = parse_prefix_key(key)
        try:
            ir = self._parse(list(prefix) + ["END"])
            panes = self._panes(region, tree) if region is not None and tree else None
            if panes is not None:
                ir["panes"] = _binarize_panes(panes)
            return {"tokens": self._tokens_of(ir), "ir": ir}
        except Exception as exc:
            return {"tokens": toks, "parse_error": str(exc)}

    @torch.no_grad()
    def predict(self, crop_path: Path, glass: list[float] | None = None) -> dict[str, Any]:
        toks, keys, trees = self.model.predict(
            self._input(crop_path, glass)[None].to(self.device),
            return_trees=True,
        )
        return self._result(toks[0], keys[0], trees[0])

    @torch.no_grad()
    def describe(self, crops: list[Image.Image], batch: int = 32, feature: str = "z") -> np.ndarray:
        """Per-crop descriptors [N, d] of tight window crops (crop = glass), for structure-aware clustering:
        ``z`` root-node embedding or ``heads`` prefix / root type / axis / V / H probabilities."""
        out = []
        for i in range(0, len(crops), batch):
            x = torch.stack([self.tfm(self.place(c.convert("RGB"), [0, 0, 1, 1], self.place_scale)[0])
                             for c in crops[i:i + batch]]).to(self.device)
            out.append(self.model.describe(x)[feature].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)

    @torch.no_grad()
    def predict_joint(self, crop_paths: list[Path], glasses: list[list[float] | None],
                      weights: list[float] | None = None) -> dict[str, Any]:
        """One structure for all crops: logits averaged over members at every tree node."""
        x = torch.stack([self._input(p, g) for p, g in zip(crop_paths, glasses)]).to(self.device)
        w = torch.tensor(weights, dtype=torch.float32) if weights is not None else None
        toks, key, tree = self.model.predict_joint(x, weights=w)
        return self._result(toks, key, tree)


def structure_fingerprint(ir: dict[str, Any]) -> str:
    """Stable key for voting (shape / pane topology; no continuous floats)."""
    from window_ast.structure import structure_view

    return json.dumps(structure_view(ir), sort_keys=True, separators=(",", ":"))


def vote_cluster_ir(
    member_preds: list[dict[str, Any]],
    *,
    prefer_unit_id: int | None = None,
) -> dict[str, Any]:
    """Majority-vote structure among cluster members; tie → prefer_unit / first.

    ``member_preds`` items: ``{unit_id, tokens?, ir?, parse_error?}``.
    Returns ``{structure_ir, structure_tokens, vote, members}``.
    """
    valid: list[dict[str, Any]] = []
    for p in member_preds:
        ir = p.get("ir")
        if not isinstance(ir, dict) or p.get("parse_error"):
            continue
        try:
            key = structure_fingerprint(ir)
        except Exception:
            continue
        valid.append({**p, "structure_key": key})

    tallies: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in valid:
        tallies[p["structure_key"]].append(p)

    vote_summary = {
        "n_members": len(member_preds),
        "n_valid": len(valid),
        "n_unique": len(tallies),
        "counts": {
            k: len(v) for k, v in sorted(tallies.items(), key=lambda kv: -len(kv[1]))
        },
    }

    if not valid:
        return {
            "structure_ir": None,
            "structure_tokens": None,
            "vote": {**vote_summary, "winner_key": None, "winner_count": 0},
            "members": member_preds,
        }

    # majority; ties → candidate containing prefer_unit_id, else largest then first
    ranked = sorted(tallies.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    best_count = len(ranked[0][1])
    tied = [k for k, vs in ranked if len(vs) == best_count]
    winner_key = tied[0]
    if prefer_unit_id is not None and len(tied) > 1:
        for k in tied:
            if any(int(p["unit_id"]) == int(prefer_unit_id) for p in tallies[k]):
                winner_key = k
                break
    winners = tallies[winner_key]
    chosen = winners[0]
    if prefer_unit_id is not None:
        for p in winners:
            if int(p["unit_id"]) == int(prefer_unit_id):
                chosen = p
                break

    members_out: list[dict[str, Any]] = []
    for p in member_preds:
        key = None
        ir = p.get("ir")
        if isinstance(ir, dict) and not p.get("parse_error"):
            try:
                key = structure_fingerprint(ir)
            except Exception:
                key = None
        members_out.append(
            {
                "unit_id": int(p["unit_id"]),
                "structure_key": key,
                "tokens": p.get("tokens"),
                "parse_error": p.get("parse_error"),
                "agrees_with_vote": key is not None and key == winner_key,
            }
        )

    return {
        "structure_ir": chosen.get("ir"),
        "structure_tokens": chosen.get("tokens"),
        "vote": {
            **vote_summary,
            "winner_key": winner_key,
            "winner_count": best_count,
            "winner_unit_id": int(chosen["unit_id"]),
            "unanimous": best_count == len(valid) and len(tallies) == 1,
        },
        "members": members_out,
    }


def joint_cluster_ir(
    predictor: TreeStructurePredictor,
    inputs: list[tuple[Path, list[float] | None, float]],
    voted: dict[str, Any],
    *,
    area_weighted: bool = False,
) -> dict[str, Any]:
    """Replace the majority-vote result of ``vote_cluster_ir`` by one jointly decoded structure.

    ``inputs`` are (crop, glass, box area) per member, in ``voted["members"]`` order. The
    majority result is kept under ``vote["majority_*"]``; ``agrees_with_vote`` is re-scored
    against the joint structure.
    """
    weights = [a for _, _, a in inputs] if area_weighted else None
    pred = predictor.predict_joint([p for p, _, _ in inputs], [g for _, g, _ in inputs], weights)
    try:
        key = structure_fingerprint(pred["ir"]) if pred.get("ir") is not None else None
    except Exception:
        key = None
    if key is None:
        return voted
    members = [{**m, "agrees_with_vote": m["structure_key"] == key} for m in voted["members"]]
    n_agree = sum(m["agrees_with_vote"] for m in members)
    old = voted["vote"]
    vote = {
        **{k: v for k, v in old.items() if k not in ("winner_key", "winner_count", "winner_unit_id", "unanimous")},
        "mode": "joint_area" if area_weighted else "joint",
        "winner_key": key,
        "winner_count": n_agree,
        "unanimous": n_agree == len(members),
        "majority_winner_key": old.get("winner_key"),
        "majority_winner_count": old.get("winner_count"),
        "majority_winner_unit_id": old.get("winner_unit_id"),
    }
    return {"structure_ir": pred["ir"], "structure_tokens": pred["tokens"], "vote": vote, "members": members}


def predict_structure_ir(
    crop_path: Path,
    *,
    ckpt_path: Path,
    device: torch.device,
    structure_only: bool = True,
) -> dict[str, Any] | None:
    """One-shot helper (loads model each call). Prefer ``StructurePredictor``."""
    try:
        pred = StructurePredictor(
            ckpt_path, device, structure_only=structure_only
        )
    except FileNotFoundError as exc:
        print(f"warn: {exc}")
        return None
    return pred.predict(crop_path)


# ---------------------------------------------------------------------------
# Facade DSL
# ---------------------------------------------------------------------------


def build_facade_dsl(
    *,
    facade_id: str,
    image_path: str,
    image_size: tuple[int, int],
    units: list[dict[str, Any]],
    types: list[dict[str, Any]],
    columns: list[tuple[float, float]] | None = None,
    door_units: list[dict[str, Any]] | None = None,
    door_types: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Recovery DSL: layout grid of type refs + window type library."""
    door_units = door_units or []
    door_types = door_types or []
    all_units = list(units) + list(door_units)
    iw, ih = image_size
    floors = sorted({int(u["floor"]) for u in all_units})
    floor_i = {f: i for i, f in enumerate(floors)}

    inter_bay_gaps_px: list[float | None] | None = None
    inter_bay_gap_norm: list[float] | None = None

    if columns is not None:
        bay_ids = list(range(len(columns)))
        envelope_units = [
            u for u in units if int(u.get("colspan", 1)) == 1
        ]
        columns_xy = infer_bay_column_bounds_from_units(
            envelope_units or units,
            len(columns),
            iw=iw,
            structural_columns=columns,
        )
        col_widths = [max(0.03, (xr - xl) / iw) for xl, xr in columns_xy]
        inter_bay_gaps_px = infer_inter_bay_gaps_robust(units, len(bay_ids))
        inter_bay_gap_norm = infer_gap_norms(units, len(bay_ids), iw)
    else:
        bay_ids = sorted({int(u["bay"]) for u in units})
        col_widths = []
        columns_xy = None
        for b in bay_ids:
            xs = [u["box_xyxy"] for u in units if int(u["bay"]) == b]
            if not xs:
                col_widths.append(1.0)
                continue
            w = float(np.mean([box[2] - box[0] for box in xs])) / iw
            col_widths.append(max(0.05, w))

    n_bays = len(bay_ids)
    placement: list[list[str | None]] = [
        [None for _ in range(n_bays)] for _ in floors
    ]
    placement_spans: list[list[int]] = [
        [1 for _ in range(n_bays)] for _ in floors
    ]
    win_units = sorted(
        units,
        key=lambda u: (
            -int(u.get("colspan", 1)),
            int(u["floor"]),
            int(u["bay"]),
        ),
    )
    for u in win_units:
        r = floor_i[int(u["floor"])]
        c = int(u["bay"])
        if c < 0 or c >= n_bays:
            continue
        tid = f"win_type_{int(u['type_id']):02d}"
        span = max(1, int(u.get("colspan", 1)))
        if placement[r][c] is not None:
            continue
        placement[r][c] = tid
        placement_spans[r][c] = span
        for j in range(1, span):
            if c + j < n_bays:
                placement[r][c + j] = None
                placement_spans[r][c + j] = 0
    for u in door_units:
        r = floor_i[int(u["floor"])]
        c = int(u["bay"])
        if c < 0 or c >= n_bays:
            continue
        tid = str(u.get("name") or door_type_token(int(u["type_id"])))
        placement[r][c] = tid
        placement_spans[r][c] = max(1, int(u.get("colspan", 1)))

    row_heights: list[float] = []
    inter_floor_gaps_px: list[float] | None = None
    inter_floor_gap_norm: list[float] | None = None
    floors_y: list[tuple[float, float]] | None = None
    if floors:
        row_heights, inter_floor_gap_norm = infer_proportional_floor_heights(
            all_units, floors, ih
        )
        inter_floor_gaps_px = infer_inter_floor_gaps_robust(all_units, floors)
        floors_y = infer_floor_row_bounds_from_units(all_units, floors, ih=ih)

    layout_note = (
        "Recovery DSL: w_norm and h_norm from box size + half inter-bay/floor gap "
        "(all ÷ image width/height), including outer wall-edge margins."
    )

    meta_extra: dict[str, Any] = {}
    if inter_bay_gaps_px is not None:
        meta_extra["inter_bay_gaps_px"] = [round(g, 1) for g in inter_bay_gaps_px]
    if inter_bay_gap_norm is not None:
        meta_extra["inter_bay_gap_norm"] = [round(g, 4) for g in inter_bay_gap_norm]
    if inter_floor_gaps_px is not None:
        meta_extra["inter_floor_gaps_px"] = [round(g, 1) for g in inter_floor_gaps_px]
    if inter_floor_gap_norm is not None:
        meta_extra["inter_floor_gap_norm"] = [round(g, 4) for g in inter_floor_gap_norm]
    if floors_y is not None:
        meta_extra["floors_y"] = [[round(a, 1), round(b, 1)] for a, b in floors_y]

    return {
        "schema": "facade_recovery_dsl_v1",
        "meta": {
            "facade_id": facade_id,
            "image": image_path,
            "image_size": [iw, ih],
            "n_units": len(units),
            "n_types": len(types),
            "n_doors": len(door_units),
            "n_door_types": len(door_types),
            "notes": layout_note,
            "columns_xy": (
                [[round(xl, 1), round(xr, 1)] for xl, xr in columns_xy]
                if columns_xy is not None
                else None
            ),
            **meta_extra,
        },
        "layout": {
            "floors": [
                {"name": f"F{f}", "id": int(f), "h_norm": float(row_heights[i])}
                for i, f in enumerate(floors)
            ],
            "bays": [
                {"name": f"B{b}", "id": int(b), "w_norm": float(col_widths[i])}
                for i, b in enumerate(bay_ids)
            ],
            "placement": placement,
            "placement_spans": placement_spans,
        },
        "window_types": types,
        "door_types": door_types,
        "instances": [{**u, "kind": "window"} for u in units] + [
            {**u, "kind": "door"} for u in door_units
        ],
    }


def render_overview(
    facade: Image.Image,
    units: list[dict[str, Any]],
    types: list[dict[str, Any]],
    out_path: Path,
) -> None:
    labels = np.array([int(u["type_id"]) for u in units], dtype=np.int32)
    items = [{"box_xyxy": u["box_xyxy"]} for u in units]
    n_types = len(set(labels.tolist()))
    ov = base.draw_overlay(
        facade,
        items,
        labels,
        title=f"e2e types={n_types} units={len(units)}",
    )
    # type strip
    cell = 96
    strip_h = cell + 28
    strip = Image.new("RGB", (max(ov.width, n_types * (cell + 8)), strip_h), (18, 18, 18))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(strip)
    for t in types:
        tid = int(t["type_id"])
        x = tid * (cell + 8) + 4
        p = Path(t["exemplar_asset"])
        if p.is_file():
            im = Image.open(p).convert("RGB")
            im.thumbnail((cell, cell), Image.Resampling.LANCZOS)
            strip.paste(im, (x, 22))
        d.text((x, 4), f"T{tid} n={t['n_instances']}", fill=(230, 230, 230), font=font)
    canvas = Image.new("RGB", (max(ov.width, strip.width), ov.height + strip_h + 8), (12, 12, 12))
    canvas.paste(ov, (0, 0))
    canvas.paste(strip, (0, ov.height + 8))
    canvas.save(out_path)


def gate_types(
    facade: Image.Image,
    facade_id: str,
    clustered: dict[str, Any],
    struct_predictor: TreeStructurePredictor,
    args: argparse.Namespace,
    axis_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Type labels of ``--cluster-mode gate`` and the facade's symmetry axis x (px)."""
    boxes = clustered["merged_boxes"]
    iw = facade.size[0]
    columns_xy = None
    if clustered.get("columns"):
        units = [
            {"box_xyxy": b, "floor": int(clustered["floor"][i]), "bay": int(clustered["bay"][i]),
             "colspan": int(clustered["colspan"][i])}
            for i, b in enumerate(boxes)
        ]
        columns_xy = infer_bay_column_bounds_from_units(
            [u for u in units if u["colspan"] == 1] or units,
            len(clustered["columns"]),
            iw=iw,
            structural_columns=clustered["columns"],
        )
    balconies = []
    if args.balcony_prompt:
        print(f"detecting '{args.balcony_prompt}'…")
        balconies = [
            d["box_xyxy"]
            for d in detect_windows(
                facade,
                prompt=args.balcony_prompt,
                threshold=args.balcony_min_score,
                min_side=8,
                max_side_frac=0.9,
                device=device,
                model_id="facebook/sam3",
            )
        ]
    ax = _load(ROOT / "scripts" / "symmetry_axis_ghost_continuous.py", "e2e_axis")
    summary = ax.run_case(ax.parse_args([]), facade_id, facade, boxes, columns_xy, balconies, axis_dir)
    cx = refine_axis(boxes, float(summary["chosen_cx"]), args.axis_refine * iw)
    partner = mirror_partners(boxes, cx)
    feat = struct_predictor.describe(
        [facade.crop(tuple(int(v) for v in b)) for b in boxes], feature=args.gate_feature
    )
    labels = gate_cluster(
        feat, np.asarray(clustered["labels_box"]), partner, thr=args.struct_split_thr, eps=args.gate_eps
    )
    print(
        f"gate: axis {summary['chosen_cx']:.1f} → {cx:.1f} ({summary['dual']['reason']}), "
        f"balconies={len(balconies)}, mirror-paired {sum(p is not None for p in partner)}/{len(boxes)}, "
        f"{len(set(clustered['labels_box'].tolist()))} box groups → {len(set(labels.tolist()))} types"
    )
    return labels.astype(np.int32), cx


def drop_units(clustered: dict[str, Any], keep: list[int]) -> None:
    """Keep only units ``keep`` in every per-unit field of ``cluster_units`` output; relabel types 0..k-1."""
    n = len(clustered["merged_boxes"])
    for key in ("merged_boxes", "members", "bay", "colspan", "floor", "labels", "labels_box", "feats",
                "feats_raw", "spatial"):
        val = clustered.get(key)
        if isinstance(val, np.ndarray) and val.ndim >= 1 and len(val) == n:
            clustered[key] = val[keep]
        elif isinstance(val, list) and len(val) == n:
            clustered[key] = [val[i] for i in keep]
    _, inv = np.unique(clustered["labels"], return_inverse=True)
    clustered["labels"] = inv.astype(np.int32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.train_up is None:
        args.train_up = default_train_up()
    if args.structure_ckpt is None:
        args.structure_ckpt = default_structure_ckpt()

    if args.blender_render:
        blender_bin = resolve_blender(args.blender)
        compiler = resolve_compiler_root(args.compiler_root)
        if blender_bin is None:
            print(
                "warn: --blender-render requested but Blender not found; "
                "continuing without render. Set --blender or $BLENDER."
            )
            args.blender_render = False
        elif compiler is None:
            print(
                "warn: --blender-render requested but window compiler not found; "
                "set FACADE_COMPILER_ROOT or --compiler-root to a package with main.py. "
                "Continuing without render."
            )
            args.blender_render = False
        else:
            args.blender = blender_bin
            args.compiler_root = compiler

    if args.image is not None:
        facade_path = Path(args.image)
        facade_id = args.facade_id or facade_path.stem
    elif args.facade_id is not None:
        facade_id = args.facade_id
        facade_path = args.train_up / f"{facade_id}.png"
    elif args.from_index is not None:
        facade_id = "unknown"
        facade_path = None
    else:
        raise SystemExit("provide --facade-id, --image, or --from-index")

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "runs" / f"facade_e2e_{facade_id}"
    assets_dir = out_dir / "assets"
    types_dir = assets_dir / "types"
    crops_dir = out_dir / "crops"
    for d in (out_dir, assets_dir, types_dir, crops_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- 1 detect ---
    source = args.windows
    if args.from_index is not None:
        facade, raw_windows, index_meta = load_from_index(args.from_index)
        facade_path = Path(index_meta.get("facade_path", facade_path or ""))
        facade_id = str(index_meta.get("facade_id", facade_id))
        source = str(index_meta.get("source", args.windows))
        print(f"reusing index: {len(raw_windows)} windows")
    else:
        if facade_path is None or not facade_path.is_file():
            raise SystemExit(f"missing facade image: {facade_path}")
        facade = Image.open(facade_path).convert("RGB")
        if args.windows == "xml":
            xml_path = facade_path.with_suffix(".xml")
            if not xml_path.is_file():
                raise SystemExit(f"missing CMP XML windows: {xml_path}")
            raw_windows = load_cmp_xml_windows(xml_path, facade.size, stem=facade_id)
            print(f"CMP XML windows: {len(raw_windows)} from {xml_path}")
            source = "xml"
        elif args.windows == "labelmap":
            png_path = facade_path.with_suffix(".png")
            if not png_path.is_file():
                raise SystemExit(f"missing CMP label map: {png_path}")
            raw_windows = load_cmp_labelmap_windows(png_path, stem=facade_id)
            print(f"CMP label-map windows: {len(raw_windows)} from {png_path}")
            source = "labelmap"
        else:
            raw_windows = detect_windows(
                facade,
                prompt=args.prompt,
                threshold=args.threshold,
                min_side=args.min_side,
                max_side_frac=args.max_side_frac,
                device=device,
                model_id=args.sam_model,
            )
            source = "sam"
            if args.shop_as_window:
                big = dict(
                    threshold=args.threshold,
                    min_side=max(16, args.min_side // 2),
                    max_side_frac=min(0.85, args.max_side_frac + 0.25),
                    device=device,
                )
                shops = detect_windows(
                    facade, prompt=args.shop_prompt, model_id=args.shop_sam_model or args.sam_model, **big
                )
                doors = (
                    detect_windows(facade, prompt=args.door_prompt, model_id=args.door_sam_model, **big)
                    if args.detect_doors
                    else []
                )
                added = shop_windows(
                    shops, raw_windows, doors, image_height=facade.height, min_y_frac=args.shop_min_y_frac
                )
                for r in added:
                    r["idx"] = len(raw_windows)
                    raw_windows.append(r)
                print(f"shop-as-window: {len(shops)} '{args.shop_prompt}' detections, added {len(added)} as windows")
            raw_windows, narrow = drop_narrow_boxes(raw_windows, args.min_rel_width)
            if narrow:
                print(f"dropped {len(narrow)} narrow windows: {[r['box_xyxy'] for r in narrow]}")
    exclude_prompts = [p.strip() for p in args.exclude_prompts.split(",") if p.strip()]
    exclude_mask = None
    excluded_windows: list[dict[str, Any]] = []
    if source == "sam" and exclude_prompts:
        exclude_mask = detect_object_mask(
            facade,
            prompts=exclude_prompts,
            threshold=args.exclude_threshold,
            device=device,
        )
        raw_windows, excluded_windows = drop_boxes_on_mask(
            raw_windows, exclude_mask, min_cover=args.exclude_min_cover
        )
        print(
            f"excluded {len(excluded_windows)} windows on {exclude_prompts}: "
            f"{[r['exclude_cover'] for r in excluded_windows]}"
        )
        if exclude_mask is not None and exclude_mask.any():
            draw_excluded(facade, exclude_mask, excluded_windows, raw_windows).save(
                out_dir / "excluded_objects.png"
            )
    if source == "sam" and args.anchor_score > 0:
        n0 = len(raw_windows)
        raw_windows = two_tier(
            [w for w in raw_windows if w["score"] >= args.threshold], args.anchor_score, args.fill_max_overlap
        )
        print(f"two-tier {args.anchor_score}/{args.threshold}: {n0} → {len(raw_windows)} windows")
    # Always persist the boxes actually used (incl. mask∩det clamp on reuse).
    (out_dir / "index_raw.json").write_text(
        json.dumps(
            {
                "facade_id": facade_id,
                "facade_path": str(facade_path),
                "source": source,
                "n_windows": len(raw_windows),
                "windows": raw_windows,
                "exclude_prompts": exclude_prompts if source == "sam" else [],
                "excluded_windows": excluded_windows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if len(raw_windows) < 2:
        raise SystemExit("need ≥2 windows")

    raw_boxes = [w["box_xyxy"] for w in raw_windows]
    for i, b in enumerate(raw_boxes):
        facade.crop(tuple(b)).save(crops_dir / f"raw_{i:03d}.png")
    raw_ov = base.draw_overlay(
        facade,
        [{"box_xyxy": b} for b in raw_boxes],
        np.arange(len(raw_boxes), dtype=np.int32),
        title=f"{facade_id}  {args.windows} raw n={len(raw_boxes)}",
    )
    raw_ov.save(out_dir / "raw_windows.png")

    # --- 2–3 unitize + cluster ---
    print(f"loading {args.dino}…")
    dino = torch.hub.load("facebookresearch/dinov2", args.dino, pretrained=True)
    dino = dino.to(device).eval()
    struct_predictor: TreeStructurePredictor | None = None
    gate = args.cluster_mode == "gate"
    if args.cluster_mode in ("gate", "shape_then_struct", "struct"):
        kind = torch.load(args.structure_ckpt, map_location="cpu", weights_only=False).get("kind")
        if kind != "pane_tree":
            raise SystemExit(f"--cluster-mode {args.cluster_mode} needs a pane-tree --structure-ckpt")
        struct_predictor = TreeStructurePredictor(
            args.structure_ckpt, device, adapter_path=args.structure_adapter, tree_root=args.tree_root
        )
    clustered = cluster_units(
        facade,
        raw_boxes,
        model=dino,
        device=device,
        facade_max_side=args.facade_max_side,
        pca_dim=args.pca_dim,
        seed=args.seed,
        k_max=args.k_max,
        col_tol=args.col_tol,
        row_tol=args.row_tol,
        spatial_strength=args.spatial_strength,
        unary_weight=args.unary_weight,
        layout_mode=args.layout_mode,
        merge_mode=args.merge_mode,
        merge_adjacent=args.merge_adjacent,
        vert_gap=args.vert_gap,
        cluster_mode="box" if gate else args.cluster_mode,
        k_max_box=args.k_max_box,
        k_max_within=args.k_max_within,
        box_method=args.box_method,
        box_conservative_k=args.box_conservative_k,
        box_conservative_k_wh_tol=args.box_conservative_k_wh_tol,
        box_merge_near=args.box_merge_near,
        box_snap_wh=args.box_snap_wh,
        struct_describe=(
            (lambda bs: struct_predictor.describe([facade.crop(tuple(int(v) for v in b)) for b in bs],
                                                  feature=args.struct_feature))
            if struct_predictor is not None else None
        ),
        struct_split_thr=args.struct_split_thr,
    )
    merged_boxes = clustered["merged_boxes"]
    members = clustered["members"]
    labels = clustered["labels"]
    feats = clustered["feats"]
    axis_x: float | None = None
    if gate:
        labels, axis_x = gate_types(facade, facade_id, clustered, struct_predictor, args, out_dir / "axis", device)
        clustered["labels"] = labels
        clustered["cluster_mode"] = "gate"
    elif args.symmetry:
        print("symmetry repair…")
        labels, sym_log = apply_symmetry_repair(
            facade,
            merged_boxes,
            labels,
            feats,
            labels_box=clustered.get("labels_box"),
            twin_mirror_min=args.twin_mirror_min,
        )
        clustered["labels"] = labels
        clustered["symmetry_log"] = sym_log
        for line in sym_log[-8:]:
            print(f"  {line}")
    n_types = len(set(int(x) for x in labels.tolist()))
    k_box = (
        len(set(clustered["labels_box"].tolist()))
        if clustered.get("labels_box") is not None
        else "n/a"
    )
    print(
        f"units: {len(raw_boxes)} raw → {len(merged_boxes)} merged  "
        f"cluster={clustered.get('cluster_mode', 'appearance')}  "
        f"merge={clustered.get('merge_mode', 'structural')}  "
        f"adjacent={'on' if clustered.get('merge_adjacent') else 'off'}  "
        f"symmetry={f'axis {axis_x:.1f}' if gate else ('on' if args.symmetry else 'off')}  "
        f"k_box={k_box}  types={n_types}  "
        f"layout={clustered.get('layout_mode', 'centroid')}  "
        f"cols={len(clustered.get('columns') or []) or 'n/a'}"
    )

    # --- 3b doors + side facades ---
    door_units: list[dict[str, Any]] = []
    door_types_out: list[dict[str, Any]] = []
    raw_doors: list[dict[str, Any]] = []
    kept_doors: list[dict[str, Any]] = []
    if args.detect_doors:
        print("detecting doors…")
        raw_doors = detect_windows(
            facade,
            prompt=args.door_prompt,
            threshold=args.door_threshold,
            min_side=max(16, args.min_side // 2),
            max_side_frac=min(0.85, args.max_side_frac + 0.25),
            device=device,
            model_id=args.door_sam_model,
        )
        raw_doors, excluded_doors = drop_boxes_on_mask(
            raw_doors, exclude_mask, min_cover=args.exclude_min_cover
        )
        if excluded_doors:
            print(f"excluded {len(excluded_doors)} doors on {exclude_prompts}")
        kept_doors = filter_door_boxes(
            raw_doors,
            merged_boxes,
            image_size=facade.size,
            min_y_frac=args.door_min_y_frac,
            iou_nms=args.door_iou_nms,
        )
    side_dropped: list[list[float]] = []
    if args.side_filter:
        side = set(
            drop_side_facades(merged_boxes, *facade.size, doors=[d["box_xyxy"] for d in kept_doors])
        )
        if side and len(side) < len(merged_boxes):
            keep = [i for i in range(len(merged_boxes)) if i not in side]
            side_dropped = [list(map(float, merged_boxes[i])) for i in sorted(side)]
            drop_units(clustered, keep)
            merged_boxes = clustered["merged_boxes"]
            members = clustered["members"]
            labels = clustered["labels"]
            feats = clustered["feats"]
            x0 = min(b[0] for b in merged_boxes)
            x1 = max(b[2] for b in merged_boxes)
            inside = [d for d in kept_doors if x0 <= 0.5 * (d["box_xyxy"][0] + d["box_xyxy"][2]) <= x1]
            side_dropped += [list(map(float, d["box_xyxy"])) for d in kept_doors if d not in inside]
            kept_doors = inside
            print(f"side facades: dropped {len(side)} windows, {len(side_dropped) - len(side)} doors")
    clustered["side_dropped"] = side_dropped

    local_sym_moves: list[list[int]] | None = None
    if gate and args.local_sym:
        prompts = [p.strip() for p in args.local_sym_occluders.split(",") if p.strip()]
        occ_mask = detect_object_mask(
            facade, prompts=prompts, threshold=args.local_sym_occluder_threshold, device=device
        )
        sym_feat = struct_predictor.describe(
            [facade.crop(tuple(int(v) for v in b)) for b in merged_boxes], feature="heads"
        )
        voted, report = local_sym_vote(merged_boxes, labels, sym_feat, occ_mask, facade.size)
        local_sym_moves = report["moves"]
        (out_dir / "local_sym.json").write_text(
            json.dumps({"before": labels.tolist(), "after": voted.tolist(), **report}, indent=1) + "\n",
            encoding="utf-8",
        )
        labels = np.unique(voted, return_inverse=True)[1].astype(np.int32)
        clustered["labels"] = labels
        print(
            f"local symmetry: {len(report['axes'])} axes, moved {len(local_sym_moves)} "
            f"{[tuple(m) for m in local_sym_moves]} → {len(set(labels.tolist()))} types"
        )

    medoids = pick_medoids(feats, labels)

    if args.detect_doors:
        door_units = assign_doors_to_layout(
            kept_doors,
            [
                {
                    "floor": int(clustered["floor"][i]),
                    "bay": int(clustered["bay"][i]),
                    "box_xyxy": merged_boxes[i],
                }
                for i in range(len(merged_boxes))
            ],
            image_size=facade.size,
        )
        for i, du in enumerate(door_units):
            crop_path = crops_dir / f"door_{i:03d}.png"
            facade.crop(tuple(du["box_xyxy"])).save(crop_path)
            du["asset"] = str(crop_path.relative_to(out_dir))
        door_types_out: list[dict[str, Any]] = []
        if door_units:
            door_units, door_types_out = cluster_door_units(
                door_units,
                facade,
                image_size=facade.size,
                base=base,
                seed=args.seed,
            )
            for dt in door_types_out:
                name = str(dt["name"])
                group = [du for du in door_units if du["name"] == name]
                doors_dir = types_dir / name
                doors_dir.mkdir(parents=True, exist_ok=True)
                exemplar = group[0]
                for g in group:
                    if g.get("is_exemplar"):
                        exemplar = g
                ex_src = out_dir / exemplar["asset"]
                canon = doors_dir / "exemplar.png"
                if ex_src.is_file():
                    shutil.copy(ex_src, canon)
                dt["exemplar_asset"] = str(canon.relative_to(out_dir))
                (doors_dir / "door.json").write_text(
                    (ROOT / "assets" / "default_door.json").read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                (doors_dir / "structure_ir.json").write_text(
                    json.dumps(
                        {
                            "kind": "door",
                            "placeholder": True,
                            "name": name,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        print(f"doors: raw={len(raw_doors)} kept={len(kept_doors)} placed={len(door_units)}")

    del dino
    torch.cuda.empty_cache()

    # --- 4 assetize ---
    units: list[dict[str, Any]] = []
    for ui, box in enumerate(merged_boxes):
        crop = facade.crop(tuple(box))
        crop_path = crops_dir / f"unit_{ui:03d}.png"
        crop.save(crop_path)
        tid = int(labels[ui])
        type_crop_dir = types_dir / f"type_{tid:02d}"
        type_crop_dir.mkdir(parents=True, exist_ok=True)
        asset_path = type_crop_dir / f"unit_{ui:03d}.png"
        shutil.copy(crop_path, asset_path)
        units.append(
            {
                "unit_id": ui,
                "box_xyxy": box,
                "floor": int(clustered["floor"][ui]),
                "bay": int(clustered["bay"][ui]),
                "colspan": int(clustered["colspan"][ui]),
                "type_id": tid,
                "member_raw_idxs": members[ui],
                "asset": str(asset_path.relative_to(out_dir)),
                "is_exemplar": ui == medoids[tid],
            }
        )

    # optional structure IR: predict all members → majority vote per type
    structure_model_note = None
    predictor: StructurePredictor | TreeStructurePredictor | None = None
    if not args.skip_structure and args.structure_ckpt.is_file():
        print(f"loading structure model {args.structure_ckpt}…")
        structure_model_note = str(args.structure_ckpt)
        kind = torch.load(args.structure_ckpt, map_location="cpu", weights_only=False).get("kind")
        if kind == "pane_tree":
            predictor = struct_predictor or TreeStructurePredictor(
                args.structure_ckpt,
                device,
                adapter_path=args.structure_adapter,
                tree_root=args.tree_root,
            )
            if args.structure_adapter is not None:
                structure_model_note += f" + adapter {args.structure_adapter}"
        elif args.structure_adapter is not None:
            raise SystemExit("--structure-adapter needs a pane-tree --structure-ckpt")
        else:
            predictor = StructurePredictor(
                args.structure_ckpt,
                device,
                structure_only=args.structure_only,
            )
    elif not args.skip_structure:
        print(f"skip structure IR (missing {args.structure_ckpt})")

    types_out: list[dict[str, Any]] = []
    for tid, med_i in sorted(medoids.items()):
        exemplar = units[med_i]
        ex_path = out_dir / exemplar["asset"]
        # canonical exemplar copy
        canon = types_dir / f"type_{tid:02d}" / "exemplar.png"
        shutil.copy(ex_path, canon)
        member_units = [u for u in units if int(u["type_id"]) == int(tid)]
        entry: dict[str, Any] = {
            "type_id": tid,
            "name": f"win_type_{tid:02d}",
            "n_instances": len(member_units),
            "exemplar_unit": med_i,
            "exemplar_asset": str(canon.relative_to(out_dir)),
            "structure_ir": None,
        }
        if predictor is not None:
            member_preds: list[dict[str, Any]] = []
            joint_inputs: list[tuple[Path, list[float] | None, float]] = []
            predict_units = (
                member_units
                if args.structure_vote
                else [u for u in member_units if int(u["unit_id"]) == int(med_i)]
            )
            for u in predict_units:
                crop_p = out_dir / u["asset"]
                glass_kw = {}
                if args.structure_glass_margin:
                    ml, mt, mr, mb = (float(v) for v in args.structure_glass_margin.split(","))
                    x0, y0, x1, y1 = u["box_xyxy"]
                    w, h = x1 - x0, y1 - y0
                    cx0, cy0 = max(0, round(x0 - ml * w)), max(0, round(y0 - mt * h))
                    cx1, cy1 = min(facade.width, round(x1 + mr * w)), min(facade.height, round(y1 + mb * h))
                    crop_p = crops_dir / f"unit_{int(u['unit_id']):03d}_ctx.png"
                    facade.crop((cx0, cy0, cx1, cy1)).save(crop_p)
                    cw, ch = cx1 - cx0, cy1 - cy0
                    glass_kw = {"glass": [(x0 - cx0) / cw, (y0 - cy0) / ch, (x1 - cx0) / cw, (y1 - cy0) / ch]}
                elif args.structure_pad > 0:
                    x0, y0, x1, y1 = u["box_xyxy"]
                    px, py = args.structure_pad * (x1 - x0), args.structure_pad * (y1 - y0)
                    crop_p = crops_dir / f"unit_{int(u['unit_id']):03d}_pad.png"
                    facade.crop((
                        max(0, round(x0 - px)), max(0, round(y0 - py)),
                        min(facade.width, round(x1 + px)), min(facade.height, round(y1 + py)),
                    )).save(crop_p)
                pred = predictor.predict(crop_p, **glass_kw)
                x0, y0, x1, y1 = u["box_xyxy"]
                joint_inputs.append((crop_p, glass_kw.get("glass"), float((x1 - x0) * (y1 - y0))))
                member_preds.append({"unit_id": int(u["unit_id"]), **pred})
                # stash per-unit prediction on the instance
                u["structure_tokens"] = pred.get("tokens")
                if pred.get("ir") is not None:
                    u["structure_ir_member"] = pred["ir"]
                if "parse_error" in pred:
                    u["structure_parse_error"] = pred["parse_error"]

            voted = vote_cluster_ir(member_preds, prefer_unit_id=med_i)
            if args.structure_joint != "off" and isinstance(predictor, TreeStructurePredictor):
                voted = joint_cluster_ir(predictor, joint_inputs, voted, area_weighted=args.structure_joint == "area")
            entry["structure_ir"] = voted["structure_ir"]
            entry["structure_tokens"] = voted["structure_tokens"]
            entry["structure_vote"] = voted["vote"]
            entry["structure_members"] = voted["members"]
            # propagate unified IR onto every unit of this type
            for u in member_units:
                u["structure_ir"] = voted["structure_ir"]
                u["structure_tokens_voted"] = voted["structure_tokens"]

            vote = voted["vote"]
            print(
                f"  type_{tid:02d}: vote "
                f"{vote.get('winner_count', 0)}/{vote.get('n_valid', 0)} "
                f"unique={vote.get('n_unique', 0)} "
                f"unanimous={vote.get('unanimous', False)}"
            )
            (types_dir / f"type_{tid:02d}" / "structure_ir.json").write_text(
                json.dumps(
                    {
                        "tokens": voted["structure_tokens"],
                        "ir": voted["structure_ir"],
                        "vote": voted["vote"],
                        "members": voted["members"],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        types_out.append(entry)

    # --- 5 DSL ---
    dsl = build_facade_dsl(
        facade_id=facade_id,
        image_path=str(facade_path) if facade_path else "",
        image_size=facade.size,
        units=units,
        types=types_out,
        columns=clustered.get("columns"),
        door_units=door_units,
        door_types=door_types_out,
    )
    if structure_model_note:
        dsl["meta"]["structure_ckpt"] = structure_model_note
    if args.structure_glass_margin:
        dsl["meta"]["structure_glass_margin"] = args.structure_glass_margin
    if args.structure_joint != "off":
        dsl["meta"]["structure_joint"] = args.structure_joint
    if axis_x is not None:
        dsl["meta"]["axis_x"] = round(axis_x, 1)
    dsl["meta"]["cluster_mode"] = clustered.get("cluster_mode")
    if local_sym_moves is not None:
        dsl["meta"]["local_sym_moves"] = local_sym_moves
    if side_dropped:
        dsl["meta"]["side_dropped"] = [[round(v, 1) for v in b] for b in side_dropped]
    if args.grid == "floor_bays":
        dsl = apply_floor_bays(dsl, row_tol=args.row_tol)
    n_types = len(types_out)
    dsl_path = out_dir / "facade_dsl.json"
    dsl_path.write_text(json.dumps(dsl, indent=2) + "\n", encoding="utf-8")

    # human-readable compact summary
    summary = {
        "facade_id": facade_id,
        "n_raw_windows": len(raw_boxes),
        "n_units": len(merged_boxes),
        "n_types": n_types,
        "layout_mode": clustered.get("layout_mode"),
        "n_columns": len(clustered.get("columns") or []),
        "floors": sorted({u["floor"] for u in units}),
        "bays": sorted({u["bay"] for u in units}),
        "types": [
            {
                "type_id": t["type_id"],
                "n": t["n_instances"],
                "exemplar": t["exemplar_asset"],
                "has_ir": t.get("structure_ir") is not None,
                "vote": t.get("structure_vote"),
            }
            for t in types_out
        ],
        "dsl": str(dsl_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    render_overview(facade, units, types_out, out_dir / "overview.png")
    print(f"\n=== facade e2e {facade_id} ===")
    print(f"  raw={len(raw_boxes)}  units={len(merged_boxes)}  types={n_types}  doors={len(door_units)}")
    print(f"  DSL → {dsl_path}")
    print(f"  assets → {types_dir}")
    print(f"  overview → {out_dir / 'overview.png'}")

    if args.blender_render:
        import os
        import subprocess

        blender_out = out_dir / "blender"
        env = os.environ.copy()
        env["FACADE_COMPILER_ROOT"] = str(args.compiler_root)
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "render_facade.py"),
            "--recovery",
            str(dsl_path),
            "--out-dir",
            str(blender_out),
            "--blender",
            str(args.blender),
            "--render",
            "--force",
        ]
        print(f"blender render → {blender_out}")
        subprocess.run(cmd, check=True, env=env)

    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
