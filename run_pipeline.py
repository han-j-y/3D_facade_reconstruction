#!/usr/bin/env python3
"""End-to-end facade recovery: photo → window types → structure IR → DSL.

Pipeline
  1. Detect   — SAM3 ``window`` boxes (or reuse ``--from-index``)
  2. Unitize  — merge adjacent same-floor panes / bay faces
  3. Layout   — structural columns from box geometry (bay + colspan); legacy ``--layout-mode centroid``
  4. Cluster  — z-scored box geometry (GMM) then optional DINO splits → window types
  4. Assetize — per-type crops; medoid exemplar
  5. Structure — predict window structure IR per unit; majority-vote within type
  6. DSL       — floor×bay layout (+ colspan) + type library (``facade_dsl.json``)
  7. (opt)     — Blender façade render via ``scripts/render_facade.py``

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
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from facade_recovery.box_cluster import (  # noqa: E402
    cluster_box_geometry_pipeline,
    cluster_within_groups,
)
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
        help="directory of {id}.png façades (default: ../data/facades/train_up)",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--from-index", type=Path, default=None, help="reuse SAM index.json")
    ap.add_argument(
        "--windows",
        choices=("sam", "xml"),
        default="sam",
        help="window boxes: SAM detect (default) or CMP XML next to the image",
    )
    ap.add_argument(
        "--merge-mode",
        choices=("structural", "ast"),
        default="structural",
        help="unitize merge: e2e structural columns, or window_ast_predictor "
        "(centroid bays + vertical same-stack merge)",
    )
    ap.add_argument(
        "--vert-gap",
        type=float,
        default=0.55,
        help="vertical merge gap (× median height) when --merge-mode ast",
    )
    ap.add_argument("--prompt", type=str, default="window")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--min-side", type=int, default=24)
    ap.add_argument("--max-side-frac", type=float, default=0.55)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dino", default="dinov2_vits14")
    ap.add_argument("--facade-max-side", type=int, default=896)
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-max", type=int, default=8, help="max k for appearance / final clustering")
    ap.add_argument(
        "--cluster-mode",
        choices=("shape_then_feat", "box", "appearance"),
        default="shape_then_feat",
        help="type clustering: z-scored box GMM then DINO splits (default), box-only, or legacy appearance+Potts",
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
    ap.add_argument("--col-tol", type=float, default=0.04)
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
    ap.add_argument("--skip-structure", action="store_true")
    ap.add_argument(
        "--detect-doors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SAM-detect ground doors and add to façade DSL (default on)",
    )
    ap.add_argument("--door-prompt", type=str, default="door")
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
) -> list[dict[str, Any]]:
    from transformers import Sam3Model, Sam3Processor

    iw, ih = image.size
    print("loading SAM3…")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    sam = Sam3Model.from_pretrained("facebook/sam3").to(device)
    sam.eval()
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
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
    for i, (box, sc) in enumerate(zip(boxes, scores)):
        det = [float(v) for v in box]
        x0, y0, x1, y1 = det
        used_mask = False
        fill = None
        if masks is not None and i < len(masks):
            m = masks[i]
            if m.ndim == 3:
                m = m[0]
            ys, xs = np.where(m > 0.5)
            if xs.size >= 16:
                x0, x1 = float(xs.min()), float(xs.max() + 1)
                y0, y1 = float(ys.min()), float(ys.max() + 1)
                fill = float(xs.size) / max(1.0, (x1 - x0) * (y1 - y0))
                used_mask = True
                n_mask_box += 1
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
        if fill is not None:
            rec["mask_fill"] = round(fill, 4)
        records.append(rec)
    print(f"after size filter: {len(records)}  tight-from-mask={n_mask_box}")
    return records


def load_from_index(index_path: Path) -> tuple[Image.Image, list[dict[str, Any]], dict]:
    index = json.loads(index_path.read_text())
    facade = Image.open(index["facade_path"]).convert("RGB")
    windows = []
    for w in index["windows"]:
        windows.append(
            {
                "idx": int(w["idx"]),
                "score": float(w.get("score", 1.0)),
                "box_xyxy": [int(v) for v in w["box_xyxy"]],
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
    vert_gap: float = 0.55,
    cluster_mode: str = "shape_then_feat",
    k_max_box: int = 6,
    k_max_within: int = 4,
    box_method: str = "gmm_diag",
    box_conservative_k: bool = True,
    box_conservative_k_wh_tol: float = 0.008,
    box_merge_near: bool = False,
    box_snap_wh: bool = True,
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
            EXP / "window_ast_predictor" / "scripts" / "overlay_facade_merge_boxes.py",
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
        "cluster_mode": cluster_mode,
    }


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
    if args.from_index is not None:
        facade, raw_windows, index_meta = load_from_index(args.from_index)
        facade_path = Path(index_meta.get("facade_path", facade_path or ""))
        facade_id = str(index_meta.get("facade_id", facade_id))
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
        else:
            raw_windows = detect_windows(
                facade,
                prompt=args.prompt,
                threshold=args.threshold,
                min_side=args.min_side,
                max_side_frac=args.max_side_frac,
                device=device,
            )
        (out_dir / "index_raw.json").write_text(
            json.dumps(
                {
                    "facade_id": facade_id,
                    "facade_path": str(facade_path),
                    "source": args.windows,
                    "n_windows": len(raw_windows),
                    "windows": raw_windows,
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
        vert_gap=args.vert_gap,
        cluster_mode=args.cluster_mode,
        k_max_box=args.k_max_box,
        k_max_within=args.k_max_within,
        box_method=args.box_method,
        box_conservative_k=args.box_conservative_k,
        box_conservative_k_wh_tol=args.box_conservative_k_wh_tol,
        box_merge_near=args.box_merge_near,
        box_snap_wh=args.box_snap_wh,
    )
    merged_boxes = clustered["merged_boxes"]
    members = clustered["members"]
    labels = clustered["labels"]
    feats = clustered["feats"]
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
        f"k_box={k_box}  types={n_types}  "
        f"layout={clustered.get('layout_mode', 'centroid')}  "
        f"cols={len(clustered.get('columns') or []) or 'n/a'}"
    )

    medoids = pick_medoids(feats, labels)

    # --- 3b doors ---
    door_units: list[dict[str, Any]] = []
    door_types_out: list[dict[str, Any]] = []
    if args.detect_doors:
        print("detecting doors…")
        raw_doors = detect_windows(
            facade,
            prompt=args.door_prompt,
            threshold=args.threshold,
            min_side=max(16, args.min_side // 2),
            max_side_frac=min(0.85, args.max_side_frac + 0.25),
            device=device,
        )
        kept_doors = filter_door_boxes(
            raw_doors,
            merged_boxes,
            image_size=facade.size,
            min_y_frac=args.door_min_y_frac,
            iou_nms=args.door_iou_nms,
        )
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
    predictor: StructurePredictor | None = None
    if not args.skip_structure and args.structure_ckpt.is_file():
        print(f"loading structure model {args.structure_ckpt}…")
        structure_model_note = str(args.structure_ckpt)
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
            predict_units = (
                member_units
                if args.structure_vote
                else [u for u in member_units if int(u["unit_id"]) == int(med_i)]
            )
            for u in predict_units:
                crop_p = out_dir / u["asset"]
                pred = predictor.predict(crop_p)
                member_preds.append({"unit_id": int(u["unit_id"]), **pred})
                # stash per-unit prediction on the instance
                u["structure_tokens"] = pred.get("tokens")
                if pred.get("ir") is not None:
                    u["structure_ir_member"] = pred["ir"]
                if "parse_error" in pred:
                    u["structure_parse_error"] = pred["parse_error"]

            voted = vote_cluster_ir(member_preds, prefer_unit_id=med_i)
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
