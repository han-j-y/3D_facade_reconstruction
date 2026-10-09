"""Balcony photo → BDSL IR (heuristic or railing classifier) → merge into window DSL.

Does not edit run.py / run_pipeline.py. Reuses SAM3 detect, box merge, DINOv2
cluster, medoid, and majority-vote *pattern* with a balcony fingerprint.

Requires an existing window facade_dsl.json (floor×bay from the window track).
Enclosed volumes (SAM3+DA3) are sliced onto that grid, snapped to column
width, and merged as ``balc_enclosed``. Open railings on the same cell are
dropped; an open balcony on the floor above keeps its slab (aligned to the
enclosed footprint by the compiler).

  python balcony_pipeline/run_balcony.py --image PHOTO.png --windows-dsl facade_dsl.json --out-dir runs/balcony_demo
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_pipeline as rp  # noqa: E402

from enclosed import (  # noqa: E402
    drop_open_under_enclosed,
    enclosed_ir,
    ensure_da3_depth,
    place_enclosed_volumes,
)
from cluster import cluster_balcony_boxes  # noqa: E402
from draw import (  # noqa: E402
    draw_boxes,
    draw_cluster_overlay,
    draw_snap,
    draw_vote,
    save,
)
from filter import filter_balcony_boxes  # noqa: E402
from heuristic_ir import (  # noqa: E402
    balcony_type_token,
    enclosure_from_ir,
    floor_shape_from_ir,
    infer_balcony_ir,
    ir_to_tokens,
    railing_kind_from_ir,
    railing_material_from_ir,
)
from merge_dsl import merge_balcony_into_windows_dsl  # noqa: E402
from recovery_profile import DEFAULT_PROFILE, PROFILES, resolve_profile  # noqa: E402
from snap import snap_units  # noqa: E402
from vote import vote_cluster_ir  # noqa: E402

from balcony_train.model import try_load_predictor  # noqa: E402
from balcony_train.paths import DEFAULT_RAILING_CKPT  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument(
        "--windows-dsl",
        type=Path,
        required=True,
        help="facade_recovery_dsl_v1 JSON from the window pipeline (layout required)",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("runs/balcony_demo"))
    ap.add_argument("--prompt", type=str, default="balcony")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--min-side", type=int, default=16)
    ap.add_argument(
        "--max-side-frac",
        type=float,
        default=1.0,
        help="drop boxes larger than this fraction of image W/H (1.0 = effectively off)",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dino", default="dinov2_vits14")
    ap.add_argument("--facade-max-side", type=int, default=896)
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--col-tol", type=float, default=0.04)
    ap.add_argument("--row-tol", type=float, default=0.055)
    ap.add_argument("--spatial-strength", type=float, default=1.8)
    ap.add_argument("--unary-weight", type=float, default=0.9)
    ap.add_argument(
        "--force",
        action="store_true",
        default=True,
        help="reset stage PNGs and stem assets before run (default: on)",
    )
    ap.add_argument(
        "--no-force",
        dest="force",
        action="store_false",
        help="keep existing stem assets (not recommended for shared out-dir)",
    )
    ap.add_argument(
        "--recovery-profile",
        default=DEFAULT_PROFILE,
        choices=sorted(PROFILES),
        help=(
            "which BDSL axes to infer+vote (default: railing_only). "
            "Use 'full' to restore structure/enclosure/floor/supports."
        ),
    )
    ap.add_argument(
        "--no-railing-vote",
        action="store_true",
        help=(
            "skip type-level majority vote; each unit keeps its own railing IR "
            "and its own floor shape"
        ),
    )
    ap.add_argument(
        "--no-floor-shape-vote",
        action="store_true",
        help=(
            "do not majority-vote floor shape; each balcony keeps its predicted "
            "plan (rectangle, triangle, circle). "
            "Implied by --no-railing-vote"
        ),
    )
    ap.add_argument(
        "--railing-ckpt",
        type=Path,
        default=None,
        help=(
            "solid/baluster classifier (default: checkpoints/railing_best.pt if present; "
            "else opaque-run heuristic)"
        ),
    )
    ap.add_argument(
        "--no-railing-ckpt",
        action="store_true",
        help="force the opaque-run heuristic even if railing_best.pt exists",
    )
    ap.add_argument(
        "--balcony-center",
        choices=("window", "bay", "photo"),
        default="window",
        help=(
            "horizontal mesh center: window (default, paired window box center), "
            "bay (mean of bays_center bands), or photo (detection box cx_norm)"
        ),
    )
    ap.add_argument(
        "--no-enclosed",
        action="store_true",
        help="skip SAM3+DA3 enclosed volumes (open-railing track only)",
    )
    ap.add_argument(
        "--depth-dir",
        type=Path,
        default=ROOT / "runs" / "da3_enclosure",
        help="DA3 ``<stem>_depth.npy`` directory for enclosed volumes",
    )
    ap.add_argument(
        "--enclosed-sam3-cache",
        type=Path,
        default=ROOT / "runs" / "sam3_da3_volumes" / "sam3_seeds.json",
        help="SAM3 seed cache for enclosed / oriel / bay prompts",
    )
    return ap.parse_args()


def _clear_stage_pngs(out_dir: Path, stem: str) -> None:
    for name in (
        f"s1_detect_balconies_{stem}.png",
        f"s1b_filtered_balconies_{stem}.png",
        f"s1c_enclosed_volumes_{stem}.png",
        f"s2_snap_floors_bays_{stem}.png",
        f"s2b_enclosed_snap_{stem}.png",
        f"s3_cluster_balcony_types_{stem}.png",
        f"s4_vote_balcony_ir_{stem}.png",
    ):
        p = out_dir / name
        if p.is_file():
            p.unlink()


def _reset_image_outputs(out_dir: Path, stem: str) -> None:
    """Drop prior stage PNGs and stem-scoped crops (shared out-dir safe)."""
    _clear_stage_pngs(out_dir, stem)
    for sub in (out_dir / "assets" / stem, out_dir / "assets" / "types"):
        if sub.is_dir():
            shutil.rmtree(sub)


def _write_image(im: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        path.unlink()
    im.save(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        path.unlink()
    path.write_text(text, encoding="utf-8")


def resolve_railing_predictor(args: argparse.Namespace, device: torch.device):
    """Classifier if ckpt exists; None → opaque-run heuristic in infer_balcony_ir."""
    force_off = bool(getattr(args, "no_railing_ckpt", False))
    ckpt = getattr(args, "railing_ckpt", None)
    predictor = try_load_predictor(ckpt, device, force_off=force_off)
    if predictor is not None:
        print(f"railing classifier {predictor.ckpt_path}")
    elif force_off:
        print("railing classifier off (--no-railing-ckpt); using heuristic")
    else:
        path = Path(ckpt) if ckpt else DEFAULT_RAILING_CKPT
        print(f"railing classifier missing ({path}); using heuristic")
    return predictor


def detect_enclosed_volume_boxes(
    image: Image.Image,
    image_path: Path,
    *,
    device: torch.device,
    depth_dir: Path,
    cache_path: Path,
    sam3_threshold: float = 0.15,
) -> list[dict]:
    """Run the SAM3+DA3 volume detector. Infers DA3 depth if the npy is missing."""
    stem = image_path.stem
    depth_path = ensure_da3_depth(image_path, depth_dir, device=str(device))
    if depth_path is None:
        return []
    from balcony_train.try_da3_enclosure import resize_depth
    from balcony_train.try_sam3_da3_volumes import detect_volumes, run_sam3

    depth = resize_depth(np.load(depth_path), image.width, image.height)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    seeds_map = run_sam3(
        [image_path],
        device=device,
        cache_path=Path(cache_path),
        threshold=sam3_threshold,
    )
    seeds = seeds_map.get(stem, [])
    if not seeds:
        print("warn: no SAM3 enclosed seeds; skip enclosed volumes")
        return []
    return detect_volumes(image, depth, seeds)


def infer_unit_ir(
    crop: Image.Image,
    unit: dict,
    *,
    image_size: tuple[int, int],
    profile_name: str,
    predictor,
) -> dict:
    kind_override = None
    material_override = None
    floor_override = None
    enclosure_override = None
    if predictor is not None:
        if hasattr(predictor, "predict_full"):
            pred = predictor.predict_full(crop)
            kind_override = pred.get("kind")
            material_override = pred.get("material")
            floor_override = pred.get("floor_shape")
            enclosure_override = pred.get("enclosure")
        else:
            kind_override = predictor.predict(crop)
    return infer_balcony_ir(
        crop,
        box=unit["box_xyxy"],
        image_size=image_size,
        profile_name=profile_name,
        rail_kind_override=kind_override,
        rail_material_override=material_override,
        floor_shape_override=floor_override,
        enclosure_override=enclosure_override,
    )


def _with_unit_floor(voted_ir: dict | None, member_ir: dict) -> dict:
    """Voted railing IR with this unit's own floor plan and enclosure."""
    if not isinstance(voted_ir, dict):
        return member_ir
    out = copy.deepcopy(voted_ir)
    floor = dict(member_ir.get("floor") or {})
    floor["shape"] = floor_shape_from_ir(member_ir)
    out["floor"] = floor
    out["enclosure"] = enclosure_from_ir(member_ir)
    member_slab = (member_ir.get("output") or {}).get("slab_thickness")
    if member_slab is not None:
        out["output"] = {**(out.get("output") or {}), "slab_thickness": member_slab}
    return out


def _types_grouped_by_token(
    units: list[dict],
    crops_dir: Path,
    out_dir: Path,
    *,
    include_floor: bool,
) -> list[dict]:
    """One Blender type per railing (+ floor shape when requested)."""
    by_token: dict[str, list[dict]] = defaultdict(list)
    for u in units:
        ir = u.get("structure_ir") or {}
        token = balcony_type_token(ir, include_floor=include_floor)
        by_token[token].append(u)
    types_out: list[dict] = []
    for tid, token in enumerate(sorted(by_token)):
        members = by_token[token]
        rep = members[0]
        canon = crops_dir / f"balc_{token}_exemplar.png"
        shutil.copy(out_dir / rep["asset"], canon)
        ir = rep["structure_ir"]
        types_out.append(
            {
                "type_id": tid,
                "name": f"balc_{token}",
                "n_instances": len(members),
                "exemplar_unit": int(rep["unit_id"]),
                "exemplar_asset": str(canon.relative_to(out_dir)),
                "structure_ir": ir,
                "structure_tokens": rep.get("structure_tokens"),
                "structure_vote": {
                    "mode": "per_unit",
                    "railing_kind": railing_kind_from_ir(ir),
                    "railing_material": railing_material_from_ir(ir),
                    "floor_shape": floor_shape_from_ir(ir) if include_floor else None,
                    "n_members": len(members),
                    "unit_ids": [int(u["unit_id"]) for u in members],
                },
            }
        )
    return types_out


def cluster_boxes(
    facade: Image.Image,
    boxes: list[list[int]],
    floors: list[int],
    bays: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    print(f"=== 3 Cluster balcony types ({args.dino}) ===")
    dino = torch.hub.load("facebookresearch/dinov2", args.dino, pretrained=True)
    dino = dino.to(device).eval()
    clustered = cluster_balcony_boxes(
        facade,
        boxes,
        floors,
        bays,
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
        patch=rp.patch,
        base=rp.base,
        reemb=rp.reemb,
    )
    del dino
    torch.cuda.empty_cache()
    return clustered


def run(args: argparse.Namespace) -> Path | None:
    """Run the balcony track. Returns merged DSL path, or None if skipped."""
    image_path = args.image.expanduser().resolve()
    dsl_path = args.windows_dsl.expanduser().resolve()
    if not image_path.is_file():
        raise SystemExit(f"image not found: {image_path}")
    if not dsl_path.is_file():
        raise SystemExit(
            f"window DSL not found: {dsl_path}\n"
            "Run the window pipeline first and pass its facade_dsl.json"
        )
    windows_dsl = json.loads(dsl_path.read_text(encoding="utf-8"))
    if not (windows_dsl.get("layout") or {}).get("floors"):
        raise SystemExit("windows-dsl missing layout.floors (need window grid)")
    if not windows_dsl.get("instances"):
        raise SystemExit("windows-dsl missing instances (need window boxes for snap)")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    facade = Image.open(image_path).convert("RGB")
    device = torch.device(args.device)
    stem = image_path.stem
    if args.force:
        _reset_image_outputs(out_dir, stem)
    crops_dir = out_dir / "assets" / stem / "types"
    crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== 1 Detect balconies  prompt={args.prompt!r} ===")
    raw = rp.detect_windows(
        facade,
        prompt=args.prompt,
        threshold=args.threshold,
        min_side=args.min_side,
        max_side_frac=args.max_side_frac,
        device=device,
    )
    raw_boxes = [r["box_xyxy"] for r in raw]
    save(
        draw_boxes(
            facade,
            raw_boxes,
            title=f"1. Detect balconies  prompt={args.prompt!r}  n={len(raw_boxes)}",
        ),
        out_dir / f"s1_detect_balconies_{stem}.png",
        force=True,
    )
    snapped: list[dict] = []
    if raw_boxes:
        print("=== 1a Filter false balconies (window band + rooftop) ===")
        filtered_boxes, drop_log = filter_balcony_boxes(
            raw_boxes,
            facade,
            windows_dsl.get("instances") or [],
        )
        n_drop = sum(1 for e in drop_log if not e["keep"])
        print(f"  {len(raw_boxes)} raw -> {len(filtered_boxes)} kept  (dropped {n_drop})")
        for e in drop_log:
            if e["keep"]:
                continue
            print(f"    drop {e['box']}  reasons={e['reasons']}")
        save(
            draw_boxes(
                facade,
                filtered_boxes,
                title=(
                    f"1a. Filtered balconies  kept={len(filtered_boxes)}/"
                    f"{len(raw_boxes)}  (below-windows | juliet-width | window-cover | no-window-above)"
                ),
                color=(40, 180, 90),
            ),
            out_dir / f"s1b_filtered_balconies_{stem}.png",
            force=True,
        )
        raw_boxes = filtered_boxes
    if raw_boxes:
        print("=== 1b Unitize (balcony merge: Pass-1 IoU/containment only) ===")
        iw, ih = facade.size
        cx = np.array([0.5 * (b[0] + b[2]) / iw for b in raw_boxes], dtype=np.float64)
        cy = np.array([0.5 * (b[1] + b[3]) / ih for b in raw_boxes], dtype=np.float64)
        merged_boxes, members, _bay_raw, _floor_raw = rp.merge_mod.merge_adjacent_boxes(
            raw_boxes,
            cx,
            cy,
            row_tol=args.row_tol,
            adj_gap=1.0,
            merge_bays=False,
            col_tol=args.col_tol,
            mode="balcony",
        )
        print(f"  {len(raw_boxes)} raw -> {len(merged_boxes)} units (mode=balcony)")
        print("=== 2 Associate to window floors/bays ===")
        snapped = snap_units(merged_boxes, windows_dsl)
        for u, mem in zip(snapped, members):
            u["member_raw_idxs"] = mem
        save(
            draw_snap(
                facade,
                snapped,
                f"2. Snap to window grid  n={len(snapped)}",
            ),
            out_dir / f"s2_snap_floors_bays_{stem}.png",
        )
    else:
        print("warn: no open balcony detections")

    enclosed_units: list[dict] = []
    if not bool(getattr(args, "no_enclosed", False)):
        print("=== 1c Enclosed volumes (SAM3+DA3) ===")
        vol_boxes = detect_enclosed_volume_boxes(
            facade,
            image_path,
            device=device,
            depth_dir=Path(getattr(args, "depth_dir", ROOT / "runs" / "da3_enclosure")),
            cache_path=Path(
                getattr(
                    args,
                    "enclosed_sam3_cache",
                    ROOT / "runs" / "sam3_da3_volumes" / "sam3_seeds.json",
                )
            ),
        )
        save(
            draw_boxes(
                facade,
                [r["box_xyxy"] for r in vol_boxes],
                title=f"1c. Enclosed volumes  n={len(vol_boxes)}",
                color=(220, 160, 40),
            ),
            out_dir / f"s1c_enclosed_volumes_{stem}.png",
            force=True,
        )
        enclosed_units = place_enclosed_volumes(
            vol_boxes, windows_dsl, unit_id0=len(snapped)
        )
        if enclosed_units:
            save(
                draw_snap(
                    facade,
                    enclosed_units,
                    f"2b. Enclosed snap  n={len(enclosed_units)}  (column width)",
                ),
                out_dir / f"s2b_enclosed_snap_{stem}.png",
            )
            print(
                f"  {len(vol_boxes)} volume(s) -> {len(enclosed_units)} floor placements"
            )
        if snapped and enclosed_units:
            snapped, dropped_open = drop_open_under_enclosed(snapped, enclosed_units)
            for u in dropped_open:
                print(
                    f"    drop open F{u.get('floor')} "
                    f"B{u.get('bay_start')}-{u.get('bay_end')} under enclosed"
                )

    units: list[dict] = []
    types_out: list[dict] = []
    profile_name = getattr(args, "recovery_profile", None) or DEFAULT_PROFILE
    axes_on = [k for k, v in resolve_profile(profile_name).items() if v]
    per_unit_railing = bool(getattr(args, "no_railing_vote", False))
    per_unit_floor = bool(getattr(args, "no_floor_shape_vote", False)) or per_unit_railing
    predictor = None
    rail_src = "heuristic"
    vote_note = f"profile={profile_name}"

    if not snapped:
        print("warn: no open balcony units after snap / enclosed conflict")
    else:
        merged_boxes = [u["box_xyxy"] for u in snapped]
        floors = [int(u["floor"]) for u in snapped]
        bays = [int(u["bay"]) for u in snapped]
        clustered = cluster_boxes(facade, merged_boxes, floors, bays, args, device)
        labels = clustered["labels"]
        feats = clustered["feats"]
        n_types = len(set(int(x) for x in labels.tolist()))
        save(
            draw_cluster_overlay(
                facade,
                merged_boxes,
                labels,
                f"3. Cluster balcony types  types={n_types} units={len(merged_boxes)}",
            ),
            out_dir / f"s3_cluster_balcony_types_{stem}.png",
        )

        medoids = rp.pick_medoids(feats, labels)
        for ui, u in enumerate(snapped):
            tid = int(labels[ui])
            type_dir = crops_dir / f"type_{tid:02d}"
            type_dir.mkdir(parents=True, exist_ok=True)
            crop_path = type_dir / f"unit_{ui:03d}.png"
            _write_image(facade.crop(tuple(u["box_xyxy"])), crop_path)
            units.append(
                {
                    **u,
                    "type_id": tid,
                    "asset": str(crop_path.relative_to(out_dir)),
                    "is_exemplar": ui == medoids[tid],
                }
            )

        predictor = resolve_railing_predictor(args, device)
        rail_src = "classifier" if predictor is not None else "heuristic"
        ir_label = "Classifier IR" if predictor is not None else "Heuristic IR"

        if per_unit_railing:
            print(
                f"=== 4 {ir_label} per unit (no vote; profile={profile_name} infer={axes_on}) ==="
            )
            for u in units:
                crop = Image.open(out_dir / u["asset"]).convert("RGB")
                ir = infer_unit_ir(
                    crop,
                    u,
                    image_size=facade.size,
                    profile_name=profile_name,
                    predictor=predictor,
                )
                tokens = ir_to_tokens(ir, profile_name=profile_name)
                kind = railing_kind_from_ir(ir)
                material = railing_material_from_ir(ir)
                u["structure_ir_member"] = ir
                u["structure_ir"] = ir
                u["structure_tokens"] = tokens
                mat_s = f" material={material}" if material else ""
                print(
                    f"  unit_{int(u['unit_id']):03d}: railing={kind}{mat_s} "
                    f"floor={floor_shape_from_ir(ir)} enclosure={enclosure_from_ir(ir)}"
                )
            types_out = _types_grouped_by_token(
                units, crops_dir, out_dir, include_floor=per_unit_floor
            )
            vote_note = (
                f"profile={profile_name} per-unit infer={axes_on} "
                f"rail={rail_src} floor=per-unit"
            )
        else:
            print(
                f"=== 4 {ir_label} + majority vote (profile={profile_name} vote={axes_on}) ==="
            )
            for tid, med_i in sorted(medoids.items()):
                exemplar = units[med_i]
                canon = crops_dir / f"type_{tid:02d}" / "exemplar.png"
                shutil.copy(out_dir / exemplar["asset"], canon)
                member_units = [u for u in units if int(u["type_id"]) == int(tid)]
                member_preds = []
                for u in member_units:
                    crop = Image.open(out_dir / u["asset"]).convert("RGB")
                    ir = infer_unit_ir(
                        crop,
                        u,
                        image_size=facade.size,
                        profile_name=profile_name,
                        predictor=predictor,
                    )
                    tokens = ir_to_tokens(ir, profile_name=profile_name)
                    member_preds.append(
                        {"unit_id": int(u["unit_id"]), "ir": ir, "tokens": tokens}
                    )
                    u["structure_ir_member"] = ir
                voted = vote_cluster_ir(
                    member_preds, prefer_unit_id=med_i, profile_name=profile_name
                )
                for u in member_units:
                    if per_unit_floor:
                        u["structure_ir"] = _with_unit_floor(
                            voted["structure_ir"], u["structure_ir_member"]
                        )
                    else:
                        u["structure_ir"] = voted["structure_ir"]
                vote = voted["vote"]
                print(
                    f"  type_{tid:02d}: vote "
                    f"{vote.get('winner_count', 0)}/{vote.get('n_valid', 0)} "
                    f"unique={vote.get('n_unique', 0)}"
                )
                if not per_unit_floor:
                    types_out.append(
                        {
                            "type_id": tid,
                            "name": f"balc_type_{tid:02d}",
                            "n_instances": len(member_units),
                            "exemplar_unit": med_i,
                            "exemplar_asset": str(canon.relative_to(out_dir)),
                            "structure_ir": voted["structure_ir"],
                            "structure_tokens": voted["structure_tokens"],
                            "structure_vote": voted["vote"],
                        }
                    )
            if per_unit_floor:
                types_out = _types_grouped_by_token(
                    units, crops_dir, out_dir, include_floor=True
                )
                for u in units:
                    ir = u.get("structure_ir") or {}
                    print(
                        f"  unit_{int(u['unit_id']):03d}: "
                        f"floor={floor_shape_from_ir(ir)} enclosure={enclosure_from_ir(ir)}"
                    )
            floor_note = " floor=per-unit" if per_unit_floor else ""
            vote_note = (
                f"profile={profile_name} vote={axes_on} rail={rail_src}{floor_note}"
            )

    if enclosed_units:
        enc_dir = crops_dir / "enclosed"
        enc_dir.mkdir(parents=True, exist_ok=True)
        for u in enclosed_units:
            crop_path = enc_dir / f"unit_{int(u['unit_id']):03d}.png"
            _write_image(facade.crop(tuple(u["box_xyxy"])), crop_path)
            u["asset"] = str(crop_path.relative_to(out_dir))
            u["type_id"] = 1000
        canon = enc_dir / "exemplar.png"
        shutil.copy(out_dir / enclosed_units[0]["asset"], canon)
        types_out.append(
            {
                "type_id": 1000,
                "name": "balc_enclosed",
                "n_instances": len(enclosed_units),
                "exemplar_unit": int(enclosed_units[0]["unit_id"]),
                "exemplar_asset": str(canon.relative_to(out_dir)),
                "structure_ir": enclosed_ir(),
                "structure_tokens": ["enclosure=enclosed"],
                "structure_vote": {
                    "mode": "enclosed_volume",
                    "n_members": len(enclosed_units),
                    "unit_ids": [int(u["unit_id"]) for u in enclosed_units],
                },
            }
        )
        units.extend(enclosed_units)
        print(f"=== 4b Enclosed IR  n={len(enclosed_units)} (column width, no railing) ===")

    if not units:
        print("warn: no balcony or enclosed detections; skip merge")
        return None

    save(
        draw_vote(types_out, out_dir, vote_note),
        out_dir / f"s4_vote_balcony_ir_{stem}.png",
    )

    print("=== 5 Merge into façade DSL ===")
    merged = merge_balcony_into_windows_dsl(
        windows_dsl,
        balcony_types=types_out,
        units=units,
        image_size=facade.size,
        per_unit_railing=per_unit_railing,
        per_unit_floor=per_unit_floor,
        center_mode=str(getattr(args, "balcony_center", "window")),
    )
    merged["meta"]["balcony_image"] = str(image_path)
    merged["meta"]["balcony_stem"] = stem
    merged["meta"]["balcony_recovery_profile"] = profile_name
    merged["meta"]["balcony_vote_axes"] = axes_on
    merged["meta"]["balcony_per_unit_railing"] = per_unit_railing
    merged["meta"]["balcony_per_unit_floor"] = per_unit_floor
    merged["meta"]["balcony_railing_source"] = rail_src
    merged["meta"]["n_enclosed_units"] = len(enclosed_units)
    if predictor is not None:
        merged["meta"]["balcony_railing_ckpt"] = str(predictor.ckpt_path)
    out_dsl = out_dir / "facade_dsl_with_balconies.json"
    _write_text(out_dsl, json.dumps(merged, indent=2) + "\n")
    print(f"  dsl -> {out_dsl}")
    _write_text(
        out_dir / "balcony_types.json",
        json.dumps(types_out, indent=2) + "\n",
    )
    print("done (meshes compile in window_compiler when you render the merged DSL)")
    return out_dsl


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
