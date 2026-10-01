#!/usr/bin/env python3
"""Symmetry repair on frozen e2e_sam3 baselines (read-only).

Does **not** modify the stage-1 run dirs. Reads
``runs/e2e_sam3_cluster_*`` boxes + type labels, re-embeds DINO
for the repair objective, applies ``repair_all_clusters`` about the ghost-
continuous ``chosen_cx`` when available (else bbox midpoint), and writes
before/after overlays (cyan axis marked) under
``runs/symmetry_from_sam3/``.

Example:
  python scripts/run_symmetry_from_sam3_e2e.py --device cuda
  python scripts/run_symmetry_from_sam3_e2e.py --stems cmp_b0079,cmp_b0001
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.overlay_facade_asset_clusters import draw_overlay  # noqa: E402
from scripts.walkthrough_symmetry_repair import (  # noqa: E402
    _load_script,
    facade_axis_center,
    facade_sym,
    match_reflected,
    reflect_box,
    repair_all_clusters,
)
from window_ast.symmetry import bounding_box  # noqa: E402

TITLE_BAR_H = 36
AXIS_COLOR = (0, 220, 255)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sam3-root",
        type=Path,
        default=ROOT / "runs",
        help="read-only root containing e2e_sam3_cluster_* dirs",
    )
    ap.add_argument(
        "--axis-root",
        type=Path,
        default=ROOT / "runs" / "symmetry_axis_sweep" / "ghost_continuous",
        help="ghost-continuous summaries with chosen_cx (preferred repair axis)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "runs" / "symmetry_from_sam3",
    )
    ap.add_argument(
        "--stems",
        default="",
        help="comma stems; default = all e2e_sam3_cluster_cmp_* under --sam3-root",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dino", default="dinov2_vits14")
    ap.add_argument("--facade-max-side", type=int, default=896)
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    # SAM-friendly defaults (looser geometric match, no private types)
    ap.add_argument("--match-iou", type=float, default=0.08)
    ap.add_argument("--match-center", type=float, default=0.55)
    ap.add_argument("--twin-mirror-min", type=float, default=0.30)
    ap.add_argument("--size-rel-tol", type=float, default=0.40)
    ap.add_argument(
        "--size-mismatch-sym-gain",
        type=float,
        default=0.004,
        help="force leftover relabel despite size mismatch if facade Sym "
        "rises by at least this much",
    )
    ap.add_argument(
        "--size-mismatch-partner-iou",
        type=float,
        default=0.45,
        help="force size-mismatch join when reflected partner IoU ≥ this "
        "(crop truncation twins)",
    )
    ap.add_argument(
        "--size-mismatch-contain",
        type=float,
        default=0.85,
        help="force size-mismatch join when smaller box is this covered by "
        "reflection of larger (pediment-vs-pane)",
    )
    ap.add_argument("--allow-private", action="store_true", default=False)
    return ap.parse_args()


def resolve_axis(
    stem: str,
    boxes: list[list[float]],
    image_size: tuple[int, int],
    axis_root: Path,
) -> tuple[float, float, list[float], str]:
    """Prefer ghost-continuous ``chosen_cx``; else bbox midpoint."""
    cx0, cy0, extent = facade_axis_center(
        boxes, extent_mode="boxes", image_size=image_size
    )
    summary_path = axis_root / stem / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        cx = float(summary["chosen_cx"])
        return cx, cy0, extent, "chosen_cx"
    return cx0, cy0, extent, "bbox_mid"


def _draw_axis(panel: Image.Image, cx: float, *, label: str) -> None:
    """Draw vertical symmetry axis on a ``draw_overlay`` panel (title-bar offset)."""
    d = ImageDraw.Draw(panel)
    x = int(round(cx))
    y0 = TITLE_BAR_H
    y1 = panel.height - 1
    d.line([(x, y0), (x, y1)], fill=AXIS_COLOR, width=3)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except OSError:
        font = ImageFont.load_default()
    d.text((x + 6, y0 + 4), label, fill=AXIS_COLOR, font=font)


def discover_stems(sam3_root: Path) -> list[str]:
    stems = sorted(
        p.name.replace("e2e_sam3_cluster_", "")
        for p in sam3_root.glob("e2e_sam3_cluster_cmp_*")
        if p.is_dir() and (p / "facade_dsl.json").is_file()
    )
    return stems


def load_sam3_baseline(run_dir: Path) -> tuple[Image.Image, list[list[float]], list[int], dict]:
    dsl = json.loads((run_dir / "facade_dsl.json").read_text())
    idx = {}
    if (run_dir / "index_raw.json").is_file():
        idx = json.loads((run_dir / "index_raw.json").read_text())
    facade_path = Path(
        idx.get("facade_path")
        or dsl.get("meta", {}).get("facade_path")
        or ""
    )
    if not facade_path.is_file():
        cand = next((b / facade_path for b in (ROOT, EXP) if (b / facade_path).is_file()), None)
        if cand is not None:
            facade_path = cand
        else:
            stem = run_dir.name.replace("e2e_sam3_cluster_", "")
            facade_path = EXP / "data" / "facades" / "base" / f"{stem}.jpg"
    facade = Image.open(facade_path).convert("RGB")
    units = [u for u in dsl.get("instances", []) if u.get("kind") != "door"]
    boxes = [list(map(float, u["box_xyxy"])) for u in units]
    labels = [int(u["type_id"]) for u in units]
    meta = {
        "facade_path": str(facade_path),
        "n_units": len(boxes),
        "n_types": len(set(labels)),
        "run_dir": str(run_dir),
    }
    return facade, boxes, labels, meta


def embed_roi(
    facade: Image.Image,
    boxes: list[list[float]],
    *,
    model,
    patch,
    base,
    device: torch.device,
    facade_max_side: int,
    pca_dim: int,
    seed: int,
) -> np.ndarray:
    int_boxes = [[int(round(v)) for v in b] for b in boxes]
    spatial, meta = patch.facade_patch_spatial(
        model, facade, device=device, max_side=facade_max_side
    )
    raw = patch.roi_pool_patches(spatial, int_boxes, meta)
    n = len(boxes)
    return base.apply_pca(raw, min(pca_dim, max(2, n - 1)), seed)


def draw_pair(
    facade: Image.Image,
    boxes: list[list[float]],
    before: list[int],
    after: list[int],
    title_l: str,
    title_r: str,
    changed: set[int],
    *,
    cx: float,
    axis_source: str,
) -> Image.Image:
    items = [{"box_xyxy": b, "id": f"u{i:03d}"} for i, b in enumerate(boxes)]
    left = draw_overlay(facade, items, np.asarray(before), title=title_l)
    right = draw_overlay(facade, items, np.asarray(after), title=title_r)
    # yellow outline on changed (facade coords → panel coords below title bar)
    if changed:
        d = ImageDraw.Draw(right)
        for i in changed:
            x0, y0, x1, y1 = boxes[i]
            d.rectangle(
                [x0, y0 + TITLE_BAR_H, x1, y1 + TITLE_BAR_H],
                outline=(255, 220, 40),
                width=3,
            )
    axis_label = f"axis cx={cx:.0f} ({axis_source})"
    _draw_axis(left, cx, label=axis_label)
    _draw_axis(right, cx, label=axis_label)
    h = max(left.height, right.height)
    gap = 10
    canvas = Image.new("RGB", (left.width + gap + right.width, h), (24, 24, 26))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    return canvas


def diagnose_matches(
    boxes: list[list[float]],
    labels: list[int],
    *,
    cx: float,
    match_iou: float,
    match_center: float,
) -> dict:
    n = len(boxes)
    same = cross = none = 0
    for i in range(n):
        rb = reflect_box(boxes[i], axis="v", cx=cx, cy=0.0)
        j, _ = match_reflected(
            rb, boxes, iou_thr=match_iou, center_thr=match_center, exclude=i
        )
        if j is None:
            none += 1
        elif int(labels[j]) == int(labels[i]):
            same += 1
        else:
            cross += 1
    return {
        "already_same": same,
        "cross_type": cross,
        "no_match": none,
        "n": n,
    }


def main() -> None:
    args = parse_args()
    stems = (
        [s.strip() for s in args.stems.split(",") if s.strip()]
        if args.stems.strip()
        else discover_stems(args.sam3_root)
    )
    if not stems:
        raise SystemExit(f"no e2e_sam3 stems under {args.sam3_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}  stems={len(stems)}  out={args.out_dir}")
    print(
        f"match_iou={args.match_iou} match_center={args.match_center} "
        f"twin_mirror_min={args.twin_mirror_min} size_rel_tol={args.size_rel_tol} "
        f"size_mismatch_sym_gain={args.size_mismatch_sym_gain} "
        f"partner_iou≥{args.size_mismatch_partner_iou} "
        f"contain≥{args.size_mismatch_contain} "
        f"allow_private={args.allow_private}"
    )

    patch = _load_script(ROOT / "scripts" / "overlay_facade_patch_layout.py", "sym_patch")
    base = _load_script(ROOT / "scripts" / "overlay_facade_asset_clusters.py", "sym_base")
    print(f"load {args.dino}…")
    model = torch.hub.load("facebookresearch/dinov2", args.dino, pretrained=True)
    model = model.to(device).eval()

    summary = []
    for stem in stems:
        run_dir = args.sam3_root / f"e2e_sam3_cluster_{stem}"
        if not run_dir.is_dir():
            print(f"skip missing {run_dir}")
            continue
        print(f"\n=== {stem} ===")
        facade, boxes, labels_before, meta = load_sam3_baseline(run_dir)
        n = len(boxes)
        if n < 2:
            print("  <2 units, skip")
            continue

        feats = embed_roi(
            facade,
            boxes,
            model=model,
            patch=patch,
            base=base,
            device=device,
            facade_max_side=args.facade_max_side,
            pca_dim=args.pca_dim,
            seed=args.seed,
        )
        iw, ih = facade.size
        extent_t = bounding_box(boxes)
        if extent_t is None:
            print("  empty extent, skip")
            continue
        cx, cy, extent, axis_source = resolve_axis(
            stem, boxes, (iw, ih), args.axis_root
        )
        windows = [
            {"id": f"u{i:03d}", "box_xyxy": boxes[i], "cluster": int(labels_before[i])}
            for i in range(n)
        ]

        diag0 = diagnose_matches(
            boxes,
            labels_before,
            cx=cx,
            match_iou=args.match_iou,
            match_center=args.match_center,
        )
        print(
            f"  baseline types={meta['n_types']} units={n}  "
            f"axis={cx:.1f} ({axis_source})  "
            f"mirror: same={diag0['already_same']} cross={diag0['cross_type']} "
            f"none={diag0['no_match']}"
        )

        fs0 = facade_sym(
            boxes, labels_before, axis="v", extent_mode="boxes", image_size=(iw, ih)
        )
        s0, g0 = float(fs0.normalized), float(fs0.normalized_global)

        labels_after, log, _panels = repair_all_clusters(
            boxes=boxes,
            labels=list(labels_before),
            feats=feats,
            windows=windows,
            facade=facade,
            axis="v",
            cx=cx,
            cy=cy,
            facade_extent=list(extent),
            img_size=(iw, ih),
            extent_mode="boxes",
            twin_mirror_min=args.twin_mirror_min,
            match_iou=args.match_iou,
            match_center=args.match_center,
            allow_private=args.allow_private,
            size_rel_tol=args.size_rel_tol,
            size_mismatch_sym_gain=args.size_mismatch_sym_gain,
            size_mismatch_partner_iou=args.size_mismatch_partner_iou,
            size_mismatch_contain=args.size_mismatch_contain,
        )
        # remap contiguous
        uniq = sorted({int(x) for x in labels_after})
        remap = {c: i for i, c in enumerate(uniq)}
        labels_after = [remap[int(x)] for x in labels_after]

        changed = [i for i in range(n) if int(labels_before[i]) != int(labels_after[i])]
        # membership change (co-members)
        def comembers(labs):
            g: dict[int, set[int]] = {}
            for i, l in enumerate(labs):
                g.setdefault(int(l), set()).add(i)
            return [frozenset(g[int(labs[i])] - {i}) for i in range(len(labs))]

        mem_chg = sum(
            1
            for i, (a, b) in enumerate(zip(comembers(labels_before), comembers(labels_after)))
            if a != b
        )
        k0 = len(set(labels_before))
        k1 = len(set(labels_after))
        print(f"  types {k0}→{k1}  label_moves={len(changed)}  Δmem={mem_chg}")
        for line in log[-6:]:
            print(f"    {line}")

        stem_dir = args.out_dir / stem
        stem_dir.mkdir(parents=True, exist_ok=True)
        panel = draw_pair(
            facade,
            boxes,
            labels_before,
            labels_after,
            f"{stem} BEFORE  k={k0}",
            f"{stem} AFTER  k={k1}  moves={len(changed)}",
            set(changed),
            cx=cx,
            axis_source=axis_source,
        )
        panel.save(stem_dir / "before_after.png")
        (stem_dir / "log.txt").write_text("\n".join(log) + "\n")
        row = {
            "stem": stem,
            "baseline": meta,
            "k_before": k0,
            "k_after": k1,
            "label_moves": len(changed),
            "membership_changed": mem_chg,
            "axis_cx": cx,
            "axis_source": axis_source,
            "sizes_before": dict(sorted(Counter(labels_before).items())),
            "sizes_after": dict(sorted(Counter(int(x) for x in labels_after).items())),
            "mirror_diag": diag0,
            "obj_before": {"facade": s0, "glob": g0},
            "changed_ids": changed,
            "log_tail": log[-8:],
            "params": {
                "match_iou": args.match_iou,
                "match_center": args.match_center,
                "twin_mirror_min": args.twin_mirror_min,
                "size_rel_tol": args.size_rel_tol,
                "size_mismatch_sym_gain": args.size_mismatch_sym_gain,
                "size_mismatch_partner_iou": args.size_mismatch_partner_iou,
                "size_mismatch_contain": args.size_mismatch_contain,
                "allow_private": args.allow_private,
            },
        }
        (stem_dir / "result.json").write_text(json.dumps(row, indent=2) + "\n")
        summary.append(row)

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n==== SUMMARY ====")
    for r in summary:
        print(
            f"{r['stem']:12}  axis={r['axis_cx']:.1f}({r['axis_source']})  "
            f"k {r['k_before']}→{r['k_after']}  "
            f"moves={r['label_moves']:3}  Δmem={r['membership_changed']:3}  "
            f"mirror cross={r['mirror_diag']['cross_type']} none={r['mirror_diag']['no_match']}"
        )
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
