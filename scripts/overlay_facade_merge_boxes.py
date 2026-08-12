#!/usr/bin/env python3
"""Merge adjacent window *boxes* (not just cluster labels), then cluster.

Asset crops often split a multi-pane unit into several boxes. This script:
  1. Finds same-floor horizontally adjacent boxes
  2. Replaces them with their axis-aligned union (one box per unit)
  3. Re-embeds each merged box via full-facade DINO patch ROI
  4. Clusters (spectral_rbf) and overlays

Also optionally merges a whole vertical bay into one tall box (``--merge-bays``).

Example:
  python scripts/overlay_facade_merge_boxes.py \\
    --facade-ids 221,252,108,121,17,15,400,270,112,263,22 \\
    --device cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


base = _load(ROOT / "scripts" / "overlay_facade_asset_clusters.py", "ov_base")
patch = _load(ROOT / "scripts" / "overlay_facade_patch_layout.py", "ov_patch")
lay = _load(ROOT / "scripts" / "overlay_facade_layout_first.py", "ov_lay")
hints = _load(ROOT / "scripts" / "overlay_facade_patch_group_hints.py", "ov_hints")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--assets-dir",
        type=Path,
        default=EXP / "data" / "facades" / "assets" / "windows",
    )
    ap.add_argument(
        "--facade-dir",
        type=Path,
        default=EXP / "data" / "facades" / "train_up",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "runs" / "facade_asset_overlays_merge_boxes",
    )
    ap.add_argument("--dino", default="dinov2_vits14")
    ap.add_argument("--facade-max-side", type=int, default=896)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-windows", type=int, default=4)
    ap.add_argument("--max-facades", type=int, default=20)
    ap.add_argument(
        "--facade-ids",
        type=str,
        default="221,252,108,121,17,15,400,270,112,263,22,189,335,328,388,204,283,339,47,74",
    )
    ap.add_argument("--match-thr", type=float, default=0.55)
    ap.add_argument("--col-tol", type=float, default=0.045)
    ap.add_argument("--row-tol", type=float, default=0.055)
    ap.add_argument(
        "--adj-gap",
        type=float,
        default=1.0,
        help="max gap between boxes as fraction of median width (pane gaps often ~0.8)",
    )
    ap.add_argument(
        "--merge-bays",
        action="store_true",
        help="also union all boxes in the same bay into one tall box",
    )
    ap.add_argument("--k-max", type=int, default=8)
    return ap.parse_args()


def union_box(boxes: list[list[int]]) -> list[int]:
    return [
        int(min(b[0] for b in boxes)),
        int(min(b[1] for b in boxes)),
        int(max(b[2] for b in boxes)),
        int(max(b[3] for b in boxes)),
    ]


def _box_area(b: list[int]) -> float:
    return max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))


def _intersection(a: list[int], b: list[int]) -> list[int] | None:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def containment_or_iou(a: list[int], b: list[int], *, contain_thr: float, iou_thr: float) -> bool:
    inter = _intersection(a, b)
    if inter is None:
        return False
    ia = _box_area(inter)
    aa, ba = _box_area(a), _box_area(b)
    if aa <= 0 or ba <= 0:
        return False
    if ia / min(aa, ba) >= contain_thr:
        return True
    if ia / (aa + ba - ia) >= iou_thr:
        return True
    return False


def y_overlap_frac(a: list[int], b: list[int]) -> float:
    y0, y1 = max(a[1], b[1]), min(a[3], b[3])
    oh = y1 - y0
    if oh <= 0:
        return 0.0
    return float(oh) / max(1.0, min(a[3] - a[1], b[3] - b[1]))


def x_gap(a: list[int], b: list[int]) -> float:
    if a[2] <= b[0]:
        return float(b[0] - a[2])
    if b[2] <= a[0]:
        return float(a[0] - b[2])
    return 0.0


def merge_adjacent_boxes(
    boxes: list[list[int]],
    cx: np.ndarray,
    cy: np.ndarray,
    *,
    row_tol: float,
    adj_gap: float,
    merge_bays: bool,
    col_tol: float,
    contain_thr: float = 0.55,
    iou_thr: float = 0.25,
    extruded_gap: float = 2.2,
    min_y_overlap: float = 0.30,
    narrow_frac: float = 0.9,
    tight_gap: float = 0.55,
    cross_bay_gap: float = 1.0,
) -> tuple[list[list[int]], list[list[int]], np.ndarray, np.ndarray]:
    """Merge pane faces of one opening; keep separate columns apart.

    - Nested/IoU always.
    - Same bay + same floor: generous gap (projecting bay faces).
    - Different bay: only if a box is narrow (flat multi-pane) and gap is small.
    - Optional: force whole vertical bay.
    """
    n = len(boxes)
    floor = lay.assign_bays(cy, row_tol)
    bay = lay.assign_bays(cx, col_tol)
    med_w = float(np.median([hints.box_width(b) for b in boxes]))
    uf = lay.UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if containment_or_iou(
                boxes[i], boxes[j], contain_thr=contain_thr, iou_thr=iou_thr
            ):
                uf.union(i, j)

    for i in range(n):
        for j in range(i + 1, n):
            if int(floor[i]) != int(floor[j]):
                continue
            if y_overlap_frac(boxes[i], boxes[j]) < min_y_overlap:
                continue
            gap = x_gap(boxes[i], boxes[j])
            wi = hints.box_width(boxes[i])
            wj = hints.box_width(boxes[j])
            narrow = min(wi, wj) <= narrow_frac * med_w
            same_bay = int(bay[i]) == int(bay[j])

            if same_bay and gap <= extruded_gap * med_w:
                # projecting / multi-face unit inside one vertical bay
                uf.union(i, j)
            elif (not same_bay) and narrow and gap <= cross_bay_gap * med_w:
                # flat coplanar panes that fell into adjacent bay ids
                uf.union(i, j)
            elif gap <= tight_gap * med_w:
                uf.union(i, j)

    if merge_bays:
        for b in sorted(set(int(v) for v in bay.tolist())):
            idxs = [i for i in range(n) if int(bay[i]) == b]
            for i in idxs[1:]:
                uf.union(idxs[0], i)

    labels = uf.labels()
    groups: dict[int, list[int]] = {}
    for i, lab in enumerate(labels.tolist()):
        groups.setdefault(int(lab), []).append(i)

    merged_boxes: list[list[int]] = []
    members: list[list[int]] = []
    ordered = sorted(
        groups.items(),
        key=lambda kv: (
            min(boxes[i][1] for i in kv[1]),
            min(boxes[i][0] for i in kv[1]),
        ),
    )
    for _lab, idxs in ordered:
        merged_boxes.append(union_box([boxes[i] for i in idxs]))
        members.append(sorted(idxs))

    return merged_boxes, members, bay, floor


def draw_before_after(
    facade: Image.Image,
    boxes_before: list[list[int]],
    items_before: list[dict[str, Any]],
    labels_before: np.ndarray,
    items_after: list[dict[str, Any]],
    labels_after: np.ndarray,
    *,
    fid: str,
    n_merged_from: int,
) -> Image.Image:
    left = base.draw_overlay(
        facade,
        items_before,
        labels_before,
        title=f"facade {fid}  raw boxes  n={len(boxes_before)}",
    )
    right = base.draw_overlay(
        facade,
        items_after,
        labels_after,
        title=(
            f"facade {fid}  merged boxes  n={len(items_after)}  "
            f"(from {n_merged_from})"
        ),
    )
    h = max(left.height, right.height)
    out = Image.new("RGB", (left.width + 8 + right.width, h), (12, 12, 12))
    out.paste(left, (0, 0))
    out.paste(right, (left.width + 8, 0))
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    facade_ids = [s.strip() for s in args.facade_ids.split(",") if s.strip()][
        : args.max_facades
    ]
    print(f"facades={len(facade_ids)} merge_bays={args.merge_bays} adj_gap={args.adj_gap}")

    print(f"loading {args.dino}…")
    model = torch.hub.load("facebookresearch/dinov2", args.dino, pretrained=True)
    model = model.to(device).eval()

    summary = []
    done_ids = []

    for fi, fid in enumerate(facade_ids):
        facade_path = args.facade_dir / f"{fid}.png"
        if not facade_path.is_file():
            continue
        items = base.load_facade_items(args.assets_dir, fid)
        if len(items) < args.min_windows:
            continue

        facade_im = Image.open(facade_path).convert("RGB")
        facade_rgb = np.array(facade_im)
        iw, ih = facade_im.size
        kept: list[dict[str, Any]] = []
        for it in items:
            crop = np.array(Image.open(it["image"]).convert("RGB"))
            box, score = base.find_box(facade_rgb, crop)
            if box is None or score < args.match_thr:
                continue
            rec = dict(it)
            rec["box_xyxy"] = list(box)
            rec["match_score"] = float(score)
            kept.append(rec)
        if len(kept) < args.min_windows:
            continue

        boxes = [it["box_xyxy"] for it in kept]
        cx = np.array([0.5 * (b[0] + b[2]) / iw for b in boxes], dtype=np.float64)
        cy = np.array([0.5 * (b[1] + b[3]) / ih for b in boxes], dtype=np.float64)

        merged_boxes, members, bay_raw, floor_raw = merge_adjacent_boxes(
            boxes,
            cx,
            cy,
            row_tol=args.row_tol,
            adj_gap=args.adj_gap,
            merge_bays=args.merge_bays,
            col_tol=args.col_tol,
        )
        # layout on merged
        mcx = np.array(
            [0.5 * (b[0] + b[2]) / iw for b in merged_boxes], dtype=np.float64
        )
        mcy = np.array(
            [0.5 * (b[1] + b[3]) / ih for b in merged_boxes], dtype=np.float64
        )
        bay_m = lay.assign_bays(mcx, args.col_tol)
        floor_m = lay.assign_bays(mcy, args.row_tol)

        n_raw, n_merged = len(boxes), len(merged_boxes)
        print(
            f"[{fi+1}/{len(facade_ids)}] facade {fid}: "
            f"boxes {n_raw} → {n_merged} "
            f"(merged {n_raw - n_merged} panes away)"
        )

        # features on merged ROIs
        spatial, meta = patch.facade_patch_spatial(
            model, facade_im, device=device, max_side=args.facade_max_side
        )
        # baseline: raw boxes
        feats_raw = base.apply_pca(
            patch.roi_pool_patches(spatial, boxes, meta),
            min(args.pca_dim, max(2, n_raw - 1)),
            args.seed,
        )
        k_raw = base.select_k(feats_raw, "spectral_rbf", args.k_max, args.seed)
        labels_raw = base.cluster_features(feats_raw, "spectral_rbf", k_raw, args.seed)

        feats_m = base.apply_pca(
            patch.roi_pool_patches(spatial, merged_boxes, meta),
            min(args.pca_dim, max(2, n_merged - 1)),
            args.seed,
        )
        if n_merged >= 4:
            k_m = base.select_k(feats_m, "spectral_rbf", min(args.k_max, n_merged - 1), args.seed)
            labels_m = base.cluster_features(feats_m, "spectral_rbf", k_m, args.seed)
        elif n_merged >= 2:
            labels_m = base.cluster_features(feats_m, "spectral_rbf", 2, args.seed)
        else:
            labels_m = np.zeros(n_merged, dtype=np.int32)

        items_before = [
            {"box_xyxy": b, "id": kept[i]["id"]} for i, b in enumerate(boxes)
        ]
        items_after = []
        crops_after = []
        for mi, mb in enumerate(merged_boxes):
            x0, y0, x1, y1 = mb
            crop = facade_im.crop((x0, y0, x1, y1))
            crops_after.append(crop)
            items_after.append(
                {
                    "box_xyxy": mb,
                    "id": f"{fid}_m{mi}",
                    "members": [kept[j]["id"] for j in members[mi]],
                    "n_members": len(members[mi]),
                }
            )

        facade_out = out_dir / f"facade_{fid}"
        facade_out.mkdir(parents=True, exist_ok=True)

        # raw / merged overlays
        ov_raw = base.draw_overlay(
            facade_im,
            items_before,
            labels_raw,
            title=f"facade {fid}  raw boxes  n={n_raw}  k={len(set(labels_raw.tolist()))}",
        )
        ov_m = base.draw_overlay(
            facade_im,
            items_after,
            labels_m,
            title=(
                f"facade {fid}  merged boxes  n={n_merged}  "
                f"k={len(set(labels_m.tolist()))}"
            ),
        )
        combo = draw_before_after(
            facade_im,
            boxes,
            items_before,
            labels_raw,
            items_after,
            labels_m,
            fid=fid,
            n_merged_from=n_raw,
        )
        side = base.make_side_by_side(ov_m, crops_after, labels_m)

        ov_raw.save(facade_out / "overlay_raw.png")
        ov_m.save(facade_out / "overlay_merged.png")
        combo.save(facade_out / "compare_raw_vs_merged.png")
        side.save(facade_out / "overlay_merged_with_crops.png")

        (facade_out / "merged.json").write_text(
            json.dumps(
                {
                    "facade_id": fid,
                    "n_raw": n_raw,
                    "n_merged": n_merged,
                    "adj_gap": args.adj_gap,
                    "merge_bays": args.merge_bays,
                    "units": [
                        {
                            "id": items_after[i]["id"],
                            "box_xyxy": merged_boxes[i],
                            "members": items_after[i]["members"],
                            "cluster": int(labels_m[i]),
                            "bay": int(bay_m[i]),
                            "floor": int(floor_m[i]),
                        }
                        for i in range(n_merged)
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        summary.append(
            {
                "facade_id": fid,
                "n_raw": n_raw,
                "n_merged": n_merged,
                "k_merged": len(set(int(c) for c in labels_m.tolist())),
            }
        )
        done_ids.append(fid)

    # overview of merged overlays
    thumbs = []
    for fid in done_ids:
        p = out_dir / f"facade_{fid}" / "compare_raw_vs_merged.png"
        if not p.is_file():
            continue
        im = Image.open(p).convert("RGB")
        im.thumbnail((720, 400), Image.Resampling.LANCZOS)
        thumbs.append((fid, im))
    if thumbs:
        cols = 2
        rows = (len(thumbs) + cols - 1) // cols
        cw = max(t.width for _, t in thumbs)
        ch = max(t.height for _, t in thumbs) + 20
        grid = Image.new("RGB", (cols * cw + 8, rows * ch + 8), (16, 16, 16))
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
            )
        except OSError:
            font = ImageFont.load_default()
        draw = ImageDraw.Draw(grid)
        for i, (fid, im) in enumerate(thumbs):
            r, c = divmod(i, cols)
            x, y = 4 + c * cw, 4 + r * ch
            draw.text((x + 4, y), f"facade {fid}", fill=(230, 230, 230), font=font)
            grid.paste(im, (x, y + 18))
        grid.save(out_dir / "overview_raw_vs_merged.png")
        print(f"overview → {out_dir / 'overview_raw_vs_merged.png'}")

    # 221 spotlight
    p221 = out_dir / "facade_221" / "compare_raw_vs_merged.png"
    if p221.is_file():
        Image.open(p221).save(out_dir / "facade_221_raw_vs_merged.png")
        print(f"221 → {out_dir / 'facade_221_raw_vs_merged.png'}")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
