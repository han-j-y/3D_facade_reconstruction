"""Side-by-side open-balcony types: DINOv2 appearance vs box geometry.

Detection and the window grid are shared. The left panel is the previous
spectral + Potts clustering. The right panel is z-scored box size, then the
window symmetry repair that scores twins with DINOv2.

Window floors and bays come from the CMP XML next to ``--xml-dir`` (same
column layout as the window track). Balcony boxes are SAM3, filtered and
snapped once.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_pipeline/compare_open_cluster.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_pipeline as rp  # noqa: E402
from cluster import cluster_balcony_boxes  # noqa: E402
from draw import draw_cluster_overlay  # noqa: E402
from facade_recovery.cmp_windows import load_cmp_xml_windows  # noqa: E402
from facade_recovery.column_layout import (  # noqa: E402
    assign_floors,
    infer_bay_column_bounds_from_units,
    infer_floor_row_bounds_from_units,
    infer_window_bay_columns,
)
from run_balcony import snap_open_units  # noqa: E402

DEFAULT_IMAGE_DIR = ROOT / "data" / "baseForBatch"
DEFAULT_XML_DIR = ROOT / "data" / "extended"
DEFAULT_OUT = ROOT / "runs" / "balcony_cluster_compare"


def windows_dsl_from_boxes(
    boxes: list[list[int]],
    image_size: tuple[int, int],
    *,
    facade_id: str,
    image_path: str,
    row_tol: float = 0.055,
) -> dict:
    """Minimal recovery DSL: floor×bay from window boxes, no structure IR."""
    iw, ih = image_size
    if len(boxes) < 2:
        raise ValueError(f"{facade_id}: need at least 2 windows")
    cy = np.array([0.5 * (b[1] + b[3]) / ih for b in boxes], dtype=np.float64)
    floors = assign_floors(cy, row_tol)
    units = [
        {
            "floor": int(floors[i]),
            "bay": 0,
            "box_xyxy": [int(v) for v in box],
            "kind": "window",
            "colspan": 1,
        }
        for i, box in enumerate(boxes)
    ]
    columns = infer_window_bay_columns(units, iw=iw)
    floor_ids = sorted({int(u["floor"]) for u in units})
    floors_y = infer_floor_row_bounds_from_units(units, floor_ids, ih=ih)
    columns_xy = infer_bay_column_bounds_from_units(
        [u for u in units if int(u.get("colspan", 1)) == 1] or units,
        len(columns),
        iw=iw,
        structural_columns=columns,
    )
    return {
        "schema": "facade_recovery_dsl_v1",
        "meta": {
            "facade_id": facade_id,
            "image": image_path,
            "image_size": [iw, ih],
            "source": "cmp_xml_grid",
            "columns_xy": [[round(a, 1), round(b, 1)] for a, b in columns_xy],
            "floors_y": [[round(a, 1), round(b, 1)] for a, b in floors_y],
        },
        "layout": {
            "floors": [{"id": int(f), "name": f"F{f}"} for f in floor_ids],
            "bays": [{"id": i, "name": f"B{i}"} for i in range(len(columns_xy))],
        },
        "instances": [
            {
                "kind": "window",
                "floor": int(u["floor"]),
                "bay": int(u["bay"]),
                "box_xyxy": u["box_xyxy"],
            }
            for u in units
        ],
    }


def windows_dsl_from_xml(image: Image.Image, xml_path: Path, stem: str) -> dict:
    raw = load_cmp_xml_windows(xml_path, image.size, stem=stem)
    boxes = [r["box_xyxy"] for r in raw]
    return windows_dsl_from_boxes(
        boxes,
        image.size,
        facade_id=stem,
        image_path=str(xml_path.with_suffix(".jpg")),
    )


def _n_types(labels: np.ndarray) -> int:
    if len(labels) == 0:
        return 0
    return len(set(int(x) for x in labels.tolist()))


def side_by_side(left: Image.Image, right: Image.Image, stem: str) -> Image.Image:
    gap = 12
    header = 36
    width = left.width + gap + right.width
    height = header + max(left.height, right.height)
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), stem, fill=(20, 20, 20))
    canvas.paste(left, (0, header))
    canvas.paste(right, (left.width + gap, header))
    return canvas


def _note(image: Image.Image, text: str) -> Image.Image:
    im = image.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    d.rectangle([8, 8, 8 + 9 * min(len(text), 90), 40], fill=(20, 20, 20))
    d.text((16, 12), text[:90], fill=(240, 240, 240))
    return im


def cluster_both(
    facade: Image.Image,
    units: list[dict],
    *,
    model,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    boxes = [u["box_xyxy"] for u in units]
    floors = [int(u["floor"]) for u in units]
    bays = [int(u["bay"]) for u in units]
    common = dict(
        model=model,
        device=device,
        facade_max_side=896,
        pca_dim=32,
        seed=seed,
        k_max=8,
        col_tol=0.04,
        row_tol=0.055,
        spatial_strength=1.8,
        unary_weight=0.9,
        patch=rp.patch,
        base=rp.base,
        reemb=rp.reemb,
    )
    old = cluster_balcony_boxes(
        facade, boxes, floors, bays, cluster_mode="appearance", **common
    )
    new = cluster_balcony_boxes(
        facade, boxes, floors, bays, cluster_mode="box", **common
    )
    labels, sym_log = rp.apply_symmetry_repair(
        facade,
        boxes,
        new["labels"],
        new["feats"],
        labels_box=new.get("labels_box"),
    )
    return old["labels"], labels, sym_log


def _load_sam(device: torch.device):
    from transformers import Sam3Model, Sam3Processor

    print("loading SAM3…", flush=True)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    sam = Sam3Model.from_pretrained("facebook/sam3").to(device)
    sam.eval()
    return sam, processor


def _load_dino(device: torch.device, name: str):
    print(f"loading {name}…", flush=True)
    model = torch.hub.load("facebookresearch/dinov2", name, pretrained=True)
    return model.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    ap.add_argument("--xml-dir", type=Path, default=DEFAULT_XML_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dino", default="dinov2_vits14")
    args = ap.parse_args()

    images = rp.iter_images_in_dir(args.image_dir)[: max(0, args.limit)]
    if not images:
        raise SystemExit(f"no images in {args.image_dir}")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"{len(images)} images  device={device}", flush=True)

    sam, processor = _load_sam(device)
    detected: dict[str, list[list[int]]] = {}
    for image_path in images:
        stem = image_path.stem
        cache = out_dir / f"{stem}_raw_boxes.json"
        facade = Image.open(image_path).convert("RGB")
        if cache.is_file():
            detected[stem] = json.loads(cache.read_text(encoding="utf-8"))
            print(f"detect cache {stem}  n={len(detected[stem])}", flush=True)
            continue
        raw = rp.detect_windows(
            facade,
            prompt="balcony",
            threshold=args.threshold,
            min_side=16,
            max_side_frac=1.0,
            device=device,
            sam=sam,
            processor=processor,
        )
        boxes = [r["box_xyxy"] for r in raw]
        cache.write_text(json.dumps(boxes), encoding="utf-8")
        detected[stem] = boxes
        print(f"detect {stem}  n={len(boxes)}", flush=True)
    del sam, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dino = _load_dino(device, args.dino)
    summary: list[dict] = []
    panels: list[Image.Image] = []
    for image_path in images:
        stem = image_path.stem
        facade = Image.open(image_path).convert("RGB")
        xml_path = args.xml_dir / f"{stem}.xml"
        raw_boxes = detected.get(stem, [])
        row: dict = {"stem": stem, "raw": len(raw_boxes)}
        if not xml_path.is_file():
            row["error"] = "missing xml"
            pair = side_by_side(
                _note(facade, f"{stem}  missing XML"),
                _note(facade, f"{stem}  missing XML"),
                stem,
            )
        else:
            try:
                dsl = windows_dsl_from_xml(facade, xml_path, stem)
                snapped, filtered, _log = snap_open_units(raw_boxes, facade, dsl)
                boxes = [u["box_xyxy"] for u in snapped]
                row["filtered"] = len(filtered)
                row["units"] = len(snapped)
                if len(snapped) < 2:
                    row["error"] = "fewer than 2 open balconies"
                    panel = _note(facade, f"{stem}  open units={len(snapped)}")
                    pair = side_by_side(panel, panel, stem)
                else:
                    old_labels, new_labels, sym_log = cluster_both(
                        facade, snapped, model=dino, device=device, seed=args.seed
                    )
                    row["appearance_types"] = _n_types(old_labels)
                    row["box_types"] = _n_types(new_labels)
                    row["symmetry"] = sym_log[-4:]
                    left = draw_cluster_overlay(
                        facade,
                        boxes,
                        old_labels,
                        f"DINOv2 appearance  types={row['appearance_types']} units={len(boxes)}",
                    )
                    right = draw_cluster_overlay(
                        facade,
                        boxes,
                        new_labels,
                        f"box geometry + symmetry  types={row['box_types']} units={len(boxes)}",
                    )
                    pair = side_by_side(left, right, stem)
                    (out_dir / f"{stem}_labels.json").write_text(
                        json.dumps(
                            {
                                "appearance": [int(x) for x in old_labels.tolist()],
                                "box": [int(x) for x in new_labels.tolist()],
                                "boxes": boxes,
                            }
                        ),
                        encoding="utf-8",
                    )
            except Exception as exc:  # noqa: BLE001 — one facade should not stop the batch
                row["error"] = f"{type(exc).__name__}: {exc}"
                panel = _note(facade, f"{stem}  {row['error']}"[:90])
                pair = side_by_side(panel, panel, stem)
                print(f"FAIL {stem}: {row['error']}", flush=True)
        dest = out_dir / f"{stem}_compare.jpg"
        pair.save(dest, quality=90)
        print(
            f"{stem}  raw={row.get('raw')}  units={row.get('units')}  "
            f"appearance={row.get('appearance_types')}  box={row.get('box_types')}  "
            f"-> {dest.name}",
            flush=True,
        )
        summary.append(row)
        thumb = pair.copy()
        thumb.thumbnail((1400, 700), Image.Resampling.LANCZOS)
        panels.append(thumb)

    del dino
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if panels:
        width = max(p.width for p in panels)
        height = sum(p.height for p in panels) + 8 * (len(panels) - 1)
        board = Image.new("RGB", (width, height), (255, 255, 255))
        y = 0
        for panel in panels:
            board.paste(panel, (0, y))
            y += panel.height + 8
        board_path = out_dir / "board.jpg"
        board.save(board_path, quality=85)
        print(f"board -> {board_path}", flush=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(summary)} compares under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
