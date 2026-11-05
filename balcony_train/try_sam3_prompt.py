"""Try SAM3 text prompts on full facade photos: every box, crops, and an overlay.

Named prompt sets (``--set``) keep a reusable mix of phrases. Each phrase is
a separate SAM3 call; boxes are then merged so a larger enclosure wins over
a nested railing.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/try_sam3_prompt.py ^
      --device cuda --in-dir "data\\test for enclosure" --set enclosure

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/try_sam3_prompt.py ^
      --device cuda --in-dir "data\\test for enclosure" ^
      --no-filter --set enclosure

Writes ``<out-dir>/<stem>_overlay.jpg`` (one color per prompt; white = merged)
and ``<out-dir>/<prompt_slug>/<stem>_NN_sX.XX.png`` crops for every box.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.recrop_sam3 import (  # noqa: E402
    clamp_box_xyxy,
    detect_balcony_records,
    list_input_images,
)

COLORS = (
    "red",
    "lime",
    "cyan",
    "yellow",
    "magenta",
    "orange",
    "deepskyblue",
    "chartreuse",
    "hotpink",
    "gold",
)
MERGE_COLOR = "white"

# Extra detectors for half_enclosed / enclosed only. Ordinary open railings
# stay on the pipeline's ``balcony`` prompt.
# Use nouns SAM3 actually returns (oriel/loggia scored 0 on the test folder).
PROMPT_SETS: dict[str, tuple[str, ...]] = {
    "balcony": ("balcony",),
    "glass": ("glass enclosed balcony",),
    "enclosure": (
        "glass enclosed balcony",
        "bay window",
        "colonnade",
    ),
    "enclosure_alt": (
        "glazed balcony",
        "bay window",
        "arcade",
    ),
    # Enclosed volume only: no window/balcony/glass. Windows stay on the
    # window pipeline; this set looks for a block that juts from the wall.
    "projecting": (
        "projecting bay",
        "oriel",
        "facade projection",
        "protruding wall",
        "canted bay",
    ),
    # Search which phrase hits the projecting enclosed volume (not half-enclosed).
    "enclosed_search": (
        "glass enclosed balcony",
        "glazed balcony",
        "enclosed balcony",
        "bay window",
        "oriel window",
        "bow window",
        "sunroom",
        "conservatory",
        "window bay",
        "enclosed porch",
    ),
    # Second search: volume / gallery / projection nouns vs yellow blocks.
    "enclosed_search2": (
        "closed balcony",
        "covered balcony",
        "glass balcony",
        "balcony enclosure",
        "enclosed veranda",
        "glazed veranda",
        "winter garden",
        "enclosed loggia",
        "glazed loggia",
        "enclosed gallery",
        "glazed gallery",
        "bay",
        "oriel window",
        "projecting bay",
        "protruding bay",
        "facade projection",
        "building projection",
        "cantilevered bay",
        "glass box",
        "glazed extension",
        "glazed structure",
    ),
    # Boxes at least ~one storey tall (pair with --min-height-frac 0.10).
    "one_floor": (
        "enclosed balcony",
        "closed balcony",
        "glazed balcony",
    ),
}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _area(box: tuple[int, int, int, int] | list[int]) -> float:
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def _inter(
    a: tuple[int, int, int, int] | list[int],
    b: tuple[int, int, int, int] | list[int],
) -> float:
    w = min(int(a[2]), int(b[2])) - max(int(a[0]), int(b[0]))
    h = min(int(a[3]), int(b[3])) - max(int(a[1]), int(b[1]))
    return max(0, w) * max(0, h)


def keep_enclosure_box(
    box: tuple[int, int, int, int] | list[int],
    *,
    iw: int,
    ih: int,
    min_width_frac: float = 0.08,
    min_height_frac: float = 0.03,
    max_aspect: float = 5.0,
    max_flat_aspect: float = 6.5,
    max_compact_area_frac: float = 0.04,
) -> bool:
    """False for sashes (narrow/tall) and handrails (wide/short)."""
    x0, y0, x1, y1 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    bw = max(0, x1 - x0)
    bh = max(0, y1 - y0)
    if bw < 1 or bh < 1:
        return False
    if bw < min_width_frac * iw:
        return False
    if bh < min_height_frac * ih:
        return False
    aspect = bh / bw
    if aspect > max_aspect:
        return False
    if bw / bh > max_flat_aspect:
        return False
    area = bw * bh
    img_area = max(1, iw * ih)
    if 0.7 <= aspect <= 1.4 and area < max_compact_area_frac * img_area:
        return False
    return True


def merge_boxes(records: list[dict], overlap: float = 0.6) -> list[dict]:
    """Keep the larger box when a smaller one is mostly inside it."""
    kept: list[dict] = []
    for rec in sorted(records, key=lambda r: -_area(r["box_xyxy"])):
        box = rec["box_xyxy"]
        area = _area(box)
        if area <= 0:
            continue
        if any(_inter(box, k["box_xyxy"]) >= overlap * area for k in kept):
            continue
        kept.append(rec)
    return kept


def resolve_prompts(prompt_set: str | None, extra: list[str] | None) -> list[str]:
    names: list[str] = []
    if prompt_set:
        if prompt_set not in PROMPT_SETS:
            known = ", ".join(PROMPT_SETS)
            raise SystemExit(f"unknown --set {prompt_set!r}; expected one of: {known}")
        names.extend(PROMPT_SETS[prompt_set])
    if extra:
        names.extend(extra)
    if not names:
        names = list(PROMPT_SETS["enclosure"])
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "sam3_prompt_test")
    ap.add_argument(
        "--set",
        dest="prompt_set",
        default=None,
        choices=sorted(PROMPT_SETS),
        help="named prompt mix (default: enclosure if --prompt is omitted)",
    )
    ap.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="extra phrase; repeat to add more. Combined with --set if both are set",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="SAM3 score cutoff for raw boxes (before any geometry filter)",
    )
    ap.add_argument(
        "--no-filter",
        action="store_true",
        help="show every raw SAM3 box; skip size/score filters (use while tuning prompts)",
    )
    ap.add_argument("--min-side", type=int, default=16)
    ap.add_argument(
        "--min-side-frac",
        type=float,
        default=0.0,
        help="SAM3 min side as a fraction of image width (both W and H; 0 = --min-side only)",
    )
    ap.add_argument(
        "--min-width-frac",
        type=float,
        default=0.08,
        help="drop boxes narrower than this fraction of image width (ordinary sashes)",
    )
    ap.add_argument(
        "--min-height-frac",
        type=float,
        default=0.03,
        help="drop boxes shorter than this fraction of image height (handrails)",
    )
    ap.add_argument(
        "--max-aspect",
        type=float,
        default=5.0,
        help="drop boxes taller than this height/width (sashes; stacked oriels stay)",
    )
    ap.add_argument(
        "--max-flat-aspect",
        type=float,
        default=6.5,
        help="drop boxes wider than this width/height (rails; one-storey galleries stay)",
    )
    ap.add_argument(
        "--min-score",
        type=float,
        default=0.18,
        help="drop SAM3 boxes below this score (weak open-balcony matches)",
    )
    ap.add_argument(
        "--max-compact-area-frac",
        type=float,
        default=0.04,
        help="drop near-square boxes smaller than this fraction of the image",
    )
    ap.add_argument("--max-side-frac", type=float, default=1.0)
    ap.add_argument(
        "--merge-overlap",
        type=float,
        default=0.6,
        help="drop a box if this share of its area sits inside a larger kept box",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    prompts = resolve_prompts(args.prompt_set, args.prompt)
    paths = list_input_images(args.in_dir)
    if not paths:
        raise SystemExit(f"no images in {args.in_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "prompts.json").write_text(
        json.dumps(
            {
                "set": args.prompt_set,
                "prompts": prompts,
                "threshold": args.threshold,
                "min_side": args.min_side,
                "min_side_frac": args.min_side_frac,
                "min_width_frac": args.min_width_frac,
                "min_height_frac": args.min_height_frac,
                "max_aspect": args.max_aspect,
                "max_flat_aspect": args.max_flat_aspect,
                "max_compact_area_frac": args.max_compact_area_frac,
                "min_score": args.min_score,
                "max_side_frac": args.max_side_frac,
                "merge_overlap": args.merge_overlap,
                "no_filter": bool(args.no_filter),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    from transformers import Sam3Model, Sam3Processor

    device = torch.device(args.device)
    print("loading SAM3…", flush=True)
    print("named sets:", flush=True)
    for name, phrases in PROMPT_SETS.items():
        mark = " *" if phrases == tuple(prompts) else ""
        print(f"  {name}: {' / '.join(phrases)}{mark}", flush=True)
    print("prompts: " + " | ".join(prompts), flush=True)
    if args.no_filter:
        print("filter=off (raw boxes only)", flush=True)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    sam = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()

    for path in paths:
        image = Image.open(path).convert("RGB")
        overlay = image.copy()
        raw_overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        raw_draw = ImageDraw.Draw(raw_overlay)
        line_w = max(2, image.width // 300)
        min_side = max(args.min_side, int(args.min_side_frac * image.width))
        pooled: list[dict] = []
        for k, prompt in enumerate(prompts):
            color = COLORS[k % len(COLORS)]
            raw = detect_balcony_records(
                image,
                processor=processor,
                sam=sam,
                device=device,
                prompt=prompt,
                threshold=args.threshold,
                min_side=min_side,
                max_side_frac=args.max_side_frac,
            )
            if args.no_filter:
                records = list(raw)
            else:
                records = [
                    rec
                    for rec in raw
                    if float(rec.get("score", 0.0)) >= args.min_score
                    and keep_enclosure_box(
                        rec["box_xyxy"],
                        iw=image.width,
                        ih=image.height,
                        min_width_frac=args.min_width_frac,
                        min_height_frac=args.min_height_frac,
                        max_aspect=args.max_aspect,
                        max_flat_aspect=args.max_flat_aspect,
                        max_compact_area_frac=args.max_compact_area_frac,
                    )
                ]
            for rec in raw:
                box = clamp_box_xyxy(rec["box_xyxy"], width=image.width, height=image.height)
                raw_draw.rectangle(box, outline=color, width=max(1, line_w - 1))
            records.sort(key=lambda r: -float(r["score"]))
            crop_dir = args.out_dir / slug(prompt)
            crop_dir.mkdir(parents=True, exist_ok=True)
            prompt_view = image.copy()
            prompt_draw = ImageDraw.Draw(prompt_view)
            for j, rec in enumerate(records):
                rec["prompt"] = k
                rec["prompt_text"] = prompt
                box = clamp_box_xyxy(rec["box_xyxy"], width=image.width, height=image.height)
                rec["box_xyxy"] = list(box)
                image.crop(box).save(crop_dir / f"{path.stem}_{j:02d}_s{rec['score']:.2f}.png")
                draw.rectangle(box, outline=color, width=line_w)
                draw.text((box[0] + 4, box[1] + 4), f"{k}:{rec['score']:.2f}", fill=color)
                prompt_draw.rectangle(box, outline=color, width=line_w)
                prompt_draw.text((box[0] + 4, box[1] + 4), f"{rec['score']:.2f}", fill=color)
            prompt_view.save(crop_dir / f"{path.stem}_overlay.jpg", quality=90)
            pooled.extend(records)
            scores = " ".join(f"{float(r['score']):.2f}" for r in records)
            print(
                f"{path.name}  [{prompt}]  raw={len(raw)} kept={len(records)}  {scores}",
                flush=True,
            )

        merged = merge_boxes(pooled, args.merge_overlap)
        merged_dir = args.out_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        for j, rec in enumerate(merged):
            box = clamp_box_xyxy(rec["box_xyxy"], width=image.width, height=image.height)
            image.crop(box).save(
                merged_dir / f"{path.stem}_{j:02d}_p{rec['prompt']}_s{rec['score']:.2f}.png"
            )
            draw.rectangle(box, outline=MERGE_COLOR, width=line_w + 2)
        print(f"{path.name}  merged={len(merged)}", flush=True)
        overlay.save(args.out_dir / f"{path.stem}_overlay.jpg", quality=90)
        raw_overlay.save(args.out_dir / f"{path.stem}_raw_overlay.jpg", quality=90)

    legend = ", ".join(f"{COLORS[k % len(COLORS)]}={p}" for k, p in enumerate(prompts))
    print(f"done -> {args.out_dir}\nlegend: {legend}\nwhite=merged (filtered)\n*_raw_overlay.jpg = SAM3 before filter")


if __name__ == "__main__":
    main()
