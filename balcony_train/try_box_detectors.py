"""Zero-shot box detectors on the enclosure test photos (not SAM3).

Runs Grounding DINO, LLMDet, and Qwen2.5-VL. Each writes axis-aligned boxes
and an overlay. Does not replace the SAM3 window/balcony pipeline.

Example::

    C:\\Users\\hiroo\\anaconda3\\envs\\glodon\\python.exe balcony_train/try_box_detectors.py ^
      --device cuda --in-dir "data\\test for enclosure" --out-dir runs\\box_detectors
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from balcony_train.recrop_sam3 import clamp_box_xyxy, list_input_images  # noqa: E402
from balcony_train.try_sam3_prompt import merge_boxes  # noqa: E402

DEFAULT_IN_DIR = ROOT / "data" / "test for enclosure"
DEFAULT_OUT_DIR = ROOT / "runs" / "box_detectors"

# Same volume phrases that SAM3 actually fired, plus stacked-gallery nouns.
DEFAULT_LABELS = (
    "enclosed balcony",
    "closed balcony",
    "glazed balcony",
    "glazed gallery",
    "oriel window",
    "projecting bay",
)

QWEN_PROMPT = (
    "Locate every projecting enclosed facade volume: glazed galleries, "
    "oriel / bay stacks, enclosed balconies, canted bays. "
    "Return ONE bounding box per stacked volume that spans all floors of "
    "the same projection. Do not box ordinary open railings, trees, cars, "
    "people, or single window sashes. "
    'Output JSON only: [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}].'
)

BACKENDS = {
    "grounding_dino": "IDEA-Research/grounding-dino-tiny",
    "llmdet": "iSEE-Laboratory/llmdet_tiny",
    "qwen25vl": "Qwen/Qwen2.5-VL-3B-Instruct",
}

COLORS = (
    "red",
    "lime",
    "cyan",
    "yellow",
    "magenta",
    "orange",
    "deepskyblue",
    "chartreuse",
)

_BBOX_RE = re.compile(
    r"\{[^{}]*?bbox_2d[^{}]*?\}",
    re.IGNORECASE | re.DOTALL,
)
_BBOX_ARRAY_RE = re.compile(
    r"bbox_2d\"?\s*:\s*\[\s*([-+eE0-9.,\s]+)\s*\]",
    re.IGNORECASE,
)
_NUMS_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def grounding_text(labels: tuple[str, ...] | list[str]) -> str:
    """Grounding DINO wants lowercase phrases separated by ``. ``."""
    return " . ".join(s.strip().lower() for s in labels if s.strip()) + " ."


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 768 * 28 * 28,
) -> tuple[int, int]:
    """Qwen2.5-VL image size (without ``qwen_vl_utils``)."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def drop_full_frame(
    records: list[dict[str, Any]],
    *,
    iw: int,
    ih: int,
    max_area_frac: float = 0.55,
) -> list[dict[str, Any]]:
    """Drop boxes that cover most of the photo (whole-facade hits)."""
    limit = max_area_frac * max(1, iw * ih)
    kept: list[dict[str, Any]] = []
    for rec in records:
        x0, y0, x1, y1 = rec["box_xyxy"]
        if (int(x1) - int(x0)) * (int(y1) - int(y0)) >= limit:
            continue
        kept.append(rec)
    return kept


def scale_box(
    box: list[float] | tuple[float, ...],
    *,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> list[int]:
    sx = dst_w / max(1, src_w)
    sy = dst_h / max(1, src_h)
    x0, y0, x1, y1 = [float(v) for v in box]
    return [
        int(round(x0 * sx)),
        int(round(y0 * sy)),
        int(round(x1 * sx)),
        int(round(y1 * sy)),
    ]


def parse_qwen_boxes(text: str) -> list[dict[str, Any]]:
    """Pull ``bbox_2d`` records out of a Qwen JSON / markdown reply."""
    records: list[dict[str, Any]] = []
    for chunk in _BBOX_RE.findall(text or ""):
        arr = _BBOX_ARRAY_RE.search(chunk)
        nums = [float(n) for n in _NUMS_RE.findall(arr.group(1))] if arr else []
        if len(nums) < 4:
            continue
        label_m = re.search(r'"label"\s*:\s*"([^"]*)"', chunk, re.IGNORECASE)
        records.append(
            {
                "box_xyxy": [nums[0], nums[1], nums[2], nums[3]],
                "label": (label_m.group(1) if label_m else "volume").strip(),
                "score": 1.0,
            }
        )
    if records:
        return records
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    items = data if isinstance(data, list) else [data]
    for item in items:
        if not isinstance(item, dict):
            continue
        box = item.get("bbox_2d") or item.get("box_xyxy")
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        records.append(
            {
                "box_xyxy": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "label": str(item.get("label") or "volume"),
                "score": float(item.get("score") or 1.0),
            }
        )
    return records


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def _records_from_grounded(result: dict, iw: int, ih: int) -> list[dict[str, Any]]:
    boxes = result.get("boxes")
    scores = result.get("scores")
    labels = result.get("text_labels") or result.get("labels") or []
    out: list[dict[str, Any]] = []
    if boxes is None:
        return out
    for i, box in enumerate(boxes):
        xyxy = [float(v) for v in box.tolist()]
        score = float(scores[i]) if scores is not None else 0.0
        lab = labels[i] if i < len(labels) else ""
        if hasattr(lab, "item"):
            lab = lab.item()
        rec = {
            "box_xyxy": list(
                clamp_box_xyxy(xyxy, width=iw, height=ih)
            ),
            "score": score,
            "label": str(lab),
        }
        out.append(rec)
    return out


@torch.no_grad()
def detect_grounding_dino(
    image: Image.Image,
    *,
    processor,
    model,
    device: torch.device,
    labels: tuple[str, ...] | list[str],
    threshold: float,
    text_threshold: float,
) -> list[dict[str, Any]]:
    text = grounding_text(labels)
    inputs = processor(images=image, text=text, return_tensors="pt")
    outputs = model(**_to_device(inputs, device))
    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs.get("input_ids"),
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[(image.height, image.width)],
    )
    return _records_from_grounded(results[0], image.width, image.height)


@torch.no_grad()
def detect_llmdet(
    image: Image.Image,
    *,
    processor,
    model,
    device: torch.device,
    labels: tuple[str, ...] | list[str],
    threshold: float,
) -> list[dict[str, Any]]:
    text_labels = [list(labels)]
    inputs = processor(images=image, text=text_labels, return_tensors="pt")
    outputs = model(**_to_device(inputs, device))
    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=[(image.height, image.width)],
        text_labels=text_labels,
    )
    return _records_from_grounded(results[0], image.width, image.height)


@torch.no_grad()
def detect_qwen25vl(
    image: Image.Image,
    *,
    processor,
    model,
    device: torch.device,
    prompt: str,
    max_new_tokens: int = 512,
    max_pixels: int = 768 * 28 * 28,
) -> tuple[list[dict[str, Any]], str]:
    iw, ih = image.size
    rh, rw = smart_resize(ih, iw, max_pixels=max_pixels)
    resized = image.resize((rw, rh), Image.Resampling.BICUBIC)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    chat = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[chat], images=[resized], return_tensors="pt", padding=True)
    if "image_grid_thw" in inputs:
        _t, gh, gw = [int(v) for v in inputs["image_grid_thw"][0].tolist()]
        patch = int(getattr(processor.image_processor, "patch_size", 14))
        rh, rw = gh * patch, gw * patch
    if "pixel_values" in inputs and hasattr(model, "dtype"):
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    inputs = _to_device(inputs, device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    in_len = inputs["input_ids"].shape[1]
    text = processor.batch_decode(generated[:, in_len:], skip_special_tokens=True)[0]
    records: list[dict[str, Any]] = []
    for rec in parse_qwen_boxes(text):
        box = scale_box(
            rec["box_xyxy"], src_w=rw, src_h=rh, dst_w=iw, dst_h=ih
        )
        rec["box_xyxy"] = list(clamp_box_xyxy(box, width=iw, height=ih))
        records.append(rec)
    return records, text


def draw_overlay(
    image: Image.Image,
    records: list[dict[str, Any]],
    *,
    labels: tuple[str, ...] | list[str],
) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    line_w = max(2, image.width // 300)
    color_for = {name: COLORS[i % len(COLORS)] for i, name in enumerate(labels)}
    for rec in records:
        box = rec["box_xyxy"]
        name = str(rec.get("label") or "")
        color = color_for.get(name, "white")
        draw.rectangle(box, outline=color, width=line_w)
        score = float(rec.get("score") or 0.0)
        tag = f"{name} {score:.2f}".strip()
        draw.text((int(box[0]) + 4, int(box[1]) + 4), tag, fill=color)
    return overlay


def _unload(*objs: Any) -> None:
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_backend(
    name: str,
    paths: list[Path],
    *,
    out_root: Path,
    device: torch.device,
    labels: tuple[str, ...],
    threshold: float,
    text_threshold: float,
    qwen_prompt: str,
) -> None:
    model_id = BACKENDS[name]
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading {name} ({model_id})…", flush=True)

    if name == "qwen25vl":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=None,
        ).to(device).eval()
    else:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()

    summary: list[dict[str, Any]] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        raw_text = ""
        if name == "grounding_dino":
            records = detect_grounding_dino(
                image,
                processor=processor,
                model=model,
                device=device,
                labels=labels,
                threshold=threshold,
                text_threshold=text_threshold,
            )
        elif name == "llmdet":
            records = detect_llmdet(
                image,
                processor=processor,
                model=model,
                device=device,
                labels=labels,
                threshold=threshold,
            )
        else:
            records, raw_text = detect_qwen25vl(
                image,
                processor=processor,
                model=model,
                device=device,
                prompt=qwen_prompt,
            )
        if name != "qwen25vl":
            records = drop_full_frame(records, iw=image.width, ih=image.height)
        merged = merge_boxes(records, 0.6)
        overlay = draw_overlay(image, merged, labels=labels)
        overlay.save(out_dir / f"{path.stem}_overlay.jpg", quality=90)
        draw_overlay(image, records, labels=labels).save(
            out_dir / f"{path.stem}_raw_overlay.jpg", quality=90
        )
        if raw_text:
            (out_dir / f"{path.stem}_qwen.txt").write_text(raw_text, encoding="utf-8")
        (out_dir / f"{path.stem}_boxes.json").write_text(
            json.dumps(
                {
                    "file": path.name,
                    "n_raw": len(records),
                    "n_merged": len(merged),
                    "boxes": merged,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        scores = " ".join(f"{r['score']:.2f}" for r in merged)
        print(
            f"{path.name}  [{name}]  raw={len(records)} merged={len(merged)}  {scores}",
            flush=True,
        )
        summary.append(
            {
                "file": path.name,
                "n_raw": len(records),
                "n_merged": len(merged),
            }
        )

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "backend": name,
                "model_id": model_id,
                "labels": list(labels),
                "threshold": threshold,
                "images": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"done -> {out_dir}", flush=True)
    _unload(model, processor)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--backend",
        action="append",
        choices=sorted(BACKENDS),
        default=None,
        help="repeat to pick a subset (default: all three)",
    )
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    paths = list_input_images(args.in_dir)
    if not paths:
        raise SystemExit(f"no images in {args.in_dir}")
    backends = args.backend or list(BACKENDS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print("backends: " + ", ".join(backends), flush=True)
    print("labels: " + " | ".join(DEFAULT_LABELS), flush=True)
    for name in backends:
        try:
            run_backend(
                name,
                paths,
                out_root=args.out_dir,
                device=device,
                labels=DEFAULT_LABELS,
                threshold=args.threshold,
                text_threshold=args.text_threshold,
                qwen_prompt=QWEN_PROMPT,
            )
        except Exception as exc:
            print(f"FAILED {name}: {exc}", flush=True)
            _unload()


if __name__ == "__main__":
    main()
