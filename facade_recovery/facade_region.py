"""The photo's facade region: where the wall ends, instead of at the outermost windows.

SAM3's `facade` box, grown to hold every window and door. With no usable detection, the whole image is the facade.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

MIN_SCORE = 0.5
MIN_ELEMENTS_INSIDE = 0.5  # a box holding fewer of the windows and doors is another building or a fragment


def choose_facade_region(
    detections: Sequence[tuple[Sequence[float], float]],
    element_boxes: Sequence[Sequence[float]],
    image_size: tuple[int, int],
) -> tuple[list[float], str]:
    """``(box_xyxy, source)``; source is "sam3" or "image". ``detections`` are ``(box, score)`` pairs."""
    iw, ih = image_size
    centers = [(0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])) for b in element_boxes]

    def inside(box: Sequence[float]) -> float:
        if not centers:
            return 1.0
        return sum(box[0] <= x <= box[2] and box[1] <= y <= box[3] for x, y in centers) / len(centers)

    usable = [(inside(b), s, b) for b, s in detections if s >= MIN_SCORE and inside(b) >= MIN_ELEMENTS_INSIDE]
    if not usable:
        return [0.0, 0.0, float(iw), float(ih)], "image"
    _, _, best = max(usable, key=lambda d: (d[0], d[1]))
    boxes = [list(best), *[list(b) for b in element_boxes]]
    return [
        max(0.0, min(float(b[0]) for b in boxes)),
        max(0.0, min(float(b[1]) for b in boxes)),
        min(float(iw), max(float(b[2]) for b in boxes)),
        min(float(ih), max(float(b[3]) for b in boxes)),
    ], "sam3"


@torch.no_grad()
def detect_facade_boxes(
    image: Image.Image, *, device: torch.device, prompt: str = "facade", threshold: float = 0.3
) -> list[tuple[list[float], float]]:
    """SAM3 boxes for ``prompt`` as ``(box_xyxy, score)``."""
    from transformers import Sam3Model, Sam3Processor

    from facade_recovery import model_cache

    processor, sam = model_cache.sam3(device)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    # Same padding box as run_pipeline.detect_windows, so the geometry encoder runs here too.
    inputs["input_boxes"] = torch.tensor([[[0.5, 0.5, 0.1, 0.1]]], device=device)
    inputs["input_boxes_labels"] = torch.tensor([[-10]], device=device)
    res = processor.post_process_instance_segmentation(
        model_cache.sam3_forward(sam, inputs, image), threshold=threshold, mask_threshold=0.5, target_sizes=inputs["original_sizes"].tolist()
    )[0]
    boxes = res["boxes"].detach().float().cpu().numpy()
    scores = np.asarray(res["scores"].detach().float().cpu().numpy())
    del sam
    torch.cuda.empty_cache()
    return [([float(v) for v in b], float(s)) for b, s in zip(boxes, scores)]
