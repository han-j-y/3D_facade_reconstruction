"""Load each heavy model once per process and reuse it (SAM3, DINOv2).

Every pipeline stage used to load its own copy: SAM3 seven times in the window track and twice in the balcony track,
DINOv2 three times, each load also checking the Hugging Face hub / GitHub over the network. Models come from the local
caches first (no network); only a missing cache falls back to downloading. Weights and eval mode are the same as before.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch


@functools.lru_cache(maxsize=None)
def _sam3(model_id: str, device: str):
    from transformers import Sam3Model, Sam3Processor

    try:
        processor = Sam3Processor.from_pretrained(model_id, local_files_only=True)
        sam = Sam3Model.from_pretrained(model_id, local_files_only=True)
    except OSError:  # not cached yet
        processor = Sam3Processor.from_pretrained(model_id)
        sam = Sam3Model.from_pretrained(model_id)
    return processor, sam.to(torch.device(device)).eval()


def sam3(device, model_id: str = "facebook/sam3"):
    """(processor, model) for SAM3 on ``device``, loaded once per process."""
    return _sam3(model_id, str(device))


@functools.lru_cache(maxsize=None)
def _dinov2(name: str, device: str):
    local = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if (local / "hubconf.py").is_file():
        net = torch.hub.load(str(local), name, source="local", pretrained=True)
    else:
        net = torch.hub.load("facebookresearch/dinov2", name, pretrained=True)
    return net.to(torch.device(device)).eval()


def dinov2(name: str, device):
    """DINOv2 backbone ``name`` on ``device``, loaded once per process from the torch.hub cache."""
    return _dinov2(name, str(device))


_VISION: dict = {}


def sam3_forward(sam, inputs: dict, image):
    """``sam(**inputs)`` with the image encoder run once per image and reused for every later prompt on it.

    SAM3 encodes the image independently of the text prompt, so passing the cached ``vision_embeds`` gives
    bit-identical outputs (checked on the test photos); each further prompt costs ~0.03 s instead of ~0.15 s."""
    import hashlib

    key = (id(sam), hashlib.md5(image.tobytes()).hexdigest(), image.size, tuple(inputs["pixel_values"].shape))
    vision = _VISION.get(key)
    if vision is None:
        if len(_VISION) >= 4:
            _VISION.clear()
        vision = _VISION[key] = sam.get_vision_features(pixel_values=inputs["pixel_values"])
    return sam(vision_embeds=vision, **{k: v for k, v in inputs.items() if k != "pixel_values"})
