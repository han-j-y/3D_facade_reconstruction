"""Frozen DINOv2 + linear head: crop → railing.kind (see labels.CLASSES)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image

from balcony_train.labels import CLASSES
from balcony_train.paths import DEFAULT_RAILING_CKPT

BACKBONE = "dinov2_vits14"
FEAT_DIM = 384
IMAGE_SIZE = 224

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def make_transform(image_size: int = IMAGE_SIZE):
    from torchvision import transforms as T

    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


class RailingClassifier(nn.Module):
    """DINOv2 ViT-S/14 (frozen) + Linear(384, n_classes). Only the head is trained."""

    def __init__(self, *, pretrained: bool = True, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.classes = CLASSES
        self.freeze_backbone = bool(freeze_backbone)
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", BACKBONE, pretrained=bool(pretrained)
        )
        self.feat_dim = int(getattr(self.backbone, "embed_dim", FEAT_DIM))
        self.head = nn.Linear(self.feat_dim, len(CLASSES))
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                feat = self.backbone(x)
            feat = feat.detach()
        else:
            feat = self.backbone(x)
        return self.head(feat)


def load_backbone(model: RailingClassifier, device: torch.device) -> None:
    model.backbone.to(device)
    model.head.to(device)
    if model.freeze_backbone:
        model.backbone.eval()


def save_checkpoint(model: RailingClassifier, path: Path, **extra: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "head": model.head.state_dict(),
        "classes": list(model.classes),
        "backbone": BACKBONE,
        "image_size": IMAGE_SIZE,
        **extra,
    }
    torch.save(payload, path)


def load_classifier(
    ckpt_path: Path,
    device: torch.device,
    *,
    pretrained_backbone: bool = True,
) -> RailingClassifier:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = RailingClassifier(pretrained=pretrained_backbone, freeze_backbone=True)
    model.head.load_state_dict(ckpt["head"])
    saved = tuple(ckpt.get("classes") or CLASSES)
    if saved != CLASSES:
        raise ValueError(f"ckpt classes {saved} != {CLASSES}")
    load_backbone(model, device)
    model.eval()
    return model


class RailingPredictor:
    """One-shot crop → railing kind. Load once per pipeline run."""

    def __init__(self, ckpt_path: Path, device: torch.device) -> None:
        self.ckpt_path = Path(ckpt_path)
        self.device = device
        self.model = load_classifier(self.ckpt_path, device)
        self.tfm = make_transform()

    @torch.no_grad()
    def predict(self, crop: Image.Image) -> str:
        x = self.tfm(crop.convert("RGB")).unsqueeze(0).to(self.device)
        idx = int(self.model(x).argmax(dim=1).item())
        return CLASSES[idx]

    def __call__(self, crop: Image.Image) -> str:
        return self.predict(crop)


def try_load_predictor(
    ckpt_path: Path | None,
    device: torch.device,
    *,
    force_off: bool = False,
) -> RailingPredictor | None:
    """Return a predictor if ckpt exists; None → caller uses the heuristic."""
    if force_off:
        return None
    path = Path(ckpt_path) if ckpt_path else DEFAULT_RAILING_CKPT
    if not path.is_file():
        return None
    return RailingPredictor(path, device)
