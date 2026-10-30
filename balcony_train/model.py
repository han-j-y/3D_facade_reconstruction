"""Frozen DINOv2 + multitask heads: kind, material, floor shape, enclosure, skin, structure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image

from balcony_train.labels import (
    CLASSES,
    ENCLOSURE_SKINS,
    ENCLOSURES,
    FLOOR_SHAPES,
    LABEL_STRUCTURES,
    MATERIALS,
)
from balcony_train.paths import DEFAULT_RAILING_CKPT, DEFAULT_STRUCTURE_CKPT

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
    """DINOv2 ViT-S/14 (frozen) + kind, material, floor, enclosure, skin, and structure heads."""

    def __init__(self, *, pretrained: bool = True, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.classes = CLASSES
        self.materials = MATERIALS
        self.floor_shapes = FLOOR_SHAPES
        self.enclosures = ENCLOSURES
        self.enclosure_skins = ENCLOSURE_SKINS
        self.structures = LABEL_STRUCTURES
        self.freeze_backbone = bool(freeze_backbone)
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", BACKBONE, pretrained=bool(pretrained)
        )
        self.feat_dim = int(getattr(self.backbone, "embed_dim", FEAT_DIM))
        self.kind_head = nn.Linear(self.feat_dim, len(CLASSES))
        self.material_head = nn.Linear(self.feat_dim, len(MATERIALS))
        self.floor_head = nn.Linear(self.feat_dim, len(FLOOR_SHAPES))
        self.enclosure_head = nn.Linear(self.feat_dim, len(ENCLOSURES))
        self.skin_head = nn.Linear(self.feat_dim, len(ENCLOSURE_SKINS))
        self.structure_head = nn.Linear(self.feat_dim, len(LABEL_STRUCTURES))
        # Legacy alias used by older single-head checkpoints / code paths.
        self.head = self.kind_head
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                feat = self.backbone(x)
            return feat.detach()
        return self.backbone(x)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.encode(x)
        return (
            self.kind_head(feat),
            self.material_head(feat),
            self.floor_head(feat),
            self.enclosure_head(feat),
            self.skin_head(feat),
            self.structure_head(feat),
        )


def load_backbone(model: RailingClassifier, device: torch.device) -> None:
    model.backbone.to(device)
    model.kind_head.to(device)
    model.material_head.to(device)
    model.floor_head.to(device)
    model.enclosure_head.to(device)
    model.skin_head.to(device)
    model.structure_head.to(device)
    if model.freeze_backbone:
        model.backbone.eval()


def save_checkpoint(model: RailingClassifier, path: Path, **extra: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "multitask": True,
        "kind_head": model.kind_head.state_dict(),
        "material_head": model.material_head.state_dict(),
        "floor_head": model.floor_head.state_dict(),
        "enclosure_head": model.enclosure_head.state_dict(),
        "skin_head": model.skin_head.state_dict(),
        "structure_head": model.structure_head.state_dict(),
        # Keep legacy key for tools that only inspect classes.
        "head": model.kind_head.state_dict(),
        "classes": list(model.classes),
        "materials": list(model.materials),
        "floor_shapes": list(model.floor_shapes),
        "enclosures": list(model.enclosures),
        "enclosure_skins": list(model.enclosure_skins),
        "skin_trained": bool(getattr(model, "_skin_trained", False)),
        "structures": list(model.structures),
        "structure_trained": bool(getattr(model, "_structure_trained", False)),
        "backbone": BACKBONE,
        "image_size": IMAGE_SIZE,
        **extra,
    }
    torch.save(payload, path)


def apply_skin_checkpoint(model: RailingClassifier, ckpt: dict[str, Any]) -> bool:
    """Load ``skin_head`` when the checkpoint trained it. Missing head stays untrained."""
    skins = tuple(ckpt.get("enclosure_skins") or ())
    has_head = "skin_head" in ckpt and skins == tuple(ENCLOSURE_SKINS)
    if "skin_trained" in ckpt:
        trained = bool(ckpt.get("skin_trained")) and has_head
    else:
        trained = has_head
    if trained:
        model.skin_head.load_state_dict(ckpt["skin_head"])
    model._skin_trained = trained  # type: ignore[attr-defined]
    return trained


def apply_structure_checkpoint(model: RailingClassifier, ckpt: dict[str, Any]) -> bool:
    """Load ``structure_head`` when the checkpoint trained it. Older checkpoints have none."""
    names = tuple(ckpt.get("structures") or ())
    trained = (
        bool(ckpt.get("structure_trained"))
        and "structure_head" in ckpt
        and names == tuple(LABEL_STRUCTURES)
    )
    if trained:
        model.structure_head.load_state_dict(ckpt["structure_head"])
    model._structure_trained = trained  # type: ignore[attr-defined]
    return trained


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
    saved = tuple(ckpt.get("classes") or CLASSES)
    if saved != CLASSES:
        raise ValueError(f"ckpt classes {saved} != {CLASSES}")
    if "kind_head" in ckpt:
        model.kind_head.load_state_dict(ckpt["kind_head"])
    elif "head" in ckpt:
        model.kind_head.load_state_dict(ckpt["head"])
    else:
        raise ValueError(f"checkpoint missing kind head: {ckpt_path}")
    mats = tuple(ckpt.get("materials") or ())
    if "material_head" in ckpt and mats == MATERIALS:
        model.material_head.load_state_dict(ckpt["material_head"])
        model._material_trained = True  # type: ignore[attr-defined]
    else:
        model._material_trained = False  # type: ignore[attr-defined]
    floors = tuple(ckpt.get("floor_shapes") or ())
    if "floor_head" in ckpt and floors == FLOOR_SHAPES:
        model.floor_head.load_state_dict(ckpt["floor_head"])
        model._floor_trained = True  # type: ignore[attr-defined]
    else:
        model._floor_trained = False  # type: ignore[attr-defined]
    encs = tuple(ckpt.get("enclosures") or ())
    if "enclosure_head" in ckpt and encs == ENCLOSURES:
        model.enclosure_head.load_state_dict(ckpt["enclosure_head"])
        model._enclosure_trained = True  # type: ignore[attr-defined]
    else:
        model._enclosure_trained = False  # type: ignore[attr-defined]
    apply_skin_checkpoint(model, ckpt)
    apply_structure_checkpoint(model, ckpt)
    load_backbone(model, device)
    model.eval()
    return model


class RailingPredictor:
    """One-shot crop → railing kind (+ material when open_work)."""

    def __init__(
        self,
        ckpt_path: Path,
        device: torch.device,
        *,
        structure_ckpt: Path | None = None,
    ) -> None:
        self.ckpt_path = Path(ckpt_path)
        self.device = device
        self.model = load_classifier(self.ckpt_path, device)
        self.structure_ckpt: Path | None = None
        if structure_ckpt is not None and not getattr(self.model, "_structure_trained", False):
            self.structure_ckpt = load_structure_head(self.model, Path(structure_ckpt), device)
        self.tfm = make_transform()

    @torch.no_grad()
    def predict_full(self, crop: Image.Image) -> dict[str, str | None]:
        x = self.tfm(crop.convert("RGB")).unsqueeze(0).to(self.device)
        kind_logits, mat_logits, floor_logits, enc_logits, skin_logits, structure_logits = self.model(x)
        kind = CLASSES[int(kind_logits.argmax(dim=1).item())]
        material: str | None = None
        if kind == "open_work":
            if getattr(self.model, "_material_trained", False):
                material = MATERIALS[int(mat_logits.argmax(dim=1).item())]
            else:
                material = "metal"
        floor_shape: str | None = None
        if getattr(self.model, "_floor_trained", False):
            floor_shape = FLOOR_SHAPES[int(floor_logits.argmax(dim=1).item())]
        enclosure: str | None = None
        if getattr(self.model, "_enclosure_trained", False):
            enclosure = ENCLOSURES[int(enc_logits.argmax(dim=1).item())]
        skin: str | None = None
        if getattr(self.model, "_skin_trained", False):
            skin = ENCLOSURE_SKINS[int(skin_logits.argmax(dim=1).item())]
        # Structure is an open-balcony label; None when the head is untrained.
        structure: str | None = None
        if getattr(self.model, "_structure_trained", False) and enclosure in (None, "open"):
            structure = LABEL_STRUCTURES[int(structure_logits.argmax(dim=1).item())]
        return {
            "kind": kind,
            "material": material,
            "floor_shape": floor_shape,
            "enclosure": enclosure,
            "skin": skin,
            "structure": structure,
        }

    @torch.no_grad()
    def predict(self, crop: Image.Image) -> str:
        return str(self.predict_full(crop)["kind"])

    def __call__(self, crop: Image.Image) -> str:
        return self.predict(crop)


def load_structure_head(
    model: RailingClassifier,
    ckpt_path: Path,
    device: torch.device,
) -> Path | None:
    """Copy a trained structure head from another checkpoint onto ``model``.

    Every checkpoint shares the frozen DINOv2 backbone, so one head can move
    between them. Returns the path when the head loaded, else None.
    """
    if not ckpt_path.is_file():
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not apply_structure_checkpoint(model, ckpt):
        return None
    model.structure_head.to(device)
    return ckpt_path


def try_load_predictor(
    ckpt_path: Path | None,
    device: torch.device,
    *,
    force_off: bool = False,
    structure_ckpt: Path | None = DEFAULT_STRUCTURE_CKPT,
) -> RailingPredictor | None:
    """Return a predictor if ckpt exists; None → caller uses the heuristic.

    ``structure_ckpt`` supplies the structure head when ``ckpt_path`` has none.
    """
    if force_off:
        return None
    path = Path(ckpt_path) if ckpt_path else DEFAULT_RAILING_CKPT
    if not path.is_file():
        return None
    return RailingPredictor(path, device, structure_ckpt=structure_ckpt)
