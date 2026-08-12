"""Image encoder + Transformer decoder for AST token sequences.

Default backbone is DINOv2 ViT-S/14 (patch tokens → decoder memory).
ResNet-18 remains available for loading older checkpoints.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import ResNet18_Weights, resnet18
except ImportError:  # pragma: no cover
    ResNet18_Weights = None
    resnet18 = None


def resolve_backbone(
    cfg: dict[str, Any] | None = None,
    state_dict: dict[str, Any] | None = None,
    *,
    default: str = "dinov2_vits14",
) -> str:
    """Pick encoder backbone from checkpoint args or state_dict keys."""
    cfg = cfg or {}
    if cfg.get("backbone"):
        return str(cfg["backbone"]).lower().strip()
    if state_dict:
        keys = list(state_dict.keys())
        if any(k.startswith("encoder.dino") for k in keys):
            return "dinov2_vits14"
        if any(
            k.startswith("encoder.stem") or k.startswith("encoder.layer1") for k in keys
        ):
            return "resnet18"
    return default


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ImageEncoder(nn.Module):
    """Spatial image features projected to ``d_model`` decoder memory tokens.

    ``backbone``:
      - ``dinov2_vits14`` (default): DINOv2 patch tokens (384-d), frozen by default
      - ``resnet18``: ImageNet ResNet-18 layer4 map (512-d)
    """

    def __init__(
        self,
        d_model: int = 256,
        pretrained: bool = True,
        *,
        backbone: str = "dinov2_vits14",
        freeze_backbone: bool | None = None,
    ) -> None:
        super().__init__()
        self.backbone_name = str(backbone).lower().strip()
        self.dino: nn.Module | None = None
        self.stem: nn.Module | None = None
        self.layer1: nn.Module | None = None
        self.layer2: nn.Module | None = None
        self.layer3: nn.Module | None = None
        self.layer4: nn.Module | None = None

        if self.backbone_name in ("dinov2_vits14", "dinov2", "dino"):
            self.dino = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14", pretrained=bool(pretrained)
            )
            self.feat_dim = int(getattr(self.dino, "embed_dim", 384))
            if freeze_backbone is None:
                freeze_backbone = True
        elif self.backbone_name in ("resnet18", "resnet"):
            if resnet18 is None:
                raise ImportError("torchvision is required for ResNet-18")
            weights = (
                ResNet18_Weights.DEFAULT
                if pretrained and ResNet18_Weights is not None
                else None
            )
            net = resnet18(weights=weights)
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.layer1 = net.layer1
            self.layer2 = net.layer2
            self.layer3 = net.layer3
            self.layer4 = net.layer4
            self.feat_dim = 512
            if freeze_backbone is None:
                freeze_backbone = False
        else:
            raise ValueError(
                f"Unknown backbone={backbone!r} (use dinov2_vits14 or resnet18)"
            )

        self.proj = nn.Linear(self.feat_dim, d_model)
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            self.set_backbone_trainable(False)

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.freeze_backbone = not trainable
        if self.dino is not None:
            for p in self.dino.parameters():
                p.requires_grad = trainable
            self.dino.train(mode=trainable)
            return
        for mod in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad = trainable

    def _dino_spatial(self, images: torch.Tensor) -> torch.Tensor:
        assert self.dino is not None
        feats = self.dino.forward_features(images)
        tokens = None
        if isinstance(feats, dict):
            tokens = feats.get("x_norm_patchtokens")
            if tokens is None and feats.get("x_norm") is not None:
                tokens = feats["x_norm"][:, 1:, :]
        if tokens is None:
            outs = self.dino.get_intermediate_layers(images, n=1, reshape=True)
            return outs[0]
        b, n, c = tokens.shape
        side = int(round(n**0.5))
        return tokens.transpose(1, 2).reshape(b, c, side, side)

    def forward_spatial(self, images: torch.Tensor) -> torch.Tensor:
        """Return spatial map ``[B, C, h, w]`` (C = feat_dim)."""
        if self.dino is not None:
            h, w = images.shape[-2:]
            if h % 14 != 0 or w % 14 != 0:
                nh = max(14, (h // 14) * 14)
                nw = max(14, (w // 14) * 14)
                images = F.interpolate(
                    images, size=(nh, nw), mode="bilinear", align_corners=False
                )
            if self.freeze_backbone:
                with torch.no_grad():
                    return self._dino_spatial(images)
            return self._dino_spatial(images)

        assert self.stem is not None
        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, H, W] → memory [B, S, D]
        x = self.forward_spatial(images)
        x = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        return self.proj(x)


class HeatmapHead(nn.Module):
    """Predict soft V/H muntin heatmaps from encoder spatial features."""

    def __init__(self, in_ch: int = 512, out_size: int = 56) -> None:
        super().__init__()
        self.out_size = out_size
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=out_size, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
        )

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:
        """``spatial [B,C,h,w]`` → logits ``[B,2,out_size,out_size]`` (ch0=V, ch1=H)."""
        return self.net(spatial)


class HeatCountHead(nn.Module):
    """Predict discrete (#V, #H) muntin counts from soft heatmaps (cascade stage)."""

    def __init__(self, heat_size: int = 56, max_count: int = 8) -> None:
        super().__init__()
        self.max_count = max_count
        self.fc_v = nn.Sequential(
            nn.Linear(heat_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, max_count + 1),
        )
        self.fc_h = nn.Sequential(
            nn.Linear(heat_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, max_count + 1),
        )

    def forward(self, heat_soft: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # heat_soft: [B,2,H,W] → column/row mass profiles
        v_prof = heat_soft[:, 0].mean(dim=1)  # [B, W]
        h_prof = heat_soft[:, 1].mean(dim=2)  # [B, H]
        return self.fc_v(v_prof), self.fc_h(h_prof)


class WindowAstModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        *,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 160,
        pad_id: int = 0,
        pretrained_encoder: bool = True,
        backbone: str = "dinov2_vits14",
        freeze_backbone: bool | None = None,
        use_heatmap: bool = False,
        heatmap_size: int = 56,
        topo_max_count: int = 8,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.max_len = max_len
        self.backbone = str(backbone).lower().strip()
        self.use_heatmap = use_heatmap
        self.heatmap_size = heatmap_size
        self.topo_max_count = topo_max_count
        self.encoder = ImageEncoder(
            d_model=d_model,
            pretrained=pretrained_encoder,
            backbone=self.backbone,
            freeze_backbone=freeze_backbone,
        )
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)
        if use_heatmap:
            self.heatmap_head = HeatmapHead(
                in_ch=self.encoder.feat_dim, out_size=heatmap_size
            )
            self.heat_proj = nn.Linear(2, d_model)
            self.heat_gate = nn.Parameter(torch.ones(1))
            self.heat_count_head = HeatCountHead(
                heat_size=heatmap_size, max_count=topo_max_count
            )
            self.count_emb_v = nn.Embedding(topo_max_count + 1, d_model)
            self.count_emb_h = nn.Embedding(topo_max_count + 1, d_model)
        else:
            self.heatmap_head = None
            self.heat_proj = None
            self.heat_gate = None
            self.heat_count_head = None
            self.count_emb_v = None
            self.count_emb_h = None

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.encoder.set_backbone_trainable(trainable)

    def _causal_mask(self, t: int, device: torch.device) -> torch.Tensor:
        # bool mask (True = blocked) to match tgt_key_padding_mask dtype
        return torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)

    def encode(
        self,
        images: torch.Tensor,
        *,
        heat_maps: Optional[torch.Tensor] = None,
        fuse_mode: str = "pred",
        count_v: Optional[torch.Tensor] = None,
        count_h: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Image → heatmaps → counts → decoder memory.

        Returns ``(memory, heat_logits, count_logits_v, count_logits_h)``.
        """
        spatial = self.encoder.forward_spatial(images)
        b, c, h, w = spatial.shape
        memory = self.encoder.proj(spatial.flatten(2).transpose(1, 2))
        heat_logits: Optional[torch.Tensor] = None
        logits_v: Optional[torch.Tensor] = None
        logits_h: Optional[torch.Tensor] = None
        if self.heatmap_head is not None:
            heat_logits = self.heatmap_head(spatial)
            pred_soft = torch.sigmoid(heat_logits)
            if heat_maps is not None and heat_maps.shape[-2:] != (self.heatmap_size, self.heatmap_size):
                heat_maps = F.interpolate(
                    heat_maps,
                    size=(self.heatmap_size, self.heatmap_size),
                    mode="bilinear",
                    align_corners=False,
                )
            if fuse_mode == "gt" and heat_maps is not None:
                heat_soft = heat_maps
            elif fuse_mode == "mix" and heat_maps is not None:
                heat_soft = 0.5 * pred_soft + 0.5 * heat_maps
            else:
                heat_soft = pred_soft
            heat_s = F.interpolate(heat_soft, size=(h, w), mode="bilinear", align_corners=False)
            heat_tok = heat_s.flatten(2).transpose(1, 2)  # [B, HW, 2]
            memory = memory + self.heat_gate * self.heat_proj(heat_tok)

            # Cascade: heatmaps → discrete counts → 2 extra memory tokens.
            logits_v, logits_h = self.heat_count_head(heat_soft)
            if count_v is not None and count_h is not None:
                nv = count_v.long().clamp(0, self.topo_max_count)
                nh = count_h.long().clamp(0, self.topo_max_count)
            else:
                nv = logits_v.argmax(dim=-1)
                nh = logits_h.argmax(dim=-1)
            extra = torch.stack([self.count_emb_v(nv), self.count_emb_h(nh)], dim=1)
            # Amplify count tokens so they compete with the spatial memory slots.
            memory = torch.cat([memory, extra * 3.0], dim=1)
        return memory, heat_logits, logits_v, logits_h

    def _decode(self, memory: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        tgt_in = token_ids[:, :-1]
        emb = self.token_emb(tgt_in) * math.sqrt(self.d_model)
        emb = self.pos(emb)
        tgt_key_padding = tgt_in.eq(self.pad_id)
        causal = self._causal_mask(tgt_in.size(1), token_ids.device)
        decoded = self.decoder(
            emb,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding,
        )
        return self.out_proj(decoded)

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        heat_maps: Optional[torch.Tensor] = None,
        fuse_mode: str = "pred",
        count_v: Optional[torch.Tensor] = None,
        count_h: Optional[torch.Tensor] = None,
        return_heat: bool = False,
    ):
        """Teacher-forced logits for token_ids[:, 1:] given token_ids[:, :-1].

        Returns logits [B, T-1, V], or ``(logits, heat_logits, count_v_logits, count_h_logits)``
        if ``return_heat``.
        """
        memory, heat_logits, lv, lh = self.encode(
            images,
            heat_maps=heat_maps,
            fuse_mode=fuse_mode,
            count_v=count_v,
            count_h=count_h,
        )
        logits = self._decode(memory, token_ids)
        if return_heat:
            return logits, heat_logits, lv, lh
        return logits

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        *,
        bos_id: int,
        eos_id: int,
        max_len: Optional[int] = None,
        heat_maps: Optional[torch.Tensor] = None,
        fuse_mode: str = "pred",
        count_v: Optional[torch.Tensor] = None,
        count_h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Greedy decode. Returns token ids [B, L] including BOS, stopping at EOS."""
        self.eval()
        max_len = max_len or self.max_len
        memory, _, _, _ = self.encode(
            images,
            heat_maps=heat_maps,
            fuse_mode=fuse_mode,
            count_v=count_v,
            count_h=count_h,
        )
        b = images.size(0)
        ys = torch.full((b, 1), bos_id, dtype=torch.long, device=images.device)
        finished = torch.zeros(b, dtype=torch.bool, device=images.device)
        for _ in range(max_len - 1):
            emb = self.token_emb(ys) * math.sqrt(self.d_model)
            emb = self.pos(emb)
            causal = self._causal_mask(ys.size(1), images.device)
            decoded = self.decoder(emb, memory, tgt_mask=causal)
            logits = self.out_proj(decoded[:, -1])
            next_id = logits.argmax(dim=-1)
            next_id = torch.where(finished, torch.full_like(next_id, self.pad_id), next_id)
            ys = torch.cat([ys, next_id.unsqueeze(1)], dim=1)
            finished = finished | next_id.eq(eos_id)
            if bool(finished.all()):
                break
        return ys


class TopologyCountHead(nn.Module):
    """Predict discrete muntin counts (#V lines, #H lines) from pooled encoder features.

    Teaches the model to count through-rails before / alongside spelling the AST —
    the inductive bias behind a general split rule.
    """

    def __init__(self, d_model: int = 256, max_count: int = 8) -> None:
        super().__init__()
        self.max_count = max_count
        self.fc_v = nn.Linear(d_model, max_count + 1)
        self.fc_h = nn.Linear(d_model, max_count + 1)

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fc_v(pooled), self.fc_h(pooled)


def topology_count_loss(
    logits_v: torch.Tensor,
    logits_h: torch.Tensor,
    target_v: torch.Tensor,
    target_h: torch.Tensor,
    *,
    max_count: int = 8,
) -> torch.Tensor:
    tv = target_v.long().clamp(0, max_count)
    th = target_h.long().clamp(0, max_count)
    return F.cross_entropy(logits_v, tv) + F.cross_entropy(logits_h, th)


def sequence_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pad_id: int,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy on teacher-forced targets token_ids[:, 1:]."""
    # logits: [B, T-1, V], targets: [B, T]
    gold = targets[:, 1:]
    v = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, v),
        gold.reshape(-1),
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
    )
