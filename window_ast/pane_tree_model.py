"""Inference-only recursive pane-tree model bundled with the standalone pipeline."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchvision.ops import roi_align

from window_ast.model import ImageEncoder
from window_ast.pane_tree import CLS_INDEX, MAX_CHILDREN, MAX_COUNT, NODE_CLASSES, assemble_tokens, parse_prefix_key

PROFILE_BINS = 16
TYPES = ("eps", "grid", "split")


def box_features(boxes: torch.Tensor) -> torch.Tensor:
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    return torch.stack(
        [boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], width, height, torch.log((width + 1e-3) / (height + 1e-3))],
        1,
    )


def decode_profile(
    presence: torch.Tensor,
    offset: torch.Tensor,
    max_cuts: int = MAX_CHILDREN - 1,
    min_gap: float = 0.04,
) -> list[float]:
    probabilities = presence.tolist()
    count = len(probabilities)
    candidates = [i for i, value in enumerate(probabilities) if value > 0.5] or [int(torch.argmax(presence))]
    kept: list[float] = []
    for index in sorted(candidates, key=lambda i: -probabilities[i]):
        value = min(0.995, max(0.005, (index + float(offset[index])) / count))
        if all(abs(value - other) >= min_gap for other in kept):
            kept.append(value)
        if len(kept) == max_cuts:
            break
    return sorted(kept)


class PaneTreeModel(nn.Module):
    def __init__(
        self,
        prefixes: list[str],
        *,
        d_model: int = 256,
        roi: int = 8,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        global_feat: str = "mean",
    ) -> None:
        super().__init__()
        self.prefixes = list(prefixes)
        self.global_feat = global_feat
        self.roi = roi
        self.encoder = ImageEncoder(
            d_model,
            pretrained=pretrained,
            backbone="dinov2_vits14",
            freeze_backbone=freeze_backbone,
        )
        channels = self.encoder.feat_dim
        self.reduce = nn.Sequential(nn.Conv2d(channels, d_model, 1), nn.GELU())
        global_input = 2 * channels if global_feat == "concat" else channels
        self.glob = nn.Sequential(nn.Linear(global_input, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.prefix_head = nn.Linear(d_model, len(self.prefixes))
        self.root_box_head = nn.Linear(d_model, 4)
        self.roi_net = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1, stride=2),
            nn.GELU(),
        )
        self.roi_fc = nn.Linear(d_model * (roi // 2) ** 2, d_model)
        self.box_emb = nn.Sequential(nn.Linear(7, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.node_mlp = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )
        self.type_head = nn.Linear(d_model, len(TYPES))
        self.axis_head = nn.Linear(d_model, 2)
        self.v_head = nn.Linear(d_model, MAX_COUNT + 1)
        self.h_head = nn.Linear(d_model, MAX_COUNT + 1)
        profile_channels = 128
        self.prof_conv = nn.Sequential(
            nn.Conv2d(d_model, profile_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(profile_channels, profile_channels, 3, padding=1),
            nn.GELU(),
        )
        self.prof_ctx = nn.Linear(d_model, profile_channels)
        self.prof_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(profile_channels, profile_channels, 3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(profile_channels, 2, 3, padding=1),
                )
                for _ in range(2)
            ]
        )

    def _global_input(self, spatial: torch.Tensor, cls: torch.Tensor | None) -> torch.Tensor:
        mean = spatial.mean(dim=(2, 3))
        if self.global_feat == "mean":
            return mean
        if cls is None:
            raise RuntimeError(f"DINO CLS token is required for global_feat={self.global_feat}")
        if self.global_feat == "cls":
            return cls
        return torch.cat([cls, mean], 1)

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        spatial = self.encoder.forward_spatial(images)
        global_features = self.glob(self._global_input(spatial, getattr(self.encoder, "last_cls", None)))
        output = {
            "prefix": self.prefix_head(global_features),
            "root_box": torch.sigmoid(self.root_box_head(global_features)),
        }
        return self.reduce(spatial), global_features, output

    def nodes(
        self,
        feature_map: torch.Tensor,
        global_features: torch.Tensor,
        batch_indices: torch.Tensor,
        boxes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        size = feature_map.shape[-1]
        regions = torch.cat([batch_indices.float().unsqueeze(1), boxes.clamp(0, 1) * size], 1)
        roi = roi_align(feature_map, regions, output_size=self.roi, spatial_scale=1.0, sampling_ratio=2, aligned=True)
        roi_features = self.roi_fc(self.roi_net(roi).flatten(1))
        node_features = self.node_mlp(
            torch.cat([roi_features, self.box_emb(box_features(boxes)), global_features[batch_indices]], 1)
        )
        profile_roi = roi_align(
            feature_map,
            regions,
            output_size=PROFILE_BINS,
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        profile_features = self.prof_conv(profile_roi) + self.prof_ctx(node_features)[:, :, None, None]
        vertical = self.prof_heads[0](profile_features.mean(dim=2))
        horizontal = self.prof_heads[1](profile_features.mean(dim=3))
        return {
            "type": self.type_head(node_features),
            "axis": self.axis_head(node_features),
            "V": self.v_head(node_features),
            "H": self.h_head(node_features),
            "profile": torch.stack([vertical, horizontal], 1),
            "z": node_features,
        }

    def nodes_tta(
        self,
        feature_maps: tuple[torch.Tensor, torch.Tensor],
        global_features: tuple[torch.Tensor, torch.Tensor],
        batch_indices: torch.Tensor,
        boxes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        predicted = self.nodes(feature_maps[0], global_features[0], batch_indices, boxes)
        flipped_boxes = torch.stack([1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], 1)
        flipped = self.nodes(feature_maps[1], global_features[1], batch_indices, flipped_boxes)
        flipped_profile = flipped["profile"].clone()
        flipped_profile[:, 0] = flipped["profile"][:, 0].flip(-1)
        flipped_profile[:, 0, 1] = -flipped_profile[:, 0, 1]
        output = {
            key: (predicted[key] + flipped[key]) / 2
            for key in ("type", "axis", "V", "H")
        }
        output["profile"] = (predicted["profile"] + flipped_profile) / 2
        return output

    @torch.no_grad()
    def describe(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        feature_map, global_features, output = self.encode(images)
        batch_indices = torch.arange(images.shape[0], device=images.device)
        predicted = self.nodes(feature_map, global_features, batch_indices, output["root_box"])
        heads = torch.cat(
            [output["prefix"].softmax(1)] + [predicted[key].softmax(1) for key in ("type", "axis", "V", "H")],
            1,
        )
        return {
            "bars": torch.sigmoid(predicted["profile"][:, :, 0]).flatten(1),
            "heads": heads,
            "z": predicted["z"],
        }

    @staticmethod
    def decode_nodes(predicted: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
        types = predicted["type"].argmax(1).tolist()
        axes = predicted["axis"].argmax(1).tolist()
        verticals = predicted["V"].argmax(1).tolist()
        horizontals = predicted["H"].argmax(1).tolist()
        output = []
        for index, type_index in enumerate(types):
            node: dict[str, Any] = {"V": verticals[index], "H": horizontals[index], "cuts": [], "children": []}
            if TYPES[type_index] == "split":
                axis = axes[index]
                cuts = decode_profile(
                    torch.sigmoid(predicted["profile"][index, axis, 0]),
                    torch.sigmoid(predicted["profile"][index, axis, 1]),
                )
                node["cls"] = CLS_INDEX[f"{'vh'[axis]}{len(cuts) + 1}"]
                node["cuts"] = cuts
            else:
                node["cls"] = CLS_INDEX[TYPES[type_index]]
            output.append(node)
        return output

    def _expand(
        self,
        feature_map: torch.Tensor,
        global_features: torch.Tensor,
        root_boxes: torch.Tensor,
        keys: list[str],
        *,
        max_depth: int = 5,
        max_nodes: int = 64,
    ) -> list[list[dict[str, Any]]]:
        trees: list[list[dict[str, Any]]] = [[] for _ in keys]
        frontier: list[tuple[int, int, list[float]]] = []
        for image_index, key in enumerate(keys):
            if parse_prefix_key(key)[1] is not None:
                frontier.append((image_index, 0, root_boxes[image_index].tolist()))
                trees[image_index].append({})
        for _ in range(max_depth + 1):
            if not frontier:
                break
            batch_indices = torch.tensor([item[0] for item in frontier], device=feature_map.device)
            boxes = torch.tensor([item[2] for item in frontier], device=feature_map.device, dtype=feature_map.dtype)
            decoded = self.decode_nodes(self.nodes(feature_map, global_features, batch_indices, boxes))
            next_frontier: list[tuple[int, int, list[float]]] = []
            for (image_index, node_index, box), node in zip(frontier, decoded):
                node["box"] = box
                cls = NODE_CLASSES[node["cls"]]
                if cls not in ("eps", "grid"):
                    child_count = len(node["cuts"]) + 1
                    if len(trees[image_index]) + child_count > max_nodes:
                        node.update(cls=CLS_INDEX["eps"], cuts=[])
                    else:
                        axis = 0 if cls[0] == "v" else 1
                        edges = [box[axis]] + [
                            box[axis] + cut * (box[axis + 2] - box[axis]) for cut in node["cuts"]
                        ] + [box[axis + 2]]
                        for child_index in range(child_count):
                            child_box = list(box)
                            child_box[axis], child_box[axis + 2] = edges[child_index], edges[child_index + 1]
                            node["children"].append(len(trees[image_index]))
                            trees[image_index].append({})
                            next_frontier.append((image_index, node["children"][-1], child_box))
                trees[image_index][node_index] = node
            frontier = next_frontier
        for tree in trees:
            for index, node in enumerate(tree):
                if not node:
                    tree[index] = {"cls": CLS_INDEX["eps"], "V": 0, "H": 0, "cuts": [], "children": []}
        return trees

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        *,
        max_depth: int = 5,
        max_nodes: int = 64,
        return_trees: bool = False,
        root_boxes: torch.Tensor | None = None,
    ) -> list[list[str]] | tuple[list[list[str]], list[str], list[list[dict[str, Any]]]]:
        self.eval()
        feature_map, global_features, output = self.encode(images)
        if root_boxes is not None:
            output["root_box"] = root_boxes.to(output["root_box"])
        keys = [self.prefixes[index] for index in output["prefix"].argmax(1).tolist()]
        trees = self._expand(
            feature_map,
            global_features,
            output["root_box"],
            keys,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )
        tokens = []
        for key, tree in zip(keys, trees):
            prefix, region = parse_prefix_key(key)
            try:
                tokens.append(assemble_tokens(prefix, region, tree))
            except Exception:
                tokens.append(prefix + ["END"])
        if return_trees:
            return tokens, keys, trees
        return tokens

    @torch.no_grad()
    def predict_joint(
        self,
        images: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
        max_depth: int = 5,
        max_nodes: int = 64,
    ) -> tuple[list[str], str, list[dict[str, Any]]]:
        self.eval()
        batch_size = images.shape[0]
        feature_map, global_features, output = self.encode(images)
        weight = (
            torch.ones(batch_size, device=images.device)
            if weights is None
            else weights.to(images.device, torch.float32)
        )
        weight = (weight / weight.sum()).to(feature_map.dtype)

        def average(value: torch.Tensor) -> torch.Tensor:
            shape = (-1,) + (1,) * (value.dim() - 1)
            return (value * weight.view(shape)).sum(0, keepdim=True)

        key = self.prefixes[int(average(output["prefix"]).argmax(1))]
        prefix, region = parse_prefix_key(key)
        tree: list[dict[str, Any]] = []
        batch_indices = torch.arange(batch_size, device=images.device)
        frontier: list[tuple[int, torch.Tensor]] = []
        if region is not None:
            frontier.append((0, output["root_box"]))
            tree.append({})

        for _ in range(max_depth + 1):
            if not frontier:
                break
            next_frontier: list[tuple[int, torch.Tensor]] = []
            for node_index, boxes in frontier:
                predicted = self.nodes(feature_map, global_features, batch_indices, boxes.to(feature_map.dtype))
                node = self.decode_nodes({name: average(value) for name, value in predicted.items()})[0]
                node["box"], node["boxes"] = boxes[0].tolist(), boxes.tolist()
                cls = NODE_CLASSES[node["cls"]]
                if cls not in ("eps", "grid"):
                    child_count = len(node["cuts"]) + 1
                    if len(tree) + child_count > max_nodes:
                        node.update(cls=CLS_INDEX["eps"], cuts=[])
                    else:
                        axis = 0 if cls[0] == "v" else 1
                        low, high = boxes[:, axis], boxes[:, axis + 2]
                        fractions = [0.0] + list(node["cuts"]) + [1.0]
                        for child_index in range(child_count):
                            child_boxes = boxes.clone()
                            child_boxes[:, axis] = low + fractions[child_index] * (high - low)
                            child_boxes[:, axis + 2] = low + fractions[child_index + 1] * (high - low)
                            node["children"].append(len(tree))
                            tree.append({})
                            next_frontier.append((node["children"][-1], child_boxes))
                tree[node_index] = node
            frontier = next_frontier

        for index, node in enumerate(tree):
            if not node:
                tree[index] = {"cls": CLS_INDEX["eps"], "V": 0, "H": 0, "cuts": [], "children": []}
        try:
            tokens = assemble_tokens(prefix, region, tree)
        except Exception:
            tokens = prefix + ["END"]
        return tokens, key, tree
