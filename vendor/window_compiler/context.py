"""Compilation state: regions, anchors, muntin segments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from geom2d import Region, Segment, Vec2


@dataclass
class Anchor:
    id: str
    position: Vec2
    kind: str = "point"


@dataclass
class CompileContext:
    spec: dict[str, Any]
    regions: dict[str, Region] = field(default_factory=dict)
    anchors: dict[str, Anchor] = field(default_factory=dict)
    segments: list[Segment] = field(default_factory=list)
    arc_muntins: dict[str, dict[str, Any]] = field(default_factory=dict)
    debug: bool = False

    def region(self, region_id: str) -> Region:
        if region_id not in self.regions:
            raise KeyError(f"unknown region: {region_id!r}")
        return self.regions[region_id]

    def add_segment(self, seg: Segment, *, region_id: str | None = None) -> None:
        if region_id:
            seg.clip_region_id = region_id
            clipped = self.regions[region_id].clip_segment(seg)
            if clipped is None:
                return
            clipped.clip_region_id = region_id
            seg = clipped
        self.segments.append(seg)

    def add_anchor(self, anchor: Anchor) -> None:
        self.anchors[anchor.id] = anchor
