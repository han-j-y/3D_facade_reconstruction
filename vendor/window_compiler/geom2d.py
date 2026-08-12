"""2D window geometry in normalized local coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Vec2:
    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Segment:
    id: str
    a: Vec2
    b: Vec2
    layer: str = "muntin"
    clip_region_id: str | None = None


@dataclass
class Region:
    id: str
    shape: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def min_side(self) -> float:
        return min(self.width, self.height)

    @property
    def center(self) -> Vec2:
        return Vec2((self.min_x + self.max_x) / 2.0, (self.min_y + self.max_y) / 2.0)

    def contains(self, p: Vec2, eps: float = 1e-9) -> bool:
        from shapes import contour_for_region, point_in_contour

        if not point_in_contour(p, contour_for_region(self)):
            return False
        return (
            self.min_x - eps <= p.x <= self.max_x + eps
            and self.min_y - eps <= p.y <= self.max_y + eps
        )

    def crop_normalized(self, u0: float, v0: float, u1: float, v1: float, new_id: str) -> Region:
        """Restrict a region to a sub-rectangle in normalized [0,1] coords."""
        u0, u1 = min(u0, u1), max(u0, u1)
        v0, v1 = min(v0, v1), max(v0, v1)
        params = dict(self.params)
        params["crop"] = {"u0": u0, "v0": v0, "u1": u1, "v1": v1}
        return Region(
            new_id,
            self.shape,
            self.min_x + u0 * self.width,
            self.min_y + v0 * self.height,
            self.min_x + u1 * self.width,
            self.min_y + v1 * self.height,
            params,
        )

    def inset(self, amount: float, new_id: str) -> Region:
        from shapes import (
            bounds_of_contour,
            contour_for_region,
            inset_contour,
            inset_head_body_contour,
        )

        if self.shape == "head_body":
            contour = inset_head_body_contour(self.params, amount)
        elif self.shape in {"arch_head", "rect_eyebrow", "springline_arch"}:
            from shapes import inset_arch_head_contour

            contour = inset_arch_head_contour(self.params, amount)
        else:
            contour = inset_contour(contour_for_region(self), amount)
        xmin, xmax, ymin, ymax = bounds_of_contour(contour)
        params = dict(self.params)
        params["contour"] = [[p.x, p.y] for p in contour]
        return Region(new_id, self.shape, xmin, ymin, xmax, ymax, params)

    def clip_segment(self, seg: Segment) -> Segment | None:
        from shapes import clip_segment_to_contour, clip_segment_to_rect, contour_for_region

        clipped = clip_segment_to_rect(seg, self.min_x, self.min_y, self.max_x, self.max_y)
        if clipped is None:
            return None
        return clip_segment_to_contour(clipped, contour_for_region(self))


def build_boundary(spec: dict[str, Any]) -> Region:
    from shapes import build_shape_region

    return build_shape_region(spec)


def normalized_point(region: Region, u: float, v: float) -> Vec2:
    return Vec2(region.min_x + u * region.width, region.min_y + v * region.height)
