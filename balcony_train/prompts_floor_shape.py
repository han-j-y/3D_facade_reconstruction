"""FLUX.2 prompts for rare balcony floor plans: triangle, trapezoid, hexagon.

Triangle and trapezoid are three photos: metal openwork, masonry openwork,
solid parapet. Hexagon also has a surface-panel photo. The triangle or
trapezoid is the floor slab only. Windows, rails, and wall panels stay
ordinary rectangles.

These crops must keep the slab outline. Do not run them through
``resize_flux_images.py`` (that rewrite is a 120×44 railing band).

Plans match ``balcony_plan.slab_outline``:
- triangle: isosceles, wall edge is the base, apex at the front center
- trapezoid: long edge on the wall, shorter front parallel to the wall,
  each side meets the wall at about 75 degrees
- hexagon: rectangle with both front corners chamfered; sides stay
  perpendicular to the wall, then meet the short front at about 30 degrees
"""

from __future__ import annotations

FLOOR_PLAN_PREFIXES: tuple[str, ...] = (
    "flux2_triangle_",
    "flux2_trapezoid_",
    "flux2_hexagon_",
)

_KIND_SUFFIXES: dict[str, tuple[str, ...]] = {
    "triangle": ("metal", "masonry", "solid"),
    "trapezoid": ("metal", "masonry", "solid"),
    "hexagon": ("metal", "masonry", "solid", "surface"),
}

# Every crop is one balcony on one flat wall. FLUX otherwise draws the
# building corner, where two facades meet. This clause is first in every prompt.
ONE_FACADE = (
    "never on a building corner, absolutely not a corner balcony, "
    "centered on one flat wall, that same wall and its windows continue "
    "on both the left and the right, the vertical corner where two facades "
    "meet is outside the frame, no second wall, the balcony does not wrap"
)

NEGATIVE_PROMPT = (
    "rectangular balcony, semicircular balcony, curved bow window, "
    "view from standing on the balcony, railing-only close-up, "
    "floor plan diagram, blueprint, top-down drawing, cartoon, cgi, studio render"
)

TRIANGLE_PROMPTS: list[str] = [
    (
        f"{ONE_FACADE}, the floor shape is a triangle, the top surface of the "
        "concrete balcony floor is an isosceles triangle and that triangular "
        "floor is what the photo shows, apex of the floor points away from the "
        "wall at the front center, the floor comes to a point, not a rectangular "
        "floor, three-quarter street photo of a 4th-floor balcony, beige stucco, "
        "dark metal vertical bars follow only the two sloping edges of that "
        "triangular floor, the bars are not triangles, the window is an ordinary "
        "rectangle, overcast daylight"
    ),
    (
        f"{ONE_FACADE}, the floor shape is a triangle, looking slightly down from "
        "the street you see the top of a triangular stone floor, the base of that "
        "floor is on the wall and the apex points outward to one front point, "
        "not a rectangular floor, oblique sidewalk view of a 6th-floor balcony, "
        "limestone facade, ordinary vertical masonry balusters follow that floor "
        "edge, the window stays rectangular, clear afternoon light"
    ),
    (
        f"{ONE_FACADE}, the floor shape is a triangle, the concrete floor slab "
        "you see from above is an isosceles triangle with a pointed front apex, "
        "not a rectangular floor, angled street photo of a 3rd-floor balcony, "
        "ochre facade, a plain solid parapet follows the two sloping edges of "
        "that triangular floor and meets at the front point, the parapet face "
        "itself is not a triangle, soft morning light"
    ),
]

TRAPEZOID_PROMPTS: list[str] = [
    (
        f"{ONE_FACADE}, the floor shape is a trapezoid, the top surface of the "
        "concrete balcony floor is a trapezoid and that floor plan is what the "
        "photo shows, only the floor is a trapezoid, long edge of the floor on "
        "the wall, short front edge of the floor parallel to the wall, sides of "
        "the floor meet the wall at about 75 degrees, not a rectangular floor "
        "seen in perspective, three-quarter street photo of a 5th-floor balcony, "
        "sandstone facade, dark metal bars follow that floor edge, the railing "
        "is not a trapezoid panel, the window is an ordinary rectangle, "
        "overcast daylight"
    ),
    (
        f"{ONE_FACADE}, the floor shape is a trapezoid, looking slightly down "
        "from the street you see the top of a trapezoidal stone floor, wider "
        "along the wall and narrower at the front, front edge parallel to the "
        "wall, floor sides at about 75 degrees, this floor plan is not a "
        "rectangle, oblique sidewalk view of a 2nd-floor balcony, red brick "
        "facade, ordinary vertical masonry balusters follow only that floor "
        "outline, clear afternoon light"
    ),
    (
        f"{ONE_FACADE}, the floor shape is a trapezoid, the concrete floor slab "
        "seen from above is a trapezoid, only the floor has that plan, long "
        "base on the wall, short front parallel to the wall, sides near 75 "
        "degrees, not a rectangular floor, angled street photo of a 7th-floor "
        "balcony, cream facade, a plain solid parapet follows that floor, the "
        "parapet face itself is not a trapezoid, soft daylight"
    ),
]

HEXAGON_PROMPTS: list[str] = [
    (
        f"{ONE_FACADE}, three-quarter street photo of a 4th-floor balcony, beige "
        "stucco facade, the floor shape is a hexagon made from a rectangle with "
        "both front corners of the floor cut off, side edges of the floor stay "
        "perpendicular to the wall, then a diagonal chamfer on each front corner "
        "of the floor meets the short front edge at about 30 degrees, dark metal "
        "railing follows that floor, six-sided floor plan, not a rectangle, not "
        "a curve, overcast daylight"
    ),
    (
        f"{ONE_FACADE}, oblique sidewalk view of a 6th-floor balcony, limestone "
        "facade, stone floor with clipped front corners, straight sides, two "
        "diagonal chamfers, short straight front, masonry balusters follow that "
        "hexagon floor outline, both chamfers of the floor visible, about 30 "
        "degree cuts, clear afternoon light"
    ),
    (
        f"{ONE_FACADE}, three-quarter street crop of a 3rd-floor balcony, ochre "
        "facade, solid parapet on a hexagon floor slab, the parapet turns where "
        "the floor is chamfered, short front of the floor, perpendicular side "
        "returns, those chamfers are about 30 degrees, no openings, soft morning light"
    ),
    (
        f"{ONE_FACADE}, angled street photo of an 8th-floor balcony, grey concrete "
        "facade, frosted glass panels follow a hexagonal floor, panels change "
        "direction at both chamfered front corners of the floor, short front "
        "panel, side panels square to the wall, about 30 degrees, not a bow "
        "window, diffuse cloudy light"
    ),
]


def prefixes_for(shape: str) -> list[str]:
    """Filename prefixes aligned with that shape's prompts."""
    return [f"flux2_{shape}_{kind}_" for kind in _KIND_SUFFIXES[shape]]


def prompt_rows(shape: str) -> list[tuple[str, str]]:
    """``(prompt, filename_prefix)`` in the same order as ``prefixes_for``."""
    bank = {
        "triangle": TRIANGLE_PROMPTS,
        "trapezoid": TRAPEZOID_PROMPTS,
        "hexagon": HEXAGON_PROMPTS,
    }[shape]
    prefixes = prefixes_for(shape)
    if len(bank) != len(prefixes):
        raise ValueError(f"{shape} prompts and prefixes differ in length")
    return list(zip(bank, prefixes))
