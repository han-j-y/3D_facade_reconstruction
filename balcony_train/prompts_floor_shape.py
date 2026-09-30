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

# Positive description of one flat wall. Naming the building edge makes
# FLUX draw it, so this clause only says what is in the frame.
ONE_FACADE = (
    "a long flat facade fills the frame, one balcony in the center of that "
    "single wall, identical windows continue on the left and on the right, "
    "the wall is one plane, the edge of the building is outside the photo, "
    "street photo from across the road, camera a little above the balcony "
    "so the top of the floor is visible"
)

NEGATIVE_PROMPT = (
    "rectangular balcony, semicircular balcony, curved bow window, "
    "view from standing on the balcony, railing-only close-up, "
    "floor plan diagram, blueprint, top-down drawing, cartoon, cgi, studio render"
)

TRIANGLE_PROMPTS: list[str] = [
    (
        f"{ONE_FACADE}, the floor shape is a triangle, the top surface of the "
        "concrete balcony floor is an isosceles triangle, apex of the floor "
        "points away from the wall, the floor comes to one front point, the "
        "floor outline itself is triangular, 4th floor, beige stucco, dark "
        "metal vertical bars follow only the two sloping edges of that "
        "triangular floor, the bars are not triangles, the window is an "
        "ordinary rectangle, overcast daylight"
    ),
    (
        f"{ONE_FACADE}, the floor shape is a triangle, the stone floor on the "
        "wall is triangular, the base of that floor is on the wall and the apex "
        "points outward to one front point, 6th floor, limestone facade, "
        "ordinary vertical masonry balusters follow that floor edge, the window "
        "stays rectangular, clear afternoon light"
    ),
    (
        f"{ONE_FACADE}, the floor shape is a triangle, the concrete floor slab "
        "is an isosceles triangle with a pointed front apex, 3rd floor, ochre "
        "facade, a plain solid parapet follows the two sloping edges of that "
        "triangular floor and meets at the front point, the parapet face itself "
        "is not a triangle, soft morning light"
    ),
]

TRAPEZOID_PROMPTS: list[str] = [
    (
        f"{ONE_FACADE}, the floor shape is a trapezoid, the top surface of the "
        "concrete balcony floor is a trapezoid and that floor plan is what the "
        "photo shows, only the floor is a trapezoid, long edge of the floor on "
        "the wall, short front edge of the floor parallel to the wall, sides of "
        "the floor meet the wall at about 75 degrees, 5th floor, sandstone "
        "facade, dark metal bars follow that floor edge, the railing is not a "
        "trapezoid panel, the window is an ordinary rectangle, overcast daylight"
    ),
    (
        f"{ONE_FACADE}, the floor shape is a trapezoid, the stone floor is "
        "wider along the wall and narrower at the front, front edge parallel "
        "to the wall, floor sides at about 75 degrees, this floor plan is the "
        "trapezoid, 2nd floor, pale limestone facade, ordinary vertical masonry "
        "balusters follow only that floor outline, clear afternoon light"
    ),
    (
        f"{ONE_FACADE}, the floor shape is a trapezoid, only the floor has that "
        "plan, long base on the wall, short front parallel to the wall, sides "
        "near 75 degrees, 7th floor, cream facade, a plain solid parapet "
        "follows that floor, the parapet face itself is not a trapezoid, "
        "soft daylight"
    ),
]

HEXAGON_PROMPTS: list[str] = [
    (
        f"{ONE_FACADE}, 4th floor, beige stucco facade, the floor shape is a "
        "hexagon made from a rectangle with both front ends of the floor cut "
        "off, side edges of the floor stay perpendicular to the wall, then a "
        "diagonal chamfer on each front end meets the short front edge at about "
        "30 degrees, dark metal railing follows that floor, six-sided floor "
        "plan, overcast daylight"
    ),
    (
        f"{ONE_FACADE}, 6th floor, limestone facade, stone floor with two "
        "diagonal chamfers at the front, straight sides, short straight front, "
        "masonry balusters follow that hexagon floor outline, both chamfers of "
        "the floor visible, about 30 degree cuts, clear afternoon light"
    ),
    (
        f"{ONE_FACADE}, 3rd floor, ochre facade, solid parapet on a hexagon "
        "floor slab, the parapet turns where the floor is chamfered, short "
        "front of the floor, perpendicular sides, those chamfers are about 30 "
        "degrees, no openings, soft morning light"
    ),
    (
        f"{ONE_FACADE}, 8th floor, grey concrete facade, frosted glass panels "
        "follow a hexagonal floor, panels change direction at both chamfered "
        "front ends of the floor, short front panel, side panels square to the "
        "wall, about 30 degrees, diffuse cloudy light"
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
