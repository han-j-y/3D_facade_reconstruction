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
# building corner, where two facades meet.
ONE_FACADE = (
    "one balcony in the middle of a single straight facade, the photo shows "
    "only that one flat wall, the same wall continues past both ends of the "
    "balcony, the building corner is outside the frame, no second facade, "
    "no corner pier, the balcony does not turn a corner, not a corner balcony, "
    "not wrapped around two walls"
)

NEGATIVE_PROMPT = (
    "rectangular balcony, semicircular balcony, curved bow window, "
    "view from standing on the balcony, railing-only close-up, "
    "floor plan diagram, blueprint, top-down drawing, cartoon, cgi, studio render"
)

TRIANGLE_PROMPTS: list[str] = [
    (
        "three-quarter street photo of a 4th-floor balcony, beige stucco facade, "
        "only the balcony floor slab is an isosceles triangle: the long base of "
        "that floor is fixed to the wall and the apex of the floor is at the front "
        "center, the concrete floor plan is the triangle, the window behind is an "
        "ordinary rectangle, the dark metal railing is straight vertical bars that "
        "follow the two sloping edges of the floor, the bars themselves are not "
        "triangles, no triangular window, no triangular wall panel, both side "
        f"returns of the floor visible, street exterior only, {ONE_FACADE}, "
        "overcast daylight"
    ),
    (
        "oblique sidewalk view of a 6th-floor balcony, limestone facade, only the "
        "stone floor slab is triangular, wall edge of the floor is the base, apex "
        "of the floor points away from the wall at the front center, masonry "
        "balusters are ordinary vertical posts following that floor edge, the "
        "balusters are not triangles, the window and the wall stay rectangular, "
        f"the triangle is the floor plan alone, {ONE_FACADE}, clear afternoon light"
    ),
    (
        "angled street photo of a 3rd-floor balcony, ochre historic facade, solid "
        "parapet standing on a floor slab whose plan is an isosceles triangle, "
        "the parapet is a normal vertical wall that follows the two sloping edges "
        "of the floor and meets at the front point of the floor, the triangle is "
        "the floor slab, not the parapet face and not a window, carved brackets "
        f"under the floor, {ONE_FACADE}, soft morning light"
    ),
]

TRAPEZOID_PROMPTS: list[str] = [
    (
        "three-quarter street photo of a 5th-floor balcony, sandstone facade, "
        "only the balcony floor slab is a trapezoid: the long parallel edge of "
        "the floor is fixed to the wall, the short parallel edge of the floor is "
        "at the front, the two non-parallel sides of the floor meet the wall at "
        "about 75 degrees, this trapezoid is the floor plan, not a rectangle "
        "distorted by perspective, the window behind is an ordinary rectangle, "
        "the dark metal railing is straight bars that follow the floor edge, the "
        "railing is not a trapezoid panel, no trapezoidal window, no trapezoidal "
        f"wall, {ONE_FACADE}, overcast daylight"
    ),
    (
        "oblique sidewalk view of a 2nd-floor balcony, red brick facade, only the "
        "stone floor is a trapezoid, wider along the wall and narrower at the "
        "front, front edge of the floor parallel to the wall, floor sides at "
        "about 75 degrees, masonry balusters are ordinary vertical posts along "
        "that floor outline, the balusters and the brick wall are not trapezoids, "
        f"the trapezoid is the floor plan alone, {ONE_FACADE}, clear afternoon light"
    ),
    (
        "angled street photo of a 7th-floor balcony, cream historic facade, solid "
        "parapet standing on a floor slab whose plan is a trapezoid, long base of "
        "the floor on the wall, short front of the floor parallel to the wall, "
        "the parapet is a normal vertical wall following that floor, the trapezoid "
        "is the floor slab, not the parapet face and not a window, heavy brackets "
        f"under the floor, {ONE_FACADE}, soft daylight"
    ),
]

HEXAGON_PROMPTS: list[str] = [
    (
        "three-quarter street photo of a 4th-floor balcony, beige stucco facade, "
        "hexagonal concrete balcony made from a rectangle with both front corners "
        "cut off, side edges stay perpendicular to the wall, then a diagonal chamfer "
        "on each front corner meeting the short front edge at about 30 degrees, "
        "dark metal railing follows all five outer edges, six-sided plan obvious, "
        f"not a rectangle, not a curve, {ONE_FACADE}, overcast daylight"
    ),
    (
        "oblique sidewalk view of a 6th-floor balcony, limestone facade, stone "
        "balcony with clipped front corners, straight sides, two diagonal corner "
        "cuts, short straight front, masonry balusters follow that hexagon outline, "
        f"both chamfers visible, {ONE_FACADE}, clear afternoon light"
    ),
    (
        "three-quarter street crop of a 3rd-floor balcony, ochre facade, solid "
        "parapet on a hexagon slab, parapet turns at two chamfered front corners, "
        "short front face, perpendicular side returns, no openings, the "
        f"clipped-corner plan is readable, {ONE_FACADE}, soft morning light"
    ),
    (
        "angled street photo of a 8th-floor balcony, grey concrete facade, frosted "
        "glass panels following a hexagonal balcony, panels change direction at "
        "both cut front corners, short front panel, side panels square to the wall, "
        f"not a bow window, {ONE_FACADE}, diffuse cloudy light"
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
