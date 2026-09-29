"""FLUX.2 prompts for rare balcony floor plans: triangle, trapezoid, hexagon.

Each shape has four photos, in this order: metal openwork, masonry openwork,
solid parapet, surface panel. Filenames use a matching prefix so
``label_from_prefix.py`` can set kind, material, and floor_shape.

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

_KIND_SUFFIXES: tuple[str, ...] = ("metal", "masonry", "solid", "surface")

# Every crop is one balcony on one flat wall. FLUX otherwise draws the
# building corner, where two facades meet.
ONE_FACADE = (
    "balcony attached to the middle of one straight flat facade, "
    "a single wall face only, the building corner and the adjacent facade "
    "are out of frame, not a corner balcony, not wrapped around two walls"
)

NEGATIVE_PROMPT = (
    "rectangular balcony, semicircular balcony, curved bow window, "
    "view from standing on the balcony, railing-only close-up, "
    "floor plan diagram, blueprint, top-down drawing, cartoon, cgi, studio render"
)

TRIANGLE_PROMPTS: list[str] = [
    (
        "three-quarter street photo of a 4th-floor balcony, beige stucco facade, "
        "isosceles triangular concrete slab, long edge fixed to the wall, both sides "
        "meet at one point at the front center, dark metal picket railing follows only "
        "those two sloping sides, no straight front rail, both side returns and the "
        "pointed front visible, full window behind, street exterior only, "
        f"{ONE_FACADE}, overcast daylight"
    ),
    (
        "oblique sidewalk view looking up at a 6th-floor balcony, limestone facade, "
        "triangular stone balcony slab pointing away from the wall, apex at the front "
        "center, masonry balusters follow the two sloping edges, wedge-shaped plan, "
        "not a rectangle, medium crop showing both side returns, "
        f"{ONE_FACADE}, clear afternoon light"
    ),
    (
        "three-quarter street crop of a 3rd-floor balcony, ochre historic facade, "
        "solid parapet on an isosceles triangular slab, parapet runs along two sides "
        "and meets at the front point, no openings, carved brackets under the slab, "
        "the pointed plan is obvious from the street, "
        f"{ONE_FACADE}, soft morning light"
    ),
    (
        "angled street photo of a 8th-floor balcony, grey concrete facade, frosted "
        "glass privacy panels in a thin metal frame following a triangular slab, "
        "panels turn at the front apex, no rectangular front, both slanted sides "
        "visible, olive-grey wall, "
        f"{ONE_FACADE}, diffuse cloudy light"
    ),
]

TRAPEZOID_PROMPTS: list[str] = [
    (
        "three-quarter street photo of a 5th-floor balcony, sandstone facade, "
        "the balcony floor plan is a trapezoid: long parallel edge fixed to the "
        "wall, short parallel edge at the front, the two non-parallel sides slant "
        "inward and meet the wall at about 75 degrees, this is the slab outline "
        "in plan, not a rectangle distorted by perspective, dark metal railing "
        "follows the two slants and the short front, both side returns visible, "
        f"{ONE_FACADE}, overcast daylight"
    ),
    (
        "oblique sidewalk view of a 2nd-floor balcony, red brick facade, floor "
        "plan is a trapezoid wider along the wall and narrower at the front, "
        "front edge parallel to the wall, slanted sides at about 75 degrees, "
        "masonry balusters follow that plan outline, not a rectangle seen in "
        f"perspective, {ONE_FACADE}, clear afternoon light"
    ),
    (
        "three-quarter street crop of a 7th-floor balcony, cream historic facade, "
        "solid parapet whose floor plan is a trapezoid, long base on the wall, "
        "short front parallel to the wall, slanted ends, no openings, the "
        "narrowing is the slab shape in plan, not perspective, heavy brackets "
        f"under the slab, {ONE_FACADE}, soft daylight"
    ),
    (
        "angled street photo of a 9th-floor balcony, white painted facade, grey "
        "metal privacy panels following a trapezoidal floor plan, front rail "
        "shorter than the wall edge and parallel to it, side panels lean inward, "
        f"{ONE_FACADE}, morning light"
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
    """Filename prefixes aligned with that shape's four prompts."""
    return [f"flux2_{shape}_{kind}_" for kind in _KIND_SUFFIXES]


def prompt_rows(shape: str) -> list[tuple[str, str]]:
    """``(prompt, filename_prefix)`` in metal, masonry, solid, surface order."""
    bank = {
        "triangle": TRIANGLE_PROMPTS,
        "trapezoid": TRAPEZOID_PROMPTS,
        "hexagon": HEXAGON_PROMPTS,
    }[shape]
    prefixes = prefixes_for(shape)
    if len(bank) != len(prefixes):
        raise ValueError(f"{shape} prompts and prefixes differ in length")
    return list(zip(bank, prefixes))
