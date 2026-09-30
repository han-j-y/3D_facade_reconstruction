"""FLUX.2 prompts for rare balcony floor plans: triangle, trapezoid, hexagon.

Each shape has 50 photos. Triangle and trapezoid rotate metal openwork,
masonry openwork, and a solid parapet. Hexagon also rotates a surface panel.
The triangle or trapezoid is the floor slab only. Windows, rails, and wall
panels stay ordinary rectangles.

These crops must keep the slab outline. Do not run them through
``resize_flux_images.py`` (that rewrite is a 120×44 railing band).

Plans match ``balcony_plan.slab_outline``:
- triangle: isosceles, wall edge is the base, apex at the front center
- trapezoid: long edge on the wall, shorter front parallel to the wall,
  each side meets the wall at about 75 degrees
- hexagon: a rectangle whose left and right edges leave the wall at a
  right angle, then a short diagonal chamfer, then a wide front parallel
  to the wall. Six edges. The outline bends twice on each side.
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

PROMPT_COUNT = 50

_FACADES: tuple[str, ...] = (
    "beige stucco",
    "pale limestone",
    "cream painted plaster",
    "white stone",
    "ochre historic plaster",
    "grey concrete",
    "sandstone",
    "pale yellow plaster",
    "tan brick",
    "soft green plaster",
)
_LIGHTS: tuple[str, ...] = (
    "overcast daylight",
    "clear afternoon light",
    "soft morning light",
    "diffuse cloudy light",
    "bright noon light",
)
_FLOORS: tuple[str, ...] = (
    "2nd",
    "3rd",
    "4th",
    "5th",
    "6th",
    "7th",
    "8th",
    "9th",
    "10th",
    "11th",
)
_EXTRAS: tuple[str, ...] = (
    "small stone brackets under the slab",
    "plain metal brackets under the slab",
    "a flower box along the outer edge of the floor",
    "two potted shrubs sitting on the floor",
    "a thin stone cornice under the floor",
)

_TRIANGLE_KIND: dict[str, str] = {
    "metal": (
        "dark metal vertical bars follow only the two sloping edges of that "
        "triangular floor, the bars are not triangles, the window is an ordinary rectangle"
    ),
    "masonry": (
        "ordinary vertical masonry balusters follow that triangular floor edge, "
        "the window stays rectangular"
    ),
    "solid": (
        "a plain solid parapet follows the two sloping edges of that triangular "
        "floor and meets at the front point, the parapet face itself is not a triangle"
    ),
}
_TRAPEZOID_KIND: dict[str, str] = {
    "metal": (
        "dark metal bars follow that floor edge, the railing is not a trapezoid "
        "panel, the window is an ordinary rectangle"
    ),
    "masonry": (
        "ordinary vertical masonry balusters follow only that floor outline, "
        "the window stays rectangular"
    ),
    "solid": (
        "a plain solid parapet follows that floor, the parapet face itself is not a trapezoid"
    ),
}
_HEXAGON_KIND: dict[str, str] = {
    "metal": (
        "dark metal vertical bars follow that six-sided floor and bend twice on "
        "each side, the window is an ordinary rectangle"
    ),
    "masonry": (
        "ordinary vertical masonry balusters follow that six-sided floor and "
        "bend twice on each side"
    ),
    "solid": (
        "a plain solid parapet follows that six-sided floor and bends twice on "
        "each side, the parapet face itself is not a hexagon"
    ),
    "surface": (
        "frosted glass panels follow that six-sided floor, one panel on each "
        "straight side and one panel on each diagonal chamfer, plus a wide front panel"
    ),
}


def _slots(count: int = PROMPT_COUNT) -> list[dict[str, str]]:
    slots: list[dict[str, str]] = []
    for i in range(count):
        slots.append(
            {
                "floor": _FLOORS[i % len(_FLOORS)],
                "facade": _FACADES[i % len(_FACADES)],
                "light": _LIGHTS[(i // len(_FACADES)) % len(_LIGHTS)],
                "extra": _EXTRAS[i % len(_EXTRAS)],
            }
        )
    return slots


def _kinds(shape: str, count: int) -> list[str]:
    cycle = _KIND_SUFFIXES[shape]
    return [cycle[i % len(cycle)] for i in range(count)]


def _triangle_prompts() -> list[str]:
    prompts: list[str] = []
    for slot, kind in zip(_slots(), _kinds("triangle", PROMPT_COUNT)):
        prompts.append(
            f"{ONE_FACADE}, the floor shape is a triangle, the top surface of the "
            "concrete balcony floor is an isosceles triangle, apex of the floor "
            "points away from the wall, the floor comes to one front point, the "
            f"floor outline itself is triangular, {slot['floor']} floor, "
            f"{slot['facade']} facade, {_TRIANGLE_KIND[kind]}, {slot['extra']}, "
            f"{slot['light']}"
        )
    return prompts


def _trapezoid_prompts() -> list[str]:
    prompts: list[str] = []
    for slot, kind in zip(_slots(), _kinds("trapezoid", PROMPT_COUNT)):
        prompts.append(
            f"{ONE_FACADE}, the floor shape is a trapezoid, only the floor is a "
            "trapezoid, this floor plan is a trapezoid, long edge of the floor on "
            "the wall, short front edge of the floor parallel to the wall, sides "
            f"of the floor meet the wall at about 75 degrees, {slot['floor']} floor, "
            f"{slot['facade']} facade, {_TRAPEZOID_KIND[kind]}, {slot['extra']}, "
            f"{slot['light']}"
        )
    return prompts


def _hexagon_prompts() -> list[str]:
    prompts: list[str] = []
    for slot, kind in zip(_slots(), _kinds("hexagon", PROMPT_COUNT)):
        prompts.append(
            f"{ONE_FACADE}, the floor shape is a hexagon, the balcony floor starts "
            "as a rectangle, the left and right edges leave the wall at a right "
            "angle and run straight out, then each side turns onto a short diagonal "
            "chamfer, then a wide front edge runs parallel to the wall, the chamfer "
            "meets that front edge at about 30 degrees, six edges, the outline "
            f"bends twice on the left and twice on the right, {slot['floor']} floor, "
            f"{slot['facade']} facade, {_HEXAGON_KIND[kind]}, {slot['extra']}, "
            f"{slot['light']}"
        )
    return prompts


TRIANGLE_PROMPTS: list[str] = _triangle_prompts()
TRAPEZOID_PROMPTS: list[str] = _trapezoid_prompts()
HEXAGON_PROMPTS: list[str] = _hexagon_prompts()


def prefixes_for(shape: str) -> list[str]:
    """Filename prefixes aligned one-for-one with that shape's prompts."""
    bank = {
        "triangle": TRIANGLE_PROMPTS,
        "trapezoid": TRAPEZOID_PROMPTS,
        "hexagon": HEXAGON_PROMPTS,
    }[shape]
    return [f"flux2_{shape}_{kind}_" for kind in _kinds(shape, len(bank))]


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
