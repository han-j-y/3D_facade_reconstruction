"""FLUX.2 prompts for the rare triangle balcony floor plan.

50 photos rotating metal openwork, masonry openwork, and a solid parapet.
The triangle is the floor slab only. Windows, rails, and wall panels stay
ordinary rectangles.

These crops must keep the slab outline. Do not run them through
``resize_flux_images.py`` (that rewrite is a 120×44 railing band).

The plan matches ``balcony_plan.slab_outline``: isosceles, wall edge is the
base, apex at the front center.
"""

from __future__ import annotations

# Full-frame crops skipped by the 120×44 resize. Hexagon and trapezoid crops
# on disk (now labeled rectangle) are full frame too.
FLOOR_PLAN_PREFIXES: tuple[str, ...] = (
    "flux2_triangle_",
    "flux2_trapezoid_",
    "flux2_hexagon_",
)

_KIND_SUFFIXES: dict[str, tuple[str, ...]] = {
    "triangle": ("metal", "masonry", "solid"),
}

# Positive description of one flat wall. Naming the building edge makes
# FLUX draw it, so this clause only says what is in the frame. The camera is
# always a pedestrian on the street; the slab outline reads from its underside.
ONE_FACADE = (
    "a long flat facade fills the frame, one balcony in the center of that "
    "single wall, identical windows continue on the left and on the right, "
    "the wall is one plane, the edge of the building is outside the photo, "
    "street photo taken by a pedestrian standing on the far sidewalk, camera "
    "held at human eye level near the ground and tilted up toward the balcony, "
    "the flat underside of the balcony floor is visible from below"
)

TRIANGLE_FACADE = (
    f"{ONE_FACADE}, every window, door, cornice and roof line on the facade is "
    "a plain rectangle made of horizontal and vertical lines"
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
# Small turns along the same sidewalk. Each one keeps the wall running past both
# sides of the frame so FLUX does not swing round to the building edge.
_VIEWS: tuple[str, ...] = (
    "pedestrian straight across the street, facing the wall square on",
    "pedestrian a few steps to the left along the sidewalk, looking at the wall "
    "about 10 degrees off square, the same wall continues past both sides of "
    "the frame",
    "pedestrian a few steps to the right along the sidewalk, looking at the wall "
    "about 10 degrees off square, the same wall continues past both sides of "
    "the frame",
)
_EXTRAS: tuple[str, ...] = (
    "small stone brackets under the slab",
    "plain metal brackets under the slab",
    "a flower box along the outer edge of the floor",
    "two potted shrubs sitting on the floor",
    "a thin stone cornice under the floor",
)

# Floor-shape scenes use their own wall/light/detail banks so a new batch
# looks different from the enclosure set and from earlier floor batches.
_FLOOR_FACADES: tuple[str, ...] = (
    "red brick",
    "dark grey brick",
    "white painted brick",
    "terracotta plaster",
    "light blue plaster",
    "travertine stone",
    "board-marked exposed concrete",
    "warm brown stone",
    "dusty pink plaster",
    "charcoal grey render",
)
_FLOOR_LIGHTS: tuple[str, ...] = (
    "late afternoon golden light",
    "wet street after light rain, grey sky",
    "hazy summer light",
    "low winter sun with long shadows",
    "early evening light with warm lit windows",
)
_FLOOR_EXTRAS: tuple[str, ...] = (
    "a glass balcony door with a plain frame behind the railing",
    "a narrow drip edge along the underside of the slab",
    "a small wall lamp beside the balcony door",
    "a folded chair standing on the floor",
    "climbing ivy on the wall beside the balcony",
)

_TRIANGLE_KIND: dict[str, str] = {
    "metal": (
        "straight dark metal vertical bars of equal height stand along the two "
        "sloping edges of the floor and meet at the front point"
    ),
    "masonry": (
        "ordinary vertical masonry balusters of equal height stand along the two "
        "sloping edges of the floor, with a flat level top rail"
    ),
    "solid": (
        "a plain solid parapet of constant height with a flat level top runs "
        "along the two sloping edges of the floor and meets at the front point"
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
                "view": _VIEWS[(i // 2) % len(_VIEWS)],
            }
        )
    return slots


def _floor_slots(count: int = PROMPT_COUNT) -> list[dict[str, str]]:
    slots: list[dict[str, str]] = []
    for i in range(count):
        slots.append(
            {
                "floor": _FLOORS[(i + 3) % len(_FLOORS)],
                "facade": _FLOOR_FACADES[i % len(_FLOOR_FACADES)],
                "light": _FLOOR_LIGHTS[(i // len(_FLOOR_FACADES)) % len(_FLOOR_LIGHTS)],
                "extra": _FLOOR_EXTRAS[(i + 2) % len(_FLOOR_EXTRAS)],
                "view": _VIEWS[(i // 2 + 1) % len(_VIEWS)],
            }
        )
    return slots


def _kinds(shape: str, count: int) -> list[str]:
    cycle = _KIND_SUFFIXES[shape]
    return [cycle[i % len(cycle)] for i in range(count)]


def _triangle_prompts() -> list[str]:
    prompts: list[str] = []
    for slot, kind in zip(_floor_slots(), _kinds("triangle", PROMPT_COUNT)):
        prompts.append(
            f"{TRIANGLE_FACADE}, {slot['view']}, the floor shape is a triangle: "
            "the balcony floor is one thin flat horizontal concrete plate of even "
            "thickness, triangular in plan, its wide base runs along the wall and "
            "its apex is one point sticking straight out from the wall at the "
            "center, the plate edge is a thin level band of the same height all "
            "the way round, the flat level underside of the plate shows the same "
            "three-sided outline pointing out from the wall, the wall under the "
            "plate is plain and flat, only the floor plate has this outline, "
            f"{slot['floor']} floor, {slot['facade']} facade, "
            f"{_TRIANGLE_KIND[kind]}, {slot['extra']}, {slot['light']}"
        )
    return prompts


TRIANGLE_PROMPTS: list[str] = _triangle_prompts()
_BANKS: dict[str, list[str]] = {"triangle": TRIANGLE_PROMPTS}


def prefixes_for(shape: str) -> list[str]:
    """Filename prefixes aligned one-for-one with that shape's prompts."""
    bank = _BANKS[shape]
    return [f"flux2_{shape}_{kind}_" for kind in _kinds(shape, len(bank))]


def prompt_rows(shape: str) -> list[tuple[str, str]]:
    """``(prompt, filename_prefix)`` in the same order as ``prefixes_for``."""
    bank = _BANKS[shape]
    prefixes = prefixes_for(shape)
    if len(bank) != len(prefixes):
        raise ValueError(f"{shape} prompts and prefixes differ in length")
    return list(zip(bank, prefixes))
