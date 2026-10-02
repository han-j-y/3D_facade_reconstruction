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

# Triangle scenes keep the camera level with the balcony; the floor point
# shows through the side views instead of a downward look.
TRIANGLE_FACADE = (
    "a long flat facade fills the frame, one balcony in the center of that "
    "single wall, identical windows continue on the left and on the right, "
    "the wall is one plane, the edge of the building is outside the photo, "
    "street photo from across the road, camera at the same height as the "
    "balcony, every window, door, cornice and roof line on the facade is a "
    "plain rectangle made of horizontal and vertical lines"
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
# Small turns along the same street. Each one keeps the wall running past both
# sides of the frame so FLUX does not swing round to the building edge.
_VIEWS: tuple[str, ...] = (
    "camera straight across the street, facing the wall square on",
    "camera a little to the left down the street, looking at the wall about "
    "15 degrees off square, the same wall continues past both sides of the frame",
    "camera a little to the right down the street, looking at the wall about "
    "15 degrees off square, the same wall continues past both sides of the frame",
    "camera further to the left down the street, looking at the wall about "
    "25 degrees off square, the side of the floor slab is visible, the same "
    "wall continues past both sides of the frame",
    "camera further to the right down the street, looking at the wall about "
    "25 degrees off square, the side of the floor slab is visible, the same "
    "wall continues past both sides of the frame",
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
_TRAPEZOID_KIND: dict[str, str] = {
    "metal": (
        "straight dark metal vertical bars of equal height stand along the front "
        "and the two angled sides of the floor"
    ),
    "masonry": (
        "ordinary vertical masonry balusters of equal height stand along the "
        "front and the two angled sides of the floor, with a flat level top rail"
    ),
    "solid": (
        "a plain solid parapet of constant height with a flat level top runs "
        "along the front and the two angled sides of the floor"
    ),
}
_HEXAGON_KIND: dict[str, str] = {
    "metal": (
        "dark metal vertical bars run along all five railing sections, the "
        "window is an ordinary rectangle"
    ),
    "masonry": (
        "ordinary vertical masonry balusters run along all five railing sections"
    ),
    "solid": (
        "a plain solid parapet runs along all five railing sections, each "
        "section is a flat rectangular panel"
    ),
    "surface": (
        "frosted glass panels fill all five railing sections, one panel on each "
        "straight side, one on each diagonal chamfer, and a wide front panel"
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
            "the concrete balcony floor slab is triangular in plan, its wide base "
            "runs along the wall and its apex is one point sticking straight out "
            "from the wall at the center, only the floor slab has this shape, "
            f"{slot['floor']} floor, {slot['facade']} facade, "
            f"{_TRIANGLE_KIND[kind]}, {slot['extra']}, {slot['light']}"
        )
    return prompts


def _trapezoid_prompts() -> list[str]:
    prompts: list[str] = []
    for slot, kind in zip(_floor_slots(), _kinds("trapezoid", PROMPT_COUNT)):
        prompts.append(
            f"{ONE_FACADE}, {slot['view']}, the floor shape is a trapezoid, only the floor is a "
            "trapezoid, this floor plan is a trapezoid, long edge of the floor on "
            "the wall, short front edge of the floor parallel to the wall, sides "
            f"of the floor meet the wall at about 75 degrees, {slot['floor']} floor, "
            f"{slot['facade']} facade, {_TRAPEZOID_KIND[kind]}, {slot['extra']}, "
            f"{slot['light']}"
        )
    return prompts


def _hexagon_prompts() -> list[str]:
    prompts: list[str] = []
    for slot, kind in zip(_floor_slots(), _kinds("hexagon", PROMPT_COUNT)):
        prompts.append(
            f"{ONE_FACADE}, seen from above so the whole floor outline shows, "
            "the floor shape is a hexagon, a deep balcony floor with six edges, "
            "the railing has five straight sections: a straight left side section "
            "that leaves the wall at a right angle, a diagonal chamfer section, "
            "a wide front section parallel to the wall, a diagonal chamfer "
            "section, and a straight right side section that leaves the wall at a "
            "right angle, the straight side sections are as long as the chamfer "
            "sections and are clearly visible, each chamfer meets the front at "
            "about 30 degrees, the railing bends twice on the left and twice on "
            f"the right, {slot['floor']} floor, {slot['facade']} facade, "
            f"{_HEXAGON_KIND[kind]}, {slot['extra']}, {slot['light']}"
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
