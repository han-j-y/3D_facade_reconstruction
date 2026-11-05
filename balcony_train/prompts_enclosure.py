"""FLUX.2 prompts for half-enclosed and enclosed balconies.

Both kinds project past the exterior wall. The slab and the enclosure sit
in front of the facade, carried on brackets, and are not recessed into it.

Half-enclosed: columns or similar elements cover the balcony, and the gaps
stay open to the outside air.
Enclosed: walls or windows close the balcony, so it is not open to the air.

50 prompts each. ``-n 50`` writes one image per prompt.
"""

from __future__ import annotations

from balcony_train.prompts_floor_shape import ONE_FACADE, _slots

ENCLOSURE_PREFIXES: tuple[str, ...] = (
    "flux2_half_enclosed_",
    "flux2_enclosed_",
)

_PROJECTING = (
    "the balcony floor cantilevers out beyond the exterior wall line, the whole "
    "balcony volume sits in front of the facade, stone brackets under the "
    "projecting slab, not recessed into the wall, not a niche inside the facade, "
    "the room wall is behind the balcony"
)

_HALF_STYLES: tuple[str, ...] = (
    "classical round columns and a stone balustrade",
    "square stone columns and a masonry balustrade",
    "slim columns with arched openings and a stone rail",
    "paired columns and an open balustrade",
    "tall columns under a flat lintel and a low stone rail",
)
_ENCLOSED_STYLES: tuple[str, ...] = (
    "tall windows with white frames",
    "a grid of narrow windows",
    "continuous windows with dark frames",
    "floor-to-ceiling windows",
    "wood-framed windows",
)


def _half_enclosed_prompts() -> list[str]:
    prompts: list[str] = []
    for i, slot in enumerate(_slots()):
        style = _HALF_STYLES[i % len(_HALF_STYLES)]
        prompts.append(
            f"{ONE_FACADE}, {slot['view']}, {slot['floor']} floor, {slot['facade']} facade, "
            f"{_PROJECTING}, half-enclosed balcony standing on that projecting "
            f"slab, {style}, open air visible between the columns, no glass walls, "
            f"still open to the outside air, {slot['extra']}, {slot['light']}"
        )
    return prompts


def _enclosed_prompts() -> list[str]:
    prompts: list[str] = []
    for i, slot in enumerate(_slots()):
        style = _ENCLOSED_STYLES[i % len(_ENCLOSED_STYLES)]
        prompts.append(
            f"{ONE_FACADE}, {slot['view']}, {slot['floor']} floor, {slot['facade']} facade, "
            f"{_PROJECTING}, enclosed balcony on that projecting slab, {style}, "
            "glass walls so the balcony is not open to the outside air, no open "
            f"railing gaps, {slot['extra']}, {slot['light']}"
        )
    return prompts


HALF_ENCLOSED_PROMPTS: list[str] = _half_enclosed_prompts()
ENCLOSED_PROMPTS: list[str] = _enclosed_prompts()
