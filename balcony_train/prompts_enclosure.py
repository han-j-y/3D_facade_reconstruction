"""FLUX.2 prompts for half-enclosed and enclosed balconies.

Half-enclosed: columns or similar elements cover the balcony, and the gaps
stay open to the outside air.
Enclosed: walls or windows close the balcony, so it is not open to the air.

Two prompts each. ``-n 2`` writes one image per prompt.
"""

from __future__ import annotations

from balcony_train.prompts_floor_shape import ONE_FACADE

ENCLOSURE_PREFIXES: tuple[str, ...] = (
    "flux2_half_enclosed_",
    "flux2_enclosed_",
)

HALF_ENCLOSED_PROMPTS: list[str] = [
    (
        "street-side photo of a 3rd-floor balcony on a pale yellow historic facade, "
        "half-enclosed balcony covered by classical columns and round arches, "
        "open air visible between the columns, no glass walls, stone balustrade "
        "along the front, both side columns in frame, medium facade crop, "
        f"street exterior only, not from on the balcony, {ONE_FACADE}, overcast daylight"
    ),
    (
        "oblique sidewalk view of a 4th-floor balcony on a cream stone facade, "
        "half-enclosed loggia carried on columns with arched openings, still open "
        "to the outside air through the arches, masonry balustrade, no windows "
        "closing the balcony, full window of the room visible behind, street "
        f"exterior only, {ONE_FACADE}, clear afternoon light"
    ),
]

ENCLOSED_PROMPTS: list[str] = [
    (
        "street-side photo of a 5th-floor enclosed balcony on a white ornate facade, "
        "balcony on one flat wall closed by tall windows with white frames, glass "
        "walls so the balcony is not open to the outside air, no open railing "
        "gaps, interior visible only through the glass, medium facade crop, "
        f"street exterior only, not from on the balcony, {ONE_FACADE}, soft daylight"
    ),
    (
        "three-quarter street photo of a 2nd-floor enclosed balcony on a pale "
        "yellow and white facade, straight projecting balcony closed by continuous "
        "windows, walls and glazing enclose the balcony, not open to the outside "
        f"air, no open balustrade, street exterior only, {ONE_FACADE}, diffuse cloudy light"
    ),
]
