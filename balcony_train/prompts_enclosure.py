"""FLUX.2 prompts for half-enclosed and enclosed balconies.

Half-enclosed: columns or similar elements cover the balcony, and the gaps
stay open to the outside air.
Enclosed: walls or windows close the balcony, so it is not open to the air.

Two prompts each. ``-n 2`` writes one image per prompt.
"""

from __future__ import annotations

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
        "street exterior only, not from on the balcony, overcast daylight"
    ),
    (
        "oblique sidewalk view of a 4th-floor balcony on a cream stone facade, "
        "half-enclosed loggia carried on columns with arched openings, still open "
        "to the outside air through the arches, masonry balustrade, no windows "
        "closing the balcony, full window of the room visible behind, street "
        "exterior only, clear afternoon light"
    ),
]

ENCLOSED_PROMPTS: list[str] = [
    (
        "street-side photo of a 5th-floor enclosed balcony on a white ornate facade, "
        "polygonal bay closed by tall windows with white frames, glass walls all "
        "around so the balcony is not open to the outside air, no open railing "
        "gaps, interior visible only through the glass, medium facade crop, "
        "street exterior only, not from on the balcony, soft daylight"
    ),
    (
        "three-quarter street photo of a 2nd-floor enclosed balcony on a pale "
        "yellow and white facade, curved bay wrapped in continuous windows, "
        "walls and glazing enclose the balcony, not open to the outside air, "
        "no open balustrade, street exterior only, diffuse cloudy light"
    ),
]
