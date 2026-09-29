"""FLUX.2 prompts for half-enclosed and enclosed balconies.

Both kinds project past the exterior wall. The slab and the enclosure sit
in front of the facade, carried on brackets, and are not recessed into it.

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

_PROJECTING = (
    "the balcony floor cantilevers out beyond the exterior wall line, the whole "
    "balcony volume sits in front of the facade, stone brackets under the "
    "projecting slab, not recessed into the wall, not a niche inside the facade, "
    "the room wall is behind the balcony"
)

HALF_ENCLOSED_PROMPTS: list[str] = [
    (
        f"{ONE_FACADE}, three-quarter street photo of a 3rd-floor balcony on a "
        f"pale yellow historic facade, {_PROJECTING}, half-enclosed balcony "
        "standing on that projecting slab, classical columns and a stone "
        "balustrade in front of the wall, open air visible between the columns, "
        "no glass walls, still open to the outside air, street exterior only, "
        "overcast daylight"
    ),
    (
        f"{ONE_FACADE}, oblique sidewalk view of a 4th-floor balcony on a cream "
        f"stone facade, {_PROJECTING}, half-enclosed balcony carried out past the "
        "wall on corbels, columns with arched openings on the projecting slab, "
        "open to the outside air through the arches, masonry balustrade, no glass "
        "walls, no windows closing the balcony, clear afternoon light"
    ),
]

ENCLOSED_PROMPTS: list[str] = [
    (
        f"{ONE_FACADE}, three-quarter street photo of a 5th-floor enclosed "
        f"balcony on a white ornate facade, {_PROJECTING}, tall windows with "
        "white frames stand on the projecting slab in front of the wall, glass "
        "walls so the balcony is not open to the outside air, no open railing "
        "gaps, the enclosure sticks out from the facade on brackets, street "
        "exterior only, soft daylight"
    ),
    (
        f"{ONE_FACADE}, oblique sidewalk view of a 2nd-floor enclosed balcony "
        f"on a pale yellow facade, {_PROJECTING}, continuous windows close the "
        "projecting balcony, walls and glazing enclose it, not open to the "
        "outside air, no open balustrade, the glazed box is outside the wall "
        "line, street exterior only, diffuse cloudy light"
    ),
]
