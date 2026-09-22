"""Prompts for synthetic solid balcony railing crops (FLUX.2).

Target look:
- solid parapet on a historic building, no openings
- color from paint or render (ochre, cream, pale yellow, pinkish plaster),
  or weathered grey stone
- heavy relief ornament across the parapet surface, recessed but not pierced
- carved stone brackets under the slab; a metal handrail only sometimes
- sometimes a person or plant at the edge

Shared photo constraints:
- street-side viewpoint
- distant, dim, horizontal crop
- balcony on floors 2–10
"""

from __future__ import annotations

_FLOORS = (
    "2nd",
    "3rd",
    "4th",
    "5th",
    "6th",
    "7th",
    "8th",
    "9th",
    "10th",
    "5th",
)

# viewpoint, painted color, relief decoration, edge interference
_SCENES: tuple[tuple[str, str, str, str], ...] = (
    (
        "street-side photo",
        "painted ochre yellow render",
        "a continuous row of recessed quatrefoil relief panels",
        "no pedestrian",
    ),
    (
        "sidewalk view looking up",
        "cream painted stucco",
        "a diamond lattice pattern carved into square panels",
        "a small pedestrian at the far left edge, not the subject",
    ),
    (
        "distant street photo",
        "warm pale yellow painted stone",
        "blind balusters carved in low relief on the wall surface, a thin metal handrail on top",
        "a potted plant at the right end",
    ),
    (
        "street view from across the road",
        "pinkish painted plaster",
        "a blind gothic arcade of small pointed trefoil arches in relief",
        "no person",
    ),
    (
        "sidewalk facade crop",
        "weathered grey stone darkened by soot",
        "a row of carved circular rosette medallions",
        "ivy climbing one corner",
    ),
    (
        "street photo",
        "beige painted render",
        "recessed rectangular panels with carved floral motifs",
        "a hanging plant and a tiny distant pedestrian near the edge",
    ),
    (
        "looking up from the sidewalk",
        "pale yellow painted wall",
        "a blind lattice grille pattern carved into solid stone",
        "no people",
    ),
    (
        "street-side view",
        "dark grey stone",
        "geometric relief panels under a heavy cornice, a thin metal handrail on top",
        "one planter with a green plant",
    ),
    (
        "medium-far street crop",
        "ochre painted plaster",
        "carved garlands and scroll ornaments across the parapet",
        "a person partly cut off at the frame edge, not a portrait",
    ),
    (
        "distant street-side crop",
        "cream painted stone",
        "small pilasters dividing ornate relief panels",
        "no pedestrian",
    ),
)


def _solid_prompt(floor: str, scene: tuple[str, str, str, str]) -> str:
    viewpoint, color, decoration, interference = scene
    return (
        f"{viewpoint} of a {floor}-floor balcony on a historic building, "
        f"solid parapet with no openings, {color}, {decoration} on the parapet "
        "surface, the ornament is recessed but not pierced, no sky or window "
        "visible through the parapet, heavy carved stone brackets under the "
        f"balcony slab, {interference}, dim overcast daylight, distant horizontal "
        "crop, low-resolution phone photo, street-side exterior only, "
        "not from on the balcony"
    )


SOLID_PROMPTS: list[str] = [
    _solid_prompt(floor, scene) for floor in _FLOORS for scene in _SCENES
]

NEGATIVE_PROMPT = (
    "glass panels, frosted glass, metal plate privacy screen, "
    "openwork balusters, pierced grille, see-through openings, "
    "sky visible through the railing, plain unornamented concrete, "
    "modern facade, red brick parapet, pale blue wall, "
    "ground-floor porch, view from on the balcony looking out, "
    "extreme close-up of railing only, cartoon, cgi, sharp studio render"
)
