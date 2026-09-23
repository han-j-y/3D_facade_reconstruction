"""Prompts for synthetic solid balcony railing crops (FLUX.2).

Target look:
- solid parapet on a historic building, no openings
- color from paint or render (ochre, cream, pale yellow, pinkish plaster),
  or weathered grey stone
- heavy relief ornament across the parapet surface, recessed but not pierced
- carved stone brackets under the slab; a metal handrail only sometimes
- sometimes a person or plant at the edge
- about 1 in 8 prompts has power lines or a tree in front of the balcony

Shared photo constraints:
- street-side viewpoint
- distant, dim, horizontal crop
- balcony on a random floor from 2 to 10
"""

from __future__ import annotations

import random

N_PROMPTS = 50
OCCLUDER_EVERY = 8
_RNG_SEED = 9100

_FLOORS = ("2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th")

_OCCLUDERS = (
    "overhead power lines crossing in front of the balcony",
    "tree branches in the foreground partly covering the parapet",
    "a street tree in front partly hiding one side of the balcony",
    "tangled utility wires running across the frame in front of the parapet",
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


def _solid_prompt(
    floor: str,
    scene: tuple[str, str, str, str],
    occluder: str | None = None,
) -> str:
    viewpoint, color, decoration, interference = scene
    foreground = f"{occluder}, " if occluder else ""
    return (
        f"{viewpoint} of a {floor}-floor balcony on a historic building, "
        f"solid parapet with no openings, {color}, {decoration} on the parapet "
        "surface, the ornament is recessed but not pierced, no sky or window "
        "visible through the parapet, heavy carved stone brackets under the "
        f"balcony slab, {foreground}{interference}, dim overcast daylight, "
        "distant horizontal crop, low-resolution phone photo, "
        "street-side exterior only, not from on the balcony"
    )


def _build_prompts(n: int = N_PROMPTS, seed: int = _RNG_SEED) -> list[str]:
    rng = random.Random(seed)
    occluded = set(rng.sample(range(n), round(n / OCCLUDER_EVERY)))
    return [
        _solid_prompt(
            rng.choice(_FLOORS),
            _SCENES[i % len(_SCENES)],
            rng.choice(_OCCLUDERS) if i in occluded else None,
        )
        for i in range(n)
    ]


SOLID_PROMPTS: list[str] = _build_prompts()

NEGATIVE_PROMPT = (
    "glass panels, frosted glass, metal plate privacy screen, "
    "openwork balusters, pierced grille, see-through openings, "
    "sky visible through the railing, plain unornamented concrete, "
    "modern facade, red brick parapet, pale blue wall, "
    "ground-floor porch, view from on the balcony looking out, "
    "extreme close-up of railing only, cartoon, cgi, sharp studio render"
)
