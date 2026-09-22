"""Prompts for synthetic solid balcony railing crops (FLUX.2).

Target look:
- opaque concrete parapet, no openings, color near RGB 137, 122, 99
- slight yellow or orange variation, still mainly concrete
- a thin metal handrail along the top of the wall
- occasional wall decoration, and sometimes a person or plant at the edge

Shared photo constraints:
- street-side viewpoint
- distant, dim, horizontal crop with only part of the window
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

# viewpoint, concrete color, wall decoration, edge interference
_SCENES: tuple[tuple[str, str, str, str], ...] = (
    (
        "street-side photo",
        "dusty warm gray-brown concrete RGB 137, 122, 99",
        "a small tile decoration band on the wall",
        "no pedestrian",
    ),
    (
        "sidewalk view looking up",
        "concrete mainly RGB 137, 122, 99 with a slight yellow stain",
        "a plain concrete face with a faint horizontal joint",
        "a small pedestrian at the far left edge, not the subject",
    ),
    (
        "distant street photo",
        "concrete RGB 137, 122, 99 with slight orange weathering on one side",
        "a stained patch on the concrete",
        "a potted plant at the right end",
    ),
    (
        "street view from across the road",
        "warm gray-brown concrete close to RGB 137, 122, 99, a bit more yellow",
        "a decorative relief panel on the wall",
        "no person",
    ),
    (
        "sidewalk facade crop",
        "concrete around RGB 137, 122, 99 with a small orange patch",
        "a narrow ornament band",
        "ivy climbing one corner",
    ),
    (
        "street photo",
        "mostly plain concrete RGB 137, 122, 99",
        "a small cornice",
        "a hanging plant and a tiny distant pedestrian near the edge",
    ),
    (
        "looking up from the sidewalk",
        "dusty concrete with slight yellow-brown variation around RGB 137, 122, 99",
        "a small wall emblem",
        "no people",
    ),
    (
        "street-side view",
        "solid concrete mainly RGB 137, 122, 99 with faint orange discoloration",
        "a stained concrete decoration",
        "one planter with a green plant",
    ),
    (
        "medium-far street crop",
        "concrete RGB 137, 122, 99, a little yellower",
        "ornamental molding on the wall",
        "a person partly cut off at the frame edge, not a portrait",
    ),
    (
        "distant street-side crop",
        "plain dusty gray-brown concrete RGB 137, 122, 99, almost no extra color",
        "no extra ornament",
        "one small potted plant and no pedestrian",
    ),
)


def _solid_prompt(floor: str, scene: tuple[str, str, str, str]) -> str:
    viewpoint, color, decoration, interference = scene
    return (
        f"{viewpoint} of a {floor}-floor balcony, opaque solid concrete parapet "
        f"with no openings, {color}, a thin metal handrail mounted along the top "
        f"of the concrete wall, {decoration}, {interference}, dim overcast daylight, "
        "distant horizontal crop, partial window behind, low-resolution phone photo, "
        "street-side exterior only, not from on the balcony"
    )


SOLID_PROMPTS: list[str] = [
    _solid_prompt(floor, scene) for floor in _FLOORS for scene in _SCENES
]

NEGATIVE_PROMPT = (
    "glass panels, frosted glass, metal plate privacy screen, "
    "openwork balusters, vertical baluster railing, pierced grille, "
    "see-through railing instead of a wall, "
    "white facade, red brick parapet, pale blue wall, charcoal black wall, "
    "ground-floor porch, view from on the balcony looking out, "
    "extreme close-up of railing only, cartoon, cgi, sharp studio render"
)
