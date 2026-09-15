"""Prompts for synthetic solid balcony railing crops (FLUX.2).

Target look:
- reads as one integral wall / waist wall / parapet
- no openings; strong mass / wall-like appearance
- concrete, mortar, stone, plaster, stucco, or parapet finishes

Shared photo constraints (aligned with other synth sets):
- street-side viewpoint
- medium facade crop with full window visible
- upper-story balcony (2nd floor or higher)
"""

from __future__ import annotations

SOLID_PROMPTS: list[str] = [
    "street-side medium shot of an upper-story balcony with a solid concrete parapet wall, no openings, heavy mass wall-like railing, full window visible above",
    "street view facade crop, 2nd-floor balcony enclosed by a solid stone waist wall, opaque parapet with no gaps, entire window behind, medium framing",
    "from the sidewalk, third-floor balcony with a continuous plastered solid parapet, wall-like mass no openings, whole window in frame, medium facade photo",
    "street-level exterior of an upper-floor balcony, solid mortar-finished parapet wall, thick opaque enclosure, no rail openings, full window visible",
    "apartment facade from the street, 2nd-floor balcony with solid stucco parapet, integral wall appearance with no openings, complete window visible, medium distance",
    "street-side photo of an upper-story balcony closed by a solid concrete block parapet, mass wall look, no balusters or panels with gaps, full window unit",
    "frontal street crop of a solid masonry balcony parapet on the 2nd floor or higher, opaque wall-like railing, full window visible, medium shot",
    "street view of ochre wall, white solid plaster parapet on the 2nd floor balcony, continuous waist wall with no openings, entire window shown",
    "facade balcony unit from the street, upper floor, thick solid stone parapet reading as one wall mass, whole window and lintel visible, pulled back",
    "street-level photo of a 3rd-floor balcony with solid rendered concrete parapet finish, no openings, wall-like appearance, full window in frame",
    "sidewalk street-side view of upper-story balcony, solid parapet in cast concrete, heavy mass enclosure without gaps, complete window behind",
    "frontal street photo of a solid balcony on the 2nd floor or higher, integral mortar and stone waist wall, opaque no openings, full window visible, medium crop",
]

NEGATIVE_PROMPT = (
    "glass panels, frosted glass, metal plate privacy screen, thin frame and panel, "
    "openwork balusters, vertical balusters, pierced grille, decorative iron, "
    "gaps between posts, see-through railing, "
    "ground-floor porch, view from on the balcony looking out, "
    "extreme close-up of railing only, cropped window, cartoon, blurry"
)
