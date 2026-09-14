"""Prompts for synthetic open_work + masonry balcony railing crops (FLUX.2).

Hard constraints (every prompt):
- viewpoint: street-side photo of the facade / balcony (never from on the balcony)
- openings: gaps must exist between posts (balusters, pierced panel, grille, etc.)
- framing: medium facade crop — whole window behind the balcony visible (not railing-only close-up)

Flexible:
- material: stone, concrete block, or concrete
- infill style between posts: balusters OR precast openwork panel (either OK)
"""

from __future__ import annotations

# Every prompt: street-side + openings + pulled-back enough to show the full window.
MASONRY_OPENWORK_PROMPTS: list[str] = [
    "street-side medium shot of a facade balcony unit, full window visible above the railing, concrete masonry posts with open gaps between posts, not a close-up",
    "street view facade crop from the sidewalk, entire window behind a stone balcony, thick posts with pierced precast panel openings, pulled back medium framing",
    "from the sidewalk facing the building, masonry balcony with square concrete posts and open baluster gaps, whole window in frame, medium facade photo",
    "street-level exterior photo of openwork concrete balcony, pier posts, grille openings between posts, full window and some wall visible, not zoomed into the rail only",
    "apartment facade from the street, limestone posts with spaced concrete balusters and openings, complete window visible behind the balcony, medium distance",
    "street-side photo of thick stone posts with perforated precast screen openings, balcony plus full window unit, slightly wider facade crop",
    "frontal street crop of balcony and full window, concrete block posts, openwork concrete panel with see-through openings, medium shot not macro",
    "street view of ochre wall with white concrete balcony posts and masonry balusters with open gaps, entire window shown, medium facade framing",
    "facade balcony unit from the street, heavy concrete posts, geometric precast grille openings, whole window and lintel visible, pulled back",
    "street-level photo of stone balcony parapet with piers and open baluster or pierced-panel gaps, full window in frame, medium architectural crop",
    "sidewalk street-side view of concrete balcony railing with openings between chunky posts, complete window behind, wider than railing close-up",
    "frontal street photo of masonry openwork balcony with corner posts and open mid-span panel or balusters, full window visible, medium facade crop",
]

# FLUX.2 text-to-image typically ignores SD-style negatives; kept for optional use.
NEGATIVE_PROMPT = (
    "view from on the balcony looking out, indoor terrace view, "
    "extreme close-up of railing only, railing fills the frame, cropped window, "
    "tight zoom on balusters, macro detail shot, "
    "solid closed wall between posts, no openings, blank panel, "
    "metal rails, thin steel rods, glass panel, wrought iron, "
    "cartoon, blurry"
)
