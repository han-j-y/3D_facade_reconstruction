"""Prompts for synthetic open_work + masonry balcony railing crops (FLUX.2).

Hard constraints (every prompt):
- viewpoint: street-side photo of the facade / balcony (never from on the balcony)
- openings: gaps must exist between posts (balusters, pierced panel, grille, etc.)

Flexible:
- material: stone, concrete block, or concrete
- infill style between posts: balusters OR precast openwork panel (either OK)
"""

from __future__ import annotations

# Indices 0–3 map to flux2_masonry_000..003 when unlabeled starts empty.
# Keep these four especially strict: street-side only + must have openings.
MASONRY_OPENWORK_PROMPTS: list[str] = [
    # 000
    "photo taken from the street looking up at a balcony railing, concrete masonry posts, open gaps between posts through which the facade is visible, frontal facade crop, realistic",
    # 001
    "street-side facade crop of a stone balcony, thick posts with a pierced precast concrete panel between them, many see-through openings, not a solid wall, daylight",
    # 002
    "from the sidewalk facing the building, masonry balcony with square concrete posts and spaced balusters, clear open voids between balusters, weathered concrete, frontal",
    # 003
    "street-level exterior photo of openwork concrete balcony railing, pier posts at sides, decorative grille with holes between posts, openings required, not viewed from the terrace",
    # 004+
    "apartment facade crop from the street, limestone posts with spaced concrete balusters and visible openings between them",
    "street-side photo of thick stone posts with a perforated precast concrete screen between them, daylight openings",
    "frontal street crop of balcony railing, concrete block posts and openwork concrete panel with see-through openings",
    "street view of ochre building wall, white concrete balcony posts and short masonry balusters with open gaps",
    "facade balcony unit crop from the street, heavy concrete posts, geometric precast grille with openings between posts",
    "street-level photo of stone balcony parapet, masonry piers and open baluster or pierced-panel infill with gaps",
    "sidewalk street-side view of concrete balcony railing, chunky posts, weathered masonry, openings between posts",
    "frontal street photo of masonry openwork balcony, corner posts, mid-span balusters or open precast panel with holes",
]

# FLUX.2 text-to-image typically ignores SD-style negatives; kept for optional use.
NEGATIVE_PROMPT = (
    "view from on the balcony looking out, indoor terrace view, "
    "solid closed wall between posts, no openings, blank panel, "
    "metal rails, thin steel rods, glass panel, wrought iron, "
    "macro close-up of holes only, cartoon, blurry"
)
