"""Prompts for synthetic open_work + masonry balcony railing crops (FLUX.2).

Target look (matches real crops used for labeling):
- material: stone, concrete block, or concrete (not metal/glass)
- viewpoint: street-side facade crop
- structure: clear masonry posts/piers
- between posts: balusters or precast concrete panels / openwork (either OK)
"""

from __future__ import annotations

# Keep prompts short and photo-like so crops resemble real facade detector crops.
MASONRY_OPENWORK_PROMPTS: list[str] = [
    "street-level photo of a concrete balcony railing with thick masonry posts, frontal facade crop",
    "street view balcony crop, stone posts and precast concrete openwork panel between posts",
    "facade photo from the street, weathered concrete pier posts with decorative pierced panel railing",
    "low-angle street photo of masonry balcony balustrade, square concrete posts and solid top rail",
    "apartment facade crop from sidewalk, limestone posts with vertical concrete balusters between them",
    "street-side photo of precast concrete balcony screen between thick stone posts, daylight",
    "frontal crop of a balcony railing made of concrete block posts and openwork concrete panel",
    "street view of ochre building wall with white concrete balcony posts and short masonry balusters",
    "facade balcony unit crop, heavy concrete posts, geometric precast grille between posts",
    "street-level architectural photo, stone balcony parapet with piers and baluster or panel infill",
    "sidewalk view of concrete balcony railing, chunky posts, weathered masonry texture, realistic photo",
    "frontal street photo of masonry openwork balcony, posts at corners, precast panel or balusters mid-span",
]

# FLUX.2 text-to-image typically ignores SD-style negatives; kept for optional use.
NEGATIVE_PROMPT = (
    "metal rails, thin steel rods, glass panel, wrought iron, "
    "macro close-up of holes only, honeycomb brick lattice, cartoon, blurry"
)
