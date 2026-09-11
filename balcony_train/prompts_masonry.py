"""Prompts for synthetic open_work + masonry balcony railing crops (FLUX.2)."""

from __future__ import annotations

# Keep prompts short and photo-like so crops resemble real facade detector crops.
MASONRY_OPENWORK_PROMPTS: list[str] = [
    "photo of a residential balcony with brick openwork railing, frontal crop",
    "close-up photo of masonry perforated balcony balustrade, daylight",
    "facade balcony crop, red brick lattice railing, concrete slab edge",
    "street-view photo of apartment balcony with brick pier and openwork rail",
    "photo of gray concrete block openwork balcony railing, front elevation",
    "building facade crop, terracotta brick balcony grille, outdoor daylight",
    "photo of white painted masonry balusters on a balcony, frontal",
    "residential balcony with thick masonry posts and open brick pattern",
    "close crop of perforated brick balcony parapet, realistic photo",
    "photo of stone openwork balcony railing on an apartment facade",
    "daylight photo of brick honeycomb balcony railing, centered crop",
    "facade photo: balcony with chunky masonry rails and open gaps",
]

# FLUX.2 text-to-image typically ignores SD-style negatives; kept for optional use.
NEGATIVE_PROMPT = "metal rails, thin steel rods, glass panel, cartoon, blurry"
