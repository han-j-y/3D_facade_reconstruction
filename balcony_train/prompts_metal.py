"""Prompts for synthetic open_work + metal balcony railing crops (FLUX.2).

Target look:
- repeated physical openings between members; background visible through the railing
- thin posts, rods, ornamental ironwork / open metal patterns
- styles: vertical bars, geometric openwork, scrollwork, filigree, lattice

Shared photo constraints (aligned with other synth sets):
- street-side viewpoint
- medium facade crop with full window visible
- upper-story balcony (2nd floor or higher)
"""

from __future__ import annotations

METAL_OPENWORK_PROMPTS: list[str] = [
    "street-side medium shot of an upper-story balcony with thin dark metal vertical bar railing, repeated open gaps showing the facade behind, full window visible",
    "street view facade crop, 2nd-floor balcony with black wrought-iron vertical rods and open spacing, background visible through the railing, entire window behind, medium framing",
    "from the sidewalk, third-floor balcony with geometric dark metal openwork, diamond motifs between vertical bars, clear openings, whole window in frame",
    "street-level exterior of an upper-floor balcony, light grey ornamental metal railing with X shapes and circular open patterns, see-through panels, full window visible",
    "apartment facade from the street, 2nd-floor long run of simple dark metal vertical pickets with open gaps, complete window visible, medium distance",
    "street-side photo of a curved upper-story balcony with ornate black wrought-iron scrollwork, open ornamental iron panels, balcony plus full window unit",
    "frontal street crop of modern dark metal baluster rods with open spacing on a 2nd-floor balcony, no solid panels, full window visible, medium shot",
    "street view of building facade, gold-colored ornate metal scrollwork railing on an upper balcony over stone base, open filigree patterns, entire window shown",
    "facade balcony unit from the street, dense dark metal lattice openwork railing with many small openings, whole window and lintel visible, pulled back",
    "street-level photo of a 3rd-floor balcony with white filigree metal openwork railing, lace-like open patterns, background visible through it, full window in frame",
    "sidewalk street-side view of upper-story balcony, thin painted metal posts and rods with repeated openings, complete window behind, medium crop",
    "frontal street photo of decorative wrought-iron openwork balcony on the 2nd floor or higher, ornamental open panels not solid, full window visible",
    "street-side medium shot of black metal geometric balcony railing, repeating open rectangles, facade seen through gaps, upper story, full window visible",
    "from the sidewalk facing the building, classic dark green wrought-iron balcony with scroll openwork, openings between members, whole window in frame",
    "street view facade crop, slim steel vertical balusters on an upper-floor balcony, airy open railing, entire window behind, medium architectural framing",
    "street-level exterior of white ornamental iron balcony railing with open floral metal patterns, see-through, 2nd floor or higher, full window visible",
]

NEGATIVE_PROMPT = (
    "solid concrete parapet, opaque waist wall, frosted glass panels, "
    "metal plate privacy screen without openings, "
    "masonry stone balusters only, brick openwork, continuous closed wall, "
    "ground-floor porch, view from on the balcony looking out, "
    "extreme close-up of railing only, cropped window, cartoon, blurry"
)
