"""Prompts for synthetic open_work + masonry balcony railing crops (FLUX.2).

Hard constraints (every prompt):
- viewpoint: street-side photo of the facade / balcony (never from on the balcony)
- openings: gaps must exist between posts (balusters, pierced panel, grille, etc.)
- framing: medium facade crop — whole window behind the balcony visible (not railing-only close-up)
- level: balcony on the 2nd floor or higher (upper-story), not a ground-floor porch

Flexible:
- material: stone, concrete block, or concrete
- slab / floor plan shape: rectangle, slight ellipse, semicircle, ellipse,
  triangle, hexagon, or trapezoid
- infill: square/rectangular masonry balusters OR classical balusters matching
  the building cornice / large brackets OR precast openwork panel
"""

from __future__ import annotations

# Mix slab shapes + baluster styles; every line keeps street-side, openings, medium frame, 2F+.
MASONRY_OPENWORK_PROMPTS: list[str] = [
    # rectangular slab + square balusters
    "street-side medium shot of an upper-story 2nd-floor balcony, rectangular concrete slab, square masonry balusters with open gaps, full window visible, not a close-up",
    # slight ellipse + classical balusters matching cornice
    "street view facade crop, slightly elliptical stone balcony on the third floor, classical balusters matching the building cornice and large brackets, openings between posts, entire window behind",
    # semicircular + square posts/balusters
    "from the sidewalk, semicircular concrete balcony above the ground floor, square posts and rectangular balusters with clear open gaps, whole window in frame, medium facade photo",
    # elliptical + classical same family as brackets
    "street-level exterior of an elliptical upper-floor limestone balcony, ornate classical balusters same style as cornice and corbels, open voids between balusters, full window visible",
    # triangular plan + openwork panel
    "apartment facade from the street, triangular-plan concrete balcony on the 2nd floor or higher, thick posts with pierced precast panel openings, complete window visible, medium distance",
    # hexagonal + square balusters
    "street-side photo of a hexagonal stone balcony slab on an upper story, square masonry balusters with open gaps between them, balcony plus full window unit, slightly wider crop",
    # trapezoidal + classical detailing
    "frontal street crop of a trapezoidal upper-story balcony, concrete posts, classical balusters matching large facade brackets and cornice moldings, see-through openings, medium shot",
    # rectangular + classical
    "street view of ochre wall, rectangular 2nd-floor white concrete balcony, classical masonry balusters matching cornice details with open gaps, entire window shown, medium framing",
    # semicircle + precast grille
    "facade balcony unit from the street, semicircular upper-floor concrete balcony, heavy posts and geometric precast grille openings, whole window and lintel visible, pulled back",
    # slight ellipse + square balusters
    "street-level photo of a slightly oval stone balcony on the third floor, square posts and rectangular balusters with open gaps, full window in frame, medium architectural crop",
    # hexagon + classical
    "sidewalk street-side view of a hexagonal upper-story masonry balcony, classical balusters same system as cornice and brackets, openings between posts, complete window behind",
    # trapezoid + square balusters
    "frontal street photo of a trapezoid-plan concrete balcony above ground floor, corner posts, mid-span square balusters with open gaps, full window visible, medium facade crop",
    # elliptical + pierced panel
    "street-side medium shot of an elliptical 2nd-floor stone balcony, masonry piers with perforated concrete openwork panel openings, full window visible above the railing",
    # triangular + classical balusters
    "from the sidewalk facing the building, triangular upper-story balcony, classical stone balusters matching ornate brackets and cornice, open baluster gaps, whole window in frame",
    # rectangular + matching classical bracket family
    "street view facade crop, rectangular upper-floor limestone balcony, balusters and posts in the same classical order as the building cornice and oversized brackets, openings, entire window behind",
    # semicircle + square section balusters
    "street-level exterior photo of a half-round concrete balcony on the 2nd floor or higher, chunky square balusters with open gaps, full window and some wall visible, not zoomed into the rail only",
]

# FLUX.2 text-to-image typically ignores SD-style negatives; kept for optional use.
NEGATIVE_PROMPT = (
    "ground-floor porch, entrance stoop only, first-floor patio at street level, "
    "view from on the balcony looking out, indoor terrace view, "
    "extreme close-up of railing only, railing fills the frame, cropped window, "
    "tight zoom on balusters, macro detail shot, "
    "solid closed wall between posts, no openings, blank panel, "
    "metal rails, thin steel rods, glass panel, wrought iron, "
    "cartoon, blurry"
)
