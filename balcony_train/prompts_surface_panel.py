"""Prompts for synthetic surface_panel balcony railing crops (FLUX.2).

Target look:
- filled with continuous panel material (not balusters / openwork)
- glass, frosted glass, metal plate, or privacy screen
- frame + panel construction
- modern, flat appearance
- few openings; continuous panel feel

Shared photo constraints (aligned with other synth sets):
- street-side viewpoint
- medium facade crop with full window visible
- upper-story balcony (2nd floor or higher)
"""

from __future__ import annotations

SURFACE_PANEL_PROMPTS: list[str] = [
    "street-side medium shot of an upper-story balcony with modern flat glass panels in a thin metal frame, continuous panel surface not balusters, full window visible",
    "street view facade crop, 2nd-floor balcony privacy screen of frosted glass panels in a dark frame, seamless panel look, entire window behind, medium framing",
    "from the sidewalk, third-floor balcony with flat metal plate panels in a slim frame, modern continuous surface railing, whole window in frame, not openwork",
    "street-level exterior of an upper-floor balcony, frosted privacy-glass panels in aluminum frames, flat modern look with almost no openings, full window visible",
    "apartment facade from the street, 2nd-floor balcony with tinted glass surface panels in a thin black frame, continuous panel band, complete window visible, medium distance",
    "street-side photo of an upper-story balcony filled with flat sheet-metal privacy panels in a frame, modern flush look, balcony plus full window unit",
    "frontal street crop of a modern balcony, continuous frosted glass panel railing in a metal frame, no balusters, full window visible, medium shot",
    "street view of ochre wall, white-framed frosted glass balcony panels on the 2nd floor, flat surface-panel railing, entire window shown, medium framing",
    "facade balcony unit from the street, upper floor, dark metal frame with opaque privacy screen panels, continuous modern surface, whole window and lintel visible",
    "street-level photo of a 3rd-floor balcony with clear glass panels in a slim frame, flat modern surface panel railing, full window in frame, medium architectural crop",
    "sidewalk street-side view of upper-story balcony, brushed metal plate panels in a rectangular frame, continuous panel feel, complete window behind",
    "frontal street photo of a modern balcony on the 2nd floor or higher, frosted glass privacy panels in frame, no open balusters, full window visible, medium facade crop",
]

NEGATIVE_PROMPT = (
    "openwork balusters, vertical baluster posts, pierced masonry grille, "
    "decorative iron scrolls, many open gaps between rails, "
    "solid concrete parapet wall without frame, thick masonry mass wall, "
    "ground-floor porch, view from on the balcony looking out, "
    "extreme close-up of railing only, cropped window, cartoon, blurry"
)
