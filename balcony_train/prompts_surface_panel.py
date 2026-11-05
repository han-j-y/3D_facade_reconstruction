"""50 surface_panel FLUX.2 prompts (wide balcony-band crops).

Constraints baked into every prompt:
- street-side, floors 2–10, frame + continuous panel (not openwork)
- wall near olive-grey RGB 103,104,99
- panel colors: white, grey, glass, dark red, light blue
- panel height:width from about 1:2 to 2:1
- about one in three prompts is a continuous panel with very weak seams
- some prompts add plant or pedestrian occlusion

After generation, match real crops with ``resize_flux_images.py`` (120×44).
"""

from __future__ import annotations

FLOORS = (
    "2nd-floor",
    "3rd-floor",
    "4th-floor",
    "5th-floor",
    "6th-floor",
    "7th-floor",
    "8th-floor",
    "9th-floor",
    "10th-floor",
)

VIEWS = (
    "street-side wide crop of a {floor} balcony, looking slightly up",
    "from the sidewalk looking up at a {floor} balcony, strong perspective, wide short crop",
    "frontal street photo of a {floor} balcony railing band",
    "low-angle sidewalk view of a {floor} balcony, foreshortened facade",
    "across-the-street view looking up at a {floor} balcony, medium-wide crop",
)

# height:width in [1:2, 2:1]
PANEL_SHAPES = (
    "one wide frosted panel about twice as wide as tall (height:width near 1:2) in a thin metal frame",
    "white opaque privacy panels, each nearly twice as wide as tall, continuous frame",
    "grey metal plate panels with height:width near 1:1, slim dark frame, no gaps",
    "clear glass panels in a black frame, slightly wider than tall",
    "light blue tinted glass panels, height:width near 2:1 (taller than wide), thin aluminum frame",
    "dark red solid privacy panels, moderately wide, height:width between 1:2 and 1:1, metal frame",
    "white framed glass panels, almost square to slightly wide, continuous surface",
    "grey frosted panels, wide band, height much smaller than width, no balusters",
    "light blue metal privacy plates, height:width near 1:2, flush modern frame",
    "dark red glass-look panels, height:width near 2:1, narrow vertical modules in one frame",
)

# About 1/3 of prompts: continuous panel, joints almost invisible (still a panel, not a parapet).
WEAK_SEAM_SHAPES = (
    "weak seams, one nearly seamless frosted panel band, only faint hairline joints, very thin frame, height:width near 1:2, continuous surface panel not a solid parapet",
    "weak seams, white continuous privacy panel with barely visible seams, low flush edge, almost no mullions, wide panel",
    "weak seams, grey glass panel reading as one smooth sheet, joints easy to miss, slim frame, height:width near 1:1",
    "weak seams, light blue tinted continuous panel, faint seams, minimal frame, slightly taller modules blended into one surface",
    "weak seams, dark red continuous privacy panel, joints nearly invisible, thin metal lip only, not a masonry wall",
    "weak seams, clear glass as one wide continuous panel, hairline joints, almost frameless look, height much smaller than width",
)

OCCLUSIONS = (
    "",
    "",
    " a leafy plant partly overlaps the left of the railing",
    " a pedestrian in the foreground partly blocks the lower panel",
    " small tree branches cross the foreground in front of the panel",
    " a potted shrub on the balcony edge occludes part of the panel",
)

LIGHTS = (
    "overcast daylight",
    "soft afternoon light",
    "diffuse cloudy light",
    "bright daylight with soft shadows",
    "morning light",
)

TAIL = (
    "olive-grey plaster wall close to RGB 103,104,99 behind the balcony, "
    "continuous surface panel railing not openwork, no balusters, "
    "full window still partly visible, street exterior only, not a close-up of the rail alone"
)


def _build_prompts(n: int = 50) -> list[str]:
    prompts: list[str] = []
    for i in range(n):
        floor = FLOORS[i % len(FLOORS)]
        view = VIEWS[i % len(VIEWS)].format(floor=floor)
        # Every third prompt: weak seams (indices 0, 3, 6, ... → 17/50).
        if i % 3 == 0:
            shape = WEAK_SEAM_SHAPES[(i // 3) % len(WEAK_SEAM_SHAPES)]
        else:
            shape = PANEL_SHAPES[i % len(PANEL_SHAPES)]
        occ = OCCLUSIONS[i % len(OCCLUSIONS)]
        light = LIGHTS[i % len(LIGHTS)]
        prompts.append(f"{view}, {shape},{occ}, {TAIL}, {light}")
    return prompts


SURFACE_PANEL_PROMPTS: list[str] = _build_prompts(50)

NEGATIVE_PROMPT = (
    "openwork balusters, vertical baluster posts, pierced masonry grille, "
    "decorative iron scrolls, many open gaps between rails, "
    "solid concrete parapet wall without frame, thick masonry mass wall, "
    "ground-floor porch, view from on the balcony looking out, "
    "extreme close-up of railing only, cropped window, cartoon, blurry"
)
