# CMP X0005 — Blender facade model

Open **cmp_x0005_facade.blend** in Blender. The file contains editable architectural geometry, procedural materials, lighting, and three cameras. The supplied photograph is packed into the file under `REFERENCE · supplied cmp_x0005.jpg`.

## Included

- Ground floor and seven residential levels.
- Two rounded corner bays, each with six glazed tiers.
- Eight curved side balconies, four rectangular upper balconies, and two stone balustrade terraces.
- Geometric wrought-iron scrollwork, shutters, window frames, pale masonry outlines, and central carved relief.
- Arched entrance with iron gates, simplified shopfronts, roof terrace, and recessed penthouse.
- Named collections organized by architectural system.
- Front elevation, three-quarter view, and central-detail cameras.

## Files

- `cmp_x0005_facade.blend` — self-contained model.
- `facade_front.png` — rendered front elevation.
- `facade_three_quarter.png` — rendered view showing depth.
- `build_facade.py` and `facade_ornaments.py` — reproducible Blender Python source.
- `model_summary.json` — scene counts and estimated scale.

## Interpretation

Only the photograph supplied in the request was used as a reference. Dimensions are visual estimates: approximately 26 m wide and 27.2 m to the penthouse roof. Front faces negative Y; Z points upward; units are meters. The unobserved side and rear surfaces form a shallow, schematic shell. Interiors and hidden structure are not reconstructed. Fine carved motifs are geometric approximations of the visible ornament.

Street vehicles, pedestrians, lamp, wires, and temporary advertising are omitted to keep the facade visible. No other pre-existing computer files were inspected.

To rebuild, run Blender with `--background --factory-startup --python` followed by the absolute path to `build_facade.py`. The source expects the original photograph at the path stated near the end of that script; the saved `.blend` already contains its own packed copy.
