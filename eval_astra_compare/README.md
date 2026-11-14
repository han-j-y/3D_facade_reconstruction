# Layout comparison with Astra (paper, tab:main)

Facade-level layout F1 of our output and Astra on Astra's 46 test facades: windows, balconies and oriels on one
floor x bay grid per image, plus oriels as whole elements. Balconies and oriels are read from each method's 3D model,
not from the renderings.

## Files

- `hitmap.py` (Blender): casts rays through the front camera and records which object each ray hits.
- `layout_grid_eval.py`: grid-to-grid cell F1 (windows, balconies, oriels).
- `oriel_object_eval.py`: oriels as whole elements.
- `paths.py`: data locations (environment variables, see below).
- Helpers: `eval_astra.py`, `eval_astra_protocol.py`, `layout_balcony.py`, `layout_floorwise.py`,
  `layout_floorwise_viz.py`, `eval_testset.py`, `soft_structure.py`. These are copies of the versions in
  `datasets/Astra` and `datasets/testset/tools` with the paths moved to `paths.py`.

## Data (`paths.py`)

| Variable | Default (research folder layout) | Contents |
|---|---|---|
| `ASTRA_DIR` | `arch/datasets/Astra` | Astra runs, `light/mapping.json`, `light_labels*`, `tids46.json` |
| `TESTSET_DIR` | `arch/datasets/testset` | `labels/` (SAM3 window labels), `photos/` |
| `LABELS_DIR` | `TESTSET_DIR/test100_labels_20261110/test100` | balcony boxes, oriel cells, `window_grid/` |
| `HITMAP_DIR` | `ASTRA_DIR/layout_hitmaps` | `ours/<tid>.json`, `astra/<tid>.json` from `hitmap.py` |
| `OUT_DIR` | `ASTRA_DIR/layout_check` | scores (json) |
| `WINDOW_AST_DIR` | `arch/experiments/window_ast_predictor` | `window_ast` for `eval_testset.py` |

## Steps

1. Hit maps, one per facade and method:
   - Ours (the saved render scene and its camera; our facade fills about 63 % of the frame, so 480 cells across):
     `blender -b <run>/eval/run/<tid>/blender_real/facade_scene.blend -P hitmap.py -- $HITMAP_DIR/ours/<tid>.json 2048 2048 480 ours`
   - Astra (its .blend, front camera, at the size of its evaluated front render, 300 cells across):
     `blender -b <astra.blend> -P hitmap.py -- $HITMAP_DIR/astra/<tid>.json <W> <H> 300 astra`
2. `python layout_grid_eval.py [strict|medium|broad]`: Astra balcony name rule, `medium` (default) is the most
   favourable to Astra and is the one reported. Ours = objects in the `Balconies` collection.
   Defaults: `BAY_MATCH=m2m ROOF_FILTER=1 MIN_BAY_W=1 ORIEL_UPPER=1`.
3. `python oriel_object_eval.py`: Astra's oriels = parts more than 0.3 m in front of the wall plus objects named as
   oriels or projecting bays, upper floors only for both methods.

Result with `runs_test100_current` (2026-11-11 outputs): layout F1 windows 0.918 (ours) / 0.868 (Astra), balconies
0.808 / 0.849; oriels as elements 0.800 / 0.857. Numbers and protocol details in `agent.md`.
