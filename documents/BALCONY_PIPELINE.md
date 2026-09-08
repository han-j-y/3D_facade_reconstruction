# Balcony pipeline — changed vs added

Window-only `python run.py` is unchanged unless you pass `--with-balconies`.

## Changed (existing files)

### `run.py`
Docstring only: documents `--with-balconies`.

### `run_pipeline.py`
- New flag `--with-balconies`.
- After writing `facade_dsl.json`, optionally calls `balcony_pipeline.run()`.
- Copies `facade_dsl_with_balconies.json` into the window `out-dir`.
- With `--blender-render`, uses the merged DSL when the balcony track succeeded.
- Window stages (detect, unitize, cluster, structure vote, `build_facade_dsl`) are not modified.

### Compiler (`vendor/window_compiler/`)
- `facade_spec.normalize_facade_spec` maps `balcony_types` + `layout.balconies` onto `balconies` / `balcony_placement` (floor×bay span from the window grid).
- `balcony_compile.py` matches the balcony catalog: plan shapes, baluster / 200 mm solid / 10 mm glass, enclosed glass box.
- `facade_compile.compile_facade_scene` calls that pass (window-only JSON still compiles as before).

### `documents/FDSL.ebnf`
- `balconies:` type library (each body is BDSL).
- `balcony_placement:` spans on the existing floor×bay grid (e.g. `F1 B1-B3 loggia`).
- Window `placement:` must not list balcony type names.

### `balcony_pipeline/run_balcony.py`
- `run()` entry for the window hook.
- Unitize uses `merge_adjacent_boxes(..., mode="balcony")` (IoU/containment only; no gap merge).
- ASCII `->` in console output (Windows cp1252).

### `scripts/overlay_facade_merge_boxes.py`
- `MERGE_PROFILES["window"|"balcony"]` and `mode=` on `merge_adjacent_boxes`.
- Window callers keep default `mode="window"` (unchanged).

## Added

### Language
- `documents/BDSL.ebnf` — one balcony type (sibling of WDSL). Catalog axes:
  structure, floor shape, railing (baluster / solid / glass); enclosure is
  authoring-only (`open`/`enclosed`). Recovery always uses `open`.
  Wall opening is the window FDSL grid.
- `vendor/window_compiler/examples/example_balcony_projecting.bdsl`
- `vendor/window_compiler/examples/example_balcony_enclosed.bdsl` (glass box; no nested WDSL mesh)
- `vendor/window_compiler/examples/example_facade_with_balcony.fdsl`

### Code (`balcony_pipeline/`)
Standalone track; does not edit window e2e internals.

| File | Role |
|------|------|
| `run_balcony.py` | CLI + `run()` |
| `snap.py` | Snap boxes to **window** floors/bays |
| `cluster.py` | DINOv2 ROI + spectral + Potts (balcony boxes only) |
| `filter.py` | Drop decoration FPs (below shrunk sill; exempt if >=20% wider than partner window), Juliet-like same-width partners (0.85–1.08×), window-inclusive boxes (partner window covers >=90% of balcony area), or top rail (no window above in the balcony's bay span) |
| `heuristic_ir.py` | Per-crop BDSL JSON IR + `balcony_view` fingerprint |
| `../balcony_train/` | Optional solid/baluster classifier (`railing.kind` only) |
| `recovery_profile.py` | Which axes to infer/vote (`railing_only` / `full`) |
| `vote.py` | Majority vote (same algorithm as windows, balcony key) |
| `merge_dsl.py` | Copy window DSL; append `balcony_types` + `layout.balconies` |
| `draw.py` | Stage overlays |

Same **methods** as windows: SAM3 detect, box merge, DINO cluster, cosine medoid, majority vote. Floor/bay are **not** re-clustered; they come from the window DSL (`layout.floors` / `layout.bays`, plus `meta.columns_xy` / `meta.floors_y` from the same structural column layout as main; `--layout-mode centroid` for the old assign_bays grid). Multi-bay balconies store `bay_start`–`bay_end` plus `bays[]` (15% width overlap for span). Instance **width** stays from the photo box (`width_norm`). Horizontal center (default **`--balcony-center window`**) uses the same **2D overlap Pair** as decoration filters (`filter._partner_windows`: window box expanded 10%, overlap with balcony); nearest row above the slab; mean box center → `window_cx_norm` (with `partner_window_unit_ids`). Fallback when no partner: bay-band mean. **`--balcony-center bay`**: `bays_center[]` → `bay_cx_norm`. **`--balcony-center photo`**: detection box `cx_norm`. Wall door vs window is the window `placement` token on those cells, not a BDSL `opening` field.

Vote fingerprint (`balcony_view`): controlled by `--recovery-profile`
(default **`railing_only`** → vote `railing.kind` only: **`baluster` | `solid`**).
`glass` is never inferred; legacy `glass` fingerprints map to `solid`.
`metal` ≡ `baluster`. Disabled axes use fixed defaults (`projecting` / `open` /
`rectangle` / supports 0). Use `--recovery-profile full` to restore other axes.

`railing.kind` uses `checkpoints/railing_best.pt` when present (frozen DINOv2 +
linear head; see `balcony_train/README.md`). Otherwise the opaque-run heuristic
in `heuristic_ir.py`. Other IR axes stay heuristic / fixed defaults. Force the
heuristic with `--no-railing-ckpt`. Meshes compile in `balcony_compile.py`
(catalog rules: baluster rods, 200 mm solid parapet, 10 mm glass, enclosed 1 m
mullions + matching ceiling slab).

## How to run

Window only (unchanged):

```text
python run.py --image PHOTO.png --out-dir runs/demo --device cuda
```

Window then balcony:

```text
python run.py --image PHOTO.png --out-dir runs/demo --device cuda --with-balconies --blender-render
```

Balcony track only (needs an existing window `facade_dsl.json`):

```text
python balcony_pipeline/run_balcony.py --image PHOTO.png --windows-dsl path/to/facade_dsl.json --out-dir runs/balcony_demo --device cuda
```

Railing classifier (after labeling crops; checkpoint is shared):

```text
python balcony_train/collect.py
python balcony_train/train.py --device cuda
python run.py --image PHOTO.png --out-dir runs/demo --device cuda --with-balconies
```

Use `pipeline_preview/.venv` if the system Python has no torch.

## Test (2026-08-19)

On `pipeline_preview/input_cmp_b0168.png` + `facade_dsl_input_cmp_b0168.json` (CUDA, preview venv):

- SAM3: 8 raw → 6 after size filter → 4 units
- 2 types; votes 2/2 and 2/2
- Wrote `runs/balcony_demo/facade_dsl_with_balconies.json` (`layout.balconies`, `balcony_types`)
