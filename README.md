# Facade E2E Recovery (standalone)

Independent package: **photo → detect → cluster → structure vote → façade DSL**.

Blender rendering is **optional**.

```
facade_e2e/
  run.py                 # entry point
  run_pipeline.py        # full pipeline
  facade_recovery/       # path helpers, column layout, doors, symmetry-gated clustering, per-floor bays
  window_ast/            # vendored structure model + symmetry.py
  scripts/               # clustering helpers, symmetry axis, templates, rendering, diagnostics
  scripts/drivers/       # batch drivers used for the experiments
  vendor/window_compiler/# Blender main.py + compile/render (optional)
  documents/             # WDSL.ebnf (window grammar), FDSL.ebnf (façade grammar)
  checkpoints/           # bundled structure checkpoint (Git LFS) + vocabulary
  data/vocab_structure.json
```

## Setup

```bash
cd facade_e2e
pip install -r requirements.txt
git lfs pull                    # fetch checkpoints/structure_best.pt
```

The default pane-tree model and its inference code are included in this repository.
Without the Git LFS checkpoint, use `--skip-structure --cluster-mode box`.

## Run (core — no Blender)

```bash
python run.py --image /path/to/facade.png --out-dir runs/demo --device cuda
python run.py --facade-id 8 --train-up data/facades/train_up --device cuda
```

The defaults are the current pipeline:

| Stage | Default |
|------|---------|
| Detect | `facebook/sam3` prompt `window`, two-tier: keep score ≥ `--anchor-score` 0.7, add boxes in [`--threshold` 0.5, 0.7) that overlap no kept box by `--fill-max-overlap` 0.2 of the smaller box; `--min-side 13`; boxes on `car` / `truck` / `van` masks dropped (`--exclude-prompts`) |
| Unitize | `--merge-mode ast`, `--col-tol 0.045`, `--row-tol 0.055` |
| Cluster | `--cluster-mode gate` (below) |
| Doors | `facebook/sam3`, prompt `door` ≥ `--door-threshold` 0.45; override with any Hugging Face ID or model directory |
| Side facades | `--side-filter`: edge window columns that sit off the main facade's floors are dropped (neighbouring buildings, side walls), plus doors outside the kept windows. An outermost column with one window on each of ≥ 3 main floors is kept (corner towers). The main floors start from the windows and doors in the middle half of the image and are then rebuilt from the kept windows until nothing changes, so a row the middle misses counts once part of it is kept. |
| Local symmetry | `--local-sym` (gate mode, below) |
| Layout | `--grid floor_bays` (below) |

**Gate clustering** (`facade_recovery/sym_gate.py`):

1. Box GMM on z-scored (w, h, aspect) gives the box groups.
2. The vertical symmetry axis comes from `scripts/symmetry_axis_ghost_continuous.py` (`run_case`). It runs on the units and the bay columns. `open balcony` detections (`--balcony-prompt`, ≥ `--balcony-min-score` 0.5) break ties between window peaks, and the axis is then shifted within ±`--axis-refine` 0.02 × width to pair the most windows. The plots go to `<out>/axis/`.
3. Windows are paired with their mirror image about the axis (IoU ≥ 0.3).
4. Inside each box group, the pane-tree descriptor (`--gate-feature heads`) is cut top-down on its average-linkage tree (cosine, `--struct-split-thr` 0.16). A split is refused if it separates more than `--gate-eps` 0.1 of the node's mirror pairs.

The axis is used during clustering, so the separate symmetry repair (below) is not needed.

**Local symmetry** (`facade_recovery/local_sym.py`, `--local-sym`, on by default with gate) runs after the side filter. It fixes single windows that the one global axis leaves in the wrong type, e.g. an occluded or badly segmented copy of a repeated window.

1. Candidate axes are the midpoints of similar window pairs (same floor band, size within 1.5×) and the window centres. Each axis matches every window's reflection to its best window (IoU ≥ 0.3). Its extent grows outward while ≥ 75% of the windows inside are matched. A pair whose own axis was merged into a stronger nearby axis, and which no longer matches there, keeps its own one-pair axis.
2. Weights are continuous: inside each axis's extent every window pairs with its best other window by reflection IoU (no 0.3 cutoff), and the axis strength is the overlap curve of Zhang et al. at the axis over those windows. A mirror pair's weight is the summed strength × IoU of the axes pairing it (relative to the strongest axis).
3. An ICM vote on a Potts energy relabels windows: cost(window, type) = 4 × mean cosine distance of its pane-tree descriptor to the type's other members + the weights of its mirror pairs that end up in another type. A window can only move into a mirror partner's type (no new types). If its box size repeats on the facade, that type must also hold a window of the same size (w and h within 12%). A one-off box can join any partner type, unless it is self-symmetric about a strong axis (≥ 0.5 × the strongest).
4. Occlusion: SAM3 masks for `--local-sym-occluders` (vehicles, fire escapes, trees, street lamps, traffic lights, people, balconies; ≥ `--local-sym-occluder-threshold` 0.4) and the image border hide window sides. An occluder can only shrink a box, so a side stays hidden only where the box is smaller than every unoccluded window in its column (horizontal overlap IoU ≥ 0.5): left / right edge inside all of theirs, or height below all of theirs for top / bottom. The pair IoU ignores hidden sides. A hidden side makes that dimension a minimum: the window may match a longer one, and the 12% size check rejects a candidate only when this box is already the longer one. Windows lying on an occluder sit out the vote.

The moves go to `meta.local_sym_moves` and `local_sym.json` (axes, pairs, vote costs). `--no-local-sym` gives plain gate clustering.

**Per-floor bays** (`facade_recovery/floor_bays.py`): every floor gets its own bays, one per window or door, split at gap midpoints. Lines that match across floors become shared lines, and a floor's empty stretches inherit lines from the nearest floor with a similar window height. The DSL columns are the union of all floors' lines, and every floor bay spans its columns (`placement_spans`). Floors and bays carry `h_m` / `w_m` so the compiler keeps them as they are. `meta.floor_bays` lists the per-floor bays. `--grid columns` keeps the older single column grid (`--layout-mode`).

Older settings remain available, e.g. CMP XML boxes with box-only clustering:

```bash
python run.py --image data/facades/base/cmp_b0250.jpg --windows xml \
  --anchor-score 0 --cluster-mode box --no-side-filter --grid columns \
  --out-dir runs/e2e_ast_cluster_cmp_b0250 --device cuda --blender-render
```

`--cluster-mode shape_then_feat` (box GMM + DINO splits), `shape_then_struct`, `struct`, `box` and `appearance` skip the axis. With them, **L–R symmetry repair** runs by default (`--symmetry` / `--no-symmetry`).

**Windows:** `--windows sam` (default) uses SAM3 instance masks as tight boxes; `--windows xml` loads CMP XML next to the image (it also annotates neighbouring buildings).

Outputs under `runs/facade_e2e_<id>/` (or `--out-dir`):

| File | Description |
|------|-------------|
| `index_raw.json` | Window detections used (after vehicle / two-tier filtering) |
| `raw_windows.png` | Raw SAM3/XML boxes before merge |
| `axis/` | Symmetry-axis score curve, overlays, `summary.json` (gate mode) |
| `local_sym.json` | Local-symmetry axes, mirror pairs and type moves (gate mode) |
| `overview.png` | Units colored by type |
| `facade_dsl.json` | Per-floor bay layout + voted structure IR (`meta.axis_x`, `meta.local_sym_moves`, `meta.side_dropped`, `meta.floor_bays`) |
| `summary.json` | Counts + vote stats |
| `assets/types/type_XX/` | Exemplar crop + `structure_ir.json` |
| `blender/compare_photo_vs_render.png` | Photo vs Blender (with `--blender-render`) |

## Optional Blender render

Needs a Blender binary on `PATH` (or `$BLENDER`). The repo already ships
`vendor/window_compiler/main.py`; override with `FACADE_COMPILER_ROOT` only if needed.

```bash
export BLENDER=/path/to/blender          # or have `blender` on PATH
python run.py --facade-id 8 --blender-render --device cuda
```

If Blender is missing, the pipeline still writes `facade_dsl.json` and prints a warning.

Every window of a type is rendered at the type's median size (centres kept) and snapped to the type's median sill on each floor. Set `meta.unify_type_size_across_spans` to `false` to unify single-bay and multi-bay placements of a type separately.

## Standalone symmetry axis + full-stack repair (older runs)

For runs made without `--cluster-mode gate`. Both read stage-1 `e2e_sam3_cluster_<stem>/` dirs read-only and
write elsewhere.

```bash
# continuous vertical symmetry axis (crop-aware ghost partners)
python scripts/symmetry_axis_ghost_continuous.py --sam3-root runs \
  --out-dir runs/symmetry_axis_sweep/ghost_continuous --stems cmp_b0226

# symmetry repair about that axis → facade_dsl_repaired.json (+ Blender before/after)
python scripts/render_sym_repair_fullstack.py --sam3-root runs \
  --axis-root runs/symmetry_axis_sweep/ghost_continuous \
  --out-dir runs/symmetry_from_sam3_fullstack --stems cmp_b0226 --device cuda
# add --skip-render to stop at the repaired DSL
```

`scripts/run_symmetry_from_sam3_e2e.py` is the same repair without Blender (overlays only).

## Floor / bay templates

Extract a floor×bay template from a (repaired) DSL, then instantiate new sizes.
Metric floor heights and bay widths are preserved; added floors/bays repeat the
typical floor / repeating bay unit.

```bash
python scripts/facade_template.py extract \
  --dsl runs/symmetry_from_sam3_fullstack/cmp_b0226/facade_dsl_repaired.json \
  --out runs/facade_templates/cmp_b0226.template.json

# shortcuts
python scripts/facade_template.py instantiate --template runs/facade_templates/cmp_b0226.template.json \
  --add-floors 2 --add-bays 2 --out runs/facade_templates/cmp_b0226_5x9/facade_dsl.json

# explicit patterns: X*k and (A B)*k
python scripts/facade_template.py instantiate --template runs/facade_templates/cmp_b0226.template.json \
  --floors "top typical*3 ground" --bays "E (B D B)*2 E" --out runs/facade_templates/custom/facade_dsl.json
```

Render an instantiated DSL with `scripts/render_facade.py` (no photo panels, since it has no `meta.image`).

## Turntable video

```bash
python scripts/render_fullstack_turntable_video.py --run-dir runs/symmetry_from_sam3_fullstack \
  --stems cmp_b0226 --frames 96 --fps 24
```

Uses `scripts/blender_facade_turntable.py` inside Blender (`$BLENDER` or `blender` on PATH).

## Grammars

`documents/WDSL.ebnf` (window JSON IR, text syntax, structure tokens, façade openings) and
`documents/FDSL.ebnf` (compiled `window_compiler_facade_v1`, recovery DSL, normalize mapping,
template/pattern language) describe the formats the code above reads and writes.

## Majority vote

Within each window type, every unit crop is predicted; votes are tallied on a discrete `structure_view` fingerprint (whole IR key, not field-by-field). Ties prefer the type medoid.

## Requirements

**Required:** torch, torchvision, numpy, Pillow, scikit-learn, scipy, transformers (SAM3)

**Optional:** Blender ≥ 3.x (uses vendored `vendor/window_compiler`)

**Downloaded at runtime:** HuggingFace `facebook/sam3` (windows, balconies, `--exclude-prompts`), `torch.hub` DINOv2

All default code and checkpoint paths are repository-relative. Input images are
user-provided via `--image`, `--train-up`, or `DATA_DIR` in the batch drivers.
