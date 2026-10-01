# Facade E2E Recovery (standalone)

Independent package: **photo → detect → cluster → structure vote → façade DSL**.

Blender rendering is **optional**.

```
facade_e2e/
  run.py                 # entry point
  run_pipeline.py        # full pipeline
  facade_recovery/       # path helpers, column layout, merge guard
  window_ast/            # vendored structure model + symmetry.py
  scripts/               # clustering helpers, stage 2/3, templates, rendering
  vendor/window_compiler/# Blender main.py + compile/render (optional)
  documents/             # WDSL.ebnf (window grammar), FDSL.ebnf (façade grammar)
  checkpoints/           # structure_best.pt (symlink or copy)
  data/vocab_structure.json
```

## Setup

```bash
cd facade_e2e
pip install -r requirements.txt
# copy structure weights into checkpoints/ (real files, not symlinks):
#   cp /path/to/best.pt checkpoints/structure_best.pt
#   cp /path/to/vocab.json checkpoints/vocab.json
```

Weights are large (~300MB) and not committed; without them the run skips structure IR.

## Run (core — no Blender)

```bash
python run.py --image /path/to/facade.png --out-dir runs/demo --device cuda
python run.py --facade-id 8 --train-up /path/to/train_up --device cuda
```

CMP XML boxes + the window-AST predictor merge/cluster (box GMM, no DINO split):

```bash
python run.py --image data/facades/base/cmp_b0250.jpg --windows xml \
  --merge-mode ast --cluster-mode box --col-tol 0.045 \
  --out-dir runs/e2e_ast_cluster_cmp_b0250 --device cuda --blender-render
```

SAM3 mask-tight boxes with the same merge/cluster:

```bash
python run.py --image data/facades/base/cmp_b0250.jpg --windows sam \
  --merge-mode ast --cluster-mode box --col-tol 0.045 \
  --out-dir runs/e2e_sam3_cluster_cmp_b0250 --device cuda --blender-render
```

**Layout:** bay count is the max number of windows on any floor. Column bounds come from that densest row; a wide box (e.g. two openings) *spans* those columns instead of merging them. Unitize merge defaults to **containment/IoU only** (`--no-merge-adjacent`); nested full-opening + pane can cross greedy floor bands. Opt in to same-floor nearby-pane merges with `--merge-adjacent`. After clustering, **L–R symmetry repair** runs by default (`--symmetry` / `--no-symmetry`).

**Windows:** `--windows sam` (default) uses SAM3 instance masks as tight boxes; `--windows xml` loads CMP XML next to the image.

Outputs under `runs/facade_e2e_<id>/` (or `--out-dir`):

| File | Description |
|------|-------------|
| `raw_windows.png` | Raw SAM3/XML boxes before merge |
| `overview.png` | Units colored by type |
| `facade_dsl.json` | Floor×bay layout + voted structure IR |
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

## Stages 2–3: symmetry axis + full-stack repair

Both read stage-1 `e2e_sam3_cluster_<stem>/` dirs read-only and write elsewhere.

```bash
# stage 2: continuous vertical symmetry axis (crop-aware ghost partners)
python scripts/symmetry_axis_ghost_continuous.py --sam3-root runs \
  --out-dir runs/symmetry_axis_sweep/ghost_continuous --stems cmp_b0226

# stage 3: symmetry repair about that axis → facade_dsl_repaired.json (+ Blender before/after)
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

**Required:** torch, torchvision, numpy, Pillow, scikit-learn, transformers (SAM3)

**Optional:** Blender ≥ 3.x (uses vendored `vendor/window_compiler`)

**Downloaded at runtime:** HuggingFace `facebook/sam3`, `torch.hub` DINOv2
