# Facade E2E Recovery (standalone)

Independent package: **photo → detect → cluster → structure vote → façade DSL**.

Blender rendering is **optional**.

```
facade_e2e/
  run.py                 # entry point
  run_pipeline.py        # full pipeline
  facade_recovery/       # path helpers
  window_ast/            # vendored structure model
  scripts/               # clustering helpers + optional render_facade.py
  vendor/window_compiler/# Blender main.py + compile/render (optional)
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

**Layout:** bay count is the max number of windows on any floor. Column bounds come from that densest row; a wide box (e.g. two openings) *spans* those columns instead of merging them. Nested/IoU merge is same-floor only — no vertical merge across floors.

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

## Majority vote

Within each window type, every unit crop is predicted; votes are tallied on a discrete `structure_view` fingerprint (whole IR key, not field-by-field). Ties prefer the type medoid.

## Requirements

**Required:** torch, torchvision, numpy, Pillow, scikit-learn, transformers (SAM3)

**Optional:** Blender ≥ 3.x (uses vendored `vendor/window_compiler`)

**Downloaded at runtime:** HuggingFace `facebook/sam3`, `torch.hub` DINOv2
