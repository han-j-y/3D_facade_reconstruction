# Facade E2E Recovery (standalone)

Independent package: **photo → detect → cluster → structure vote → façade DSL**.

Blender rendering is **optional** and skipped cleanly when unavailable.

```
facade_e2e/
  run.py                 # entry point
  run_pipeline.py        # full pipeline
  facade_recovery/       # path helpers
  window_ast/            # vendored structure model
  scripts/               # clustering helpers + optional recovery_to_blender
  checkpoints/           # structure_best.pt (symlink or copy)
  data/vocab_structure.json
  vendor/facade_dsl8 ->  # optional, for --blender-render
```

## Setup

```bash
cd arch/experiments/facade_e2e
pip install -r requirements.txt
# place or symlink a structure checkpoint:
#   ln -s /path/to/best.pt checkpoints/structure_best.pt
#   ln -s /path/to/vocab.json checkpoints/vocab.json
```

## Run (no Blender)

```bash
python run.py --image /path/to/facade.png --out-dir runs/demo --device cuda

# or from the shared train_up set (sibling data/)
python run.py --facade-id 8 --device cuda
```

Outputs under `runs/facade_e2e_<id>/`:

| File | Description |
|------|-------------|
| `overview.png` | Units colored by type |
| `facade_dsl.json` | Floor×bay layout + voted structure IR |
| `summary.json` | Counts + vote stats |
| `assets/types/type_XX/` | Exemplar crop + `structure_ir.json` |

## Optional Blender render

Requires a Blender binary **and** `vendor/facade_dsl8` (symlink included when packaging next to `../facade_dsl8`).

```bash
export BLENDER=/path/to/blender   # optional if `blender` is on PATH
python run.py --facade-id 8 --blender-render --device cuda
```

If Blender (or facade_dsl8) is missing, the pipeline still writes `facade_dsl.json` and prints a warning — it does **not** abort.

## Majority vote

Within each window type, every unit crop is predicted; votes are tallied on a discrete `structure_view` fingerprint (whole IR key, not field-by-field). Ties prefer the type medoid. See vote fields in `assets/types/type_XX/structure_ir.json`.

## Requirements

**Required:** torch, torchvision, numpy, Pillow, scikit-learn, transformers (SAM3)

**Optional:** Blender ≥ 3.x, sibling [`facade_dsl8`](../facade_dsl8)

**Downloaded at runtime:** HuggingFace `facebook/sam3`, `torch.hub` DINOv2
