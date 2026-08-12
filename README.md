# Facade E2E Recovery (standalone)

Independent package: **photo → detect → cluster → structure vote → façade DSL**.

Does **not** require `facade_dsl8` or Blender.

```
facade_e2e/
  run.py                 # entry point
  run_pipeline.py        # full pipeline
  facade_recovery/       # path helpers
  window_ast/            # vendored structure model
  scripts/               # clustering helpers
  checkpoints/           # structure_best.pt (symlink or copy)
  data/vocab_structure.json
```

## Setup

```bash
cd facade_e2e
pip install -r requirements.txt
# place or symlink a structure checkpoint:
#   ln -s /path/to/best.pt checkpoints/structure_best.pt
#   ln -s /path/to/vocab.json checkpoints/vocab.json
```

## Run

```bash
python run.py --image /path/to/facade.png --out-dir runs/demo --device cuda

# or from a folder of {id}.png façades
python run.py --facade-id 8 --train-up /path/to/train_up --device cuda
```

Outputs under `runs/facade_e2e_<id>/` (or `--out-dir`):

| File | Description |
|------|-------------|
| `overview.png` | Units colored by type |
| `facade_dsl.json` | Floor×bay layout + voted structure IR |
| `summary.json` | Counts + vote stats |
| `assets/types/type_XX/` | Exemplar crop + `structure_ir.json` |

## Majority vote

Within each window type, every unit crop is predicted; votes are tallied on a discrete `structure_view` fingerprint (whole IR key, not field-by-field). Ties prefer the type medoid. See vote fields in `assets/types/type_XX/structure_ir.json`.

## Requirements

**Required:** torch, torchvision, numpy, Pillow, scikit-learn, transformers (SAM3)

**Downloaded at runtime:** HuggingFace `facebook/sam3`, `torch.hub` DINOv2

**Not included / not required:** facade_dsl8, Blender, rendering
