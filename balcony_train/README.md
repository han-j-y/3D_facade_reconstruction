# Balcony railing classifier (multitask: kind + material)

Trains frozen DINOv2 + two linear heads:

- **kind**: `open_work` | `surface_panel` | `solid`
- **material** (only when kind is `open_work`): `metal` | `masonry`

Checkpoint: `checkpoints/railing_best.pt`. Crop PNGs stay under `runs/` (gitignored).

## 1. Collect crops

```text
python balcony_train/collect.py
```

### Optional: synthetic masonry open_work (FLUX.2)

When `open_work` + `masonry` labels are scarce, generate extra crops locally with
FLUX.2 [dev] (images stay under `runs/`; not published):

```text
C:\Users\hiroo\anaconda3\envs\glodon\python.exe -m pip install -r requirements-flux2.txt
# Accept https://huggingface.co/black-forest-labs/FLUX.2-dev then: huggingface-cli login
python balcony_train/generate_flux2.py -n 40 --quantized --cpu-offload --device cuda
```

Outputs go to `crops/unlabeled/`. Full weights need a large GPU; `--quantized`
uses `diffusers/FLUX.2-dev-bnb-4bit` for consumer cards.

## 2. Label (kind + open_work material)

```text
python balcony_train/label_ui.py
```

Writes `runs/balcony_clf/labels.jsonl` and moves images into `crops/{kind}/`.

## 3. Train (multitask)

After labeling (need both metal and masonry among open_work for a useful material head):

```text
python balcony_train/train.py --device cuda --refresh-split
```

```text
python balcony_train/evaluate.py --device cuda
python balcony_train/evaluate.py --split val
```

Material loss is applied only on `open_work` samples. Stratified splits use keys
`open_work|metal`, `open_work|masonry`, `surface_panel`, `solid`.

## 4. Pipeline / compile

- Predictor returns `{kind, material}`; IR stores `railing.material` for open_work.
- Vote fingerprint includes `railing_material` when kind is open_work.
- Blender: `open_work` + `metal` → thin balusters; `open_work` + `masonry` →
  250 mm posts / φ150 balusters (100 mm gaps, centered) / 300×150 top rail
  (front rail full-width; side rails trimmed — no corner notch).
  All balcony slabs use **1.5 m** depth.

Force heuristic (no ckpt): `--no-railing-ckpt`.
