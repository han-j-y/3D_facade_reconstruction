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
python balcony_train/generate_flux2.py -n 40 --quantized --device cuda --prompt-set masonry
python balcony_train/generate_flux2.py -n 300 --quantized --device cuda --prompt-set metal
python balcony_train/generate_flux2.py -n 40 --quantized --device cuda --prompt-set surface_panel
python balcony_train/generate_flux2.py -n 40 --quantized --device cuda --prompt-set solid
```

Prefixes: `flux2_masonry_`, `flux2_metal_`, `flux2_surface_`, `flux2_solid_`. Label in UI as
`open_work`+`masonry`, `open_work`+`metal`, `surface_panel`, or `solid` respectively.

Outputs go to `crops/unlabeled/`. After labeling (so files sit in a kind
folder), replace each ``flux2_*`` training copy with a centered horizontal
band resized to 120×44 (median real ``cmp`` crop) and a JPEG quality-60
round trip. The original moves to
``crops/_Flux Original/{unlabeled|open_work|surface_panel|solid}/`` and is
not used for training:

```text
C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_train/resize_flux_images.py
```

Full weights need a large GPU; `--quantized`
uses `diffusers/FLUX.2-dev-bnb-4bit` and loads on CPU first then offloads
(avoids OOM on ~24–32 GB cards during `from_pretrained`).

Optional: tighten synth frames with the same SAM3 ``balcony`` detect as the
pipeline (max-score box crop; originals go to `unlabeled_precrop/`):

```text
python balcony_train/recrop_sam3.py --device cuda ^
  --in-dir "runs/balcony_clf/Before crops" ^
  --out-dir runs/balcony_clf/crops/unlabeled
```

Reads ``png`` / ``jpg`` / ``jpeg`` / ``webp`` (default pattern = all of them).

### Bulk: all balconies from ``data/base`` (SAM3)

Crop **every** detected balcony box (not just max-score) into ``crops/unlabeled/``
as ``{stem}_bal_000.png``, ``…_001.png``, …:

```text
C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_train/crop_balconies_sam3.py --device cuda
```

Defaults: ``--in-dir data/base`` → ``--out-dir runs/balcony_clf/crops/unlabeled``.
If few boxes, try ``--threshold 0.35``.

## 2. Label (kind + open_work material)

Interactive UI:

```text
python balcony_train/label_ui.py
```

Or bulk-label FLUX.2 synth crops from filename prefixes (no UI):

```text
C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_train/label_from_prefix.py --unlabeled-dir runs/balcony_clf/crops/unlabeled
```

Optional dry-run: add `--dry-run`. Writes `runs/balcony_clf/labels.jsonl` and moves
images into `crops/{kind}/`.

## 3. Train (multitask)

After labeling (need both metal and masonry among open_work for a useful material head):

```text
python balcony_train/train.py --device cuda --refresh-split
```

```text
python balcony_train/evaluate.py --device cuda
python balcony_train/evaluate.py --split val
```

Evaluate defaults to **real facade crops only** (filename starts with ``cmp``),
so FLUX.2 ``flux2_*`` synth does not inflate metrics. To score everything:

```text
python balcony_train/evaluate.py --device cuda --include-synth
```

List the exact images in the eval set (no GPU):

```text
python balcony_train/evaluate.py --list
python balcony_train/evaluate.py --list-out runs/balcony_clf/eval_list_test_cmp.txt
python balcony_train/evaluate.py --copy-to runs/balcony_clf/eval_view_test_cmp
```

Copy only mistakes (runs the checkpoint). Each run gets a new
``YYYY-MM-DD_HHMMSS`` folder, then
``kind/true-{label}__pred-{label}/`` and ``material/true-{label}__pred-{label}/``:

```text
python balcony_train/evaluate.py --device cuda --errors-to runs/balcony_clf/Wrong
```

Material loss is applied only on `open_work` samples. Stratified splits use keys
`open_work|metal`, `open_work|masonry`, `surface_panel`, `solid`.

### Domain check: DINOv2 t-SNE (real vs FLUX.2)

```text
C:\Users\hiroo\anaconda3\envs\glodon\python.exe -m pip install scikit-learn matplotlib
C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_train/tsne_real_vs_synth.py --device cuda
```

Writes ``runs/balcony_clf/tsne_real_vs_synth.png`` (and ``*_by_kind.png``).
Use ``--components 3`` for 3D, ``--max-per-source 0`` for all images.

Quantitative domain gap (logistic regression AUC on the same features):

```text
C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_train/domain_auc_real_vs_synth.py --device cuda
```

Writes ``runs/balcony_clf/domain_auc_real_vs_synth.json`` (AUC near 1 ⇒ strong
real/synth separation in DINOv2 space).

### Held-out eval set (``crops_ext_eval``)

Keep a **separate** crop tree so training never sees these labels.

1. Put raw facades in ``runs/balcony_clf/crops_ext_eval/Before Crop/``
   (if both ``.jpg`` and ``.png`` duplicates exist, crop one suffix only).

2. SAM3 all-balcony crops → ``unlabeled/``:

```text
C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_train/crop_balconies_sam3.py --device cuda --in-dir "runs/balcony_clf/crops_ext_eval/Before Crop" --out-dir runs/balcony_clf/crops_ext_eval/unlabeled --pattern "*.jpg"
```

3. Label UI (writes ``labels_ext_eval.jsonl`` only):

```text
C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_train/label_ui.py --crops-dir runs/balcony_clf/crops_ext_eval --labels runs/balcony_clf/labels_ext_eval.jsonl
```

4. Optional: build a ~100-image mixed subset (caps open_work; keeps all surface):

```text
C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_train/make_balanced_eval_subset.py --target 100
```

5. Evaluate the train checkpoint on this set only:

```text
C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_train/evaluate.py --device cuda --data-dir runs/balcony_clf/crops_ext_eval_balanced --labels runs/balcony_clf/labels_ext_eval_balanced.jsonl --split all --include-synth --errors-to runs/balcony_clf/eval_view_ext_balanced/Wrong
```

(Or point ``--data-dir`` / ``--labels`` at the full ``crops_ext_eval`` / ``labels_ext_eval.jsonl`` to score everything labeled.)

## 4. Pipeline / compile

- Predictor returns `{kind, material}`; IR stores `railing.material` for open_work.
- Vote fingerprint includes `railing_material` when kind is open_work.
- Blender: `open_work` + `metal` → thin balusters; `open_work` + `masonry` →
  250 mm posts / φ150 balusters (100 mm gaps, centered) / 300×150 top rail
  (front rail full-width; side rails trimmed — no corner notch).
  All balcony slabs use **1.5 m** depth.

Force heuristic (no ckpt): `--no-railing-ckpt`.
