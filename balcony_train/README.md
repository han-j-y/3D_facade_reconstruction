# Balcony railing classifier (baluster vs solid)

Trains a frozen DINOv2 + linear head. The checkpoint fills `railing.kind` in
balcony IR. Crop PNGs stay under `runs/` (gitignored). The small head checkpoint
is shared in `checkpoints/railing_best.pt`.

## 1. Collect crops

After a `--with-balconies` batch run:

```text
python balcony_train/collect.py
```

Copies `runs/batch/**/balcony/**/unit_*.png` into
`runs/balcony_clf/crops/unlabeled/`. Duplicates (same file hash) are skipped.

### 1b. Optional: synthetic balcony crops (SDXL)

Generate a **mix of railing styles** (baluster, solid, glass, etc.) when you
need more training data:

```text
pip install -r requirements-sdxl.txt
huggingface-cli login
python balcony_train/generate_sdxl.py -n 5 --device cuda
python balcony_train/generate_sdxl.py --count 100 --device cuda
```

Use `-n` / `--count` for how many images (default **10**). Outputs are tight
**single-balcony** crops in `unlabeled/`. **You label manually:** move keepers to
`solid/` or `baluster/` (pick only the solid ones if baluster data is enough).

On 8 GB VRAM laptops, add `--cpu-offload` if CUDA runs out of memory.

Writes `sdxl_balcony_*.png` into `runs/balcony_clf/crops/unlabeled/`.

Prompt templates live in `balcony_train/prompts_balcony.py`.

## 2. Label

Move each PNG:

```text
runs/balcony_clf/crops/unlabeled/   →  still unlabeled
runs/balcony_clf/crops/solid/       →  opaque panel / parapet
runs/balcony_clf/crops/baluster/    →  vertical bars
```

Do not use the heuristic's guess as ground truth. Need at least one of each class.

## 3. Train

Stratified **80% / 10% / 10%** train / val / test split (per class). Written to
`runs/balcony_clf/split.json` on first run. After adding new solid images:

```text
python balcony_train/train.py --device cuda --refresh-split
```

```text
python balcony_train/train.py --device cuda
```

Writes `checkpoints/railing_best.pt` (linear head only). Checkpoint selection
uses the **validation** fold; test is held out.

```text
python balcony_train/evaluate.py --device cuda
python balcony_train/evaluate.py --split val
python balcony_train/evaluate.py --split test
```

Default evaluate fold is **test**. Use `--split all` only for debugging (data leakage).

## 4. Pipeline

If that checkpoint exists, `infer_balcony_ir` uses it for `railing.kind`.
Otherwise the opaque-run heuristic runs. Force heuristic with `--no-railing-ckpt`.
