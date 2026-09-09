# Balcony railing classifier (3 infill types)

Trains a frozen DINOv2 + linear head. The checkpoint fills `railing.kind` in
balcony IR. Crop PNGs stay under `runs/` (gitignored). The small head checkpoint
is shared in `checkpoints/railing_best.pt`.

Classes: **`open_work`** | **`surface_panel`** | **`solid`**.
`baluster` and `lined_panel` labels map to `open_work`.

## 1. Collect crops

After a `--with-balconies` batch run:

```text
python balcony_train/collect.py
```

Copies `runs/batch/**/balcony/**/unit_*.png` into
`runs/balcony_clf/crops/unlabeled/`. Duplicates (same file hash) are skipped.

### 1b. Optional: synthetic balcony crops (SDXL)

Generate a **mix of railing styles** when you need more training data:

```text
pip install -r requirements-sdxl.txt
huggingface-cli login
python balcony_train/generate_sdxl.py -n 5 --device cuda
python balcony_train/generate_sdxl.py --count 100 --device cuda
```

Use `-n` / `--count` for how many images (default **10**). Outputs are tight
**single-balcony** crops in `unlabeled/`. **You label manually** into one of
the three class folders below.

On 8 GB VRAM laptops, add `--cpu-offload` if CUDA runs out of memory.

Writes `sdxl_balcony_*.png` into `runs/balcony_clf/crops/unlabeled/`.

Prompt templates live in `balcony_train/prompts_balcony.py`.

## 2. Label

### Option A — Label UI (kind + open_work material)

```text
python balcony_train/collect.py
python balcony_train/label_ui.py
```

- **Kind** (`1`/`2`/`3`): `open_work` | `surface_panel` | `solid`
- **Material** (only when kind is `open_work`): `M` metal | `N` masonry
- **Enter** save, **S** skip, **U** undo, **Q** quit

Writes `runs/balcony_clf/labels.jsonl` and moves the PNG into
`crops/{kind}/` (so the existing kind-only trainer still works).
`material` is stored only in the JSONL (`null` for non-open_work).

Re-open already labeled crops:

```text
python balcony_train/label_ui.py --review-all
```

### Option B — Folder move (kind only)

Move each PNG:

```text
runs/balcony_clf/crops/unlabeled/        →  still unlabeled
runs/balcony_clf/crops/open_work/        →  high-openness infill (balusters, bars, lattice)
runs/balcony_clf/crops/surface_panel/    →  low-opening surface panel (incl. glass)
runs/balcony_clf/crops/solid/            →  no opening
```

Folder moves do not set `material`; open those again in the Label UI if you
need metal vs masonry.

Do not use the heuristic's guess as ground truth. Need at least one of each class.

## 3. Train

Stratified **80% / 10% / 10%** train / val / test split (per class). Written to
`runs/balcony_clf/split.json` on first run. After adding labeled images:

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
Otherwise the opaque-run heuristic runs (`open_work` vs `solid` only).
Force heuristic with `--no-railing-ckpt`.
