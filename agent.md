# Enclosed-balcony detection: session summary

Detect enclosures (oriel windows, enclosed and half-enclosed balconies; not told
apart) on the window floor × bay grid of a facade photo.

Python: `C:\Users\hiroo\anaconda3\envs\glodon\python.exe` only.

## Data

| Set | Photos | With enclosure | Labels | Photos dir |
|---|---|---|---|---|
| Train (cmp_b) | 375 | 29 | `runs/enclosed_grid/labels_train.jsonl` | `data/base_jpg/` |
| Eval (cmp_x) | 61 | 25 | `runs/enclosed_grid/labels_eval.jsonl` | `runs/balcony_clf/crops_ext_eval/Before Crop/` |

- Train photos are the jpgs of `data/base` (the same-stem `.png` files there are CMP
  label maps, never run them). Window grids: `runs/batch/<stem>/facade_dsl.json`
  (made with `run_pipeline.py --image-dir data/base_jpg_todo --out-dir runs/batch`;
  `cmp_b0045`, `0301`, `0342` failed with fewer than 2 windows).
- Eval set: `cmp_x0002` and `cmp_x0003` were replaced by `cmp_x0010` and `cmp_x0013`
  (all four are no-enclosure photos). Caches and features were rebuilt after the swap.
- The eval set is never used for training or for choosing parameters.
- Label UI:
  `balcony_train/enclosed_grid_ui.py --in-dir data/base_jpg --all-photos --labels runs/enclosed_grid/labels_train.jsonl`

## Metrics

- exact (bay_floor): predicted cell set equals the label set. Strict; one wrong cell fails.
- detect: an enclosure photo hits at least one labeled cell; a no-enclosure photo predicts nothing.
- Reported per group: enclosure (25), none (36), total (61); plus cell tp / fp / fn.

## Methods and results (eval set, after the photo swap)

| Method | Total exact | Enclosure exact / detect | None correct |
|---|---|---|---|
| OP1 rule-based, Stage 5 (SAM3 seeds → depth filter → grid) | 34/61 | 6/25 / 15 | 28/36 |
| OP1 + Stage 6 symmetry repair (gate 0.75) | 36/61 | 7/25 / 15 | 29/36 |
| OP2 depth-first rule (`depth_cells.py`) | 22/61 | 1/25 / 17 | 21/36 |
| OP3 cell classifier (gate 0.4) | 40/61 | 6/25 / 9 | 34/36 |
| OP3 classifier only, no gate | 35/61 | 10/25 / 23 | 25/36 |
| **OP3 gate ≥ 0.2 → OP1 cells (current best)** | **41/61** | **7/25 / 13** | **34/36** |
| OP3 rectangle + touching OP1 cells (gate ≥ 0.2) | 39/61 | 8/25 / 16 | 31/36 |
| Pure classifier + touching OP1 (no gate) | 34/61 | 9/25 / 23 | 25/36 |
| Pure classifier + three cell rules | 39/61 | 14/25 / 23 | 25/36 |

### OP1: SAM3 → depth

1. SAM3 finds enclosure candidates (prompts: enclosed / closed / glazed balcony,
   oriel window, bay window; fire escape excluded).
2. Depth (DA3) filters them: 1.5–22% in front of the side wall.
3. Rules fix the final rectangles: merge, grow down, snap to the grid, and Stage 6
   symmetry repair (`--symmetry`).

### OP2: depth-first rule

Cell steps: depth L or R ≥ 1.5% → upper 2/3 in front (drops open balconies) →
SAM3 box on the cell → ≥ 2 floors in one bay → mirror add / drop → drop photo-edge bays.
Fails because depth also picks up building massing (risalits), walls, eaves and trees.

### OP3: cell classifier

- Features per cell: depth (L, R, upper / lower front fraction, neighbours, row and
  column medians), SAM3 scores and cover, OP1 outputs (in_cells_5/6, stage covers),
  position and mirror features, plus DINOv2 `dinov2_vits14` of the cell crop (25%
  context, PCA 16 fitted on training rows).
- Two per-cell binary HistGradientBoosting models:
  - gate: every train photo, hand features without OP1 features; photo gate = max cell probability.
  - cell: enclosure photos only, all features.
- Decoding: score = a·p_cell + (1−a)·in_cells_6, then `rect_decode`: one contiguous
  floor range shared by all chosen bays; mirror bays are paired when window symmetry ≥ 0.75.
- Parameters (a=1.0, t=0.3, gate=0.4) were chosen by grouped 5-fold CV on cmp_b only.

### Current best: gate → OP1

The OP3 gate removes photos without an enclosure, and OP1 places the cells. The
threshold 0.2 was chosen on cmp_b out-of-fold scores (3 seeds); every value in
0.1–0.3 gives 41/61 on eval. It works because the gate's misses are mostly photos
OP1 misses anyway, and OP1's false positives on no-enclosure photos have gate < 0.1.

### Tried: OP3 rectangle, then add touching OP1 cells

`balcony_train/op3_touch_op1.py`. Model is `runs/cell_clf/model_base.joblib` (trained on cmp_b). A photo with gate max < 0.2 predicts nothing. Otherwise the classifier rectangle (t=0.3) is kept, and an OP1 Stage-6 cell is added only when all three hold:

- it 4-connects to the rectangle (same bay and adjacent floor, or same floor and adjacent bay; diagonals do not count);
- the bay is one the rectangle already uses, its neighbor, or its mirror, and the floor is at most one storey outside that bay's rectangle range;
- weak evidence: depth L ≥ 1.5%, SAM3 enclosure cover ≥ 0.25, or classifier probability ≥ 0.10.

Eval once: total exact 39/61, enclosure exact 8/25 (detect 16), none 31/36, cells tp/fp/fn 85/27/48. Groups: exact 8, wrong 8, missed 9, none_ok 31, false_pos 5. Images: `runs/cell_clf/compare_touch/` (`summary.txt`, `_board_*.jpg`).

Adding fixed `0020` (four side cells) and `0179` (one upper floor). The `0006` gallery stayed out because it only touches the oriels on a diagonal. `0143` was exact from the classifier alone and became wrong after three OP1 cells were added. Below the current best (41/61).

A raw union was rejected: it would copy OP1 cells that are a different volume. Dropping a top floor whose box covers under 85% of the storey height was discussed and not implemented. OP1 already drops a spilling top storey at 90% cover (`MIN_UPPER_FLOOR_COVER` in `balcony_pipeline/enclosed.py`); Stage-1 seed height uses 85% of one floor (`SEED_MIN_FLOOR_FRAC`).

### Tried: pure classifier (no gate), then the same touching OP1 cells

`balcony_train/op3_touch_op1.py --gate 0`. Same three conditions, but every photo keeps the classifier rectangle (t=0.3); the gate is not applied. Same trainer as `draw_clf_compare.py`. Images: `runs/cell_clf/compare_clf_op1/`.

Images for the classifier alone are `runs/cell_clf/compare_clf/` (`summary.txt`, `_board_*.jpg`): 35/61, enclosure exact 10/25 (detect 23), none 25/36, cells 93/35/40. Groups: exact 10, wrong 15, missed 0, none_ok 25, false_pos 11. `0119` is exact once bay 0 is gone, and `0123`'s label is now two cells (`f1b4`, `f2b4`) so the rectangle matches. Adding the touching OP1 cells then scores 34/61 (enclosure exact 9/25, detect 23, none 25/36, cells tp/fp/fn 100/51/33). Groups: exact 9, wrong 16, missed 0, none_ok 25, false_pos 11.

Gains: `0020` (four side cells) and `0179` (one upper floor) become exact. `0006` stays out; the gallery only touches the oriels on a diagonal. Losses: `0119`, `0123`, and `0143` were exact from the classifier and become wrong after the added cells. No-enclosure photos are unchanged (nothing was added). Below both the pure classifier on this data (35/61) and gate → OP1 (41/61).

### Tried: pure classifier plus three cell rules

`balcony_train/clf_three_rules.py`. Same rectangle (t=0.3, no gate), then:

1. Add an OP1 cell when L ≥ 1.5%, upper ≥ 0.55, SAM3 enclosure cover ≥ 0.40, and the classifier probability is at least 0.10.
2. Drop the top floor of the rectangle when that band is under 85% of the photo's median floor height.
3. Unless window symmetry ≥ 0.75, drop a cell on that top floor whose probability is under 0.05. Other cells on the floor stay.

Eval once, after `0125` gained `f3b0` in the label: total exact 39/61, enclosure exact 14/25 (detect 23), none 25/36, cells tp/fp/fn 97/33/36. Groups: exact 14, wrong 11, missed 0, none_ok 25, false_pos 11. Images: `runs/cell_clf/compare_clf_rules/`.

The previous 10 exact enclosure photos stay exact. `0005` becomes exact by dropping its short top floor (2 cells). `0179` becomes exact by adding `f0b0`. `0020` becomes exact by adding `f0b0`, `f0b4`, and `f1b0`. `0006` gains nothing. The probability floor keeps out `0125` `f2b5` (p=0.04). No none-photo that was correct became a false positive.

## What was learned

- Exact match is hard for every method: errors are mostly one floor or one bay off.
  SAM3 rarely boxes a whole oriel. Depth cannot separate oriels from massing. The
  grid boundaries and the labels are ambiguous. There are few enclosure examples
  (29 train, 25 eval), and the class balance differs (8% train vs 41% eval).
- On cmp_b, train CV scores (balanced 0.52–0.55) could not rank variants. Class
  weighting, flip augmentation and IoU-based selection did not beat the simple
  classifier. The learning curve is still rising (cell IoU 0.588 → 0.631 → 0.651
  at 50 / 75 / 100% of the data).
- The classifier does help localisation: on gated-on photos its mean IoU is 0.80 vs
  OP1 0.61. The two are complementary:
  - classifier-only exact: `0097`, `0137`, `0032`, `0212`, `0143`;
  - OP1-only exact: `0005`, `0006`, `0020`, `0179`.
- `rect_decode` shares one floor range across bays. Low-probability cells can be
  added by the shared range (`0125` f0 b0) or by mirror pairing (`0149` b5, correct there).

## Resolve priority (agreed)

1. Detect enclosures that are missed entirely. OP1 misses `cmp_x0028`, `0032`, `0074`,
   `0078`, `0097`, `0116`, `0121`, `0129`, `0137`, `0147`.
2. Fix wrong rectangles.
3. Suppress detections on no-enclosure photos (largely solved by the gate).

Tune rules on cmp_b, then check the eval set once.

## Rule ideas from the 17 "wrong" photos (pure classifier), not yet tested

Error types:

- A: two enclosures at different heights (`0006`, `0125`).
- B: the side column of an oriel is missed (`0020`, `0078`, `0116`).
- C: top or bottom floor off by one (`0005`, `0090`, `0147`, `0179`, `0121`, `0129`).
- D: wrong bay: a neighbour building at the photo edge or a strong-depth column
  (`0127`, `0081`, `0119`, `0028`).
- E: strong depth but no SAM3, so the classifier ignores it (`0074`).

Proposed rules, try 1 → 2 → 3 first:

1. Choose the floor range per bay (max-sum run per column); mirror pairs share a range.
2. Extend horizontally: add a neighbour bay over the same floors if SAM3 enclosure
   cover ≥ 0.8 and L ≥ 1.5% (or OP1 picked it).
3. Drop bays touching the photo border (2%) unless their mirror is chosen.
4. Strong-depth columns as candidates (L ≥ 4%, upper ≥ 0.75, ≥ 2 floors, not ground
   floor, not edge, row otherwise flat). Risky: massing.
5. Prefer OP1 cells when they overlap the classifier and are well covered by SAM3.
   Fixes `0005`, `0006`, `0020`, `0179`; hurts `0143`, `0212`.
   A stricter version (add only OP1 cells that touch the OP3 rectangle) was run;
   see "Tried: OP3 rectangle, then add touching OP1 cells". It fixed `0020` and
   `0179` and hurt `0143`.

C looks like a labeling-policy question (turret tops, balcony bases) more than a rule problem.

## Scripts (balcony_train/)

| Script | Purpose |
|---|---|
| `enclosed_symmetry.py` | OP1 Stage 6 symmetry repair |
| `evaluate_enclosed_grid.py` | OP1 evaluation (`--symmetry`), enclosure / none / total report |
| `depth_cells.py`, `evaluate_depth_cells.py` | OP2 |
| `cache_volume_cells.py` | Cache OP1 stage boxes and cells (`volume_cells*.json`) |
| `cell_features_build.py` | Per-cell features + DINOv2 → `runs/cell_clf/cells*.npz` |
| `cell_clf_train.py` | OP3: train on cmp_b, eval once; model `runs/cell_clf/model_base.joblib` |
| `cell_clf_search.py` | Variant search (weighting, flip, OP1 fallback; `--metric exact/iou`), nested CV, learning curve |
| `cell_dino_flip.py` | Flipped DINOv2 embeddings for flip augmentation |
| `gate_op1.py` | Current best: gate threshold chosen on cmp_b, eval once |
| `op3_touch_op1.py` | OP3 rectangle plus touching OP1 cells. Default gate ≥ 0.2 → `runs/cell_clf/compare_touch/`. `--gate 0` is the pure classifier → `runs/cell_clf/compare_clf_op1/` |
| `draw_clf_compare.py` | Pure-classifier compare images with probabilities, grouped boards → `runs/cell_clf/compare_clf/` |
| `draw_cell_clf_compare.py` | Grouped compare boards from a results json |
| `draw_depth_cells.py`, `draw_sam3_raw.py` | Depth / raw SAM3 panels (`runs/cell_clf/wrong_board/{depth,sam3_raw}/` for the 17 wrong photos) |

## Open items

- Touching-OP1 add is below gate → OP1. With the gate it is 39/61; on the pure classifier (no gate) it is 34/61 and breaks classifier-exact photos (`0119`, `0123`, `0143`). If retried, do not add cells on a photo the classifier already matched, and decide the top-floor cut (85% vs the existing 90%) on cmp_b.
- Test rules 1–3 (then 4–5) on cmp_b, then evaluate gate → OP1 / classifier once.
- Rerun `cell_clf_search.py` on the swapped eval set if the search table is needed again.
- Integrate the chosen method into the pipeline (for example `--method cell_clf`):
  features → saved model → gate → OP1 / decoding.
- Add labels for photos with enclosures; avoid buildings that appear in the eval set.
