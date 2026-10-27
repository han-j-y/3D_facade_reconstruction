# Enclosed-balcony detection: session summary

Detect enclosures (oriel windows, enclosed and half-enclosed balconies; not told
apart) on the window floor × bay grid of a facade photo.

Python: `C:\Users\hiroo\anaconda3\envs\glodon\python.exe` only.

## Data

| Set | Photos | With enclosure | Labels | Photos dir |
|---|---|---|---|---|
| Train (cmp_b) | 375 | 29 | `runs/enclosed_grid/labels_train.jsonl` | `data/base_jpg/` |
| Eval (cmp_x) | 61 | 26 | `runs/enclosed_grid/labels_eval.jsonl` | `runs/balcony_clf/crops_ext_eval/Before Crop/` |

- Train photos are the jpgs of `data/base` (the same-stem `.png` files there are CMP
  label maps, never run them). Window grids: `runs/batch/<stem>/facade_dsl.json`
  (made with `run_pipeline.py --image-dir data/base_jpg_todo --out-dir runs/batch`;
  `cmp_b0045`, `0301`, `0342` failed with fewer than 2 windows).
- Eval set: `cmp_x0002` and `cmp_x0003` were replaced by `cmp_x0010` and `cmp_x0013`
  (all four are no-enclosure photos). Caches and features were rebuilt after the swap.
- The eval set is never used for training or for choosing parameters.
- Label UI:
  `balcony_train/enclosed_grid_ui.py --in-dir data/base_jpg --all-photos --labels runs/enclosed_grid/labels_train.jsonl`
- Train labels hold 29 enclosure photos, 37 balconies (4-connected cell groups), 110 cells. `labels_train.jsonl` still has all 29.
- Shared SAM3 masks: `balcony_train/export_enclosed_masks.py` writes `runs/enclosed_grid/gt_masks/`. One white-on-black PNG per balcony, same size as the photo. The white region is the window-cell rectangle, not the balcony outline. `check/` is a color overlay for looking; do not train on it.
- Left out of the share because the window grid does not cover the enclosure: `cmp_b0291`, `cmp_b0304`, `cmp_b0305`. Shared set is 26 photos, 33 balconies, 103 cells. `0291` mixes an open railing and windows in one cell. `0304` and `0305` only have window boxes, so the oriel body is not a cell.

## Metrics

- exact (bay_floor): predicted cell set equals the label set. Strict; one wrong cell fails.
- detect: an enclosure photo hits at least one labeled cell; a no-enclosure photo predicts nothing.
- Enclosure photos in `labels_eval.jsonl` can also store one `floor_shape` and one `skin`. Empty cell lists stay unlabeled for both. The eval grid UI asks for them (R/T/C and P/G). Score with `score_enclosed_plan.py` on the official predicted cells: each cell is cropped, then the same majority vote as the pipeline. Triangle counts as rectangle. No predicted cells is wrong on both. `cmp_x` stays out of the training jsonl.
- Reported per group: enclosure (26), none (35), total (61); plus cell tp / fp / fn.
  `0185` was a no-enclosure photo and is now `f0b1`, `f1b1`, `f2b1`, `f3b1`.
  Rows in the table below that still say 25 and 36 were scored before that label edit.

## Methods and results (eval set, after the photo swap)

| Method | Total exact | Enclosure exact / detect | None correct |
|---|---|---|---|
| OP1 rule-based, Stage 5 (SAM3 seeds → depth filter → grid) | 34/61 | 6/25 / 15 | 28/36 |
| OP1 + Stage 6 symmetry repair (gate 0.75) | 36/61 | 7/25 / 15 | 29/36 |
| OP2 depth-first rule (`depth_cells.py`) | 22/61 | 1/25 / 17 | 21/36 |
| OP3 cell classifier (gate 0.4) | 40/61 | 6/25 / 9 | 34/36 |
| OP3 classifier only, no gate | 35/61 | 10/26 / 24 | 25/35 |
| OP3 gate ≥ 0.2 → OP1 cells | 41/61 | 7/25 / 13 | 34/36 |
| OP3 rectangle + touching OP1 cells (gate ≥ 0.2) | 39/61 | 8/25 / 16 | 31/36 |
| Pure classifier + touching OP1 (no gate) | 34/61 | 9/25 / 23 | 25/36 |
| Pure classifier + four cell rules | 41/61 | 16/26 / 24 | 25/35 |
| **Four cell rules + gate ≥ 0.10** | **47/61** | **16/26 / 21** | **31/35** |
| SAM3 fold-0 fine-tune, raw boxes snapped to the grid | 33/61 | 6/26 / 21 | 27/35 |

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

### Tried: gate → OP1

The OP3 gate removes photos without an enclosure, and OP1 places the cells. The
threshold 0.2 was chosen on cmp_b out-of-fold scores (3 seeds); every value in
0.1–0.3 gives 41/61 on eval. It works because the gate's misses are mostly photos
OP1 misses anyway, and OP1's false positives on no-enclosure photos have gate < 0.1.
Superseded by four cell rules + gate ≥ 0.10 (47/61). `--enclosed-method sam3` is
this older SAM3+DA3 volume path, not the 0.2 gate.

### Tried: OP3 rectangle, then add touching OP1 cells

`balcony_train/op3_touch_op1.py`. Model is `runs/cell_clf/model_base.joblib` (trained on cmp_b). A photo with gate max < 0.2 predicts nothing. Otherwise the classifier rectangle (t=0.3) is kept, and an OP1 Stage-6 cell is added only when all three hold:

- it 4-connects to the rectangle (same bay and adjacent floor, or same floor and adjacent bay; diagonals do not count);
- the bay is one the rectangle already uses, its neighbor, or its mirror, and the floor is at most one storey outside that bay's rectangle range;
- weak evidence: depth L ≥ 1.5%, SAM3 enclosure cover ≥ 0.25, or classifier probability ≥ 0.10.

Eval once: total exact 39/61, enclosure exact 8/25 (detect 16), none 31/36, cells tp/fp/fn 85/27/48. Groups: exact 8, wrong 8, missed 9, none_ok 31, false_pos 5. Images: `runs/cell_clf/compare_touch/` (`summary.txt`, `_board_*.jpg`).

Adding fixed `0020` (four side cells) and `0179` (one upper floor). The `0006` gallery stayed out because it only touches the oriels on a diagonal. `0143` was exact from the classifier alone and became wrong after three OP1 cells were added. Below gate → OP1 (41/61).

A raw union was rejected: it would copy OP1 cells that are a different volume. Dropping a top floor whose box covers under 85% of the storey height was discussed and not implemented. OP1 already drops a spilling top storey at 90% cover (`MIN_UPPER_FLOOR_COVER` in `balcony_pipeline/enclosed.py`); Stage-1 seed height uses 85% of one floor (`SEED_MIN_FLOOR_FRAC`).

### Tried: pure classifier (no gate), then the same touching OP1 cells

`balcony_train/op3_touch_op1.py --gate 0`. Same three conditions, but every photo keeps the classifier rectangle (t=0.3); the gate is not applied. Same trainer as `draw_clf_compare.py`. Images: `runs/cell_clf/compare_clf_op1/`.

Images for the classifier alone are `runs/cell_clf/compare_clf/` (`summary.txt`, `_board_*.jpg`): 35/61, enclosure exact 10/25 (detect 23), none 25/36, cells 93/35/40, scored before `0185` was labeled as an enclosure. Groups: exact 10, wrong 15, missed 0, none_ok 25, false_pos 11. `0119` is exact once bay 0 is gone, and `0123`'s label is now two cells (`f1b4`, `f2b4`) so the rectangle matches. On the current labels the same rectangle, printed at the top of `runs/cell_clf/compare_clf_rules/summary.txt`, is 35/61, enclosure exact 10/26 (detect 24), none 25/35, cells 97/31/40. Adding the touching OP1 cells then scores 34/61 (enclosure exact 9/25, detect 23, none 25/36, cells tp/fp/fn 100/51/33) on the earlier labels. Groups: exact 9, wrong 16, missed 0, none_ok 25, false_pos 11.

Gains: `0020` (four side cells) and `0179` (one upper floor) become exact. `0006` stays out; the gallery only touches the oriels on a diagonal. Losses: `0119`, `0123`, and `0143` were exact from the classifier and become wrong after the added cells. No-enclosure photos are unchanged (nothing was added). Below both the pure classifier on this data (35/61) and gate → OP1 (41/61).

### Tried: pure classifier plus four cell rules

`balcony_train/clf_three_rules.py`. Same rectangle (t=0.3, no gate). Applied in this order:

1. Drop the top floor of the rectangle when that band is under 85% of the photo's median floor height.
2. Unless window symmetry ≥ 0.75, drop a cell on that top floor whose probability is under 0.05. Other cells on the floor stay.
3. Add an OP1 cell when L ≥ 1.5%, upper ≥ 0.55, SAM3 enclosure cover ≥ 0.40, and the classifier probability is at least 0.10.
4. Add the cell one floor above a kept cell in the same bay when its depth L is within 0.02 of the cell directly below, upper ≥ 0.75, and that floor band is at least 85% of the median floor height. Classifier probability, OP1 membership, and SAM3 cover are not required. The new cell does not grow another floor.

Eval once, after `0125` gained `f3b0`, `0020` lost `f1b4`, and `0185` became an enclosure: total exact 41/61, enclosure exact 16/26 (detect 24), none 25/35, cells tp/fp/fn 104/33/33. Groups: exact 16, wrong 10, missed 0, none_ok 25, false_pos 10. Images: `runs/cell_clf/compare_clf_rules/`.

Exact enclosure photos: `0005`, `0020`, `0032`, `0090`, `0097`, `0118`, `0119`, `0123`, `0125`, `0137`, `0143`, `0144`, `0149`, `0179`, `0185`, `0212`. The earlier 10 exact photos stay exact. `0005` drops its short top floor. `0020` adds `f0b0`, `f0b4`, `f1b0`. `0179` adds `f0b0`. `0090` adds `f0b6` (stored probability 0.0986; L 0.074 vs 0.065 below). `0185` adds `f0b1` (probability 0.04; L 0.050 vs 0.042 below). `0125` `f0b0` stays out because its depth is 0.041 away from the floor below. `0006` is still not exact: the prediction is `f4b0`, `f4b4` and the gallery `f5b1`, `f5b2`, `f5b3` is missing.

Rule 4 also adds `0074` `f3b5` and `0129` `f2b4` (both in the label; neither photo becomes exact) and three cells that are not in the label: `0121` `f0b2`, `0073` `f1b0`, `0134` `f1b0`. No none-photo that was correct became a false positive.

### Tried: four cell rules, then gate ≥ 0.10

`balcony_train/clf_three_rules.py --gate 0.10`. Same four rules. A photo whose gate-model max is below 0.10 predicts nothing; every other photo keeps the four-rule cells. The threshold leaves the four no-enclosure photos at gate 0.23 and above as false positives, and it is below the lowest exact enclosure (`0119` at 0.14, then `0032` at 0.15). Images: `runs/cell_clf/compare_clf_rules_gate/`.

Eval once:

```text
enclosure n=26  detect=21/26  detect_acc=0.808  bay_floor=16/26  bay_floor_acc=0.615
none n=35  detect=31/35  detect_acc=0.886  bay_floor=31/35  bay_floor_acc=0.886
total n=61  detect=52/61  detect_acc=0.852  bay_floor=47/61  bay_floor_acc=0.770
cells tp=97 fp=11 fn=40  recall=0.708  precision=0.898
```

Groups: exact 16, wrong 6, missed 4, none_ok 31, false_pos 4. The 16 exact enclosure photos are unchanged. Six no-enclosure photos become correct: `0009`, `0046`, `0077`, `0134`, `0186`, `0191`. The four false positives left on purpose (gate 0.23 and above) are `0073`, `0120`, `0173`, `0180`. Four enclosure photos that were wrong become missed: `0028` (already IoU 0), `0074`, `0081`, `0121`. Detect falls by the three that had overlapped the label.

### Pipeline default

`balcony_train/enclosed_detect.py`, called from `balcony_pipeline/run_balcony.py` and from `run_pipeline.py --with-balconies`. Default `--enclosed-method cell_clf`. `--enclosed-method sam3` restores the previous SAM3+DA3 volumes (`detect_enclosed_volume_boxes`).

cell_clf builds the same per-cell features as training (depth, SAM3, OP1 stage cells, DINOv2), decodes with the four rules, then blanks the photo when the gate max is below 0.10. Each kept cell becomes one `balc_enclosed` unit via `place_enclosed_cells` in `balcony_pipeline/enclosed.py`. The classifier is the `cell_clf_search` base model, fit on cmp_b and cached at `runs/cell_clf/model_rules_gate.joblib`. Decoding the eval feature matrix with that cache matches `runs/cell_clf/compare_clf_rules_gate/preds.json` on all 61 photos.

Enclosed floor plan is a second step. The railing checkpoint's floor head labels each enclosed crop; `triangle` counts as `rectangle`. One shape is chosen for the whole photo by majority vote (`vote_enclosed_floor_shape`); a tie is `rectangle`. That plan is written onto every enclosed unit. Open balconies are not in this photo-level vote.

Enclosed skin is the same vote on a separate head (`punched_window` / `curtain_wall`). A tie, or an untrained head, is `punched_window`. Curtain wall writes `walls.infill = glass` plus `curtain_wall: true` and the compiler drops windows in those bays. Punched writes `walls.infill = wall`, so windows move onto the front as before. Crops come from `balcony_train/export_enclosed_crops.py` (cmp_b only) and are labeled in `label_ui.py` (P / G). Train with `--heads-only floor,skin`.

Railing crops stay `open`. `infer_unit_ir` does not read the railing classifier's enclosure head, so a handrail is never `half_enclosed` or `enclosed`. Enclosed units come only from the cell detector.

Open-balcony floor shape is a separate majority inside each appearance cluster (`apply_floor_majority`), using the same rule: circle only on a strict majority, otherwise rectangle. The railing vote (kind and material) is unchanged, and the winning railing IR receives that floor. `--no-floor-shape-vote` and `--no-railing-vote` keep each unit's own floor. The cluster vote record stores `floor_shape` and `floor_counts`.

A curtain-wall circle is one semicircular drum. Touching cells (same floor, neighboring bays, or the floor above in overlapping bays) fuse in `fuse_curtain_circles`: width is the union of the bays, depth is half that width, glass runs from the lowest sill to the top of the stack. Each storey gets a floor slab on that outline. A horizontal mullion sits 2100 mm above each slab top (`CURTAIN_TRANSOM_ABOVE_FLOOR_M`) and the glass splits there; a storey shorter than that gets no transom. A rectangle curtain wall already has a slab per floor and only gains the same transom. Punched enclosures are unchanged.

### Tried: SAM3 fold-0 fine-tune, raw boxes

`balcony_train/eval_sam3_enclosed_ft.py`. Checkpoint `proj-facade/sam3-enclosed-facade-fold0` (base `facebook/sam3`, trained on 301 of 378 CMP base facades in fold 0). The held-out fold is cmp_b stems. This run scores cmp_x only and does not read cmp_b. Inference follows the model card: prompt `enclosed balcony`, score threshold 0.45, mask threshold 0.5, and one padding box `input_boxes=[[[0.5, 0.5, 0.1, 0.1]]]` with label `-10` so the geometry encoder stays on. Boxes are snapped with `cells_from_boxes`. The DA3 stage filters and the cell classifier are not applied to this score.

Eval once, current labels (26 enclosure / 35 none, 2022 cells):

```text
61 images, 2022 cells
TP 104  FP 54  FN 33  TN 1831
Precision 0.658
Recall 0.759
F1 0.705
IoU 0.545
exact 33/61
enclosure exact 6/26  detect 21/26
none 27/35
```

Exact enclosure photos: `0005`, `0119`, `0129`, `0137`, `0147`, `0149`. Missed (no overlapping cell): `0032`, `0097`, `0123`, `0144`, `0185`. Wrong rectangle: `0006`, `0020`, `0028`, `0074`, `0078`, `0081`, `0090`, `0116`, `0118`, `0121`, `0125`, `0127`, `0143`, `0179`, `0212`. False positives on no-enclosure photos: `0045`, `0055`, `0073`, `0120`, `0128`, `0167`, `0171`, `0180`. Outputs: `runs/sam3_ft_fold0/cmp_x/` (`summary.json`, `summary.txt`, `board.png`, `compare/{stem}.jpg`, cache `boxes.json`). `cmp_x0005` smoke check: 2 boxes, tp 12, fp 0, fn 0, tn 51, exact.

`--stages` draws the six OP1 panels from those cached boxes and the existing depth maps `runs/da3_enclosure/{stem}_depth.npy` (61/61 present). Symmetry is on. Strips: `runs/sam3_ft_fold0/cmp_x/stages/{stem}_stages.jpg` and the stacked `stages_board.jpg`. Stage 1 already drops a seed shorter than 0.85 of a floor, so the strip counts are a different result from the cell table. `cmp_x0005` is exact in the cell table and `n=0` on every panel of its strip.

## What was learned

- Exact match is hard for every method: errors are mostly one floor or one bay off.
  SAM3 rarely boxes a whole oriel. Depth cannot separate oriels from massing. The
  grid boundaries and the labels are ambiguous. There are few enclosure examples
  (29 train, 26 eval), and the class balance differs (8% train vs 43% eval).
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
3. Suppress detections on no-enclosure photos. Gate ≥ 0.10 leaves four false positives on purpose: `0073`, `0120`, `0173`, `0180`.

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
| `gate_op1.py` | Gate threshold chosen on cmp_b, then OP1 cells; eval once |
| `op3_touch_op1.py` | OP3 rectangle plus touching OP1 cells. Default gate ≥ 0.2 → `runs/cell_clf/compare_touch/`. `--gate 0` is the pure classifier → `runs/cell_clf/compare_clf_op1/` |
| `clf_three_rules.py` | Pure classifier plus the four cell rules → `runs/cell_clf/compare_clf_rules/`. `--gate 0.10` blanks photos under that gate → `runs/cell_clf/compare_clf_rules_gate/` |
| `enclosed_detect.py` | Pipeline detector: same four rules and gate ≥ 0.10. Model cache `runs/cell_clf/model_rules_gate.joblib` |
| `enclosed_floor_ui.py` | Label rectangle/circle on enclosed cells cropped from `data/base`. One plan per photo → `runs/enclosed_floor/` |
| `export_enclosed_crops.py` | Train enclosed volumes → `runs/balcony_clf/crops/unlabeled/<stem>_enclosed_<NN>.png` for floor and skin labels |
| `export_enclosed_masks.py` | Shared binary masks → `runs/enclosed_grid/gt_masks/`. Skips `cmp_b0291`, `cmp_b0304`, `cmp_b0305` |
| `draw_clf_compare.py` | Pure-classifier compare images with probabilities, grouped boards → `runs/cell_clf/compare_clf/` |
| `draw_cell_clf_compare.py` | Grouped compare boards from a results json |
| `score_enclosed_plan.py` | Eval floor shape and skin from `labels_eval.jsonl` and official predicted cells |
| `draw_depth_cells.py`, `draw_sam3_raw.py` | Depth / raw SAM3 panels (`runs/cell_clf/wrong_board/{depth,sam3_raw}/` for the 17 wrong photos) |
| `eval_sam3_enclosed_ft.py` | Fold-0 SAM3 fine-tune on cmp_x. Raw boxes → `runs/sam3_ft_fold0/cmp_x/`. `--stages` writes the six-panel strips under `stages/` |

## Open-balcony type clustering

Open balconies are typed after snap. Two modes live in `balcony_pipeline/cluster.py`. The pipeline default is still appearance.

- `appearance` (default): DINOv2 ROI, spectral clustering, then Potts on neighboring floors and bays. This is the old window appearance path.
- `box` (`run_balcony.py --cluster-mode box`): z-scored width, height, and aspect, same GMM as the window track's `--cluster-mode box` (`box_geometry_labels`). DINOv2 is still computed. It is used by `apply_symmetry_repair` (cosine twin merge / steal), not to form the types. `labels_box` reverts a flip that would mix two size groups. `--k-max-box` defaults to 6. `--twin-mirror-min` defaults to 0.35.

Same-size balconies with different railings become one type. The railing vote then writes one kind for that size group.

### Visual compare (not scored)

`balcony_pipeline/compare_open_cluster.py --limit 20`. First 20 names in `data/baseForBatch` (`cmp_x0004` through `cmp_x0121`). SAM3 balcony boxes are detected once. Both panels use those boxes after the usual filter and snap. The window floor × bay grid is built from `data/extended/<stem>.xml` with the same column functions as the window track (`windows_dsl_from_boxes`). It is not a `runs/batch` facade DSL, so unit counts will not match an older appearance overlay of the same photo. `cmp_x0006` here is 22 units; an earlier appearance overlay of that facade was 19 units and 5 types.

Outputs: `runs/balcony_cluster_compare/{stem}_compare.jpg` (left appearance, right box + symmetry), `board.jpg`, `summary.json`, `{stem}_labels.json`, `{stem}_raw_boxes.json`.

Five photos had fewer than 2 open units and were not clustered: `0004`, `0008`, `0013`, `0015`, `0019`.

Type counts on the other 15 (appearance → box, after the symmetry gate):

| Stem | Units | Appearance | Box |
|---|---|---|---|
| `0005` | 22 | 4 | 2 |
| `0006` | 22 | 2 | 5 |
| `0009` | 4 | 2 | 2 |
| `0010` | 3 | 2 | 2 |
| `0011` | 5 | 2 | 2 |
| `0012` | 2 | 1 | 2 |
| `0014` | 2 | 2 | 2 |
| `0016` | 9 | 2 | 3 |
| `0017` | 4 | 2 | 2 |
| `0018` | 5 | 2 | 3 |
| `0020` | 5 | 3 | 2 |
| `0084` | 3 | 1 | 2 |
| `0094` | 6 | 3 | 3 |
| `0116` | 9 | 3 | 3 |
| `0121` | 6 | 2 | 4 |

Equal type count does not mean the same grouping. Symmetry changed the box-mode type count only on `0018` (4→3) and `0116` (4→3). On `0006`, `0020`, `0094`, and `0121` the repair moved labels and the size gate reverted those moves, so the type count stayed at the box clustering.

`detect_windows` accepts a loaded SAM3 model so this batch does not reload it per photo.

## Extended label distribution

Two finished censuses of `data/extended` (`cmp_x`). Neither is the 61-photo enclosure eval set. Do not retune the gate or the cell rules from these counts. Reprint the crop margins with `balcony_train/summarize_xml_labels.py`.

### Open crops

`runs/balcony_clf/crops_ext_xml/labels.jsonl`. 738 rows. 737 are finished open balconies. One row, `cmp_x0093_bal_000`, is `enclosed` with floor and skin still empty, so it is left out of the tables below. `half_enclosed` is 0. `none.jsonl` lists 84 photos with no balcony crop. Counts are crops, not photos. An empty structure label is `projecting`. Material is counted only on `open_work`.

| kind | crops |
|---|---:|
| open_work | 643 |
| solid | 89 |
| surface_panel | 5 |

| material (open_work only) | crops |
|---|---:|
| metal | 381 |
| masonry | 262 |

| floor | crops |
|---|---:|
| rectangle | 628 |
| circle | 106 |
| triangle | 3 |

| structure | crops |
|---|---:|
| projecting | 561 |
| juliet | 176 |

The axes are not independent. Multiplying the margins invents combinations that do not occur and splits the largest cell. Use the joint. Seven cells cover 714/737 (96.9%):

| kind | material | floor | structure | crops |
|---|---|---|---|---:|
| open_work | metal | rectangle | projecting | 268 |
| open_work | masonry | rectangle | juliet | 129 |
| open_work | masonry | rectangle | projecting | 108 |
| open_work | metal | circle | projecting | 72 |
| solid | — | rectangle | projecting | 72 |
| open_work | metal | rectangle | juliet | 40 |
| open_work | masonry | circle | projecting | 25 |

The other 23 crops are six cells: solid circle projecting 8, solid rectangle juliet 6, surface_panel rectangle projecting 5, solid triangle projecting 2, open_work metal triangle projecting 1, solid circle juliet 1.

### Enclosed photos

`runs/enclosed_grid/labels_ext_all.jsonl`. One floor shape and one skin per photo. An empty cell list is no enclosure, so those photos have neither. 215 photos, all finished.

| | photos |
|---|---:|
| no enclosure | 142 |
| enclosure | 73 |

The 73 enclosure photos cover 508 cells.

| floor | photos |
|---|---:|
| rectangle | 55 |
| circle | 17 |
| triangle | 1 |

| skin | photos |
|---|---:|
| punched_window | 71 |
| curtain_wall | 2 |

Joint: rectangle × punched_window 54, circle × punched_window 16, rectangle × curtain_wall 1, circle × curtain_wall 1, triangle × punched_window 1.

## Open items

- Pipeline default is four cell rules + gate ≥ 0.10: 47/61 total, enclosure exact 16/26, detect 21/26, none 31/35. False positives left: `0073`, `0120`, `0173`, `0180`. The gate blanks `0074`, `0081`, and `0121`. `0006` is still short the gallery floor. Rule 4's extra cells on `0121`, `0073`, and `0134` are not in the labels. Switch back with `--enclosed-method sam3`.
- The fold-0 SAM3 fine-tune was scored once on cmp_x (raw boxes, 33/61, cells TP 104 / FP 54 / FN 33). It is not wired into `--enclosed-method`. The stage strips under `runs/sam3_ft_fold0/cmp_x/stages/` were not scored as cells.
- Touching-OP1 add is below both gate → OP1 (41/61) and the pipeline default (47/61). With the gate it is 39/61; on the pure classifier (no gate) it is 34/61 and breaks classifier-exact photos (`0119`, `0123`, `0143`). If retried, do not add cells on a photo the classifier already matched, and decide the top-floor cut (85% vs the existing 90%) on cmp_b.
- Rules 1–3 in the "not yet tested" list (per-bay floor range, horizontal extend, border-bay drop) are still untried. Test them on cmp_b, then check eval once.
- Rerun `cell_clf_search.py` on the swapped eval set if the search table is needed again.
- Add labels for photos with enclosures; avoid buildings that appear in the eval set.
- Open-balcony types still default to DINOv2 appearance. `--cluster-mode box` was compared visually on 15 of the first 20 `data/baseForBatch` photos (`runs/balcony_cluster_compare/`). It was not scored against railing labels. Same-size balconies with different railings collapse to one type.
