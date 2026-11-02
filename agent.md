# Enclosed-balcony detection: session summary

Detect enclosures (oriel windows, enclosed and half-enclosed balconies; not told
apart) on the window floor × bay grid of a facade photo.

Python: `C:\Users\hiroo\anaconda3\envs\glodon\python.exe` only.

## Data

| Set | Photos | With enclosure | Labels | Photos dir |
|---|---|---|---|---|
| Train (cmp_b) | 375 | 29 | `runs/enclosed_grid/labels_train.jsonl` | `data/base_jpg/` |
| Test (cmp_x) | 133 | 56 | `runs/enclosed_grid/labels_ext_all.jsonl` | `data/extended_selected/` |

The default test is the 133-photo row. `paper_eval.py` still reads `runs/balcony_clf/crops_ext_eval/Before Crop` for the CMP xml files and label maps.

- Train photos are the jpgs of `data/base` (the same-stem `.png` files there are CMP
  label maps, never run them). Window grids: `runs/batch/<stem>/facade_dsl.json`
  (made with `run_pipeline.py --image-dir data/base_jpg_todo --out-dir runs/batch`;
  `cmp_b0045`, `0301`, `0342` failed with fewer than 2 windows).
- Eval set: `cmp_x0002` and `cmp_x0003` were replaced by `cmp_x0010` and `cmp_x0013`
  (all four are no-enclosure photos). Caches and features were rebuilt after the swap.
- The eval set is never used for training or for choosing parameters.
- Label UI:
  `balcony_train/enclosed_grid_ui.py --in-dir data/base_jpg --all-photos --labels runs/enclosed_grid/labels_train.jsonl`
  - 2026-10-29: the UI draws the per-floor bays (`meta.floor_bays`). Click toggles a whole floor bay; Shift+click toggles one union column. Labels are still stored per union column `(floor, column)`, so the scorers and old labels are unchanged. A red outline marks a half-labeled bay. `--review-partial` queues only those photos (test: 16, train: 7); `--stems` queues named photos. Before this the UI drew only the union columns. Some half-labeled bays are deliberate (an oriel narrower than its floor bay, e.g. `cmp_x0078` f2–f3 b5–6).
  - 2026-10-29: ignore cells. Right-click (Shift+right-click: one column) or Ignore mode (I) marks a cell whose label is ambiguous from the photo (the grid line cuts through the oriel, or the cell mixes it with a neighbouring building). Saved as `"ignore": [[f, c], ...]` next to `cells`, without changing `cells`, so the main per-column score is unchanged. `score_enclosed_bays.py` adds supplementary rows that drop ignore cells from TP / FP / FN and prints how many cells were ignored. Mark ignore from the photo only, never from the prediction boards.
  - Test DSLs `cmp_x0020`, `0094`, `0121`, `0224` (and `0005`, still exact) were rebuilt on Oct 28, after `labels_ext_all.jsonl` was saved on Oct 27. Their cell labels need a check in the UI.
- Train labels hold 29 enclosure photos, 37 balconies (4-connected cell groups), 110 cells. `labels_train.jsonl` still has all 29.
- Shared SAM3 masks: `balcony_train/export_enclosed_masks.py` writes `runs/enclosed_grid/gt_masks/`. One white-on-black PNG per balcony, same size as the photo. The white region is the window-cell rectangle, not the balcony outline. `check/` is a color overlay for looking; do not train on it.
- Left out of the share because the window grid does not cover the enclosure: `cmp_b0291`, `cmp_b0304`, `cmp_b0305`. Shared set is 26 photos, 33 balconies, 103 cells. `0291` mixes an open railing and windows in one cell. `0304` and `0305` only have window boxes, so the oriel body is not a cell.

## Metrics

- exact (bay_floor): predicted cell set equals the label set. Strict; one wrong cell fails.
- detect: an enclosure photo hits at least one labeled cell; a no-enclosure photo predicts nothing.
- Enclosure photos in `labels_ext_all.jsonl` can also store one `floor_shape` and one `skin`. Empty cell lists stay unlabeled for both. The eval grid UI asks for them (R/T/C and P/G). Score with `score_enclosed_plan.py` on the official predicted cells: each cell is cropped, then the same majority vote as the pipeline. Triangle counts as rectangle. No predicted cells is wrong on both. `cmp_x` stays out of the training jsonl.
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

**2026-10-29 rescore on the 133-photo test set, current grids.** The old test cache (`cells_ext_no0227.npz`, 6,754 cells) was built on old grids for 10 photos and made the scorer crash. Rebuilt: `cache_volume_cells.py --out runs/enclosed_grid/volume_cells_floor_bays_20261029.json`, then `cell_features_build.py --volume-cache <that> --out runs/cell_clf/cells_ext_20261029.npz` (7,009 cells), then `clf_three_rules.py --gate 0.10 --eval-cells <that> --out-dir runs/cell_clf/compare_clf_rules_gate_ext_20261029`. Train cache unchanged (all 375 photos match their grids). Labels: `labels_ext_all.jsonl` as of this date, before the half-labeled-bay review. Pooled cells:

| set | photos | TP | FP | FN | P | R | F1 | IoU | exact photos |
|---|---|---|---|---|---|---|---|---|---|
| test, all | 133 (56 with oriels) | 262 | 160 | 266 | 0.621 | 0.496 | 0.552 | 0.381 | 74/133 |

Groups: exact 7, wrong 38, missed 11, none_ok 67, false_pos 10. Without the gate, the four rules give 272 / 210 / 256.

Compare images (`render_compare`, `draw_probs`) now draw the per-floor bays when given `floor_bays` (`clf_three_rules.py`, `op3_touch_op1.py`, `draw_clf_compare.py`, `draw_cell_clf_compare.py` pass it). Cells are still colored per union column; each floor bay gets one outline, one name (`f1 b2-4`) and one probability, the highest among its columns. A white outline marks a bay whose columns differ. Scores are unchanged.

Supplementary, per floor bay (`balcony_train/score_enclosed_bays.py --preds runs/cell_clf/compare_clf_rules_gate_ext_20261029/preds.json`). Each floor bay counts once however many union columns it spans; it is labeled / predicted when any of its columns is. Same predictions and labels as the per-column table above:

| set | photos | floor bays | TP | FP | FN | P | R | F1 | IoU | exact photos |
|---|---|---|---|---|---|---|---|---|---|---|
| test, all | 133 | 4656 | 208 | 112 | 208 | 0.650 | 0.500 | 0.565 | 0.394 | 77/133 |

### Pipeline default

`balcony_train/enclosed_detect.py`, called from `balcony_pipeline/run_balcony.py` and from `run_pipeline.py --with-balconies`. Default `--enclosed-method cell_clf`. `--enclosed-method sam3` restores the previous SAM3+DA3 volumes (`detect_enclosed_volume_boxes`).

cell_clf builds the same per-cell features as training (depth, SAM3, OP1 stage cells, DINOv2), decodes with the four rules, then blanks the photo when the gate max is below 0.10. Each kept cell becomes one `balc_enclosed` unit via `place_enclosed_cells` in `balcony_pipeline/enclosed.py`. The classifier is the `cell_clf_search` base model, fit on cmp_b and cached at `runs/cell_clf/model_rules_gate.joblib`. Decoding the eval feature matrix with that cache matches `runs/cell_clf/compare_clf_rules_gate/preds.json` on all 61 photos.

Enclosed floor plan is a second step. The railing checkpoint's floor head labels each enclosed crop; `triangle` counts as `rectangle`. One shape is chosen for the whole photo by majority vote (`vote_enclosed_floor_shape`); a tie is `rectangle`. That plan is written onto every enclosed unit. Open balconies are not in this photo-level vote.

Enclosed skin is the same vote on a separate head (`punched_window` / `curtain_wall`). A tie, or an untrained head, is `punched_window`. Curtain wall writes `walls.infill = glass` plus `curtain_wall: true` and the compiler drops windows in those bays. Punched writes `walls.infill = wall`, so windows move onto the front as before. Crops come from `balcony_train/export_enclosed_crops.py` (cmp_b only) and are labeled in `label_ui.py` (P / G). Train with `--heads-only floor,skin`.

Railing crops stay `open`. `infer_unit_ir` does not read the railing classifier's enclosure head, so a handrail is never `half_enclosed` or `enclosed`. Enclosed units come only from the cell detector.

Open-balcony floor shape is a separate majority inside each appearance cluster (`apply_floor_majority` → `vote_open_floor_shape`). Since 2026-10-29 triangle is its own shape there: a circle or a triangle wins only with more votes than each of the other two shapes, otherwise rectangle; juliet members do not vote. The enclosed photo vote above still maps triangle to rectangle. The railing vote (kind and material) is unchanged, and the winning railing IR receives that floor. `--no-floor-shape-vote` and `--no-railing-vote` keep each unit's own floor. The cluster vote record stores `floor_shape` and `floor_counts`.

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
| `clf_three_rules.py` | Pure classifier plus the four cell rules → `runs/cell_clf/compare_clf_rules/`. `--gate 0.10` blanks photos under that gate → `runs/cell_clf/compare_clf_rules_gate_ext_relabel/` |
| `enclosed_detect.py` | Pipeline detector: same four rules and gate ≥ 0.10. Model cache `runs/cell_clf/model_rules_gate.joblib` |
| `enclosed_floor_ui.py` | Label rectangle/circle on enclosed cells cropped from `data/base`. One plan per photo → `runs/enclosed_floor/` |
| `export_enclosed_crops.py` | Train enclosed volumes → `runs/balcony_clf/crops/unlabeled/<stem>_enclosed_<NN>.png` for floor and skin labels |
| `export_enclosed_masks.py` | Shared binary masks → `runs/enclosed_grid/gt_masks/`. Skips `cmp_b0291`, `cmp_b0304`, `cmp_b0305` |
| `draw_clf_compare.py` | Pure-classifier compare images with probabilities, grouped boards → `runs/cell_clf/compare_clf/` |
| `draw_cell_clf_compare.py` | Grouped compare boards from a results json |
| `score_enclosed_plan.py` | Test floor shape and skin from `labels_ext_all.jsonl` and official predicted cells |
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

## Open-balcony box detection (CMP label maps)

Scored open-balcony boxes against balcony rectangles in the CMP label PNGs. This is not the enclosure cell metric above. Ground truth is palette index 7 in `data/extended_selected/<stem>.png`: 8-connected components, tight boxes scaled onto the photo.

Photos: 123 of the 145 jpgs in `data/extended_selected`. The other 22 were left out because their balcony labels are wrong:

- `0006`, `0011`, `0012`, `0014`, `0016`, `0021`, `0043`, `0116`, `0173`, `0227`. `0006`, `0011`, `0012`, `0014`, `0016`, and `0116` were also moved out of `data/baseForBatch` into `data/baseForBatch_excluded`; the other four were never in that folder, so `data/baseForBatch` now has 17 jpgs.
- `0002`, `0003`, `0008`, `0009`, `0036`, `0044`, `0047`, `0067`, `0107`, `0127`, `0162`, `0176`, removed in a second pass.

Matching: one-to-one, greedy from the highest IoU, IoU ≥ 0.5 is a true positive. An unmatched prediction is a false positive and an unmatched ground-truth unit a false negative. This is one operating point at score ≥ 0.45, not mean average precision.

The labels sometimes split one balcony into several boxes (piers through a railing). Label boxes that sit side by side form a group: vertical overlap at least half the shorter box height, horizontal gap at most that height, transitive. In each photo every group counts either as one bounding box or as its label boxes, whichever gives more true positives; a tie keeps the label boxes. Recall uses the count of the chosen units. The vertical-overlap test keeps balconies stacked in one column from merging across floors. Counting every group as one box drops the score (raw boxes, 123 photos: precision 0.514, recall 0.704), because it joins the twin railings under paired windows that SAM3 finds one by one (`0089` goes from 8 TP to 0).

Cite the pipeline row. Pipeline boxes are what `run_balcony.py` passes to the window grid: SAM3 `balcony` boxes at score ≥ 0.45, clamped to the mask, sides under 16 px dropped, then `filter_balcony_boxes`. `filter.py` currently only drops the smaller of two overlapping boxes; its window rules are disabled in the working tree. `merge_adjacent_boxes(mode="balcony")` changed no box after that filter, and snapping does not move pixel boxes. `drop_open_under_enclosed` is not applied.

| Boxes | Pred | GT units | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAM3 raw | 738 | 634 | 478 | 260 | 156 | 0.648 | 0.754 | 0.697 |
| Mask clamp + 16 px | 699 | 634 | 457 | 242 | 177 | 0.654 | 0.721 | 0.686 |
| Pipeline (+ overlap filter) | 612 | 633 | 432 | 180 | 201 | 0.706 | 0.682 | 0.694 |

Mask clamping loses 21 TP (shrunk boxes fall under IoU 0.5). The overlap filter removes 62 FP and 25 TP. Five groups were counted as one box on the raw boxes (`0095`, `0113`, `0121`, `0130`, `0144`, one TP each).

### IoU ≥ 0.3 next to 0.5 (paper table)

Same protocol and the same 123 photos; only the IoU threshold changes. The group rule picks its best split per threshold, so GT units shift slightly. The paper reports both thresholds. The mask-clamp rows stay out of the paper table.

| Boxes | IoU | Pred | GT units | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAM3 raw | ≥ 0.5 | 738 | 634 | 478 | 260 | 156 | 0.648 | 0.754 | 0.697 |
| SAM3 raw | ≥ 0.3 | 738 | 631 | 523 | 215 | 108 | 0.709 | 0.829 | 0.764 |
| Mask clamp + 16 px | ≥ 0.5 | 699 | 634 | 457 | 242 | 177 | 0.654 | 0.721 | 0.686 |
| Mask clamp + 16 px | ≥ 0.3 | 699 | 631 | 506 | 193 | 125 | 0.724 | 0.802 | 0.761 |
| Pipeline (+ overlap filter) | ≥ 0.5 | 612 | 633 | 432 | 180 | 201 | 0.706 | 0.682 | 0.694 |
| Pipeline (+ overlap filter) | ≥ 0.3 | 612 | 629 | 491 | 121 | 138 | 0.802 | 0.781 | 0.791 |

From 0.5 to 0.3 the pipeline gains 59 TP (F1 0.694 → 0.791). Most of the misses at 0.5 are box extent, not missed balconies.

Example, `cmp_x0010`: 4 GT boxes and 4 SAM3 boxes, all four balconies found, but only 2 pass at IoU 0.5.

- The middle balcony's SAM3 box covers only the railing (y 374–444). The label runs down to y 503. IoU 0.48.
- The roof balustrade's SAM3 box reaches the cornice (y 80). The label is a thin strip (y 6–42). IoU 0.44.
- All four pass at IoU ≥ 0.4.

Also tested, not adopted:

- IoU ≥ 0.4: pipeline 466 TP, F1 0.750.
- Center rule: horizontal center offset ≤ half the GT width plus IoU ≥ 0.4. Pipeline 465 TP, F1 0.748. The center test removes 1 TP compared with plain IoU 0.4, so it adds nothing. As a custom rule written after looking at errors, it is not used.

Paper text (draft): "We match predicted and ground-truth balcony boxes one-to-one in descending IoU order and report results at IoU ≥ 0.5 and IoU ≥ 0.3, since CMP balcony annotations vary in extent (railing only vs. including the slab)."

The IoU 0.3 rows come from `score_sam3_either_gt.score_image(..., iou=0.3)` on `extended_selected_sam3_stage_boxes.json`. To rerun without overwriting the cited json: `score_sam3_pipeline_stages.py --iou 0.3 --out runs/batch/_rows/extended_selected_sam3_pipeline_stages_iou30.json`.

Same photos with label boxes as drawn (no grouping): raw 473/738 precision 0.641, 473/640 recall 0.739; pipeline 426/612 precision 0.696, 426/640 recall 0.666.

Files:

- Per-image sheet, one tab for pipeline boxes and one for raw: `runs/batch/_rows/extended_selected_sam3_pipeline_123.xlsx`.
- Scores per stage and image: `runs/batch/_rows/extended_selected_sam3_pipeline_stages.json`. Boxes per stage, so rescoring skips SAM3: `runs/batch/_rows/extended_selected_sam3_stage_boxes.json`.
- Raw boxes only, 125 photos: `runs/batch/_rows/extended_selected_sam3_either_gt.json` and `_either_gt_123.json` (`score_sam3_either_gt.py`, reads the box cache `extended_selected_sam3_boxes.json`).

```
C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_pipeline/score_sam3_pipeline_stages.py --device cuda
```

Older scores, not cited:

- 135 photos (only the first ten removed), raw boxes, label boxes as drawn, IoU 0.5: TP 488, precision 0.616, recall 0.701, F1 0.656 (`runs/batch/_rows/extended_selected_sam3_inside.json`).
- 60%-of-the-prediction rule: a predicted box is a true positive when 60% of its area lies inside a label box, and several boxes can match one balcony. 135 photos: precision 564/792 = 0.712, recall 520/696 = 0.747. 125 photos: 0.727 / 0.772 (`_inside.xlsx`, `_inside_125.xlsx`). A box much smaller than the balcony passes, so it is not a detection score.
- Cover of ground truth (box covers ≥ 60% of the balcony rectangle), 135 photos: recall 508/696, precision 507/792.

Analysis strips: `runs/batch/_rows/extended_selected_sam3_analysis/<stem>.jpg`. Left is the CMP label map; right are SAM3 raw, mask clamp + 16 px, and the overlap filter. Photo | label pairs for all 145 stems: `runs/extended_selected_pairs/`.

`balcony_pipeline/score_sam3_merged_gt.py` was rewritten by another session (its own iterative merge, label dir `data/extended`). Its `extended_selected_sam3_merged_gt.json` is not the score above.

## Open-balcony structure (projecting / juliet)

New label axis for open balconies: `projecting` (slab out from the wall), `juliet` (railing, almost no floor), `rooftop` (railing on the roof or setback below, which is the floor). Plan: label train crops → structure head on the railing classifier → pipeline vote → compiler geometry for juliet and rooftop → re-review cmp_x for rooftop and score once. Only labeling setup is done.

- Train crops had no structure labels. cmp_x (`crops_ext_xml`, `crops_pipeline_ext`) is test; never train on it.
- `label_store.py` now stores `"projecting"` explicitly. Null means not labeled. cmp_x files saved projecting as null, so cmp_x readers keep treating null as projecting.
- Juliet has no floor: `floor_shape` is null on juliet rows (the 162 temporary `rectangle` labels were cleared on 2026-10-28, backup `runs/balcony_clf/labels.jsonl.bak_before_juliet_floor`). `label_ui.py` disables the floor row for juliet and `queue_stems` does not ask for a floor (`floor_required`). Training ignores a null floor.
- 2026-10-28 structure head: `RailingClassifier.structure_head` (projecting / juliet / rooftop; samples are 7-tuples, `structure_target`). `train.py --extend-split --out checkpoints/railing_structure.pt` (all heads, 20 epochs; log `runs/balcony_clf/train_structure.log`). `--extend-split` keeps old folds by stem and splits only new rows, stratified with structure (old file `split.before_extend.json`); now train 830 / val 106 / test 105, structure train 72/130/3, val 9/16/2, test 9/16/1. Internal test fold (`evaluate.py --split test --include-synth`), railing_best -> railing_structure: kind 0.866 -> 0.856, material 0.859 -> 0.906, floor 0.941 -> 0.918, enclosure 0.390 -> 1.000, skin 0.875 -> 0.875; structure 0.808 (n=26: projecting 8/9, juliet 13/16, rooftop 0/1). Rooftop is not learned (3 train crops). `railing_best.pt` is unchanged and is still the pipeline default. `test_train_init.test_railing_best_has_no_skin_head` was already failing (railing_best has had a skin head since Oct 19).
- 2026-10-29 rooftop removed from the structure vocabulary (user): `LABEL_STRUCTURES = ("projecting", "juliet")`. Rooftop is decided from the facade (no window above the balcony, see `no_window_above` in `balcony_pipeline/filter.py`), not by the classifier. The 6 rooftop rows in `labels.jsonl` were relabeled projecting (backup `labels.jsonl.bak_before_drop_rooftop`); the F key is gone from the label UIs. The 3-class model is kept as `checkpoints/railing_structure_rooftop.pt` (log `train_structure_rooftop.log`). `checkpoints/railing_structure.pt` is now 2-class, same folds (structure train 75/130, val 11/16, test 10/16). Internal test: kind 0.907, material 0.906, floor 0.953, enclosure 0.990, skin 0.875, structure 0.962 (25/26). On cmp_x XML crops (`crops_ext_xml`, 737, null structure = projecting): kind macro F1 0.482, material 0.884, floor 0.586, structure acc 0.783 / macro F1 0.737 (juliet precision 0.532). Provisional on pipeline crops (`crops_pipeline_ext`, 512 prefilled, not reviewed): kind 0.548, material 0.853, floor 0.680, structure acc 0.805 / macro F1 0.654. Scripts: scratchpad `eval_ext_xml.py`, `eval_pipeline_prefill.py` (not in the repo).
- `balcony_train/export_train_pipeline_crops.py`: pipeline boxes (same as `export_pipeline_crops.py`) on 378 `data/base_jpg` photos, cache `runs/batch/_rows/base_jpg_sam3_stage_boxes.json`. Old cmp_b crops (any folder under `crops/`, `_Archive` included) are found in their photo by template match (921/923; the two misses are `cmp_b0004_type_00_unit_001/002`). 997 boxes: new 255 (70 photos), check 5, duplicate 737. Duplicates by best match: `_Archive` 435, `open_work` 242, `solid` 37, `surface_panel` 23. The 255 new crops were copied to `crops/unlabeled/<stem>_pbal_<NNN>.png`. The 5 checks were not (IoU 0.31–0.43, partial railing overlaps); see `runs/balcony_clf/crops_pipeline_train/dup_board.jpg` and pass `--commit --accept <names>` to add them.
- Only `unlabeled/` and the kind folders are training data; `_Archive` (448 open cmp_b crops) is not. The active open cmp_b set is about 380 crops.
- 2026-10-30 more juliet candidates (user: pipeline crops over label-map crops).
  - Box detection is unchanged since the dry run: same SAM3 cache, and the manifest's `filter_sha256` equals today's `filter.py`. Rerunning the pipeline on data/base adds no boxes.
  - Took the 435 dry-run "duplicate" pipeline boxes whose best match is an `_Archive` crop (all IoU ≥ 0.5). Each was re-checked by template match against every active cmp_b crop (kind folders + `unlabeled`, all located).
  - 23 have an active twin at IoU ≥ 0.3 (0.93–0.99) and were dropped. The other 412 were copied to `crops/unlabeled/<stem>_pbal_<NNN>.png` with `--commit --accept`.
  - Selection script and board: session scratchpad `select_archive_dups.py`, `archive_dups/keep_board.jpg`. Not in the repo.
  - `commit()` now skips crops already present anywhere under `crops/` instead of raising.
  - Labels:
    - 375 of the 412 were prefilled from their archived twin: open_work + material (326 metal / 49 masonry), enclosure open, floor and structure empty. These now train kind and material.
    - The other 37 have unlabeled twins.
    - Backup: `labels.before_archive_pbal.jsonl`.
  - After the copy, the cmp_b structure queue (`label_ui.py --structure-pass`) is 491. Real kind (no flux2, no LSAA) is open_work 890 / solid 68 / surface 30.
- 2026-10-30 real-only retrain with LSAA (user plan: no flux2, add LSAA).
  - **Merge.** The 818 LSAA crops (`crops_lsaa/`, from lsaa-dataset `export_balcony_labels.py`) were copied into `crops/{kind,unlabeled}` and their rows appended to `labels.jsonl`. Backups: `labels.before_lsaa.jsonl`, `split.before_lsaa.json`.
  - **Data used** (2,253 loaded, 1,858 without flux2):
    - kind 1002 / 347 / 373;
    - material metal 688 / masonry 314;
    - floor rectangle 638 / circle 245 / triangle 30;
    - structure projecting 413 / juliet 219.
  - **`train.py --exclude-prefix`** (repeatable): drops matching file names from all folds after `indices_from_record`. Class weights follow, and the prefixes are stored in the checkpoint. Test: `test_train_init.DropPrefixedTests`.
  - **Split.** `--extend-split` kept all 1,041 old rows in their folds (0 moved) and split 1,212 new rows (LSAA + the 412 new pbal). Without flux2: train 1484 / val 184 / test 190. The split is per crop, so LSAA facades and pbal photos leak across folds.
  - **Checkpoints** (all heads, 20 epochs, defaults, best epoch on val). `railing_best.pt` is still the pipeline default.
    - A `checkpoints/railing_real.pt` (sha256 b2df33d8…): `--exclude-prefix flux2_ --exclude-prefix lsaa_`, train 828. Best val 0.926. Log `train_real.log`.
    - B `checkpoints/railing_real_lsaa.pt` (56b306ca…): `--exclude-prefix flux2_`. Best val 0.953. Log `train_real_lsaa.log`.
  - **Choice on val** (real crops, n=184; scratchpad `eval_val_real.py`). Macro F1, A / B / railing_best / railing_structure:
    - kind 0.602 / **0.897** / 0.767 / 0.791;
    - material 0.919 / **0.965** / 0.964 / 0.929;
    - floor 0.602 / **0.739** / 0.604 / 0.604;
    - structure 0.926 / **0.942** / – / 0.777. B juliet P 0.94 / R 0.89.
    - B was chosen as the candidate. No juliet threshold was needed.
    - Supplementary: on the cmp_b-only part of val (n=102), A beats B on kind (0.892 vs 0.795; B's solid recall is 0.57 with few solids).
  - **cmp_x XML crops, scored once** (`crops_ext_xml`, 737, null structure = projecting; scratchpad `eval_ext_xml.py`, output `ext_xml_eval/real_lsaa_20261030.txt`). Macro F1, railing_best / railing_structure / A / B:

    | Head | railing_best | railing_structure | A | B |
    |---|---|---|---|---|
    | kind | 0.425 | 0.482 | 0.461 | 0.456 |
    | material | 0.906 | 0.884 | 0.900 | 0.871 |
    | floor | 0.570 | 0.586 | 0.634 | 0.516 |
    | circle F1 | 0.577 | 0.573 | 0.571 | 0.613 |
    | structure | – | 0.737 | 0.785 | 0.770 |
    | juliet P / R | – | 0.532 / 0.761 | 0.667 / 0.682 | 0.695 / 0.597 |
    | open_work recall | – | 0.882 | 0.858 | 0.848 |
    | solid F1 | – | 0.465 | 0.411 | 0.322 |

    - B's floor macro F1 drops because of triangle (0/3; B predicts 12 triangles).
    - **The val choice (B) does not carry over to cmp_x on kind, material or floor.** Both real-only models improve structure over `railing_structure`.
    - No default changed.
  - **Provisional: cmp_x pipeline crops** (`crops_pipeline_ext`, 512, prefill labels not reviewed; scratchpad `eval_pipeline_prefill.py`, output `pipeline_ext_eval/real_lsaa_20261030.txt`). Macro F1, railing_structure / A / B:
    - kind 0.548 / 0.535 / 0.542;
    - material 0.853 / 0.886 / 0.887;
    - floor 0.680 / 0.691 / 0.579 (B: triangle 0/3, circle 69/87);
    - structure 0.654 / 0.728 / 0.736; juliet P/R 0.36/0.51, 0.50/0.58, 0.59/0.50.
- Label UI: `label_ui.py` has P/J/F structure (shown only for open; P is skin when enclosed). `--structure-pass` queues open `cmp_b*` crops without structure. `label_pipeline_crops_ui.py --train [--structure-pass]` is the same on the training crops, with the photo context from `crops_pipeline_train/context/` (new and old crops). `xml_label_ui.py` has F rooftop.
- Next: label, check the rooftop count, then add the structure head (`--heads-only structure`, new checkpoint, keep `railing_best.pt`).
- 2026-10-29 structure in the pipeline (user asked the code to match the paper text). `run_balcony.py` loads `railing_best.pt` as before and copies only the structure head from `checkpoints/railing_structure.pt` (`--structure-ckpt`; `--no-structure` keeps every open balcony projecting). Per type, `vote_structure` votes projecting / juliet only, juliet on a strict majority. Rooftop is a placement flag, not a type value (user, 2026-10-29): a balcony on the top floor of the window grid with no window above it in its bays (`no_window_above` from `filter.py`, `mark_rooftop_units`) gets `"structure": "rooftop"` on its `layout.balconies` record; `normalize_facade_spec` copies it to `balcony_placement` and `balcony_layout._frame` uses it over the type's structure. The balcony keeps its type. Two earlier same-day versions (rooftop split into its own type; rooftop as a third vote option) were removed. BDSL `structure` gains `juliet` and `rooftop`. Compiler: juliet = front railing 0.15 m in front of the wall from the sill, no slab, no stacking; rooftop = front railing on the top edge of its storey (back face on the wall plane), no slab, except when an enclosed stack below shares its slab as roof (`carries_stack`), where it compiles as projecting. `cmp_x0005` smoke run: 2/24 open units rooftop (the railings on top of the corner oriel towers, which carry the stack), no juliet; a forced juliet/rooftop copy rendered as expected (scratchpad, not cited). Not scored: no structure or floor numbers changed in the cited tables.

## Open-balcony attributes on pipeline crops (test set, in progress)

The earlier attribute score used crops cut from the CMP segmentation (`crops_ext_xml`). That isolates the classifier, but the pipeline never sees those crops. This test labels the crops the pipeline actually cuts.

- `balcony_train/export_pipeline_crops.py`: pipeline boxes (SAM3 `balcony` ≥ 0.45, mask clamp, 16 px, `filter_balcony_boxes`), `photo.crop(box)` with no margin, as `run_balcony.py` feeds the classifier. All 145 photos of `data/extended_selected`, the 22 wrong-segmentation photos included, because hand labels do not depend on the segmentation.
  - Cache: `runs/batch/_rows/extended_selected_sam3_stage_boxes_145.json`. It is the 123-photo cache plus SAM3 on the other 22. Re-filtering the cached `sized` boxes with the current `filter.py` matched the cached boxes on all 123 photos.
  - Output in `runs/balcony_clf/crops_pipeline_ext/`: 710 crops in `unlabeled/` (612 on the 123 photos), `context/` (photo with the box drawn), `manifest.jsonl` (box and `filter.py` sha256).
  - `prefill.jsonl`: 512 crops matched by IoU ≥ 0.3 to an earlier `crops_ext_xml` label (median IoU 0.52). These are starting values from human labels, never model output.
- Labeling: `balcony_train/label_pipeline_crops_ui.py` (no flag: this test set). Its docstring holds the rules:
  - Label the real balcony in the red box, not only what the crop shows.
  - `X` = not a balcony, stored in `not_balcony.jsonl`. These are left out of attribute accuracy and counted separately.
  - Every crop is reviewed. A crop counts as reviewed once it is saved to `labels.jsonl` or marked X.
- Not started on 2026-10-29: no `labels.jsonl` yet, and the scorer `eval_pipeline_crops.py` is not written.
- Scoring plan:
  - Use the frozen `checkpoints/railing_best.pt` (record its sha256) and `predict_full` per crop.
  - Report kind, material (open_work), floor (triangle counts as rectangle) and the `not_balcony` count, for 145 photos and the 123-photo subset.
  - Report how many prefilled labels were changed in review.
  - Score once, after every crop is reviewed.

## Base + LSAA 100-photo test set (2026-10-30, in progress)

The user's plan has these targets:
- 100 photos from Base (cmp_b) + LSAA;
- ≥20 pipeline crops per class value;
- 30–40 photos with oriels;
- triangle counts as rectangle.

Step 1, done: the Base part has ≥10 of each class and 10 oriel photos.

- Script: `balcony_train/select_test_base.py` (tests `test_select_test_base.py`).
  - Counts open crops per photo from `labels.jsonl`, using only crop names that start with the stem.
  - Oriel photos are the rows of `runs/enclosed_grid/labels_train.jsonl` that have cells.
  - Selection: 10 oriel photos first, then a greedy cover to target 12 (10 + margin, rare classes weighted 1 / Base total), then pruning.
  - Output in `runs/test100/`: `base_selected.json`, `heldout_stems.txt`, `board.jpg`.
- Chosen (20). The list is `runs/test100/heldout_stems.txt`; the photos are copied to `runs/test100/photos/`.
  - 12 from the greedy run with `--exclude cmp_b0008 cmp_b0305 cmp_b0316` (user):
    - oriel: `0007 0010 0041 0100 0155 0157 0162 0217 0235 0303`;
    - non-oriel: `0017 0317`.
    - The greedy run picked `0015`. The user asked to swap it out (it has triangle floors and half-enclosed crops), and it was replaced by hand with `0162`: 4 crops, fully labeled, clear oriels.
  - 8 hand-picked "standard" photos (user: few training crops, standard facades, no triangle or half-enclosed): `0062 0159 0161 0168 0171 0178 0310 0340`.
    - Each has 3–6 open_work crops: no surface_panel, solid or circle, no triangle, no enclosure. These keep scarce training crops in training.
    - Fire-escape photos (`0052 0056 0057`) were skipped.
  - Recounted with `select_test_base.py --stems <list>`, which now rewrites the outputs for a given list.
    - On current labels: open_work 119, surface_panel 23, solid 24, metal 80, masonry 39, rectangle 76, circle 14, projecting 86, juliet 25.
    - Still unlabeled: floor 51, structure 55.
- Step 2, done: the LSAA part, 80 facades. Script: `balcony_train/select_test_lsaa.py`.
  - Pool: the 519 facades with a FiftyOne tag in `crops_lsaa` (the 2026-10-29 annotation, 818 crops).
  - At most one facade per building (OSM `relation…` / `way…` id; user).
  - Greedy on the gain in Σ log(1 + class count), divided by that building's LSAA training crops. This keeps classes balanced and the hold-out small.
  - Ties go to facades whose tags cover LSAA's `num_balconies_filtered`.
  - Two earlier variants (same day) were dropped:
    - target 30 per class: solid overshot to 61;
    - plain log gain: 329 held-out crops.
  - Result: 80 facades from 80 buildings.
    - Tagged counts: open_work 40, surface_panel 26, solid 26, metal 36, masonry 4, rectangle 47, circle 45.
    - Masonry is capped: only 4 annotated facades have it.
    - Countries: France 37, UK 24, Italy 5, Austria 4, Belgium 4, USA 4, Netherlands 2.
  - Hold-out: every LSAA crop of the 80 buildings, 93 crops. Prefixes are in `runs/test100/lsaa_heldout_prefixes.txt` (for `--exclude-prefix`).
  - Outputs: `lsaa_selected.json`, `lsaa_facades.txt`, `board_lsaa.jpg`. The images are copied into `runs/test100/photos/`, which now holds all 100.
  - Base + LSAA tagged totals: open_work 159, surface_panel 49, solid 50, metal 116, masonry 43, rectangle 123, circle 59, projecting 86, juliet 25. Every class is ≥20, but juliet only has its 25 Base crops.
  - **Revised the same day (user: frontal photos, no complex facades). This supersedes the numbers above.**
    - The user flagged 18 of the facades they looked at; the other 17 they saw are kept by hand (`--keep`).
    - New picks need `|view_angle| ≤ 30` (`--max-view-angle 30`).
    - The agent rejected 50 more by eye over six rounds: ornate or gabled, curved bays, staggered or deep loggias, occluded (bus, wall, tree), two buildings in one image.
    - All 68 rejected building ids are in `runs/test100/lsaa_rejected_buildings.txt`.
    - Final 80 (`lsaa_facades.txt`, 80 buildings), tagged counts: open_work 21, surface_panel 51, solid 40, metal 18, masonry 3, rectangle 37, circle 23. Fully tagged 27.
    - Hold-out: 142 LSAA crops.
    - Base + LSAA tagged totals: open_work 140, surface_panel 74, solid 64, metal 98, masonry 42, rectangle 113, circle 37, projecting 86, juliet 25.
    - `--keep` refuses a bare building id that matches several facades; pass the facade name instead.
  - Caveats:
    - Only 27 of the 80 facades are fully tagged (33 before the revision).
- Step 3, label UI (2026-10-30).
  - `balcony_train/export_test100_crops.py` cuts pipeline crops for every photo in `runs/test100/photos`, using the same boxes as `export_pipeline_crops.py`.
    - cmp_b photos reuse the `base_jpg` stage cache, so their crop names equal the training `*_pbal_*` names. SAM3 ran on the LSAA facades.
    - Cache: `runs/batch/_rows/test100_sam3_stage_boxes.json`.
  - Prefill comes from hand labels only:
    - cmp_b: the training label of the crop, else of its duplicate's old crop.
    - LSAA: the FiftyOne crop whose box, shifted by `balconies_ctx.csv` `ctx_x0/ctx_y0`, has the best IoU ≥ 0.3.
  - Output `runs/test100/crops/`:
    - 773 crops: cmp_b 122, LSAA 651.
    - Prefilled 187: cmp_b 118, LSAA 69 of 116 tags on these facades.
    - 7 photos have no crop.
  - UI: `label_pipeline_crops_ui.py --dir runs/test100/crops`.
    - Open-only mode (user: enclosure and triangle are not used): no enclosure row, every save is `open`, no T key or triangle floor; a triangle prefill shows as rectangle.
    - On by default for this dir (`--open-only` elsewhere). Implemented as `LabelApp(open_only=...)` in `label_ui.py`; other label UIs are unchanged.
    - **Cancel** button (same as X, no dialog): removes all labels of the crop and adds it to `not_balcony.jsonl`; U undoes it. For crops wrongly saved as balconies (reach saved crops with `--review-all`). First pass done 2026-10-30: 683 of 773 crops saved, none marked X.
  - Labeling finished (user, 2026-10-30): 676 labeled, 15 not_balcony, 82 never saved (24 LSAA photos, up to 14 per photo).
    - Labeled crops, all `open`: open_work 343 (metal 276, masonry 67), surface_panel 236, solid 97; projecting 384 (rectangle 318, circle 66), juliet 292.
    - cmp_b 116 / LSAA 560. Every class value has ≥ 66 crops.
  - **Held-out retrain** (2026-10-30): `checkpoints/railing_test100_heldout.pt` (sha256 ce962bc9d426…).
    - Run: `train.py --exclude-prefix flux2_` plus the 96 prefixes in `runs/test100/train_exclude_prefixes.txt`: 20 cmp_b stems, and 76 LSAA buildings taken from the current `photos/`.
      - It drops 280 training crops (cmp_b 180, LSAA 100).
      - Otherwise as checkpoint B. Best val 0.954. Log `runs/balcony_clf/train_test100_heldout.log`.
    - The 43 number-only rows (`0003.png` …) have no image files and cannot be traced to a photo.
  - **Test scores** (`balcony_train/evaluate_test100.py --by-source`, output `runs/test100/eval_test100_20261030.md`).
    - 676 crops, pooled; a triangle prediction counts as rectangle.
    - Held-out checkpoint, accuracy / macro F1:
      - kind 0.788 / 0.778: open_work R 0.726, 71 crops predicted surface_panel;
      - material 0.915 / 0.878: masonry P 0.721;
      - floor 0.938 / 0.895: circle F1 0.829;
      - structure 0.636 / 0.531: juliet R 55/292 = 0.188.
    - LSAA has no structure labels in training, so the structure head has seen no LSAA juliet. 267 of the 292 test juliets are LSAA (R 0.150).
    - cmp_b part (n=116), supplementary: kind 0.843, material 0.933, floor 0.928, structure 0.793 macro F1.
    - `railing_real_lsaa` and `railing_best` were trained on these photos (leak). Their scores in the same file are reference only, not test scores.
  - **LSAA structure labels** (next step for juliet): `label_ui.py --structure-pass --structure-prefix lsaa_ --skip-prefixes runs/test100/train_exclude_prefixes.txt`. The new `--skip-prefixes` option hides the held-out crops. The queue is 718 of the 818 LSAA training crops; 100 are held out.
  - **LSAA juliet round** (2026-10-31, user request after viewing the test juliet errors).
    - Errors (`runs/test100/errors_juliet/`, scratchpad `juliet_errors.py`): 237 juliet predicted projecting, mostly Paris (138, iron/glass window railings) and Vienna (61, stone balustrades and ornaments under windows); 9 projecting predicted juliet.
    - **Not blind:** this round was chosen after looking at test errors. Any structure test score after retraining on it must be reported as test-informed.
    - Selection: `balcony_train/select_juliet_lsaa.py`.
      - Pool: frontal (`|view_angle|` ≤ 15), ≥ 1 balcony, no test building, not in the 519 annotated facades, one facade per building.
      - Quota: Vienna 26, Paris 18, Rome 11, Brussels 3, Bucharest 2 (user chose a mixed set of about 60). Sorted by LSAA `deco`, then balcony count.
      - Output: `runs/lsaa_juliet/{photos,facades.json}`.
    - Crops: `export_test100_crops.py --photos-dir runs/lsaa_juliet/photos --out-dir runs/lsaa_juliet/crops --cache runs/batch/_rows/lsaa_juliet_sam3_stage_boxes.json`.
      - 575 crops from 55 facades; 5 facades have none. No prefill.
      - Fixed a bug: a relative `--cache` path crashed `relative_to`.
    - Label: `label_pipeline_crops_ui.py --dir runs/lsaa_juliet/crops`, open-only by default for that dir.
    - Merged 2026-10-30 (first 145 labels, all juliet: open_work masonry 76, metal 41, solid 28) into `crops/{kind}/lsaa_<stem>_pbal_NNN.png` and `labels.jsonl` (backups `labels.before_lsaa_juliet.jsonl`, `split.before_lsaa_juliet.json`). LSAA training rows then had structure on 193 of 818 (projecting 117, juliet 76; user structure pass in progress).
    - Retrain (ran after the other session's oriel eval exited): `train.py --extend-split --exclude-prefix flux2_` plus the test prefixes.
      - Output `checkpoints/railing_test100_juliet.pt` (sha256 b2b5959e2198…). Best val 0.954. Log `train_test100_juliet.log`.
      - Structure folds: train 424 projecting / 348 juliet; val 54 / 44; test 52 / 48.
    - Test set (`evaluate_test100.py`, `runs/test100/eval_test100_juliet_20261030.md`).
      - **The structure score is test-informed, not blind.**
      - Macro F1, heldout → juliet checkpoint:
        - kind 0.778 → 0.768;
        - material 0.878 → 0.886;
        - floor 0.895 → 0.912;
        - **structure 0.531 → 0.750**: juliet R 0.188 → 0.771, P 0.859 → 0.690; projecting R 0.977 → 0.737.
      - LSAA part: structure 0.486 → 0.718. cmp_b part: 0.793 → 0.885.
      - New errors: 101 projecting predicted juliet. Solid precision falls to 0.555: 42 open_work and 31 surface_panel are predicted solid.
  - **Test set update** (2026-10-31, user changed `photos/`): 11 photos out, 11 in. Still 100 photos.
    - Out: `cmp_b0303`, Vienna `relation2139383_Fid8872`, `way100965071_Fid3026`, `way187527040_Fid945`, `way39771306`, Paris `way64039962`, `way64695046`, `way69999763`, `way82239207`, London `way626264112`, New York `way665486060`.
    - In: `cmp_x0005/0006/0126/0149`, Vienna `way177275868`, Copenhagen `relation5364728`, `way569041948`, `way88264123`, Paris `relation946705`, `way63649040`, `way63863024`.
    - Sources now: cmp_b 19, cmp_x 4, LSAA 77.
    - Grids: cmp_x copied from `runs/batch`; the 7 LSAA photos ran through the window track (`runs/test100/batch_window_new.log`, 7/7 ok).
    - Crops: `export_test100_crops.py --photos-dir runs/test100/new_photos --out-dir runs/test100/crops_new` gave 98 crops. The user labeled 61; 37 were never saved.
      - Merged into `runs/test100/crops`. Dropped photos' crops and label rows are in `crops/_dropped_20261031/`; full backup in `crops.before_20261031/`.
      - Now 653 labeled crops and 14 not_balcony.
    - `train_exclude_prefixes.txt` rebuilt from the current `photos/`: 95 prefixes (19 cmp_b, 76 LSAA buildings). Backup: `train_exclude_prefixes.before_20261031.txt`.
      - Back in training: `cmp_b0303` and the 7 LSAA buildings that left.
      - `relation2139383`, `way100965071` and `way187527040` stay excluded, because another photo of each building is still in the test set.
      - The 7 new LSAA buildings have no training crops, so `railing_test100_juliet.pt` does not leak on the new set either.
    - `oriel_labels.jsonl`: dropped photos removed, 100 → 89 rows. Backup: `oriel_labels.before_20261031.jsonl`. The 11 new photos still have no oriel labels.
  - **Retrain v2** (2026-10-31): `train.py --extend-split --exclude-prefix flux2_` plus the 95 prefixes, on the current training labels.
    - Output `checkpoints/railing_test100_v2.pt` (sha256 cfc1ee60df69…). Train 1424 / val 181 / test 180. Best val 0.947. Log `runs/balcony_clf/train_test100_v2.log`.
    - Test (`evaluate_test100.py --by-source`, `runs/test100/eval_test100_v2_20261031.md`): 653 crops.
      - Macro F1, v2 / juliet checkpoint: kind 0.831 / 0.800, material 0.843 / 0.894, floor 0.904 / 0.910, structure 0.750 / 0.759.
      - Accuracy, v2 / juliet: kind 0.844 / 0.815, material 0.885 / 0.927, floor 0.937 / 0.942, structure 0.757 / 0.761.
      - The structure score is still test-informed (juliet round).
      - `--by-source` counts the 36 cmp_x crops as `lsaa`, since only `cmp_b` is split out.
    - Juliet checkpoint, old set (676 crops) → new set (653 crops), macro F1: kind 0.768 → 0.800, material 0.886 → 0.894, floor 0.912 → 0.910, structure 0.750 → 0.759. Same model, so the change comes from the swapped photos.
      - Excel for researchers: `runs/test100/balcony_type_eval_test100.xlsx`: juliet checkpoint on the new set, pooled; confusion matrices, then P/R/F1 tables as formulas (user asked for this format only, no old-set sheets). Built from the two eval .md files by scratchpad `export_xlsx.py`.
      - New set, per class: open_work R 0.746 (54 → surface_panel, 43 → solid); solid P 0.622; masonry P 0.724; circle F1 0.855; juliet P 0.726 / R 0.741, projecting P 0.790 / R 0.777.
  - **`photos/` was changed by another session before the export (with `oriel_labels.jsonl`, `batch/`).** 40 LSAA facades differ from `lsaa_facades.txt`, which is now stale.
    - 27 of the new facades are outside the 519 annotated facades.
    - Four buildings appear twice: `relation2139383`, `relation2367958`, `way100965071`, `way187527040`.
    - Not reverted; flagged to the user. The rest need every pipeline crop labeled.
    - LSAA has no structure or oriel tags. Juliet and the oriel target (30–40 photos; Base gives 10) were skipped for now (user).
- **Leakage.** These photos are still in training: all railing checkpoints, and the enclosure cell model (10 of its 31 oriel photos).
  - Before any test score, retrain without them, e.g. `train.py --exclude-prefix cmp_bNNNN_` per stem, and drop them from `labels_train.jsonl` for the cell model.
  - User: retrain later.
- **Oriel cells on the 100 photos** (2026-10-30).
  - Labels: `runs/test100/oriel_labels.jsonl`, from `enclosed_grid_ui.py --in-dir runs/test100/photos --dsl-dir runs/test100/batch --labels runs/test100/oriel_labels.jsonl --all-photos --plan`.
    - 26 photos with oriels (10 Base, 16 LSAA), 232 cells, 15 ignore cells.
    - The 20 Base rows started as copies of their `labels_train.jsonl` rows.
  - Window grids: `runs/test100/batch/`. Base DSLs were copied from `runs/batch`; the 80 LSAA photos were run through the window track only (`run.py`, log `runs/test100/batch_window.log`).
  - Script: `balcony_train/eval_test100_oriel.py`.
    - Official detector steps per photo (SAM3 seeds, DA3, OP1 volumes, cell features, gate 0.10, rect decode, four rules).
    - Caches: `runs/test100/enclosed_sam3_cache.json`, `runs/test100/da3_depth/`.
    - Output: `runs/test100/oriel_eval/` (`summary.md`, `preds_*.json`).
  - Main row: `heldout`, the cell model refit on `cells_train.npz` without the 20 Base test photos (355 train photos).
    - Per column: TP 123, FP 32, FN 109, TN 3739; P 0.794, R 0.530, F1 0.636, IoU 0.466; exact 74/100.
    - With ignore: P 0.799, R 0.531, F1 0.638.
    - Base 20: P 0.742, R 0.469, F1 0.575. LSAA 80: P 0.806, R 0.546, F1 0.651.
  - Supplementary: `official` (cached `model_rules_gate.joblib`, trained on the 20 Base photos).
    - Per column: TP 161, FP 47, FN 71; P 0.774, R 0.694, F1 0.732.
    - It is better on LSAA too (R 0.634 vs 0.546), although LSAA was in neither model's training: the held-out fit loses 10 of 31 oriel training photos.
- Next:
  - Label floor and structure on these photos' crops (`label_pipeline_crops_ui.py --train --structure-pass`).
  - Re-check with `select_test_base.py --stems runs/test100/heldout_stems.txt` (report only).
  - Then select the LSAA part.

- **Open-balcony boxes on test100** (2026-11-01). Only the 23 CMP photos have box ground truth (palette 7). LSAA balcony boxes (`balcony_detection.json`, `balconies_ctx.csv`) are detector output, so the 77 LSAA photos have no box score.
  - Same protocol as "Open-balcony box detection" (one-to-one greedy, group rule via `score_image`, `either`). Boxes from the cache `runs/batch/_rows/test100_sam3_stage_boxes.json`, no SAM3 rerun. Labels `data/base` / `data/extended`. Scratchpad script `score_test100_boxes.py`.
  - 22 photos (`cmp_x0006` left out, wrong label as before), pipeline (`filtered`): IoU ≥ 0.5 pred 141 / GT 134, TP 103, FP 38, FN 31, P 0.730, R 0.769, F1 0.749; IoU ≥ 0.3 TP 114, P 0.809, R 0.857, F1 0.832. Raw: F1 0.725 / 0.784.
  - All 23: pipeline F1 0.746 (0.5) / 0.826 (0.3). cmp_b 19: 0.753 / 0.838.
  - Best IoU per GT box (pipeline, 22 photos): mean 0.621, median 0.739; 18 of 134 below 0.1 (no box at all).
  - Worst photos: `cmp_b0171` (TP 0/8: SAM3 boxes a whole row of balustrades as one strip, the label boxes each balustrade under a window), `cmp_b0157` (0/2, box extent), `cmp_b0062` (1/5). `cmp_b0235`: 11 boxes for 4 GT (window ornaments, a row strip).
  - **Supplementary, not ground truth:** the 77 LSAA photos scored against LSAA `annotations/balcony/balcony_detection.json` (the LSAA paper's detector output, box `[x, y, w, h]` in facade pixels; scratchpad `score_test100_lsaa.py`). All 77 photos have LSAA boxes; 8 have no pipeline box.
    - vs `balcony_all` (475 boxes), pipeline: IoU ≥ 0.5 pred 623, TP 361, FP 262, FN 114, P 0.579, R 0.760, F1 0.658; IoU ≥ 0.3 P 0.626, R 0.821, F1 0.710. Raw: F1 0.640 / 0.667.
    - vs `balcony_filtered` (251 boxes): pipeline R 0.845 / 0.912, P 0.340 / 0.368.
    - The board (12 photos) shows the LSAA boxes are incomplete and partly wrong: modern projecting balconies are often unboxed, while rooftop dormers, a window and a shop sign are boxed. Precision against them is not meaningful. Using them as test ground truth needs a human pass (add missing, delete wrong).
  - Side-by-side images for all 100 photos: `runs/test100/compare_boxes/<stem>.jpg` (left: CMP label boxes or LSAA boxes; right: pipeline). LSAA boxes alone: `runs/test100/lsaa_boxes_vis/`.
- **Open-balcony box ground truth, test100** (2026-11-01, in progress). Tool `balcony_train/gt_box_ui.py` (store `gt_box_store.py`, test `test_gt_box_store.py`): draw / move / resize / delete boxes, saves `runs/test100/balcony_gt_boxes.jsonl` (one row per reviewed photo, photo pixels).
  - Prefill (user choice): cmp_b / cmp_x from the CMP label boxes; LSAA from the pipeline `filtered` boxes. The LSAA labels therefore start from our own output; say so when citing the LSAA score.
  - The user also edits CMP boxes. That is a revised ground truth, not the CMP labels: keep the original-label score (above) next to any score on the revised boxes.
  - Labeling rule, fixed before labeling: a balcony has a railing / balustrade, or a solid parapet that projects from the wall (`surface_panel`, `solid`). Window sills, ledges, flower boxes and flush decorative panels are not balconies. Railings split by piers get one box per section; box covers railing plus slab.
  - Ignore regions (user, 2026-11-01; key I / T in the tool, saved as `"ignore"`): balconies too small and cut by the image border, or hidden by glare. Scoring rule fixed before any scoring: match balcony boxes as usual; an unmatched prediction with ≥ 50% of its own area inside one ignore region is dropped (neither TP nor FP); ignore regions never count as FN.
  - Labeling done 2026-11-01 (16:00–16:28): 100/100 photos saved. CMP 23: 11 edited, 150 → 159 boxes, 4 ignore. LSAA 77: 36 edited, 623 pipeline boxes → 539, 24 ignore, 11 photos with no balcony.
  - **Score** (`balcony_pipeline/score_test100_gt_boxes.py` → `runs/test100/balcony_box_score.json`). Pipeline (`filtered`) boxes, group rule as in the CMP score (plain matching gives the same or ±1 TP):

    | subset | IoU | pred | GT | TP | FP | FN | ignored pred | P | R | F1 |
    |---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
    | all 100 | ≥ 0.5 | 736 | 695 | 606 | 130 | 89 | 48 | 0.823 | 0.872 | 0.847 |
    | all 100 | ≥ 0.3 | 736 | 694 | 619 | 117 | 75 | 48 | 0.841 | 0.892 | 0.866 |
    | CMP 23 | ≥ 0.5 | 158 | 159 | 119 | 39 | 40 | 3 | 0.753 | 0.748 | 0.751 |
    | CMP 23 | ≥ 0.3 | 158 | 158 | 130 | 28 | 28 | 3 | 0.823 | 0.823 | 0.823 |
    | LSAA 77 | ≥ 0.5 | 578 | 536 | 487 | 91 | 49 | 45 | 0.843 | 0.909 | 0.874 |
    | LSAA 77 | ≥ 0.3 | 578 | 536 | 489 | 89 | 47 | 45 | 0.846 | 0.912 | 0.878 |

    Raw SAM3 boxes, all 100: F1 0.806 (0.5) / 0.823 (0.3).
    Table updated after a GT fix (user, 2026-11-01): `France_Paris_way63863024` lost three boxes on the two oriels (left oriel railing and its bracketed underside, right oriel whole bay); now 6 balconies. Backup `runs/test100/balcony_gt_boxes.before_paris_fix.jsonl`. Other rows of this section are from before the fix and move by at most 3 boxes.
  - **LSAA is not an independent score.** 480 of the 539 LSAA GT boxes are pipeline boxes left exactly as they were (41 of 77 photos untouched); median best IoU per GT box is 1.0. It measures how much the annotator had to correct, and favours the pipeline. Cite CMP (labels independent of the pipeline) as the test score. CMP revised GT: 141 of 159 boxes are the original CMP label boxes; F1 0.751 vs 0.746 on the original labels (23 photos), so the revision barely moves the score.
  - Per image (`runs/test100/balcony_box_score_per_image.csv`, pipeline, IoU 0.5, mean over photos where defined): CMP edited 11 P 0.690 R 0.636; CMP unedited 12 P 0.842 R 0.837; LSAA edited 36 P 0.683 R 0.786; LSAA unedited 41 P = R = 1.000 by construction. 56 of 100 photos are perfect, 41 of them unedited LSAA. Worst photos: whole rows of Vienna balustrades boxed as one strip (`relation2445885` 0/8, `cmp_b0171` 0/7, `relation2139386` and `relation2367958` 1/8), many false boxes on ornament (`relation2148482` 25 boxes for 5, `way177275868` 12 boxes, no balcony).
  - **Post-rules** (user, 2026-11-01; ideas from viewing test100 errors). AGENTS.md line "Do not change ground truth or invent matching rules after looking at errors" removed at the user's request (test set is fixed). Code `balcony_pipeline/postfilter.py` (parameters fixed when written, not tuned), scored by `score_test100_postrules.py` → `runs/test100/balcony_postrules_score.json`. Classifier labels `railing_test100_v2.pt` (cache `runs/test100/railing_preds_v2.json`); oriel cells `runs/oriel_grid_compare/grid2_test100/preds_heldout.json` (89 photos). **Wired into `run_balcony.py` (2026-11-01): `side_wall` (rule 1, `--no-side-wall`) and `masonry_juliet_split` (`--no-juliet-split`), both default on. The other rules in `postfilter.py` are not wired.**
    - Rules: off_facade (< 50% of width over the window-grid x-extent, or touching the border and < 0.5 × median box area); oriel (open box on a predicted oriel cell, same floor + bay); no_window_near (no window centred over the box with its bottom ≤ 0.5 window height above the box top, top floor exempt); solid_juliet (classifier solid + juliet); juliet_split (juliet box over ≥ 2 such windows → one box per window).
    - IoU 0.5, no ignore, all 100 (baseline P 0.773 R 0.872 F1 0.819): only off_facade 0.823; only oriel 0.779; only no_window_near 0.788; only solid_juliet 0.822; only juliet_split 0.809; all five 0.750. off_facade + solid_juliet 0.824 (with ignore 0.831 vs baseline 0.847).
    - Why: oriel drops 111 boxes, 73 of them real balconies (oriel cells P ≈ 0.49 on this grid). no_window_near drops 108, 66 real; 53 of those have no detected window above (window track misses windows). off_facade 19 real of 53; solid_juliet 22 real of 58. juliet_split: 28 of 79 pieces match a GT box.
  - **Default is now without ignore regions** (user, 2026-11-01). `score_test100_gt_boxes.py` ignores the `ignore` boxes unless `--use-ignore` (supplementary, `runs/test100/balcony_box_score_ignore.json`). Main test100 row, pipeline, IoU 0.5 / 0.3: all 100 P 0.773 R 0.872 F1 0.819 / 0.838; CMP 23 F1 0.744 / 0.815; LSAA 77 F1 0.840 / 0.844. With ignore: 0.847 / 0.866. The tables above that say "ignored pred" are the with-ignore numbers.
  - **User's three rules, each alone** (`score_test100_user_rules.py` → `runs/test100/balcony_user_rules_score.json`, images `runs/test100/compare_rules/<stem>.jpg`: GT | baseline | rule 1 | rule 2 | rule 3, removed boxes blue dashed). Rule 1 `edge_depth` as the user worded it: small box (< 0.5 × photo median area) within 3% of the border, or median DA3 depth > 20% off the main wall (median depth over the window-grid extent). DA3 depth for the 11 photos added on 2026-10-31 was generated into `runs/test100/da3_depth/`. No ignore, all 100:

    | variant | IoU | pred | TP | FP | FN | P | R | F1 | removed |
    |---|---|---:|---:|---:|---:|---:|---:|---:|---:|
    | baseline | 0.5 | 784 | 606 | 178 | 89 | 0.773 | 0.872 | 0.819 | 0 |
    | 1 edge / depth | 0.5 | 753 | 598 | 155 | 97 | 0.794 | 0.860 | 0.826 | 31 |
    | 2 solid juliet | 0.5 | 726 | 584 | 142 | 111 | 0.804 | 0.840 | 0.822 | 58 |
    | 3 no window above | 0.5 | 676 | 540 | 136 | 155 | 0.799 | 0.777 | 0.788 | 108 |
    | baseline | 0.3 | 784 | 619 | 165 | 75 | 0.790 | 0.892 | 0.838 | 0 |
    | 1 edge / depth | 0.3 | 753 | 611 | 142 | 83 | 0.811 | 0.880 | 0.845 | 31 |
    | 2 solid juliet | 0.3 | 726 | 596 | 130 | 98 | 0.821 | 0.859 | 0.839 | 58 |
    | 3 no window above | 0.3 | 676 | 553 | 123 | 141 | 0.818 | 0.797 | 0.807 | 108 |

    CMP 23 at 0.5: baseline 0.744, rule 1 0.746, rule 2 0.745, rule 3 0.752. LSAA 77 at 0.5: 0.840, 0.849, 0.844, 0.798.
  - **Rule 1 v2 `side_wall`** (2026-11-01, replaces `edge_depth` in `USER_RULES`). The user marked 23 pipeline boxes on side walls / neighbouring buildings in six photos (`relation2139383`, `relation2148482`, `way177765882`, Paris `way64040028`, London `way350450350`, `way640943795`). Findings: touching the border alone removes 92 boxes, 42 real (wide balconies cut by the frame); recessed depth is clean; projecting balconies are closer than the wall (−12% to −17%). Rule: drop if median DA3 depth > 8% farther than the main wall, or the box touches the left/right border (≤ 1% of width) and is taller than wide. Chosen on test100 by looking at these boxes.
    - All 100, no ignore: IoU 0.5 P 0.815 R 0.858 F1 0.836 (baseline 0.819); IoU 0.3 F1 0.855 (0.838). Removes 53 boxes in 22 photos: depth 22 (4 real), edge 31 (6 real). 20 of the 23 marked boxes go. Of the 10 real ones, 3 are the cut-off slivers in `UK_London_way418371200` and 2 the cut-off column in `France_Paris_relation946705`, both on the frame edge.
    - Images: `runs/test100/compare_rule1/<stem>.jpg` (GT | baseline | rule 1, removed boxes blue dashed with reason), list `_removed.json`.
  - **`side_wall` is in the pipeline** (2026-11-01, user). `run_balcony.py` applies it after `filter_balcony_boxes` inside `snap_open_units` (default on, `--no-side-wall` turns it off). Depth: `ensure_da3_depth` in `--depth-dir` (default `runs/da3_enclosure`), resized to the photo; without depth only the border test runs. Drops appear in the filter log as `side_wall:depth` / `side_wall:edge`. Checked on `Austria_Vienna_way177765882` (`--no-enclosed`): the 5 left-edge boxes dropped, 3 kept, same as the scorer. `compare_open_cluster.py` calls `snap_open_units` without it (default `side_wall=False`).
  - **Rule 2 `solid_juliet` on top of rule 1** (images `runs/test100/compare_rule2/<stem>.jpg`: GT | pipeline with rule 1 | + rule 2, list `_removed.json`). No ignore, all 100: IoU 0.5 F1 0.836 → 0.833 (P 0.815 → 0.840, R 0.858 → 0.826); IoU 0.3 0.855 → 0.851. CMP 0.734 → 0.733, LSAA 0.865 → 0.862. Removes 48 boxes in 19 photos, 22 real. Real losses are classifier errors: thin metal window railings called solid, and projecting panel balconies called juliet (`France_Paris_way69694010` loses 6 real). False removals are mostly ornament under windows (`relation2148482` 13). Not adopted.
    - Training crops labeled solid + juliet (`balcony_train/export_label_combo.py`, out `runs/balcony_clf/review_solid_juliet/`, per-fold folders + `sheet_<fold>.jpg` + `index.csv`): only 44 of 4585 rows (train 29, val 4, test 4, test100 photos 7 not trained). 23 of the 37 non-test100 crops come from three Vienna buildings (way621703814 10, way130741255 8, way144468649 5), nearly all grey sills with brackets. Doubtful labels: Paris way64623265 (curved projecting panel balcony, not juliet), cmp_b0263_bal_002 (pierced ironwork panel), cmp_b0155_pbal_002 (balusters, test100). For the user to review.
    - All solid training crops (user, 2026-11-01: solid + juliet is not a real category): `export_label_combo.py --kind solid` → `runs/balcony_clf/review_solid/` (default `--structure any`), 507 crops (kind = crop folder): train 281, val 38, test 36, test100 photos 22, flux2_ synthetic 130 (not used by v2). Sheets `sheet_<fold>_NN.jpg`, 64 crops a page; `index.csv` has the structure label (projecting 100, juliet 44, none 363).
    - Label fix (user, 2026-11-01): 5 crops solid → open_work masonry, files moved `crops/solid` → `crops/open_work`, other labels kept: `lsaa_Austria_Vienna_way204687396_…_pbal_001`, `lsaa_Italy_Rome_way306601493_…O-NayEECV0fxeBO6wCJ7SA`, `lsaa_Austria_Vienna_way621703814_…_pbal_012` / `_pbal_018`, `cmp_b0156_type_00_unit_000`. Backup `runs/balcony_clf/labels.before_masonry_fix_20261101.jsonl`. Folds unchanged (split.json matches by stem). Solid now 502 crops. Not retrained yet; `review_solid/` is from before the fix.
  - **Option C: SAM3 `window sill` veto** (`balcony_pipeline/try_sam3_sill.py`; sill boxes cached in `runs/test100/sam3_window_sill_boxes.json` at score >= 0.30; images `runs/test100/compare_sill/<stem>.jpg`, left pipeline with rule 1, right + C; scores `runs/test100/balcony_sill_score.json`). Rule fixed beforehand: drop a pipeline box when a sill box (score >= 0.45) has IoU >= 0.5 with it or covers >= 70% of it. No ignore, all 100, IoU 0.5: F1 0.836 → 0.840 (P 0.815 → 0.841, R 0.858 → 0.839); IoU 0.3 0.855 → 0.857. CMP 0.734 → 0.729, LSAA 0.865 → 0.872. 416 sill boxes; removes 38 boxes in 9 photos, 13 real. Half the removals are one photo (`relation2148482`, 19 removed, 5 real). SAM3 'window sill' also fires on low baluster balconies under windows (`Denmark_Copenhagen_way136103697` loses both real ones), so it cannot tell a sill from a baluster balcony.
  - **GT fix 2 (user, 2026-11-01): `France_Paris_way53309361`** — the five stacked balconies in the middle column were missing; the five pipeline boxes added ([764,324,939,546], [769,645,941,805], [766,953,940,1104], [772,1269,937,1361], [797,1559,960,1635]); 25 → 30 boxes, LSAA total 541. Backup `runs/test100/balcony_gt_boxes.before_paris53309361_fix.jsonl`. New scores (no ignore, IoU 0.5, all 100): pipeline filtered (no rules) F1 0.823 (P 0.779 R 0.873; LSAA 0.845, CMP 0.744 unchanged), IoU 0.3 0.842; with rule 1 F1 0.840 (P 0.822 R 0.859), IoU 0.3 0.859; + option C 0.844, IoU 0.3 0.861. Earlier rows in this section are from before this fix; all move by these five boxes on one LSAA photo only.
    - Same five balconies in the test100 classifier labels (`runs/test100/crops/labels.jsonl`, crops `_pbal_001/003/004/012/014`): solid + juliet → solid + projecting, floor_shape rectangle (user: projecting + solid; rectangle is my reading of the photo). Backup `runs/test100/crops/labels.before_paris53309361_fix.jsonl`. The test100 classifier scores in this log (eval_test100_v2) are from before this fix.
  - **Retrain v3 + option D** (2026-11-01). v3: same command as v2 (`--extend-split --exclude-prefix flux2_` + the 95 prefixes) on the current labels (5 crops solid → open_work masonry). `checkpoints/railing_test100_v3.pt` (sha256 037388b07970…), train 1424 / val 181 / test 180, best val 0.954 (v2 0.947). Log `runs/balcony_clf/train_test100_v3.log`.
    - test100 crops (`evaluate_test100.py`, `runs/test100/eval_test100_v3_20261101.md`, corrected test labels), macro F1 v2 → v3: kind 0.831 → 0.783, material 0.843 → 0.849, floor 0.904 → 0.884, structure 0.748 → 0.763. v3 calls more open_work crops solid (open_work recall 0.804 → 0.709, 46 → solid; solid P 0.752 → 0.608). v2 stays the reference checkpoint.
    - Option D (`balcony_pipeline/score_test100_rule2_conf.py`, probs cached `runs/test100/railing_probs_<ckpt>.json`, images `runs/test100/compare_rule2_conf/` for v3: GT | pipeline (rule 1) | + rule 2 argmax | + rule 2 P >= 0.8, removed boxes labelled P(solid)/P(juliet); scores `runs/test100/balcony_rule2_conf_score.json`). Threshold 0.8 fixed before scoring. No ignore, all 100, IoU 0.5 F1 (removed / real): pipeline 0.840; v2 argmax 0.833 (48 / 25); v2 P >= 0.8 0.844 (18 / 5); v3 argmax 0.834 (62 / 30); v3 P >= 0.8 0.843 (27 / 9). IoU 0.3: 0.859; 0.851; 0.861; 0.852; 0.862. CMP 0.734 → v2 conf 0.741, v3 conf 0.739; LSAA 0.870 → 0.873 / 0.873.
    - Remaining confident real removals are the projecting solid balconies the test labels now call projecting (`France_Paris_way53309361` middle column: v3 P 1.00/0.87 and 0.96/0.92); that fix was in the test labels only, so the model still learns them as sills. Same ballpark as option C (0.844).
  - **Rule 3 `no_window_near` on top of rule 1** (`balcony_pipeline/score_test100_rule3.py`; images `runs/test100/compare_rule3/<stem>.jpg`: GT | pipeline (rule 1) | + rule 3 with detected windows green and the top floor band shaded; `_removed.json`; scores `runs/test100/balcony_rule3_score.json`). No ignore, all 100, IoU 0.5: F1 0.840 → 0.800 (P 0.822 → 0.830, R 0.859 → 0.773); IoU 0.3 0.859 → 0.820. CMP 0.734 → 0.743, LSAA 0.870 → 0.817. Removes 79 boxes, 60 real. Cause: the rule needs a detected window just above, and the window track misses the doors and windows behind balconies (`France_Paris_way70145482`: no window found in either balcony column, 7 real lost; `France_Paris_way53309361`: the top-row projecting balconies and side balconies have no window found above, 6 real lost). Not adopted.
  - **Rule 3 v2 `below_floor` (user, 2026-11-01): box top below its floor's window row** (`postfilter.below_floor`; `score_test100_rule3.py --rule below_floor`; images `runs/test100/compare_rule3_below_floor/`, window rows shaded yellow; scores `runs/test100/balcony_rule3_below_floor_score.json`). Window row = top of highest to bottom of lowest window of a floor (the structural `floors_y` bands have no gaps, so a first try on them removed nothing). The box's row is the nearest one starting above its top; boxes above every row are kept. No ignore, all 100, IoU 0.5: F1 0.840 → 0.842 (P 0.822 → 0.843, R 0.859 → 0.841); IoU 0.3 0.859 → 0.860. CMP 0.734 → 0.746 (5 removed, 0 real), LSAA 0.870 → 0.869. Removes 32 in 16 photos, 12 real. Works on ornament between floors (`relation2148482`: 12 removed, all false). Real losses: low baluster balconies whose top sits just below the window bottom (`France_Paris_way79639959` 3, `UK_London_way51171075` 2, `UK_London_way132678806` 2).
    - **Correction (user spotted, 2026-11-01):** the version above took the nearest row starting above the box top, so tall balconies whose top reaches above their own floor's windows were compared with the floor above and removed (`UK_London_way132678806` box [508,193,741,392] vs row 134–192; `UK_London_way51171075` box [796,417,941,545] vs row 290–391). Its numbers are void. Fixed: a box is dropped when it overlaps no window row vertically (in a gap or below the lowest row); boxes above every row are kept. Result: F1 0.840 → 0.832 (P 0.828, R 0.837), IoU 0.3 0.859 → 0.850; removes 23, 15 real. The ornament in `relation2148482` overlaps the rows and stays (2 removed, both false). Real losses are low balustrades that sit just under the window bottom with a few px gap (`relation2367958_Fid5615` 2, `France_Paris_way79639959` 3, `cmp_b0159` 2). `removed_only/` holds the 13 photos with removals. Not adopted.
  - **Hood rule `window_below` (user, 2026-11-01; target: decoration above windows in `relation2148482`)** (`postfilter.window_right_below`; `score_test100_rule3.py --rule window_below`; images `runs/test100/compare_rule3_window_below/`, photos with removals also in `removed_only/`; scores `runs/test100/balcony_rule3_window_below_score.json`). Drop a box when a window centred in its x-range has its top below the box centre and within 0.25 window heights of the box bottom, and no window is just above the box (`windows_just_above`). 0.25 fixed before scoring. No ignore, all 100, IoU 0.5: F1 0.840 → 0.837 (P 0.822 → 0.829, R 0.859 → 0.846); IoU 0.3 0.859 → 0.854. CMP 0.734 → 0.735, LSAA 0.870 → 0.866. Removes 17 in 10 photos, 9 real. Target photo: 3 of 7 hoods removed; the other hoods sit 0.26–0.28 window heights above the detected window glass (frame + cornice between), just past 0.25. Real losses: large projecting balconies resting on the window of the floor below whose own door is not detected (`France_Paris_way53309361` 3, `France_Paris_way288358394` 2). Not adopted.
  - **Oriel rule on top of rule 1** (user, 2026-11-01; `score_test100_rule3.py --rule oriel --oriel-cells pred|gt`; images `runs/test100/compare_rule3_oriel_pred/` and `_oriel_gt/` with oriel cells purple, `removed_only/` in each; scores `runs/test100/balcony_rule3_oriel_{pred,gt}_score.json`). Drops an open box on an oriel cell (same floor, shared bay; `drop_open_under_enclosed`, the step `run_balcony.py` already runs after snapping). No ignore, all 100, IoU 0.5, from 0.840 (P 0.822 R 0.859):
    - Predicted cells (held-out cell model, `grid2_test100/preds_heldout.json`, 89 photos): F1 0.798 (P 0.845 R 0.756), IoU 0.3 0.813; removes 105, 72 real. Wrong oriel cells on balcony columns (`France_Paris_way53309361` 12 real lost, `France_Paris_way72230705` 6).
    - Hand-labeled cells (`oriel_labels.jsonl`, upper bound): F1 0.846 (P 0.857 R 0.836), IoU 0.3 0.864; CMP 0.734 → 0.764, LSAA 0.870 → 0.869; removes 48, 16 real. The 16 are boxes the box GT calls balconies on cells the oriel labels call oriel (`France_Paris_way81233669` 4: the slab under each oriel floor; `way68804360` 3, `way72230705` 3, `UK_London_way640943795` 2) — the two label sets disagree there.
    - So the rule only pays with a much better oriel detector. Not adopted as a test100 post-rule.
  - **Final open-balcony box score (2026-11-01)**: SAM3 `filtered` boxes + rule 1 `side_wall` (what `run_balcony.py` now does before snapping; oriel step not included), current GT (both Paris fixes), group rule. `runs/test100/balcony_final_score.json`. No ignore (main): all 100 IoU 0.5 pred 731 / GT 700 / TP 601 / FP 130 / FN 99, P 0.822 R 0.859 F1 0.840; IoU 0.3 F1 0.859. CMP 23 F1 0.734 (0.3: 0.806); LSAA 77 F1 0.870 (0.3: 0.874). With ignore (supplementary): all 100 F1 0.845 (0.3: 0.864), 8 predictions ignored. The older with-ignore 0.849 was the plain pipeline: rule 1 removes most boxes that the ignore regions used to absorb (48 → 8 ignored) and 10 real balconies, so with ignore it scores slightly lower; without ignore it gains 0.823 → 0.840.
    - Images: `balcony_pipeline/draw_test100_final.py` → `runs/test100/compare_gt_final/<stem>.jpg`, GT (left) | final pipeline (right; since 2026-11-01 rule 1 + masonry_juliet_split, v2 labels), all boxes plain red, header TP / FP / FN at IoU 0.5.
    - **GT fix 3 (user, 2026-11-01): `France_Paris_way81233669`** — 8 window railings in the side columns added as balconies (boxes read from the user's screenshot, 1:1 with the photo, offset (17, 41)): [58,296,130,337], [340,299,413,337], [469,299,541,337], [58,407,130,446], [340,407,413,446], [469,407,541,446], [340,527,413,557], [469,527,541,557]; 7 → 15 boxes. Test crop labels `_pbal_001/002/005/008` solid + projecting → solid + juliet (floor shape cleared). Backups `runs/test100/balcony_gt_boxes.before_paris81233669_fix.jsonl`, `runs/test100/crops/labels.before_paris81233669_fix.jsonl`. Final score above recomputed after this fix (see `balcony_final_score.json`).
  - **Error breakdown of the final pipeline** (`balcony_pipeline/error_breakdown_test100.py` → `runs/test100/balcony_error_breakdown.json`; paper-style write-up `runs/test100/balcony_error_analysis.md`). Current GT, IoU 0.5, no ignore. Without rule 1 on current GT: F1 0.822 (CMP 0.744, LSAA 0.843); final 0.838 (CMP 0.734, LSAA 0.867), so rule 1 lowers CMP. 53/100 photos error-free; 10 worst photos hold 47% of the 233 errors, 20 worst 75%.
    - FN 105: merged into a larger box 49 (continuous balustrade boxed once, GT per segment; the group rule needs touching boxes, e.g. `relation2445885` whole-row IoU 0.88), SAM3 no box 26, SAM3 wrong extent 10, overlap filter 10, side-wall rule 10. Attributes: on oriel 4, large 3, overlapping 2.
    - FP 128: below a window (sills / ornament panels) 63, on a hand-labeled oriel cell 28, misplaced on a real balcony 20, other 11, on top of a window (hoods) 6. Position tags use detected windows (proxy).
  - **Split wide boxes per window, on top of rule 1** (user, 2026-11-01, for the 49 "merged" misses; `balcony_pipeline/score_test100_split.py`, `postfilter` `juliet_split` and new `wide_split`; images `runs/test100/compare_split/` GT | pipeline | + juliet split | + any wide box split, pieces blue, 42 photos with splits in `removed_only/`; scores `runs/test100/balcony_split_score.json`). Current GT, no ignore, IoU 0.5: pipeline 0.838 (P 0.825 R 0.852); juliet_split (v2 classifier labels) 0.826 (P 0.787 R 0.870; 27 boxes → 79 pieces, 28 match; CMP 0.727, LSAA 0.854); wide_split 0.733 (90 → 239 pieces, 37 match). IoU 0.3: 0.860 / 0.857 / 0.780. Fixes the target balustrades (`relation2445885` TP 0 → 8, `relation2139386` +7, `relation2139383` +4, `cmp_b0171` +4) but splits real continuous balconies over two windows that the classifier calls juliet (`France_Paris_way63642281` TP 12 → 9, FP +6; `UK_London_way640943795`, `Austria_Vienna_way123353819`, `cmp_x0005`). Not adopted.
    - Oracle check (hand structure labels from `runs/test100/crops/labels.jsonl`, matched to pipeline boxes via `crops/manifest.jsonl`): juliet_split F1 0.838 → 0.840 at IoU 0.5 (P 0.800 R 0.886), 0.860 → 0.873 at IoU 0.3; 27 boxes → 80 pieces. Of the 27 boxes the v2 classifier split, the hand labels call 13 juliet, 10 projecting, 4 have no crop. So wrong juliet labels cost 0.014 F1 (0.826 vs 0.840); with correct labels the split is about neutral at IoU 0.5 because the pieces take the window's x-range, not the balustrade segment's, and some juliet railings are one GT box.
    - **`masonry_juliet_split` (user, 2026-11-01): split only boxes the classifier calls juliet + open_work + masonry** (baluster balustrades), one box per window just above (>= 2). Images: 5th panel of `runs/test100/compare_split/`; `--labels hand` → `compare_split_hand/`, `balcony_split_hand_score.json`. Current GT, no ignore. v2 classifier labels: IoU 0.5 F1 0.838 → **0.854** (P 0.825 → 0.824, R 0.852 → 0.886), IoU 0.3 0.860 → 0.878; CMP 0.734 → 0.748, LSAA 0.867 → 0.884; 11 boxes → 41 pieces, 28 match. Hand labels: 0.852 (IoU 0.3 0.877). Gains: `relation2445885` +8, `relation2139386` +7, `relation2139383` +4, `cmp_b0171` +4, `cmp_b0235` +3, `relation2139368` +2. Losses: balustrades under paired windows are split per window (`Austria_Vienna_way123353819` TP −2 FP +4, `relation2367958_Fid5615` −1, `cmp_x0005` −1); the grid gives each window of a pair its own bay, so splitting per bay would not change this. Condition chosen after seeing the test results (not tuned on cmp_b).
    - **Wired into `run_balcony.py`** (user, 2026-11-01; default on, `--no-juliet-split` to skip). `snap_open_units(..., split_predictor=)`: after the overlap filter and side_wall, each filtered box is classified (`predict_full`) and `apply_rules(..., rules=("masonry_juliet_split",))` splits it before merge / snap; the railing predictor is loaded once before snapping and reused for the IR step. Smoke test `relation2445885` (`--no-enclosed --depth-dir runs/test100/da3_depth`): with `--railing-ckpt checkpoints/railing_test100_v2.pt` the balustrade box → 8 boxes; with the default `railing_best.pt` + `railing_structure.pt` the structure head calls it projecting and nothing is split. The test100 gain (0.838 → 0.854) is measured with the v2 labels; the default checkpoints give different structure labels.
  - **Oriel rule alone on the current pipeline (rule 1 + masonry_juliet_split)** (user, 2026-11-01; `balcony_pipeline/score_test100_oriel.py`; images two panels pipeline | + oriel, `runs/test100/compare_oriel_pred/` and `compare_oriel_gt/`, `removed_only/` 39 and 18 photos; scores `runs/test100/balcony_oriel_{pred,gt}_score.json`). No ignore, all 100, IoU 0.5 from 0.854 (P 0.824 R 0.886): predicted cells F1 **0.814** (P 0.847 R 0.782), removes 107, 73 real; CMP 0.748 → 0.764, LSAA 0.884 → 0.828. Hand-labeled cells (upper bound) **0.860** (P 0.857 R 0.863), removes 48, 16 real; CMP 0.748 → 0.778, LSAA 0.884 → 0.883. IoU 0.3: 0.878 → 0.833 / 0.884. `run_balcony.py` runs this step whenever the oriel detector is on, so with the current oriel model the full pipeline scores about 0.814, not 0.854.
  - **Current final score (2026-11-01; oriel step left out until the oriel model is improved, user)**: rule 1 + masonry_juliet_split (v2 labels), `runs/test100/balcony_final_score.json` (overwritten; was rule 1 only). No ignore, all 100: IoU 0.5 pred 761 / GT 708 / TP 627 / FP 134 / FN 81, P 0.824 R 0.886 F1 0.854; IoU 0.3 F1 0.878. CMP F1 0.748, LSAA 0.884.
    - Error breakdown of this version (`error_breakdown_test100.py`, now includes the split by default, `--no-split` for rule 1 only; `balcony_error_breakdown.json` overwritten): FN 81 = SAM3 no box 27, merged into a larger box 17, SAM3 wrong extent 13, overlap filter 10, side-wall rule 10, misplaced 4 (attributes: on oriel 4, large 4, overlapping 2). FP 134 = below a window 60, misplaced on a real balcony 29, on an oriel cell 28, other 11, on top of a window 6. 55/100 photos error-free; 10 worst photos hold 52% of the 215 errors, 20 worst 77%.
  - **Classifier labels for the final boxes** (2026-11-01). Of the 761 final boxes, 627 have a labeled test crop and 8 are not_balcony. 85 have a crop in `runs/test100/crops` that was never saved (20 of them match a GT box at IoU ≥ 0.5). The other 41 are `masonry_juliet_split` pieces with no crop (28 match a GT box). `balcony_train/export_test100_split_crops.py` writes the 41 pieces to `runs/test100/crops_split/` (`<parent crop>_s<k>.png`, context, manifest with the parent box, prefill = parent's hand label). Two parents are hand-labeled projecting (`cmp_b0171_pbal_001`, `cmp_x0005_pbal_023`). Label: `label_pipeline_crops_ui.py --dir runs/test100/crops_split --open-only`; not yet merged into `runs/test100/crops`.
    - User labeled all 41 (2026-11-01): 39 open_work masonry juliet, 2 open_work masonry rectangle projecting, none X.
    - **Classifier on the final-pipeline crops** (2026-11-01). Set = the labeled crops of the 761 final boxes: 627 from `crops` + 41 from `crops_split` = 668 (93 boxes have no label: 85 unsaved, 8 not_balcony). The split parents and the boxes `side_wall` drops are left out. Built by `balcony_train/build_test100_final_set.py <out_dir>` (copies crops + labels into one dir), scored with `evaluate_test100.py --dir <out_dir>`; output `runs/test100/eval_test100_final_boxes_20261101.md`.
      - Macro F1, juliet checkpoint / v2: kind 0.807 / 0.831, material 0.914 / 0.873, floor 0.906 / 0.901, structure 0.763 / 0.763. Accuracy: material 0.930 / 0.893, floor 0.941 / 0.936, structure 0.763 / 0.766.
      - Caveat: which boxes are split depends on the v2 labels, so v2 partly chose its own test crops. Structure stays test-informed (juliet round).
      - **GT-matched set (user, 2026-11-01: a balcony not in the box ground truth is not scored by the classifier).** `build_test100_final_set.py` now drops a labeled crop whose box has IoU < 0.5 with every `balcony_gt_boxes.jsonl` box (`--gt-iou`, default 0.5; `--gt-iou 0` gives the 668 set). Label files unchanged; the dropped list is `dropped_not_in_gt.jsonl` in the out dir. 62 dropped (best IoU 0: 34, 0–0.3: 10, 0.3–0.5: 18; 13 are split pieces), 606 scored, out dir `runs/test100/crops_final_gt`, report `runs/test100/eval_test100_final_gt_20261101.md`, error images `runs/test100/errors_final_juliet/` (regenerated on this set; the 668-set images were replaced).
        - Macro F1, juliet / v2: kind 0.774 / 0.812, material 0.905 / 0.855, floor 0.904 / 0.893, structure 0.761 / 0.764. Kind falls from the 668 set because 36 dropped crops were true solid (oriel tops, sills), mostly predicted solid: solid n 94 → 58, juliet-ckpt solid P 0.627 → 0.514.
        - This is now the main classifier row on test100 (scored only on detections that match a GT box).
        - **Structure label fix (user, 2026-11-01), chosen from the error images:** 43 crops projecting → juliet, floor_shape cleared (was rectangle 32, circle 11) in `runs/test100/crops/labels.jsonl` (backup `labels.before_juliet_fix_20261101.jsonl`). Photos: Paris `way68822623` 010, `way70002774` 008, `way82684045` 001–008/010/011/013–016, London `way119104403` 000/002/003/005, `way188549802` 000/001/002/004–007/010/013–015, `way345786998` 000/001/005/007/009/012/013/015/017–020. Rescored (same 606 crops), macro F1 juliet / v2: kind 0.774 / 0.812, material 0.905 / 0.855, floor 0.890 / 0.878 (n 344 → 301), **structure 0.835 / 0.787** (juliet ckpt acc 0.764 → 0.835). Not blind: only crops the model got wrong were reviewed. The 26 crops of the same 6 photos still labeled projecting (22 predicted projecting) are in `runs/test100/review_projecting_same_photos/` (`review.csv`, `sheet_00.jpg`) for the user to check under the same rule.
        - **Second fix (user, 2026-11-01), also from the error images:** 4 crops juliet → projecting: Paris `way70002774` `_pbal_001/002/003`, `way72230705` `_pbal_007`. floor_shape left null (not scored on floor). Backup `labels.before_projecting_fix_20261101.jsonl`. Rescored, macro F1 juliet / v2: kind 0.774 / 0.812, material 0.905 / 0.855, floor 0.890 / 0.878, **structure 0.841 / 0.793** (juliet ckpt acc 0.842; confusion projecting 268 / 37, juliet 59 / 242). Error images regenerated: 256 errors.
        - **Third fix (user, 2026-11-01), from the error images:** 2 crops open_work → solid, material cleared, file moved to `crops/solid/`: Paris `way69694010` `_pbal_015` (patterned panel), `cmp_b0100_pbal_000` (curved solid parapet). Backup `labels.before_solid_fix_20261101.jsonl`. Not in the training `labels.jsonl`. Rescored, macro F1 juliet / v2: **kind 0.780 / 0.819**, material 0.904 / 0.854, floor 0.890 / 0.878, structure 0.841 / 0.793. Error images: 254 errors.
        - **Fourth fix (user, 2026-11-01), from the error images:** 23 crops open_work metal → surface_panel, material cleared, files moved to `crops/surface_panel/`: Paris `way64040028` `_pbal_003–006/008–013/015/016/020/021/023–028` (20, glass-infill window guards), `way68822623` `_pbal_009/025/028`. Backup `labels.before_surface_fix_20261101.jsonl`. Rescored, macro F1 juliet / v2: **kind 0.813 / 0.848**, **material 0.912 / 0.881**, floor 0.890 / 0.878, structure 0.841 / 0.793. Error images: 228 errors.
        - The four fixes (43 + 4 + 2 + 23 crops) were all chosen from the error images.
  - **Training structure review** (user, 2026-11-01; juliet = a balustraded guard with no walkable projection beyond the facade plane, whatever the depth of the window reveal behind it). `export_label_combo.py --kind <k> --structure <s> --out runs/balcony_clf/review_structure/<s>/<k>` for both structures and all three kinds: juliet 440 (open_work 328, surface_panel 71, solid 41), projecting 530 (367 / 63 / 100); per fold, `sheet_<fold>_NN.jpg`, `index.csv`. Crop only, no photo context.
    - Fix 1: 9 LSAA training crops (surface_panel, train fold) projecting → juliet, floor_shape rectangle → null, in `runs/balcony_clf/labels.jsonl` (backup `labels.before_juliet_fix_20261101.jsonl`): Paris `relation1002642` Bid34940, `way175217108` Bid132139, `way44701813` Bid137075/137076, `way63240131` Bid142158/142160/142162/142163/142164. Folds unchanged. Not retrained; `review_structure/` still shows them under projecting.
    - Fix 2: `cmp_b0030_pbal_000` (open_work masonry, train fold) projecting → juliet, floor_shape rectangle → null. Backup `labels.before_b0030_fix_20261101.jsonl`.
    - Fix 3: 3 train-fold crops open_work masonry → solid (plain slabs / parapet), material cleared, files moved `crops/open_work` → `crops/solid` (training reads kind from the folder): `cmp_b0325_pbal_000`, Vienna `way621703814_Fid7464` `_pbal_013`, `_pbal_014`. Structure stays juliet. Backup `labels.before_solid_fix_20261101.jsonl`.
    - Fix 4: 2 crops solid → open_work masonry, files moved `crops/solid` → `crops/open_work`: `cmp_b0155_pbal_002` (balusters; a test100 photo, not trained by v2) and Vienna `way621703813_Fid9339_…_pbal_000` (val fold). Backup `labels.before_masonry_fix2_20261101.jsonl`. `cmp_b0155_pbal_002` also changed in `runs/test100/crops/labels.jsonl` (backup `labels.before_masonry_fix_20261101.jsonl`, file moved to `crops/open_work`). Test100 rescored, macro F1 juliet / v2: kind 0.815 / 0.851, material 0.912 / 0.882, floor 0.890 / 0.878, structure 0.841 / 0.793. Error images: 227.
    - **Retrain v4** (2026-11-01, after the 15 training fixes): same command as v2/v3 (`--extend-split --exclude-prefix flux2_` + the 95 prefixes). `checkpoints/railing_test100_v4.pt` (sha256 c328c44de4f3…), best val 0.951, log `runs/balcony_clf/train_test100_v4.log`. Structure folds train 414 projecting / 358 juliet.
      - test100 GT-matched set (606, current test labels; `runs/test100/eval_test100_v4_gt_20261101.md`), macro F1 v4 / v2 / juliet: kind **0.846** / 0.851 / 0.815, material **0.900** / 0.882 / 0.912, floor **0.889** / 0.878 / 0.890, structure **0.806** / 0.793 / 0.841. v4 kind: solid P 55/85 = 0.647 (juliet ckpt 0.533), open_work R 0.858. v4 structure: juliet R 0.728 (82 juliet → projecting), P 0.862.
      - 668 set (`--gt-iou 0`, `runs/test100/eval_test100_v4_668_20261101.md`), v4 / juliet: kind 0.867 / 0.845, material 0.910 / 0.920, floor 0.898 / 0.894, structure 0.798 / 0.834.
      - All-crop review images for a ground-truth check (user, 2026-11-01): `export_test100_errors.py checkpoints/railing_test100_v4.pt --dir runs/test100/crops_final_gt --out runs/test100/all_crops_v4 --all` → `<head>/<true label>/` with every crop (1866 images over 4 heads; v4 wrong ones prefixed `ERR_` and marked WRONG), `sheet_NN.jpg` 20 per page, `all.csv`.
      - Floor fix (user, 2026-11-01, from `all_crops_v4`): test crop `cmp_b0100_pbal_000` floor circle → rectangle (backup `runs/test100/crops/labels.before_floor_fix_20261101.jsonl`; not in training labels). Floor macro F1 v4 / v2 / juliet: 0.894 / 0.882 / 0.895 (acc 0.937 / 0.930 / 0.940); other heads unchanged. `all_crops_v4` regenerated (238 errors).
      - Kind fix (user, 2026-11-01, from `all_crops_v4`): test crop Vienna `relation2139368_Fid496_…_pbal_001` solid → open_work masonry (baluster balustrade), file moved to `crops/open_work` (backup `labels.before_masonry_fix2_20261101.jsonl`; not in training labels). Macro F1 v4 / v2 / juliet: kind 0.849 / 0.853 / 0.817, material 0.901 / 0.883 / 0.913, floor 0.894 / 0.882 / 0.895, structure 0.806 / 0.793 / 0.841. `all_crops_v4` regenerated (237 errors).
      - Structure fix (user, 2026-11-02, from `all_crops_v4`, correct and wrong crops alike): 13 test crops projecting → juliet, floor_shape cleared (backup `runs/test100/crops/labels.before_juliet_fix2_20261102.jsonl`): London `way188549802` `_pbal_003/008/009/011/012`, `way119104403` `_001/004/006/008/009`, `way50810363` `_000`, Paris `way79639641` `_007`, `way239222098` `_009`. Macro F1 v4 / v2 / juliet: kind 0.849 / 0.853 / 0.817, material 0.901 / 0.883 / 0.913, floor 0.884 / 0.872 / 0.893 (n 301 → 288), structure 0.808 / 0.786 / 0.830. `all_crops_v4` regenerated (236 errors).
      - Kind fix (user, 2026-11-02): test crop Paris `way64040028_…_pbal_022` open_work metal → surface_panel (file moved; backup `labels.before_surface_fix2_20261102.jsonl`). Material macro F1 v2 0.883 → 0.886; v4 / juliet kind and material unchanged at 3 decimals. `way64040028` still has `_pbal_002/007/014` as open_work.
      - **Share package for Jiyeon** (user, 2026-11-02): `balcony_train/export_test100_share.py checkpoints/railing_test100_v4.pt --out runs/test100/share_jiyeon` (zip `runs/test100/share_jiyeon.zip`). `balcony_classification.xlsx`: sheet Balconies (606 rows: facade file, balcony file, truth / prediction for kind, material, floor, structure; wrong predictions red, `-` = no truth), sheet Summary (accuracy, per-class P/R/F1, macro; equals the v4 numbers above). `photos/combined` (crop | facade with red box, 606), `photos/facade` (87 raw photos), `photos/balcony` (606 raw crops).
      - Metrics Excel (user, 2026-11-02): `balcony_train/export_eval_xlsx.py runs/test100/eval_test100_v4_gt_20261101.md runs/test100/balcony_type_eval_test100_v4.xlsx` (first checkpoint in the report). Confusion matrices as values; accuracy, per-class P/R/F1 and macro as formulas on those cells; checked in Excel (all equal the report). Copied into the share package as `balcony_type_eval.xlsx`, zip rebuilt.
      - Test labels were corrected from error images (72 + 1 crops), so all three are test-informed; the juliet checkpoint's structure also came from a test-informed juliet round. Before a final number, review the matching correct crops too (`review_projecting_same_photos/` is still open).
      - Error images (juliet checkpoint): `balcony_train/export_test100_errors.py` → `runs/test100/errors_final_juliet/<head>/<true>_as_<pred>/` (crop | photo with box, `sheet_NN.jpg` 24 per page), `errors.csv`. Regenerated after the juliet fix: 260 errors (structure: juliet→projecting 63, projecting→juliet 37).

## Paper evaluation runs on the 61-photo dev set (2026-10-24)

`balcony_train/paper_eval.py`, `paper_edit_demo.py`, `paper_figures.py`. Results are in `runs/paper_eval/results/quantitative.md` and `runs/paper_eval/figures/`. All numbers are dev (the old 61-photo set), not test.

- Data:
  - Balcony track and renders for all 61 photos: `runs/paper_eval/<stem>/`.
  - Window-track ablations from saved SAM3 detections: control / nosym / boxonly in `runs/paper_eval_abl/`. control equals `runs/batch` on 60/61.
- Pooled results (IoU ≥ 0.5 unless noted):

  | Item | Precision | Recall | F1 |
  |---|---:|---:|---:|
  | Windows, DSL units | 0.792 | 0.496 | 0.610 |
  | Windows, compiled 3D projected back onto the photo | 0.595 | 0.349 | 0.440 |
  | Grid cells with an opening | 0.887 | 0.899 | 0.893 |
  | Enclosure cells | 0.898 | 0.708 | 0.792 |

  - Floor count exact 32/61, bay count exact 20/61.
  - The compiled drop comes from a few photos where the grid breaks: `0054` has two buildings; `0128`, `0171` and `0011` have only 2–8 SAM3 windows.
  - Symmetry repair and box-only clustering do not change geometric scores. They only change the type counts: 321 (control) / 335 (no symmetry) / 182 (box only).
- Editing demo on `cmp_x0005`:

  | Edit | DSL fields changed | Instances changed |
  |---|---:|---:|
  | Window type to arched | 3 | 10 |
  | Railings to solid | 6 | 14 |
  | Enclosure to punched, rectangle | 3 | 12 |
- Fixed during the runs:
  - `merge_dsl.py`: door partner ids such as `door_0` crashed `int()`.
  - `facade_spec.py`: `columns_xy` was indexed by bay id instead of column.
  - Recompiling all 183 outputs changed nothing else.

## Threshold search on cmp_b, scored on dev (2026-10-21)

`balcony_train/threshold_f1_search.py`, output in `runs/cell_clf/threshold_f1/`. Nine ways of choosing the cell thresholds were compared on cmp_b out-of-fold probabilities with nested CV: grids over t × gate, a coordinate search over all ten thresholds, relative t, a joint score, and a gmax × cmax gate.

- None beat the hand-set t = 0.30 / gate = 0.10 on dev pooled F1 (0.792). A sweep on dev itself peaks at 0.795, so thresholds have almost no headroom.
- Gate 0.10 sits in a gap of the dev set: none photos reach 0.09, enclosure photos start at 0.11. Train out-of-fold and dev gate distributions differ, so a gate chosen on cmp_b does not transfer.
- Of the 40 dev FN, 23 are on four photos the gate blanks (`0028`, `0074`, `0081`, `0121`). Most of the rest have p_cell < 0.3 on labeled cells. The next gain has to come from the models or more labels, not from thresholds.

## Paper draft decisions (2026-10-29)

CVPR 2027 draft. Section 3.3.3 (Oriels) is now separate from 3.3.2 (Balconies).

- Table 2:
  - Balcony columns are Railing, Material, Floor, Struct., LPIPS. Enclosed is dropped because enclosure comes from the oriel track, not the railing classifier.
  - New Oriel group: Cell F1 (TP / FP / FN pooled over all photos), Plan, Skin, LPIPS. Plan and skin come from `score_enclosed_plan.py`.
  - Facade-level Mask IoU stays window / door / balcony. CMP label maps have no oriel class, and a mask built from our cells would repeat Cell F1. If a reviewer asks about oriel location, add cell IoU to the Oriel group.
  - Curtain-wall oriels drop the windows in their bays. CMP still labels those pixels as window, so Window Mask IoU can drop on those photos.
- 3.4 Implementation Details lists:
  - balcony SAM3 ≥ 0.45 and < 16 px dropped; enclosure prompts ≥ 0.15;
  - classifier heads: kind, material, floor, structure, oriel plan, oriel skin;
  - spectral clustering, PCA 32;
  - oriel: HistGradientBoosting, crop context 25 %, PCA 16, t = 0.3, γ = 0.10;
  - compiled geometry: slab 1.5 m, railing 1.1 m, juliet railing 0.15 m in front of the wall at the sill, rooftop railing on the top edge of the storey.
- The paper says the heads are "trained jointly", but the structure head is copied from `railing_structure.pt`, a separate run on the same frozen backbone.
- The Depth Anything 3 bib key is `depthanything3` (arXiv:2511.10647).

## Open items

- **Open-balcony box score on test100, current state (2026-11-01).** Pipeline = SAM3 `balcony` ≥ 0.45 → overlap filter → `side_wall` → `masonry_juliet_split` (v2 labels); oriel step left out until the oriel model is improved (user). No ignore, IoU 0.5: F1 0.854 (P 0.824, R 0.886); IoU 0.3 0.878; CMP 0.748, LSAA 0.884 (LSAA GT started from pipeline boxes, so CMP is the independent number). `runs/test100/balcony_final_score.json`; images `runs/test100/compare_gt_final/`; error breakdown `runs/test100/balcony_error_breakdown.json`; paper text `runs/test100/balcony_error_analysis.md` (results / errors / limitations, error tables there are the rule-1-only version) and `runs/test100/balcony_method.md` (method + protocol). Open points:
  - `side_wall` and `masonry_juliet_split` were designed after viewing test100 errors; confirm on cmp_b before citing their gains as unbiased.
  - The split needs juliet labels: the scored run uses `railing_test100_v2.pt`; the pipeline default `railing_best.pt` + `railing_structure.pt` called the `relation2445885` balustrade projecting (no split). Decide the pipeline checkpoint, then rescore end to end.
  - Oriel removal (`drop_open_under_enclosed`) lowers the score with the current oriel model (0.854 → 0.814; hand-labeled cells 0.860). Revisit after the oriel model is improved.
  - Remaining false positives are mostly decorated sills (60 of 134) and oriel parts (28); position rules failed. Next candidate: a balcony-or-not crop check trained on training photos only (option A). Not started.
  - Balustrades under paired windows are split per window (`Austria_Vienna_way123353819`); continuous-balustrade GT convention (one box vs per segment) is still the user's call.
  - `runs/test100/balcony_box_score_per_image.csv` is stale (written before the GT fixes, with ignore); rewrite it if needed (it was open in Excel).

- 2026-10-31 merged `main` (window track up to `4ec7868`) into `hiroo/balcony-pipeline`. Not committed: the merge is staged with MERGE_HEAD, for the user to commit.
  - New from main:
    - local symmetry: continuous pair weights, balcony occluder, hidden side smaller than its column mates;
    - side-facade filter rebuilds the main floors until stable;
    - per-floor bays: gap-merge lines, adopted lines, one-floor lines dropped in gaps that already hold a line;
    - door leaves inside a larger door dropped;
    - `--exclude-prompts car,truck,van`.
  - Conflicts:
    - `floor_bays.py`, `local_sym.py`, `doors.py`, `paths.py` take main (the branch held older main copies plus cosmetic edits).
    - `run_pipeline.py` and `README.md` keep the branch side of each hunk: `--image-dir`, `--with-balconies` and its options, the SAM3 model reuse, the Windows PosixPath patch. Main's other changes auto-merged.
  - Checks:
    - `test_merge_dsl`, `test_parse_bdsl`, `test_balcony_layout` and `test_balcony_plan` pass.
    - `run.py --with-balconies` on `cmp_b0007` runs end to end, and `render_facade.py` compiles its DSL. Scratchpad, not cited.
  - **The grid changes.** `cmp_b0007` goes from 5 to 4 union columns (22 → 24 floor bays).
    - Every cell label (`labels_train.jsonl`, `labels_ext_all.jsonl`, `labels_eval.jsonl`, `runs/test100/oriel_labels.jsonl`) and `cells_*.npz` is indexed on the old grids in `runs/batch` and `runs/test100/batch`.
    - Those runs were not regenerated. Regenerating them needs the cell labels remapped or relabeled, and the cell model retrained. All cited oriel and balcony numbers are on the old grid.
  - 2026-10-31 new-grid rerun, done (user asked to move the oriel evaluation to the new grid).
    - Window track on 148 cmp_x test/dev photos, 375 cmp_b training photos, the 77 current test100 LSAA photos and the 22 extra oriel photos → `runs/batch_grid2/` and `runs/test100/batch_grid2/` (test100 cmp DSLs copied there). All got a grid. `runs/batch` is untouched.
    - It is slow, about 60 s per photo: main's pipeline loads SAM3 four times per photo and adds nine occluder prompts.
    - The first attempt crashed with out-of-memory, next to another session's `embed_facades.py`.
    - New scripts:
      - `remap_grid_labels.py` (+ test): old → new cell labels by pixel overlap ≥ 0.5, with a review list.
      - `cell_features_pipeline.py`: training and eval features through `enclosed_detect.photo_cell_data`, now shared with `detect_enclosed_cells`.
  - **Oriel training extra** (user, 2026-10-31): `runs/test100/photo Archive/Training/`, 60 oriel photos, stems in `runs/enclosed_grid/oriel_train_extra_stems.txt`. AGENTS.md has the exception.
    - 50 cmp_x (33 in the 133 test set, 32 of its 55 oriel photos; 16 in dev 61) and 10 LSAA (no building shared with test100).
    - User chose to train on all 60 and to score the cmp_x oriel test without them. Old cmp_x oriel numbers are not comparable to the new ones.
    - 22 of them need a new grid (`runs/batch_grid2/_todo/extra`).
    - Need labels on the new grid: the 10 LSAA, `cmp_x0031` (unlabeled), and `cmp_x0072/0117/0122/0145/0148` (grid failures on the old grid).
  - **2026-10-31 new-grid labels, features, model, scores.**
    - Labels remapped with `remap_grid_labels.py` (old files kept as `*.grid1.jsonl`; review lists in `*.review.txt`):
      - `labels_train` 375 photos, 134 → 188 cells, 12 to review;
      - `labels_eval` 61, 206 → 229, 8 to review;
      - `labels_ext_all` 226 → 159 rows, 613 → 582 cells, 22 to review. The 67 dropped rows are cmp_x photos outside test and dev with no new grid; they stay in `.grid1`. All 145 `extended_selected` photos are kept, 12 still grid failures (status copied, not relabeled);
      - `runs/test100/oriel_labels` 89, 193 → 177, 1 to review.
    - Refactor check: `cell_features_pipeline.py` on the old grid reproduces `cells_train.npz` exactly (5 photos, max |ΔX| = 0, DINO identical).
    - Training: `labels_train_oriel.jsonl` = `labels_train.jsonl` + 43 extra photos with cells (from `labels_ext_all`). 17 extra photos are not labeled yet: the 10 LSAA, `cmp_x0031`, `0072`, `0117`, `0122`, `0134`, `0145`, `0148`. `cells_train.npz` is rebuilt on the new grid: 418 photos, 17,279 cells, 615 positive. `cell_clf_train.DEFAULT_TRAIN_LABELS` points at `labels_train_oriel.jsonl`. `model_rules_gate.joblib` is refit (sha256 `947604ad…`). Gate, thresholds and rules are unchanged.
    - Old files: `cells_train.grid1.npz`, `model_rules_gate.grid1.joblib` (`8ad657e4…`). New grid, cmp_b only: `cells_train_grid2_cmpb.npz`, `model_rules_gate.grid2_cmpb.joblib` (`37c46602…`).
    - Scores: `runs/oriel_grid_compare/run_all.sh` and `run_cmpb.sh` (`eval_test100_oriel.py`, now with `--model`, `--train-cells`, `--train-labels`, `--skip-stems`). Per column, pooled, no ignore. Test = 133 minus the extra photos (100 photos); dev = 61 minus extra (45). The extra photos are not in test100. Qualitative panels are in each `<run>/qualitative_*`.

      | set | grid, model | TP | FP | FN | P | R | F1 | IoU | exact | no-oriel photos correct |
      |---|---|---|---|---|---|---|---|---|---|---|
      | test 100 | grid1, cmp_b | 105 | 91 | 64 | 0.536 | 0.621 | 0.575 | 0.404 | 72 | 66 |
      | test 100 | grid2, cmp_b | 84 | 59 | 71 | 0.587 | 0.542 | 0.564 | 0.393 | 73 | 67 |
      | test 100 | grid2, cmp_b + extra (default) | 111 | 179 | 44 | 0.383 | 0.716 | 0.499 | 0.332 | 55 | 48 |
      | dev 45 | grid1, cmp_b | 56 | 17 | 26 | 0.767 | 0.683 | 0.723 | 0.566 | 34 | 29 |
      | dev 45 | grid2, cmp_b | 44 | 35 | 40 | 0.557 | 0.524 | 0.540 | 0.370 | 30 | 26 |
      | dev 45 | grid2, cmp_b + extra (default) | 53 | 91 | 31 | 0.368 | 0.631 | 0.465 | 0.303 | 25 | 21 |
      | test100 89, held out | grid1, cmp_b | 98 | 23 | 95 | 0.810 | 0.508 | 0.624 | 0.454 | 68 | 64 |
      | test100 89, held out | grid2, cmp_b | 101 | 38 | 76 | 0.727 | 0.571 | 0.639 | 0.470 | 67 | 64 |
      | test100 89, held out | grid2, cmp_b + extra (default) | 142 | 146 | 35 | 0.493 | 0.802 | 0.611 | 0.440 | 49 | 42 |

      - Units differ between grids (test cells 5133 vs 3763), so compare F1 / IoU / exact, not raw counts.
      - The grid change alone is about neutral on test and test100 and costs dev F1 0.18. The gate and rules were tuned on dev with the old grid.
      - Adding the 43 extra oriel photos raises recall but triples false positives on no-oriel photos with the unchanged gate 0.10 and t = 0.3. The training positive rate goes from 1.3 % to 3.6 % of cells. Re-picking gate and threshold would need a choice on training folds; not done. User to decide the default (switch: copy `model_rules_gate.grid2_cmpb.joblib` over `model_rules_gate.joblib`).

- Pipeline default is four cell rules + gate ≥ 0.10: 47/61 total, enclosure exact 16/26, detect 21/26, none 31/35. False positives left: `0073`, `0120`, `0173`, `0180`. The gate blanks `0074`, `0081`, and `0121`. `0006` is still short the gallery floor. Rule 4's extra cells on `0121`, `0073`, and `0134` are not in the labels. Switch back with `--enclosed-method sam3`.
- The fold-0 SAM3 fine-tune was scored once on cmp_x (raw boxes, 33/61, cells TP 104 / FP 54 / FN 33). It is not wired into `--enclosed-method`. The stage strips under `runs/sam3_ft_fold0/cmp_x/stages/` were not scored as cells.
- Touching-OP1 add is below both gate → OP1 (41/61) and the pipeline default (47/61). With the gate it is 39/61; on the pure classifier (no gate) it is 34/61 and breaks classifier-exact photos (`0119`, `0123`, `0143`). If retried, do not add cells on a photo the classifier already matched, and decide the top-floor cut (85% vs the existing 90%) on cmp_b.
- Rules 1–3 in the "not yet tested" list (per-bay floor range, horizontal extend, border-bay drop) are still untried. Test them on cmp_b, then check eval once.
- Rerun `cell_clf_search.py` on the swapped eval set if the search table is needed again.
- Add labels for photos with enclosures; avoid buildings that appear in the eval set.
- Open-balcony types still default to DINOv2 appearance. `--cluster-mode box` was compared visually on 15 of the first 20 `data/baseForBatch` photos (`runs/balcony_cluster_compare/`). It was not scored against railing labels. Same-size balconies with different railings collapse to one type.
- 22 CMP balcony maps are wrong and are left out of the open-balcony box score: `0002`, `0003`, `0006`, `0008`, `0009`, `0011`, `0012`, `0014`, `0016`, `0021`, `0036`, `0043`, `0044`, `0047`, `0067`, `0107`, `0116`, `0127`, `0162`, `0173`, `0176`, `0227`. The labels were not redrawn. `data/baseForBatch` no longer contains the six of the first ten that used to be there.
- The open-balcony score uses `filter.py` with its window rules disabled (uncommitted). If those rules come back, rerun `score_sam3_pipeline_stages.py`; the filter step needs the window grid, which the script does not load.
