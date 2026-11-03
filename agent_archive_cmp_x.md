# Archive: cmp_x (extended) test and dev results

Moved out of `agent.md` on 2026-11-02 (user: cmp_x is no longer used for testing; test100 is the only test set). Text is verbatim; headings show the `agent.md` section each block came from. Grids, labels, models and defaults have changed since; these numbers are history, not current results.

## From: Data

The default test is the 133-photo row. `paper_eval.py` still reads `runs/balcony_clf/crops_ext_eval/Before Crop` for the CMP xml files and label maps.

- Eval set: `cmp_x0002` and `cmp_x0003` were replaced by `cmp_x0010` and `cmp_x0013`
  (all four are no-enclosure photos). Caches and features were rebuilt after the swap.
- The eval set is never used for training or for choosing parameters.
  - Test DSLs `cmp_x0020`, `0094`, `0121`, `0224` (and `0005`, still exact) were rebuilt on Oct 28, after `labels_ext_all.jsonl` was saved on Oct 27. Their cell labels need a check in the UI.

## From: Metrics

- Enclosure photos in `labels_ext_all.jsonl` can also store one `floor_shape` and one `skin`. Empty cell lists stay unlabeled for both. The eval grid UI asks for them (R/T/C and P/G). Score with `score_enclosed_plan.py` on the official predicted cells: each cell is cropped, then the same majority vote as the pipeline. Triangle counts as rectangle. No predicted cells is wrong on both. `cmp_x` stays out of the training jsonl.
- Reported per group: enclosure (26), none (35), total (61); plus cell tp / fp / fn.
  `0185` was a no-enclosure photo and is now `f0b1`, `f1b1`, `f2b1`, `f3b1`.
  Rows in the table below that still say 25 and 36 were scored before that label edit.

## From: Methods and results (eval set, after the photo swap)

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

## From: Tried: gate → OP1

The OP3 gate removes photos without an enclosure, and OP1 places the cells. The
threshold 0.2 was chosen on cmp_b out-of-fold scores (3 seeds); every value in
0.1–0.3 gives 41/61 on eval. It works because the gate's misses are mostly photos
OP1 misses anyway, and OP1's false positives on no-enclosure photos have gate < 0.1.
Superseded by four cell rules + gate ≥ 0.10 (47/61). `--enclosed-method sam3` is
this older SAM3+DA3 volume path, not the 0.2 gate.

## From: Tried: OP3 rectangle, then add touching OP1 cells

`balcony_train/op3_touch_op1.py`. Model is `runs/cell_clf/model_base.joblib` (trained on cmp_b). A photo with gate max < 0.2 predicts nothing. Otherwise the classifier rectangle (t=0.3) is kept, and an OP1 Stage-6 cell is added only when all three hold:

- it 4-connects to the rectangle (same bay and adjacent floor, or same floor and adjacent bay; diagonals do not count);
- the bay is one the rectangle already uses, its neighbor, or its mirror, and the floor is at most one storey outside that bay's rectangle range;
- weak evidence: depth L ≥ 1.5%, SAM3 enclosure cover ≥ 0.25, or classifier probability ≥ 0.10.

Eval once: total exact 39/61, enclosure exact 8/25 (detect 16), none 31/36, cells tp/fp/fn 85/27/48. Groups: exact 8, wrong 8, missed 9, none_ok 31, false_pos 5. Images: `runs/cell_clf/compare_touch/` (`summary.txt`, `_board_*.jpg`).

Adding fixed `0020` (four side cells) and `0179` (one upper floor). The `0006` gallery stayed out because it only touches the oriels on a diagonal. `0143` was exact from the classifier alone and became wrong after three OP1 cells were added. Below gate → OP1 (41/61).

A raw union was rejected: it would copy OP1 cells that are a different volume. Dropping a top floor whose box covers under 85% of the storey height was discussed and not implemented. OP1 already drops a spilling top storey at 90% cover (`MIN_UPPER_FLOOR_COVER` in `balcony_pipeline/enclosed.py`); Stage-1 seed height uses 85% of one floor (`SEED_MIN_FLOOR_FRAC`).

## From: Tried: pure classifier (no gate), then the same touching OP1 cells

`balcony_train/op3_touch_op1.py --gate 0`. Same three conditions, but every photo keeps the classifier rectangle (t=0.3); the gate is not applied. Same trainer as `draw_clf_compare.py`. Images: `runs/cell_clf/compare_clf_op1/`.

Images for the classifier alone are `runs/cell_clf/compare_clf/` (`summary.txt`, `_board_*.jpg`): 35/61, enclosure exact 10/25 (detect 23), none 25/36, cells 93/35/40, scored before `0185` was labeled as an enclosure. Groups: exact 10, wrong 15, missed 0, none_ok 25, false_pos 11. `0119` is exact once bay 0 is gone, and `0123`'s label is now two cells (`f1b4`, `f2b4`) so the rectangle matches. On the current labels the same rectangle, printed at the top of `runs/cell_clf/compare_clf_rules/summary.txt`, is 35/61, enclosure exact 10/26 (detect 24), none 25/35, cells 97/31/40. Adding the touching OP1 cells then scores 34/61 (enclosure exact 9/25, detect 23, none 25/36, cells tp/fp/fn 100/51/33) on the earlier labels. Groups: exact 9, wrong 16, missed 0, none_ok 25, false_pos 11.

Gains: `0020` (four side cells) and `0179` (one upper floor) become exact. `0006` stays out; the gallery only touches the oriels on a diagonal. Losses: `0119`, `0123`, and `0143` were exact from the classifier and become wrong after the added cells. No-enclosure photos are unchanged (nothing was added). Below both the pure classifier on this data (35/61) and gate → OP1 (41/61).

## From: Tried: pure classifier plus four cell rules

`balcony_train/clf_three_rules.py`. Same rectangle (t=0.3, no gate). Applied in this order:

1. Drop the top floor of the rectangle when that band is under 85% of the photo's median floor height.
2. Unless window symmetry ≥ 0.75, drop a cell on that top floor whose probability is under 0.05. Other cells on the floor stay.
3. Add an OP1 cell when L ≥ 1.5%, upper ≥ 0.55, SAM3 enclosure cover ≥ 0.40, and the classifier probability is at least 0.10.
4. Add the cell one floor above a kept cell in the same bay when its depth L is within 0.02 of the cell directly below, upper ≥ 0.75, and that floor band is at least 85% of the median floor height. Classifier probability, OP1 membership, and SAM3 cover are not required. The new cell does not grow another floor.

Eval once, after `0125` gained `f3b0`, `0020` lost `f1b4`, and `0185` became an enclosure: total exact 41/61, enclosure exact 16/26 (detect 24), none 25/35, cells tp/fp/fn 104/33/33. Groups: exact 16, wrong 10, missed 0, none_ok 25, false_pos 10. Images: `runs/cell_clf/compare_clf_rules/`.

Exact enclosure photos: `0005`, `0020`, `0032`, `0090`, `0097`, `0118`, `0119`, `0123`, `0125`, `0137`, `0143`, `0144`, `0149`, `0179`, `0185`, `0212`. The earlier 10 exact photos stay exact. `0005` drops its short top floor. `0020` adds `f0b0`, `f0b4`, `f1b0`. `0179` adds `f0b0`. `0090` adds `f0b6` (stored probability 0.0986; L 0.074 vs 0.065 below). `0185` adds `f0b1` (probability 0.04; L 0.050 vs 0.042 below). `0125` `f0b0` stays out because its depth is 0.041 away from the floor below. `0006` is still not exact: the prediction is `f4b0`, `f4b4` and the gallery `f5b1`, `f5b2`, `f5b3` is missing.

Rule 4 also adds `0074` `f3b5` and `0129` `f2b4` (both in the label; neither photo becomes exact) and three cells that are not in the label: `0121` `f0b2`, `0073` `f1b0`, `0134` `f1b0`. No none-photo that was correct became a false positive.

## From: Tried: four cell rules, then gate ≥ 0.10

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

## From: Tried: SAM3 fold-0 fine-tune, raw boxes

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

## From: What was learned

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

## From: Resolve priority (agreed)

1. Detect enclosures that are missed entirely. OP1 misses `cmp_x0028`, `0032`, `0074`,
   `0078`, `0097`, `0116`, `0121`, `0129`, `0137`, `0147`.
2. Fix wrong rectangles.
3. Suppress detections on no-enclosure photos. Gate ≥ 0.10 leaves four false positives on purpose: `0073`, `0120`, `0173`, `0180`.

Tune rules on cmp_b, then check the eval set once.

## From: Rule ideas from the 17 "wrong" photos (pure classifier), not yet tested

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

## From: Scripts (balcony_train/)

| `score_enclosed_plan.py` | Test floor shape and skin from `labels_ext_all.jsonl` and official predicted cells |
| `eval_sam3_enclosed_ft.py` | Fold-0 SAM3 fine-tune on cmp_x. Raw boxes → `runs/sam3_ft_fold0/cmp_x/`. `--stages` writes the six-panel strips under `stages/` |

## From: Extended label distribution

Two finished censuses of `data/extended` (`cmp_x`). Neither is the 61-photo enclosure eval set. Do not retune the gate or the cell rules from these counts. Reprint the crop margins with `balcony_train/summarize_xml_labels.py`.

## From: Open crops

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

## From: Enclosed photos

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

## From: Open-balcony box detection (CMP label maps)

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

## From: IoU ≥ 0.3 next to 0.5 (paper table)

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

## From: Open-balcony structure (projecting / juliet)

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

## From: Open-balcony attributes on pipeline crops (test set, in progress)

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

## From: Paper evaluation runs on the 61-photo dev set (2026-10-24)

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

## From: Threshold search on cmp_b, scored on dev (2026-10-21)

`balcony_train/threshold_f1_search.py`, output in `runs/cell_clf/threshold_f1/`. Nine ways of choosing the cell thresholds were compared on cmp_b out-of-fold probabilities with nested CV: grids over t × gate, a coordinate search over all ten thresholds, relative t, a joint score, and a gmax × cmax gate.

- None beat the hand-set t = 0.30 / gate = 0.10 on dev pooled F1 (0.792). A sweep on dev itself peaks at 0.795, so thresholds have almost no headroom.
- Gate 0.10 sits in a gap of the dev set: none photos reach 0.09, enclosure photos start at 0.11. Train out-of-fold and dev gate distributions differ, so a gate chosen on cmp_b does not transfer.
- Of the 40 dev FN, 23 are on four photos the gate blanks (`0028`, `0074`, `0081`, `0121`). Most of the rest have p_cell < 0.3 on labeled cells. The next gain has to come from the models or more labels, not from thresholds.

## From: Open items

    - User chose to train on all 60 and to score the cmp_x oriel test without them. Old cmp_x oriel numbers are not comparable to the new ones.
      - `labels_eval` 61, 206 → 229, 8 to review;
      - `labels_ext_all` 226 → 159 rows, 613 → 582 cells, 22 to review. The 67 dropped rows are cmp_x photos outside test and dev with no new grid; they stay in `.grid1`. All 145 `extended_selected` photos are kept, 12 still grid failures (status copied, not relabeled);
      | test 100 | grid1, cmp_b | 105 | 91 | 64 | 0.536 | 0.621 | 0.575 | 0.404 | 72 | 66 |
      | test 100 | grid2, cmp_b | 84 | 59 | 71 | 0.587 | 0.542 | 0.564 | 0.393 | 73 | 67 |
      | test 100 | grid2, cmp_b + extra (default) | 111 | 179 | 44 | 0.383 | 0.716 | 0.499 | 0.332 | 55 | 48 |
      | dev 45 | grid1, cmp_b | 56 | 17 | 26 | 0.767 | 0.683 | 0.723 | 0.566 | 34 | 29 |
      | dev 45 | grid2, cmp_b | 44 | 35 | 40 | 0.557 | 0.524 | 0.540 | 0.370 | 30 | 26 |
      | dev 45 | grid2, cmp_b + extra (default) | 53 | 91 | 31 | 0.368 | 0.631 | 0.465 | 0.303 | 25 | 21 |
      - Units differ between grids (test cells 5133 vs 3763), so compare F1 / IoU / exact, not raw counts.
        | test 100 | 0.70 / 0.5 | 105 | 60 | 50 | 0.636 | 0.677 | 0.656 | 0.488 | 75 | 69 |
        | test 100 | 0.10 / 0.5 | 113 | 111 | 42 | 0.504 | 0.729 | 0.596 | 0.425 | 58 | 50 |
        | cmp_x test 100 | 456, 0.80 / 0.4 (default) | 98 | 80 | 57 | 0.551 | 0.632 | 0.589 | 0.417 | 78 |
        | cmp_x test 100 | 456, 0.70 / 0.5 | 105 | 54 | 50 | 0.660 | 0.677 | 0.669 | 0.502 | 77 |
        | cmp_x test 100 | 475, 0.70 / 0.5 | 105 | 60 | 50 | 0.636 | 0.677 | 0.656 | 0.488 | 75 |
        | test 100 | 97 | 96 | 58 | 0.503 | 0.626 | 0.557 | 0.386 | 75 | 68 |
        | dev 45 | 47 | 60 | 37 | 0.439 | 0.560 | 0.492 | 0.326 | 30 | 26 |
- Pipeline default is four cell rules + gate ≥ 0.10: 47/61 total, enclosure exact 16/26, detect 21/26, none 31/35. False positives left: `0073`, `0120`, `0173`, `0180`. The gate blanks `0074`, `0081`, and `0121`. `0006` is still short the gallery floor. Rule 4's extra cells on `0121`, `0073`, and `0134` are not in the labels. Switch back with `--enclosed-method sam3`.
- The fold-0 SAM3 fine-tune was scored once on cmp_x (raw boxes, 33/61, cells TP 104 / FP 54 / FN 33). It is not wired into `--enclosed-method`. The stage strips under `runs/sam3_ft_fold0/cmp_x/stages/` were not scored as cells.
- Touching-OP1 add is below both gate → OP1 (41/61) and the pipeline default (47/61). With the gate it is 39/61; on the pure classifier (no gate) it is 34/61 and breaks classifier-exact photos (`0119`, `0123`, `0143`). If retried, do not add cells on a photo the classifier already matched, and decide the top-floor cut (85% vs the existing 90%) on cmp_b.
- Rules 1–3 in the "not yet tested" list (per-bay floor range, horizontal extend, border-bay drop) are still untried. Test them on cmp_b, then check eval once.
- Rerun `cell_clf_search.py` on the swapped eval set if the search table is needed again.
- Add labels for photos with enclosures; avoid buildings that appear in the eval set.
- 22 CMP balcony maps are wrong and are left out of the open-balcony box score: `0002`, `0003`, `0006`, `0008`, `0009`, `0011`, `0012`, `0014`, `0016`, `0021`, `0036`, `0043`, `0044`, `0047`, `0067`, `0107`, `0116`, `0127`, `0162`, `0173`, `0176`, `0227`. The labels were not redrawn. `data/baseForBatch` no longer contains the six of the first ten that used to be there.
- The open-balcony score uses `filter.py` with its window rules disabled (uncommitted). If those rules come back, rerun `score_sam3_pipeline_stages.py`; the filter step needs the window grid, which the script does not load.

## From: Open-balcony structure (projecting / juliet), 2026-10-29 rooftop bullet (tail)

On cmp_x XML crops (`crops_ext_xml`, 737, null structure = projecting): kind macro F1 0.482, material 0.884, floor 0.586, structure acc 0.783 / macro F1 0.737 (juliet precision 0.532). Provisional on pipeline crops (`crops_pipeline_ext`, 512 prefilled, not reviewed): kind 0.548, material 0.853, floor 0.680, structure acc 0.805 / macro F1 0.654. Scripts: scratchpad `eval_ext_xml.py`, `eval_pipeline_prefill.py` (not in the repo).
