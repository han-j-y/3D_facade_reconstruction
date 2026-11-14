# Element-level balcony attributes: Astra vs ours (2026-11-13)

How well each method reads the four balcony attributes of `tab:main` (Element level, Balcony: railing kind,
material, floor shape, structure), scored on the same test100 balconies. Everything here reruns from `data/`
alone; `runs/` and `test_Astra Result/` are not needed.

## Result

487 balconies are in all three sources: the hand GT, our final output and Astra's rendering. Paper rule
(conditional macro F1). In each cell: macro F1 (accuracy, n).

| Method | Railing (kind) | Material | Floor | Structure | Mean of 4 |
|---|---|---|---|---|---|
| Ours  | 0.838 (0.881, 487) | 0.887 (0.955, 221) | **0.920** (0.970, 164) | 0.812 (0.815, 487) | 0.864 (acc 0.905) |
| Astra | **0.863** (0.875, 487) | **0.974** (0.992, 258) | 0.905 (0.953, 193) | **0.848** (0.848, 487) | **0.897** (acc 0.917) |

Other crop sets (`results/score_all.md`):
- Astra on all 526 linked crops: 0.837 / 0.954 / 0.866 / 0.846, mean 0.876.
- Both methods on 495 crops (ours also matched by unit IoU >= 0.5): ours 0.842 / 0.898 / 0.920 / 0.812 (0.868), Astra 0.857 / 0.952 / 0.905 / 0.846 (0.890).
- Ours in `tab:main` (its own crop set): 0.829 / 0.929 / 0.886 / 0.829 (0.868).

Main errors on the 487 (confusion tables in `results/score_gtiou05.md`):
- Ours, kind: open_work read as panel or solid; solid precision 0.62.
- Ours, structure: projecting read as juliet 73 times; projecting recall 0.70.
- Astra, kind: surface_panel read as open_work 49 times. Every open_work is correct.
- Astra, structure: projecting read as juliet 44 times.

Robustness (`results/extra_stats.txt`):
- IoU of the scored GT crops with the hand GT boxes: median 1.00, 10th percentile 0.79.
- Mean F1 over IoU thresholds 0 to 0.8: ours stays 0.864–0.871, Astra 0.890–0.904. The gap does not depend on the threshold.

SAM3 `balcony` boxes on the 100 facades, same prompt and filter (photo | Astra front view):

| Stage | Photo | Astra front view |
|---|---|---|
| raw | 898 | 684 |
| after the size filter | 883 | 684 |
| after the overlap filter | 764 | 648 |

The hand GT has 690 balconies on 87 photos.

### Caveats (read before citing)
- **Selection favours Astra.** Only balconies Astra reproduced are scored. Its misses do not enter the
  attribute scores.
- **Astra labels are not blind.** Hiroo labelled Astra's crops knowing they were Astra's. The paper caption
  says annotators did not know the method.
- **Material and Floor use different crops for the two methods.** These heads only score crops where truth
  and prediction both pass the condition. Astra calls every open_work open_work, so more of its crops pass
  (material n 258 vs 221).
- **GT crop labels.** They come from `runs/test100/crops` (+ `add5_crops`), copied on 2026-11-13. The
  paper's ours numbers were rescored with Jiyeon's `test100_labels_20261110`.

## How it was derived

1. **Astra crops.** SAM3 `balcony` was run on Astra's front views (`test_Astra Result/Front View/facade_NN.png`).
   The same size and overlap filters as the pipeline, without snapping, give 648 crops. Hiroo labelled them
   with `balcony_train/label_pipeline_crops_ui.py --dir "test_Astra Result/Cropped Balcony"`: 621 labelled,
   1 not a balcony. `astra_index.csv` maps `facade_NN` (index; 2 digits up to 50, 3 digits from 51) to the
   test100 stem.
2. **GT crops.** These are the hand-labelled SAM3 crops of the test100 photos: 642, in `gt_crops` plus
   `gt_add5_crops`.
3. **Hand links.** `balcony_train/astra_link_ui.py` shows each photo with its GT crops next to Astra's front
   view with its crops. Hiroo linked them by eye, one to one. All 100 facades were reviewed, 74 have links, and
   there are 526 pairs (`data/gt_links.jsonl`).
4. **Ours.** These are the `final` voted type labels of the run behind the paper numbers
   (`score_test100_types_final.py`, classifier v5, `data/ours_final_v5_20261110.json`). A GT crop takes the
   row with the same crop name; for split pieces it takes the largest piece.
5. **487 set.** `--gt-iou 0.5` keeps the GT crops with IoU >= 0.5 with a hand GT box (514 of 526) that also
   have an ours row (487). It has no unit-IoU fallback.
6. **Scoring.** This is `head_metrics(..., conditional=True)` from
   `balcony_pipeline/score_test100_types_final.py`, the same function as the paper.
   - Kind and structure are scored on every crop.
   - Material is scored only where truth and prediction are both open_work.
   - Floor is scored only where both are projecting.
   - Each head reports its macro F1 over its classes.

## Rerun

```
C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_train/score_astra_links.py --snapshot eval_astra_compare/balcony_attributes/data --gt-iou 0.5 --out eval_astra_compare/balcony_attributes/results/score_gtiou05.md
C:\Users\hiroo\anaconda3\envs\glodon\python.exe balcony_train/score_astra_links.py --snapshot eval_astra_compare/balcony_attributes/data --out eval_astra_compare/balcony_attributes/results/score_all.md
C:\Users\hiroo\anaconda3\envs\glodon\python.exe eval_astra_compare/balcony_attributes/extra_stats.py
```

Without `--snapshot` the scripts read the live files: `test_Astra Result/Cropped Balcony/gt_links.jsonl`,
`runs/test100/...`. Use that mode after relabelling or relinking; this folder stays as of 2026-11-13. To
relink, run `balcony_train/astra_link_ui.py`. It needs `test_Astra Result/Front View` and `data/test100/photos`.

## Files

| Path | What |
|---|---|
| `data/gt_links.jsonl` | Hand links, one row per facade: `links` = [[GT crop, Astra crop], ...], `done` = reviewed |
| `data/astra_crops/` | Astra crop `manifest` (box on the front view), `labels`, `not_balcony` |
| `data/gt_crops/`, `data/gt_add5_crops/` | test100 GT crop manifests and hand labels |
| `data/balcony_gt_boxes.jsonl` | Hand GT balcony boxes (test100) |
| `data/ours_final_v5_20261110.json` | Ours, per crop and per unit: crop / unit / final labels |
| `data/astra_index.csv` | Astra `facade_NN` to test100 stem |
| `data/*_sam3_stage_boxes.json` | SAM3 balcony boxes per stage (photo, Astra) |
| `results/score_gtiou05.md` (+ `.rows.json`) | The 487-crop result with confusion tables |
| `results/score_all.md` (+ `.rows.json`) | 526 / 495-crop result |
| `results/extra_stats.txt` | SAM3 counts, IoU distribution, threshold check |
