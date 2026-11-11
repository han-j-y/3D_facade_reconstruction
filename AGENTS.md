# AGENTS.md

Instructions for every coding agent in this repository (Claude Code, Cursor, Codex). Keep this file short and stable. Results, experiments and open items go in `agent.md`, the shared work log.

## Read first

`agent.md` is the running log shared by all sessions: data splits, every method tried with its score, pipeline defaults, cited evaluation protocols and open items. Read it before answering questions about results or starting work. Add new results there, not here.

Several agents work on this repo in parallel. Files and label files change between turns: take on-disk changes as the current state and do not revert them.

## Environment

- Windows. Run all Python with `C:\Users\hiroo\anaconda3\envs\glodon\python.exe`. Never use `python`, `py`, Anaconda base, `dtsc670`, `dtsc680` or the system Python 3.12. Install packages only with that interpreter's `-m pip`.
- Set `PYTHONIOENCODING=utf-8` when running pipeline scripts. Console output contains characters such as `∩` that crash under cp1252.
- Blender: `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe` (pass via `$BLENDER`). It is optional; without it the pipeline still writes the DSL.
- The GPU is 8 GB. Two SAM3 processes at once fill it and deadlock. Run SAM3 jobs one after another.
- SAM3 (`facebook/sam3`) and DINOv2 (`torch.hub`) download at runtime.

## Commands

```
# full pipeline: windows + balconies (+ optional render)
C:\Users\hiroo\anaconda3\envs\glodon\python.exe run.py --image <photo.jpg> --out-dir runs/demo --device cuda --with-balconies [--blender-render]
C:\Users\hiroo\anaconda3\envs\glodon\python.exe run.py --image-dir <dir> --out-dir runs/batch --device cuda --with-balconies

# rerun the window track from saved SAM3 detections (no SAM3)
... run_pipeline.py --from-index runs/batch/<stem>/index_raw.json --out-dir <dir> --device cuda

# balcony track only, on an existing window DSL
... balcony_pipeline/run_balcony.py --image <photo> --windows-dsl runs/batch/<stem>/facade_dsl.json --out-dir <dir>

# compile (and render) a DSL; without --render it only writes blender/facade_compile.json
... scripts/render_facade.py --recovery <facade_dsl[_with_balconies].json> --out-dir <dir>/blender [--render]
```

Tests are `unittest` modules next to the code (`test_*.py`); pytest is not installed. Run them from the repo root, except the compiler tests, which import siblings and must run from their own directory:

```
C:\Users\hiroo\anaconda3\envs\glodon\python.exe -m unittest balcony_pipeline/test_merge_dsl.py
C:\Users\hiroo\anaconda3\envs\glodon\python.exe -m unittest balcony_train/test_label_store.py
cd vendor/window_compiler && C:\Users\hiroo\anaconda3\envs\glodon\python.exe -m unittest test_parse_bdsl
```

## Architecture

Photo → detected elements → typed, voted structure → JSON DSL → parametric Blender mesh. The DSL is the product: repeated elements share one type definition, so editing a type changes every instance.

**Window track** (`run_pipeline.py`, helpers in `facade_recovery/`, `scripts/`, `window_ast/`):
1. SAM3 `window` detection (or CMP XML). Boxes are clamped to the mask.
2. Unitize: merge panes into openings.
3. Structural column layout gives the floor × bay grid (`facade_recovery/column_layout.py`).
4. Type clustering (`--cluster-mode gate`, `--gate-mode fold`, `facade_recovery/fold_gate.py`): box-geometry GMM groups, per-floor fold-symmetry pairs and vertical pairs (sizes within 1.3x since 2026-11-10, `--vertical-size-ratio`) as must-links, a structure-descriptor split; then the joint vote (`facade_recovery/joint_vote.py`, default since 2026-11-10, `--no-joint-vote`): floor-symmetry twins and column neighbours pull windows into one type, occlusion-aware (balcony-hidden sides = lower bounds; SAM3 `shutter` along a side = width unknown, `--shutter-uncertain`); then the local-symmetry vote (`local_sym.py`) after the side-facade drop.
5. A structure model (`window_ast`, `checkpoints/structure_best.pt`) predicts each unit's window IR. A majority vote inside each type on a discrete `structure_view` fingerprint gives one IR per type.
6. Output: `facade_dsl.json` (`facade_recovery_dsl_v1`), the layout plus the type library. Every later stage reads this grid.

**Balcony track** (`balcony_pipeline/`, called by `--with-balconies`):
- Open balconies:
  - SAM3 `balcony` boxes → `filter.py` → snapped to the window grid (`snap.py`, never re-clustered).
  - Typed (`cluster.py`, appearance, `--cluster-mode box` available): must-link rows and columns (same window relation) first (`--cluster-links`, default since 2026-11-07), k = 1 allowed, and units whose box widths differ 2x or more never share a type.
  - The railing classifier (`balcony_train/model.py`, default `checkpoints/railing_test100_v5.pt` since 2026-11-10 (user; = `railing_curve30_lowres_x0005_20261110.pt`, v4's recipe + 30 curved-front user crops + the 16 `cmp_x0005` crops; v4 = `railing_test100_v4.pt`), set in `balcony_train/paths.py`: frozen DINOv2 + linear heads, trained without the test100 photos and without FLUX.2, own structure head) labels each crop, then a per-type vote (`vote.py`): kind / material / floor by soft vote (mean softmax of the members, default since 2026-11-07; `--vote majority` = old rule), structure by majority. Its skin head never saw a curtain wall, so the pipeline does not use it for oriels. Structure (projecting / juliet) does not come from it either since 2026-11-07 (user): DA3 depth decides, projecting when the box sticks out of the wall beside it by > 0.10 m (`balcony_pipeline/structure_depth.py`, `--structure-depth`); the classifier's structure head is only the fallback (no wall strip) and the tie-break of the type vote. The old default `railing_best.pt` saw the test100 photos.
- Enclosed balconies / oriels are a separate cell detector: `balcony_train/enclosed_detect.py` (`--enclosed-method cell_clf`).
  - Per-cell features (DA3 depth, SAM3, DINOv2) → HistGradientBoosting gate + cell model → rectangle decode (t 0.5) → four cell rules → photo gate ≥ 0.70 (454-photo model without test100 and without the val30 photos, refit 2026-11-10 with the 4 photos moved out of test100 and without London `way341319207`).
  - Floor shape (railing classifier) and skin (oriel skin model `checkpoints/oriel_skin.joblib`, `balcony_train/oriel_skin.py`, default since 2026-11-03; `--no-oriel-skin-model` falls back to the railing classifier) are one majority vote per photo.
- `merge_dsl.py` appends `balcony_types` and `layout.balconies`, giving `facade_dsl_with_balconies.json`.
- After the merge (default since 2026-11-08): a projecting balcony straddling two windows is fixed for side-view parallax when the DA3 wall tilt and the window offsets agree on the camera side (`parallax.py`, `--no-parallax-fix`).

**Compiler** (`vendor/window_compiler/`, runs inside Blender via `main.py`):
- `facade_spec.normalize_facade_spec` maps the recovery DSL to `window_compiler_facade_v1`.
- `facade_compile.py` places windows and doors; `balcony_compile.py` / `balcony_layout.py` / `balcony_plan.py` build slabs, railings and enclosure shells.
- Grammars: `documents/WDSL.ebnf` (window), `documents/BDSL.ebnf` (balcony), `documents/FDSL.ebnf` (facade, recovery ↔ compiled mapping).

**Training and evaluation** (`balcony_train/`): crop export, label UIs (Tkinter), classifier training (`train.py`, `--heads-only`), the enclosure cell classifier and its experiments, and scoring scripts. `agent.md` records which script produced which cited number.

## Model defaults

- When a newly trained model scores better on test100 than the pipeline default and passes the leakage check, make it the pipeline default right away (code, `agent.md`, this file) instead of leaving it as "not wired in" (user, 2026-11-03). Report the switch; the user does not need to ask each time.

## Data and evaluation rules

- CMP Facade Database:
  - `cmp_b*` (base) is the only training data.
  - `cmp_x*` (extended) is no longer a test set (user, 2026-11-02). Do not train on it except as below. Its old test and dev results are in `agent_archive_cmp_x.md`.
  - Exception (user, 2026-10-31): the photos in `runs/enclosed_grid/oriel_train_extra_stems.txt` (53 cmp_x, 10 LSAA; `oriel_train_add_261105_stems.txt` adds more LSAA, minus London `way341319207` since 2026-11-10; `cmp_x0196` added 2026-11-03, `cmp_x0149` and `cmp_x0006` added 2026-11-10 from test100) train the oriel models only: the cell model and the oriel skin model (`balcony_train/oriel_skin.py`, user 2026-11-03). They never train the railing classifier. Oriel skin no longer comes from the railing classifier.
  - Exception (user, 2026-11-10): `cmp_x0005` left val30 (val30 = 30 photos, backup `data/val30/val30_ids.before_x0005_to_train_20261110.json`, photo in `data/val30/_moved_to_train/`) and its pipeline balcony crops (`runs/balcony_clf/crops_cmp_x0005`) may train the railing classifier. It is the only cmp_x photo allowed there.
  - Label maps are palette PNGs: 3 window, 4 door, 7 balcony.
  - In CMP XML, `<x>` holds rows and `<y>` holds columns (swapped from the usual meaning).
- Enclosure splits:
  - Train: `runs/enclosed_grid/labels_train_oriel.jsonl` (oriel cell model; see below).
  - Test: test100 only, `runs/test100/oriel_labels.jsonl` (photos `runs/test100/photos`). No test100 photo may be in any training set.
  - test100 has 100 photos again since 2026-11-10 16:38 (user): 5 new photos (`cmp_x0058`, Amsterdam `way278102588`, London `way341319207`, `way497722711`, `way78230839`) were labeled and added (backups `*.before_test100add5_20261110.*`; `way341319207` was removed from the oriel training first). Earlier the same day (user), `cmp_b0100`, `cmp_b0168`, `cmp_x0149`, `cmp_x0005`, `cmp_x0006` left the test set (photos in `runs/test100/photos/Exclude`, their test crops in `runs/test100/excluded_test95/`, label backups `*.before_test95_20261110.*`). `cmp_x0005` joined val30 (`v30`); the other 4 joined the training sets (oriel cell and skin models; the 2 cmp_b may also train the railing classifier; the `railing_test95_v5.pt` retrain with them scored lower and is unused, see `agent.md`). The name test100 stays.
  - `labels_ext_all.jsonl` and `labels_eval.jsonl` (cmp_x) are no longer scored or reported.
  - Cell numbers refer to the window grid in `runs/batch_grid2/<stem>/facade_dsl.json` (test100 LSAA: `runs/test100/batch_grid2`). The old-grid labels are kept as `*.grid1.jsonl` for `runs/batch`.
  - `labels_train_oriel.jsonl` = cmp_b + the extra oriel photos, minus the test100 photos and the val30 cmp_b photos (user, 2026-11-10). `labels_train.jsonl` stays cmp_b only.
- Leakage: never score, show or cite a test result from a model whose training data contains any test photo (e.g. `railing_best.pt` on test100). Not even as a "reference" row with a caveat, and not in `agent.md`. Check the training set before scoring a checkpoint.
- Scoring:
  - Enclosure cells: pool TP / FP / FN over all cells (precision, recall, F1, IoU), not per-photo averages.
  - Open-balcony boxes: one-to-one greedy IoU matching, reported at IoU ≥ 0.5 and ≥ 0.3, on test100 (`runs/test100/balcony_gt_boxes.jsonl`); protocol in `agent.md`.
  - Open balconies and their types are scored on the pipeline output after the oriel removal (user, 2026-11-03): boxes with `balcony_pipeline/score_test100_pipeline.py`, the classifier on the crops of those boxes (`build_test100_final_set.py --pipeline-score`). Do not report or mix in older box / classifier numbers (cached v2 split labels, or no oriel removal).
- Every model file the pipeline loads lives in `checkpoints/`, pushed with Git LFS (`*.pt`, `*.joblib`; e.g. `oriel_cell_gate.joblib`, `oriel_skin.joblib`). `runs/` is git-ignored and never pushed (user, 2026-11-03), so no default path in pipeline code may point into `runs/`.
- `runs/` holds large generated outputs and cached detections (e.g. `runs/batch/_rows/*_boxes.json`) that let scoring skip SAM3. Prefer reusing caches over rerunning detection.
