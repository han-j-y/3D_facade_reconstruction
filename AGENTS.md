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
4. Type clustering: box-geometry GMM, then DINOv2 appearance splits, then L–R symmetry repair.
5. A structure model (`window_ast`, `checkpoints/structure_best.pt`) predicts each unit's window IR. A majority vote inside each type on a discrete `structure_view` fingerprint gives one IR per type.
6. Output: `facade_dsl.json` (`facade_recovery_dsl_v1`), the layout plus the type library. Every later stage reads this grid.

**Balcony track** (`balcony_pipeline/`, called by `--with-balconies`):
- Open balconies:
  - SAM3 `balcony` boxes → `filter.py` → snapped to the window grid (`snap.py`, never re-clustered).
  - Typed (`cluster.py`, appearance by default, `--cluster-mode box`).
  - The railing classifier (`balcony_train/model.py`, `checkpoints/railing_best.pt`: frozen DINOv2 + linear heads) labels each crop, then a per-type vote (`vote.py`).
- Enclosed balconies / oriels are a separate cell detector: `balcony_train/enclosed_detect.py` (`--enclosed-method cell_clf`).
  - Per-cell features (DA3 depth, SAM3, DINOv2) → HistGradientBoosting gate + cell model → rectangle decode → four cell rules → photo gate ≥ 0.10.
  - Floor shape and skin (punched window / curtain wall) are one majority vote per photo.
- `merge_dsl.py` appends `balcony_types` and `layout.balconies`, giving `facade_dsl_with_balconies.json`.

**Compiler** (`vendor/window_compiler/`, runs inside Blender via `main.py`):
- `facade_spec.normalize_facade_spec` maps the recovery DSL to `window_compiler_facade_v1`.
- `facade_compile.py` places windows and doors; `balcony_compile.py` / `balcony_layout.py` / `balcony_plan.py` build slabs, railings and enclosure shells.
- Grammars: `documents/WDSL.ebnf` (window), `documents/BDSL.ebnf` (balcony), `documents/FDSL.ebnf` (facade, recovery ↔ compiled mapping).

**Training and evaluation** (`balcony_train/`): crop export, label UIs (Tkinter), classifier training (`train.py`, `--heads-only`), the enclosure cell classifier and its experiments, and scoring scripts. `agent.md` records which script produced which cited number.

## Data and evaluation rules

- CMP Facade Database:
  - `cmp_b*` (base) is the only training data.
  - `cmp_x*` (extended) is evaluation only. Never train or tune thresholds on it.
  - Label maps are palette PNGs: 3 window, 4 door, 7 balcony.
  - In CMP XML, `<x>` holds rows and `<y>` holds columns (swapped from the usual meaning).
- Enclosure splits:
  - Train: `runs/enclosed_grid/labels_train.jsonl`.
  - Test: `runs/enclosed_grid/labels_ext_all.jsonl` (skip `"status": "grid_failure"` lines).
  - Dev: the older 61-photo `labels_eval.jsonl` set, which the current gate and rules were tuned on.
  - Cell numbers refer to the window grid in `runs/batch/<stem>/facade_dsl.json`.
- Scoring:
  - Enclosure cells: pool TP / FP / FN over all cells (precision, recall, F1, IoU), not per-photo averages.
  - Open-balcony boxes: one-to-one greedy IoU matching, reported at IoU ≥ 0.5 and ≥ 0.3, with the protocol and exclusions in `agent.md`.
- Do not change ground truth or invent matching rules after looking at errors. Report extra criteria only as supplementary.
- `runs/` holds large generated outputs and cached detections (e.g. `runs/batch/_rows/*_boxes.json`) that let scoring skip SAM3. Prefer reusing caches over rerunning detection.
