"""Data locations for the Astra comparison. Override with environment variables; the defaults assume the research
folder layout  arch/{experiments/facade_e2e_combined, experiments/window_ast_predictor, datasets/Astra, datasets/testset}."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent                     # this folder (scorers + helpers)
REPO = HERE.parent                                          # facade_e2e_combined
ARCH = REPO.parents[1]
_e = lambda k, d: Path(os.environ[k]).expanduser().resolve() if os.environ.get(k) else d
ASTRA_DIR = _e("ASTRA_DIR", ARCH / "datasets" / "Astra")    # Astra runs, mapping.json, light_labels*, tids46.json
TESTSET_DIR = _e("TESTSET_DIR", ARCH / "datasets" / "testset")   # labels/ (SAM3 window labels), photos/
LABELS_DIR = _e("LABELS_DIR", TESTSET_DIR / "test100_labels_20261110" / "test100")  # balcony boxes, oriel cells, window_grid
HITMAP_DIR = _e("HITMAP_DIR", ASTRA_DIR / "layout_hitmaps")  # hitmap.py output: ours/<tid>.json, astra/<tid>.json
OUT_DIR = _e("OUT_DIR", ASTRA_DIR / "layout_check")          # scores (json)
WINDOW_AST_DIR = _e("WINDOW_AST_DIR", ARCH / "experiments" / "window_ast_predictor")  # window_ast (eval_testset)
