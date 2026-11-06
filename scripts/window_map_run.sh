#!/usr/bin/env bash
# Open balconies on their windows on every facade, or on their own box with no window behind them: balcony track,
# front renders and SAM IoU with balconies for the 20 validation facades with balconies, and the 9 candidates with blocks.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."; E=$PWD; P=/localhome/jiyeonh/miniconda3/envs/arch/bin/python; F=$E/runs_window_map; rm -rf $F/front; mkdir -p $F/logs $F/oblique; TS=/local-scratch/localhome/jiyeonh/Research/arch/datasets/testset; B=$E/runs_val30_combined_eval/window_map
IDS="v04 v07 v08 v09 v11 v13 v14 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28 c031 c032 c041 c047 c050 c051 c052 c092 c094"
for T in $IDS; do
  case $T in v*) SRC=$E/runs_val30_combined;; *) SRC=$E/runs_cand;; esac
  mkdir -p $F/$T; cp $SRC/$T/facade_dsl.json $F/$T/
  IMG=$($P -c "import json;print(json.load(open('$F/$T/facade_dsl.json'))['meta']['image'])")
  $P -c "
import sys; sys.argv = ['run.py']; sys.path.insert(0, '.')
import run
from pathlib import Path
run.run_balconies(Path('$IMG'), Path('$F/$T'), 'cuda')" > $F/logs/$T.balcony.log 2>&1 || echo "fail balcony $T"
  case $T in v*) mkdir -p $B/run/$T; cp $F/$T/balcony/facade_dsl_with_balconies.json $B/run/$T/facade_dsl.json;; esac
  echo "[$(date +%T)] $T"
done
python3 -c "import json;print(json.dumps('$IDS'.split()))" > $F/ids.json
bash scripts/val_render_sheet.sh $F $F/ids.json &
(cd $TS && FACADE_E2E=$E $P tools/sam_iou_testset.py --run $B/run --out $B/iou --style real --detect pipeline --force-render > $B/iou.log 2>&1 || echo "fail iou"; tail -1 $B/iou.log | cut -c1-110) &
wait; echo ALLDONE
