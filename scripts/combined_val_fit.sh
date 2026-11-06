#!/usr/bin/env bash
# Enclosed blocks fitted to their windows: balcony track again on the 4 validation facades with enclosed units (open
# balconies beside a narrowed block now survive), then front renders and SAM IoU for those 4.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."; E=$PWD; P=/localhome/jiyeonh/miniconda3/envs/arch/bin/python; M=$E/../facade_e2e; O=$E/runs_val30_combined; F=$E/runs_val30_fit; mkdir -p $F/logs
TS=/local-scratch/localhome/jiyeonh/Research/arch/datasets/testset; IDS="v14 v16 v24 v26"
B=$E/runs_val30_combined_eval/fit
for T in $IDS; do
  mkdir -p $F/$T; cp $O/$T/facade_dsl.json $F/$T/facade_dsl.json
  IMG=$($P -c "import json;print(json.load(open('$M/runs_val30/$T/index_raw.json'))['facade_path'])")
  $P -c "
import sys; sys.argv = ['run.py']; sys.path.insert(0, '.')
import run
from pathlib import Path
run.run_balconies(Path('$IMG'), Path('$F/$T'), 'cuda')" > $F/logs/$T.balcony.log 2>&1 || echo "fail balcony $T"
  mkdir -p $B/run/$T; cp $F/$T/balcony/facade_dsl_with_balconies.json $B/run/$T/facade_dsl.json
  echo "[$(date +%T)] $T done"
done
python3 -c "import json;print(json.dumps('$IDS'.split()))" > $F/ids.json
bash scripts/val_render_sheet.sh $F $F/ids.json &
(cd $TS && FACADE_E2E=$E $P tools/sam_iou_testset.py --run $B/run --out $B/iou --style real --detect pipeline --force-render > $B/iou.log 2>&1 || echo "fail iou"; tail -1 $B/iou.log | cut -c1-120) &
wait; echo ALLDONE
