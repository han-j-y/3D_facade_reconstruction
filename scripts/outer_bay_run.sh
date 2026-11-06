#!/usr/bin/env bash
# Balconies past the window grid in outer bays: balcony track, front renders and SAM IoU on the 20 validation facades with balconies.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."; E=$PWD; P=/localhome/jiyeonh/miniconda3/envs/arch/bin/python; F=$E/runs_outer_bay2; TS=/local-scratch/localhome/jiyeonh/Research/arch/datasets/testset; B=$F/eval; mkdir -p $F/logs $B/run
IDS="v04 v07 v08 v09 v11 v13 v14 v16 v17 v18 v19 v20 v21 v22 v23 v24 v25 v26 v27 v28"
for T in $IDS; do
  mkdir -p $F/$T; cp $E/runs_region/$T/facade_dsl.json $F/$T/
  IMG=$($P -c "import json;print(json.load(open('$F/$T/facade_dsl.json'))['meta']['image'])")
  $P -c "
import sys; sys.argv = ['run.py']; sys.path.insert(0, '.')
import run
from pathlib import Path
run.run_balconies(Path('$IMG'), Path('$F/$T'), 'cuda')" > $F/logs/$T.balcony.log 2>&1 || echo "fail balcony $T"
  mkdir -p $B/run/$T; cp $F/$T/balcony/facade_dsl_with_balconies.json $B/run/$T/facade_dsl.json
done
python3 -c "import json;print(json.dumps('$IDS'.split()))" > $F/ids.json
bash scripts/val_render_sheet.sh $F $F/ids.json &
(cd $TS && FACADE_E2E=$E $P tools/sam_iou_testset.py --run $B/run --out $B/iou --style real --detect pipeline --force-render > $B/iou.log 2>&1 || echo "fail iou"; tail -1 $B/iou.log | cut -c1-110) &
wait; echo ALLDONE
