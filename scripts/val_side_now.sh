#!/usr/bin/env bash
# Validation facades with blocks, pushed code, side windows on, against runs_outer_bay2 (same code, off): renders and SAM IoU.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."; E=$PWD; P=/localhome/jiyeonh/miniconda3/envs/arch/bin/python; OFF=$E/runs_outer_bay2; F=$E/runs_val_side_now; TS=$E/../../datasets/testset; B=$F/eval
BL=/localhome/jiyeonh/local_install/usr/share/blender-5.1.0-linux-x64/blender; mkdir -p $F/logs $B/run $F/oblique $F/oblique_off
IDS="v14 v16 v24 v26"
for T in $IDS; do
  mkdir -p $F/$T; cp $OFF/$T/facade_dsl.json $F/$T/
  IMG=$($P -c "import json;print(json.load(open('$F/$T/facade_dsl.json'))['meta']['image'])")
  $P -c "
import sys; sys.argv = ['run.py']; sys.path.insert(0, '.')
import run
from pathlib import Path
run.run_balconies(Path('$IMG'), Path('$F/$T'), 'cuda', side_windows=True)" > $F/logs/$T.balcony.log 2>&1 || echo "fail balcony $T"
  mkdir -p $B/run/$T; cp $F/$T/balcony/facade_dsl_with_balconies.json $B/run/$T/facade_dsl.json
done
python3 -c "import json;print(json.dumps('$IDS'.split()))" > $F/ids.json
bash scripts/val_render_sheet.sh $F $F/ids.json &
(cd $TS && FACADE_E2E=$E $P tools/sam_iou_testset.py --run $B/run --out $B/iou --style real --detect pipeline --force-render > $B/iou.log 2>&1 || echo "fail iou"; tail -1 $B/iou.log | cut -c1-110) &
wait
for T in $IDS; do
  $BL -b -P $E/scripts/oblique_ours.py -- $F/$T/balcony/facade_dsl_with_balconies.json $F/oblique/$T.png 1000 > /dev/null 2>&1 || echo "fail oblique $T"
  $BL -b -P $E/scripts/oblique_ours.py -- $OFF/$T/balcony/facade_dsl_with_balconies.json $F/oblique_off/$T.png 1000 > /dev/null 2>&1 || echo "fail oblique off $T"
done
echo ALLDONE
