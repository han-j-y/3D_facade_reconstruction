#!/usr/bin/env bash
# Bay blocks between the front and side windows: balcony track again on the 4 validation facades with enclosed
# units, then front and oblique renders and SAM IoU for those 4.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."; E=$PWD; P=/localhome/jiyeonh/miniconda3/envs/arch/bin/python; M=$E/../facade_e2e; O=$E/runs_val30_combined; F=$E/runs_val30_mid; mkdir -p $F/logs
TS=/local-scratch/localhome/jiyeonh/Research/arch/datasets/testset; IDS="v14 v16 v24 v26"
B=$E/runs_val30_combined_eval/mid
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
wait
BL=/localhome/jiyeonh/local_install/usr/share/blender-5.1.0-linux-x64/blender; mkdir -p $F/oblique
for T in $IDS; do
  export FACADE_DSL_FRAME_RGB=0.85,0.85,0.83 FACADE_DSL_GLASS_RGB=0.03,0.04,0.05 FACADE_DSL_GLASS_TRANSMISSION=0.0 FACADE_DSL_RECESS=0.15 FACADE_DSL_WALL_RGB=0.72,0.68,0.6
  $BL -b -P $E/scripts/oblique_ours.py -- $F/$T/balcony/facade_dsl_with_balconies.json $F/oblique/${T}_side.png 1000 > $F/logs/$T.oblique.log 2>&1 || echo "fail oblique $T"
  $BL -b -P $E/scripts/oblique_ours.py -- $E/runs_val30_side/$T/balcony/facade_dsl_with_balconies.json $F/oblique/${T}_before.png 1000 > $F/logs/$T.oblique_before.log 2>&1 || echo "fail oblique before $T"
done
echo ALLDONE
