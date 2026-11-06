#!/usr/bin/env bash
# Validation on the merged code: windows from the same saved detections (must match the control), balcony track with
# the branch defaults, front renders, SAM IoU with balconies for this run and for the earlier sam3 run.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."; E=$PWD; P=/localhome/jiyeonh/miniconda3/envs/arch/bin/python; M=$E/../facade_e2e; O=$E/runs_val30_combined; mkdir -p $O/logs
TS=/local-scratch/localhome/jiyeonh/Research/arch/datasets/testset
IDS=$(python3 -c "import json;print(' '.join(f'v{i:02d}' for i in range(30)))")
for T in $IDS; do
  [ -f $O/$T/facade_dsl.json ] || $P run.py --from-index $M/runs_val30/$T/index_raw.json --windows sam --out-dir $O/$T --device cuda > $O/logs/$T.log 2>&1 || echo "fail main $T"
  [ -f $O/$T/balcony/facade_dsl_with_balconies.json ] || { IMG=$($P -c "import json;print(json.load(open('$M/runs_val30/$T/index_raw.json'))['facade_path'])")
    $P -c "
import sys; sys.argv = ['run.py']; sys.path.insert(0, '.')
import run
from pathlib import Path
run.run_balconies(Path('$IMG'), Path('$O/$T'), 'cuda')" > $O/logs/$T.balcony.log 2>&1 || echo "fail balcony $T"; }
  echo "[$(date +%T)] $T done"
done
$P - <<PY
import json
bad=[t for t in "$IDS".split() if any(json.load(open(f"$O/{t}/facade_dsl.json"))[k]!=json.load(open(f"$M/runs_val30_control/{t}/facade_dsl.json"))[k] for k in("instances","window_types","door_types","layout"))]
print("windows differ from control:",bad or "none")
PY
python3 -c "import json;print(json.dumps('$IDS'.split()))" > $O/ids.json
bash scripts/val_render_sheet.sh $O $O/ids.json
for k in combined:$O old:$M/runs_val30; do N=${k%%:*}; D=${k#*:}; B=$E/runs_val30_combined_eval/balc_$N; mkdir -p $B
  for T in $IDS; do [ -f $D/$T/balcony/facade_dsl_with_balconies.json ] && { mkdir -p $B/run/$T; cp $D/$T/balcony/facade_dsl_with_balconies.json $B/run/$T/facade_dsl.json; }; done
  (cd $TS && $P tools/sam_iou_testset.py --run $B/run --out $B/iou --style real --detect pipeline --force-render > $B/iou.log 2>&1 || echo "fail iou $N") &
done; wait
echo ALLDONE
