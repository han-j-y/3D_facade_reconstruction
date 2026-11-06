#!/usr/bin/env bash
# Front renders and SAM IoU with the facade region, windows only and with balconies; windows only without it, same compiler.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."; E=$PWD; P=/localhome/jiyeonh/miniconda3/envs/arch/bin/python; TS=/local-scratch/localhome/jiyeonh/Research/arch/datasets/testset; F=$E/runs_region; B=$F/eval
$P scripts/facade_region_apply.py | tail -1
IDS=$(python3 -c "print(' '.join(f'v{i:02d}' for i in range(30)))"); python3 -c "import json;print(json.dumps('$IDS'.split()))" > $F/ids.json
mkdir -p $B/win/run $B/balc/run $B/base/run
for T in $IDS; do
  mkdir -p $B/win/run/$T $B/base/run/$T; cp $F/$T/facade_dsl.json $B/win/run/$T/; cp $E/runs_val30_combined/$T/facade_dsl.json $B/base/run/$T/
  [ -f $F/$T/balcony/facade_dsl_with_balconies.json ] && { mkdir -p $B/balc/run/$T; cp $F/$T/balcony/facade_dsl_with_balconies.json $B/balc/run/$T/facade_dsl.json; }
done
bash scripts/val_render_sheet.sh $F $F/ids.json &
for k in win balc base; do (cd $TS && FACADE_E2E=$E $P tools/sam_iou_testset.py --run $B/$k/run --out $B/$k/iou --style real --detect pipeline --force-render > $B/$k/iou.log 2>&1 || echo "fail iou $k"; echo $k $(tail -1 $B/$k/iou.log | cut -c1-110)) & done
wait
# The real path once: detection through the DSL, from saved window detections.
$P run.py --from-index $E/../facade_e2e/runs_val30/v28/index_raw.json --windows sam --out-dir $F/pipeline_v28 --device cuda > $F/pipeline_v28.log 2>&1 || echo "fail pipeline"
$P -c "import json;m=json.load(open('$F/pipeline_v28/facade_dsl.json'))['meta'];print('pipeline v28',m.get('facade_region_xyxy'),m.get('facade_region_source'),m['columns_xy'][0][0],m['columns_xy'][-1][1])"
echo ALLDONE
