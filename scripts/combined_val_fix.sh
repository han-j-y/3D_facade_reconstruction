#!/usr/bin/env bash
# Validation after the balcony scale fix in balcony_layout.photo_x_span: same DSLs as runs_val30_combined, new renders and SAM IoU.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."; E=$PWD; O=$E/runs_val30_combined; F=$E/runs_val30_scalefix; P=/localhome/jiyeonh/miniconda3/envs/arch/bin/python
TS=/local-scratch/localhome/jiyeonh/Research/arch/datasets/testset
mkdir -p $F; for T in $(python3 -c "print(' '.join(f'v{i:02d}' for i in range(30)))"); do [ -e $F/$T ] || ln -s $O/$T $F/$T; done
bash scripts/val_render_sheet.sh $F $O/ids.json
B=$E/runs_val30_combined_eval/balc_scalefix; mkdir -p $B
for T in $(ls $O/*/balcony/facade_dsl_with_balconies.json | xargs -n1 dirname | xargs -n1 dirname | xargs -n1 basename); do mkdir -p $B/run/$T; cp $O/$T/balcony/facade_dsl_with_balconies.json $B/run/$T/facade_dsl.json; done
(cd $TS && $P tools/sam_iou_testset.py --run $B/run --out $B/iou --style real --detect pipeline --force-render > $B/iou.log 2>&1 || echo "fail iou")
echo ALLDONE
