#!/usr/bin/env bash
set -u
# Current pipeline (run_pipeline.py defaults) on the 14 e2e facades, with Blender renders.
export FACADE_DSL_GPU=OPTIX
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
P=${PYTHON:-python}
DATA_DIR=${DATA_DIR:-data/facades/base}
STEMS="cmp_b0001 cmp_b0005 cmp_b0010 cmp_b0020 cmp_b0031 cmp_b0050 cmp_b0071 cmp_b0083 cmp_b0100 cmp_b0125 cmp_b0150 cmp_b0170 cmp_b0198 cmp_b0215"
R=${R:-runs_current}
mkdir -p "$R/logs"
for S in ${ONLY:-$STEMS}; do
  "$P" run.py --image "$DATA_DIR/$S.jpg" --windows sam --out-dir "$R/e2e_sam3_cluster_$S" --device cuda \
    --blender-render > "$R/logs/$S.log" 2>&1 && echo "ok $S" || echo "fail $S"
done
