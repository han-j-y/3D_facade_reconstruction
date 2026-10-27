#!/bin/bash
# Current pipeline (run_pipeline.py defaults) on the 14 e2e facades, with Blender renders.
export FACADE_DSL_GPU=OPTIX
export HF_HUB_OFFLINE=1
cd /localhome/jiyeonh/Research/arch/experiments/facade_e2e
P=~/miniconda3/envs/arch/bin/python
B=/local-scratch/localhome/jiyeonh/local_install/usr/share/blender-5.1.0-linux-x64/blender
STEMS="cmp_b0001 cmp_b0005 cmp_b0010 cmp_b0020 cmp_b0031 cmp_b0050 cmp_b0071 cmp_b0083 cmp_b0100 cmp_b0125 cmp_b0150 cmp_b0170 cmp_b0198 cmp_b0215"
R=${R:-runs_current}
for S in ${ONLY:-$STEMS}; do
  $P run.py --image ../data/facades/base/$S.jpg --windows sam --out-dir $R/e2e_sam3_cluster_$S --device cuda \
    --blender-render --blender $B > /tmp/current_$S.log 2>&1 && echo "ok $S" || echo "fail $S"
done
