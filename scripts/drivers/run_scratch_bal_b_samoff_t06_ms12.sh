#!/bin/bash
export FACADE_DSL_GPU=OPTIX
export HF_HUB_OFFLINE=1
cd /localhome/jiyeonh/Research/arch/experiments/facade_e2e
P=~/miniconda3/envs/arch/bin/python
B=/local-scratch/localhome/jiyeonh/local_install/usr/share/blender-5.1.0-linux-x64/blender
STEMS="cmp_b0020 cmp_b0198 cmp_b0001 cmp_b0005 cmp_b0010 cmp_b0031 cmp_b0050 cmp_b0071 cmp_b0083 cmp_b0100 cmp_b0125 cmp_b0150 cmp_b0170 cmp_b0215"
G=scratch_bal_b_samoff_t06_ms12
T=scratch_bal_b
R=runs_tree_$G
for S in ${ONLY:-$STEMS}; do
  for i in 1 2 3; do
    $P run.py --image ../data/facades/base/$S.jpg --windows sam --min-side 13 --threshold 0.6 --sam-model facebook/sam3 --merge-mode ast --cluster-mode shape_then_struct --col-tol 0.045 \
      --structure-ckpt ../window_ast_predictor/runs/$T/best.pt \
      --out-dir $R/e2e_sam3_cluster_$S --device cuda --blender-render > /tmp/more_${G}_$S.log 2>&1 && { echo "ok $G $S"; break; }
    echo "fail $G $S $i"; sleep 15
  done
done
CS=$(echo $STEMS | tr ' ' ',')
$P scripts/symmetry_axis_ghost_continuous.py --sam3-root $R --out-dir $R/axis --stems $CS > /tmp/more_axis_$G.log 2>&1
$P scripts/render_sym_repair_fullstack.py --sam3-root $R --axis-root $R/axis --out-dir $R/fullstack --stems $CS --device cuda --blender $B > /tmp/more_full_$G.log 2>&1
echo "fullstack $G: $(rg -c 'blender=yes' /tmp/more_full_$G.log)"
echo ALLDONE
