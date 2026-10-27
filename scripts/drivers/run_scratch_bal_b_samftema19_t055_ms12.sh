#!/usr/bin/env bash
set -u
export FACADE_DSL_GPU=OPTIX
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
P=${PYTHON:-python}
DATA_DIR=${DATA_DIR:-data/facades/base}
SAM_MODEL=${SAM_MODEL:-facebook/sam3}
STEMS="cmp_b0020 cmp_b0198 cmp_b0001 cmp_b0005 cmp_b0010 cmp_b0031 cmp_b0050 cmp_b0071 cmp_b0083 cmp_b0100 cmp_b0125 cmp_b0150 cmp_b0170 cmp_b0215"
G=scratch_bal_b_samftema19_t055_ms12
R=runs_tree_$G
mkdir -p "$R/logs"
for S in ${ONLY:-$STEMS}; do
  for i in 1 2 3; do
    "$P" run.py --image "$DATA_DIR/$S.jpg" --windows sam --min-side 13 --threshold 0.55 --sam-model "$SAM_MODEL" --merge-mode ast --cluster-mode gate --col-tol 0.045 \
      --structure-ckpt checkpoints/structure_best.pt \
      --out-dir "$R/e2e_sam3_cluster_$S" --device cuda --blender-render > "$R/logs/$S.log" 2>&1 && { echo "ok $G $S"; break; }
    echo "fail $G $S $i"; sleep 15
  done
done
CS=$(echo $STEMS | tr ' ' ',')
"$P" scripts/symmetry_axis_ghost_continuous.py --sam3-root "$R" --out-dir "$R/axis" --stems "$CS" > "$R/logs/axis.log" 2>&1
"$P" scripts/render_sym_repair_fullstack.py --sam3-root "$R" --axis-root "$R/axis" --out-dir "$R/fullstack" --stems "$CS" --device cuda > "$R/logs/fullstack.log" 2>&1
echo "fullstack $G: $(rg -c 'blender=yes' "$R/logs/fullstack.log")"
echo ALLDONE
