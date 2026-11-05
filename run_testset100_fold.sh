#!/usr/bin/env bash
# Full test-set run with the fold gate (run_pipeline default --gate-mode fold):
#   main (+ balcony track), WindowOpens, leave-one-out ablations, cumulative build-up; eval_testset on all,
#   SAM IoU (real style, two-tier, doors excluded) on all. Usage: ./run_testset100_fold.sh [TAG]
set -u
TESTSET=/localhome/jiyeonh/Research/arch/datasets/testset
E=/localhome/jiyeonh/Research/arch/experiments/facade_e2e
P=/localhome/jiyeonh/miniconda3/envs/arch/bin/python
TAG=${1:-testset100_fold}
MAIN=runs_${TAG}_main
BALC=runs_${TAG}_balconies
OUT="$E/runs_${TAG}_eval"
cd "$E"; mkdir -p "$MAIN/logs" "$BALC" "$OUT"
IDS=$(python3 -c "import json;print(' '.join(json.load(open('$TESTSET/labels/ids.json'))))")
stem() { python3 -c "import json;print(json.load(open('$TESTSET/labels/ids.json'))['$1'])"; }

echo "[$(date +%T)] main"
for T in $IDS; do
  [ -f "$MAIN/$T/facade_dsl.json" ] && continue
  $P run.py --image "$TESTSET/photos/$(stem $T).jpg" --windows sam --out-dir "$MAIN/$T" --device cuda \
    > "$MAIN/logs/$T.log" 2>&1 || echo "fail main $T"
done
echo "[$(date +%T)] main done"

balconies() {
  for T in $IDS; do
    [ -f "$BALC/$T/facade_dsl.json" ] && continue
    [ -f "$MAIN/$T/facade_dsl.json" ] || continue
    $P -c "
import sys; sys.argv = ['run.py']; sys.path.insert(0, '.')
import run
from pathlib import Path
run.run_balconies(Path('$TESTSET/photos/$(stem $T).jpg'), Path('$MAIN/$T'), 'cuda')" > "$MAIN/logs/$T.balcony.log" 2>&1 \
      || echo "fail balcony $T"
    if [ -f "$MAIN/$T/facade_dsl_with_balconies.json" ]; then
      mkdir -p "$BALC/$T"; cp "$MAIN/$T/facade_dsl_with_balconies.json" "$BALC/$T/facade_dsl.json"
    fi
  done
  echo "[$(date +%T)] balconies done"
}

A="$TESTSET/tools/run_ablation.sh"
export SRC=$MAIN PREFIX=runs_${TAG}_abl
# leave-one-out (from the full system)
#   nosym: no fold pairs, no vertical merges (no symmetry links at all; cross-group merging has nothing to merge)
loo1() { $A nofold --no-fold-pairs; $A novertical --no-vertical-merge; $A nocross --no-cross-group; }
loo2() { $A nosplit --no-structure-split; $A novote --no-local-sym; $A nosym --no-fold-pairs --no-vertical-merge; }
# cumulative build-up: tree only -> + fold -> + vertical -> + cross-group -> (+ vote = main)
cum() {
  $A cum0_tree --no-fold-pairs --no-vertical-merge --no-cross-group --no-local-sym
  $A cum1_fold --no-vertical-merge --no-cross-group --no-local-sym
  $A cum2_vertical --no-cross-group --no-local-sym
  $A cum3_cross --no-local-sym
  $P "$TESTSET/tools/run_windowopens.py" --run "$MAIN" --out runs_${TAG}_windowopens > "$MAIN/logs/windowopens.log" 2>&1 \
    || echo "fail windowopens"
  echo "[$(date +%T)] cumulative + windowopens done"
}
balconies & loo1 & loo2 & cum & wait
echo "[$(date +%T)] all runs done"

RUNS="$MAIN runs_${TAG}_windowopens"
for N in nosym nofold novertical nocross nosplit novote cum0_tree cum1_fold cum2_vertical cum3_cross; do RUNS="$RUNS runs_${TAG}_abl_$N"; done
cd "$TESTSET"
for R in $RUNS; do
  $P tools/eval_testset.py --run "$E/$R" --out "$OUT/$R.json" > "$OUT/$R.txt" 2>&1 || echo "fail eval $R"
done
$P tools/eval_testset.py --run "$E/$MAIN" --no-types --out "$OUT/${MAIN}_notypes.json" > "$OUT/${MAIN}_notypes.txt" 2>&1
echo "[$(date +%T)] eval done"

sam() {  # $1 = run dir name
  mkdir -p "$OUT/sam_iou_$1"
  $P tools/sam_iou_testset.py --run "$E/$1" --out "$OUT/sam_iou_$1" --style real --detect pipeline --force-render \
    > "$OUT/sam_iou_$1/log.txt" 2>&1 || echo "fail sam_iou $1"
}
set -- $MAIN $BALC $RUNS
ALL=$(printf "%s\n" "$@" | awk '!seen[$0]++')
i=0
for R in $ALL; do
  sam "$R" & i=$((i + 1))
  if [ $((i % 3)) -eq 0 ]; then wait; fi
done
wait
echo "[$(date +%T)] sam_iou done -> $OUT"
