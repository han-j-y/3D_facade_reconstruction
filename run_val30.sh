#!/usr/bin/env bash
# Validation set (datasets/valset, 30 facades): main pipeline (fold gate) -> balcony track (railing v4, cell_clf enclosed)
# -> oblique renders (photo wall colour) of windows-only and windows+balconies. Usage: ./run_val30.sh [OUT]
set -u
VAL=/localhome/jiyeonh/Research/arch/datasets/valset
E="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P=/localhome/jiyeonh/miniconda3/envs/arch/bin/python
B=/localhome/jiyeonh/local_install/usr/share/blender-5.1.0-linux-x64/blender
OBL=${OBL:-$E/scripts/oblique_ours.py}
OUT=${1:-$E/runs_val30}
cd "$E"; mkdir -p "$OUT/logs" "$OUT/oblique"
IDS=$(python3 -c "import json;print(' '.join(json.load(open('$VAL/labels/ids.json'))))")
stem() { python3 -c "import json;print(json.load(open('$VAL/labels/ids.json'))['$1'])"; }
for T in $IDS; do
  S=$(stem $T); IMG=$VAL/photos/$S.jpg; D=$OUT/$T
  [ -f "$D/facade_dsl.json" ] || $P run.py --image "$IMG" --windows sam --out-dir "$D" --device cuda > "$OUT/logs/$T.log" 2>&1 || echo "fail main $T"
  [ -f "$D/balcony/facade_dsl_with_balconies.json" ] || [ ! -f "$D/facade_dsl.json" ] || \
    $P balcony_pipeline/run_balcony.py --image "$IMG" --windows-dsl "$D/facade_dsl.json" --out-dir "$D/balcony" --device cuda > "$OUT/logs/$T.balcony.log" 2>&1 || echo "fail balcony $T"
  echo "[$(date +%T)] $T done"
done
echo "[$(date +%T)] pipeline done"
for T in $IDS; do
  D=$OUT/$T; [ -f "$D/facade_dsl.json" ] || continue
  WALL=$($P -c "
import json,sys; sys.path.insert(0,'/localhome/jiyeonh/Research/arch/datasets/testset/tools'); import sam_iou_testset as m
print(m.photo_wall_rgb(json.load(open('$D/facade_dsl.json'))))" 2>/dev/null | tail -1)
  export FACADE_DSL_FRAME_RGB=0.85,0.85,0.83 FACADE_DSL_GLASS_RGB=0.03,0.04,0.05 FACADE_DSL_GLASS_TRANSMISSION=0.0 FACADE_DSL_RECESS=0.15 \
    FACADE_DSL_SUN_ROT=0.6,0.0,0.5 FACADE_DSL_SUN_ENERGY=4.0 FACADE_DSL_DOOR_RGB=0.06,0.035,0.02 FACADE_DSL_WALL_RGB=$WALL
  $B -b -P "$OBL" -- "$D/facade_dsl.json" "$OUT/oblique/${T}_ours.png" 1000 > "$OUT/logs/$T.obl.log" 2>&1
  BD=$D/balcony/facade_dsl_with_balconies.json
  [ -f "$BD" ] && $B -b -P "$OBL" -- "$BD" "$OUT/oblique/${T}_balc.png" 1000 > "$OUT/logs/$T.obl_balc.log" 2>&1
done
echo "[$(date +%T)] renders done -> $OUT"
