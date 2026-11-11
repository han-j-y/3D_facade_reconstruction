#!/usr/bin/env bash
# test100 with the current pipeline (windows + balconies), then a Blender oblique render per facade (Windows).
# Resumable: a photo whose oblique render exists is skipped; a photo whose merged DSL exists only renders.
# Output: runs/test100_current/<stem>/, runs/test100_current/oblique/<stem>.png, logs/, progress per photo on stdout.
#   bash scripts/test100_current.sh [photo_dir] [out_dir]
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT_W=$(pwd -W)   # Windows form (C:/...): oblique_ours.py changes its working directory, so paths must be absolute
PY=/c/Users/hiroo/anaconda3/envs/glodon/python.exe
BL="/c/Program Files/Blender Foundation/Blender 5.2/blender.exe"
PH=${1:-data/test100/photos}
OUT=${2:-runs/test100_current}
export PYTHONIOENCODING=utf-8
mkdir -p "$OUT/logs" "$OUT/oblique"
n=0; total=$(ls "$PH"/*.jpg | wc -l)
for IMG in "$PH"/*.jpg; do
  n=$((n + 1)); S=$(basename "$IMG" .jpg)
  if [ -f "$OUT/oblique/$S.png" ]; then echo "[$n/$total] skip $S"; continue; fi
  t0=$(date +%s)
  if [ ! -f "$OUT/$S/facade_dsl_with_balconies.json" ]; then
    "$PY" run.py --image "$IMG" --out-dir "$OUT/$S" --device cuda --with-balconies > "$OUT/logs/$S.log" 2>&1 || echo "fail pipeline $S"
  fi
  DSL="$OUT/$S/facade_dsl_with_balconies.json"; [ -f "$DSL" ] || DSL="$OUT/$S/facade_dsl.json"
  if [ -f "$DSL" ]; then
    "$BL" -b -P "$ROOT_W/scripts/oblique_ours.py" -- "$ROOT_W/$DSL" "$ROOT_W/$OUT/oblique/$S.png" 1000 \
      > "$OUT/logs/$S.blender.log" 2>&1 || echo "fail blender $S"
  fi
  echo "[$n/$total] $S $(( $(date +%s) - t0 ))s $( [ -f "$OUT/oblique/$S.png" ] && echo ok || echo NO_RENDER )"
done
echo ALLDONE
