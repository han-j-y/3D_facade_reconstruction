#!/usr/bin/env bash
# Front renders (photo wall colour) of windows-only and windows+balconies DSLs for every run dir, then a sheet.
set -u
RUN=$1; IDS=$2
E="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P=/localhome/jiyeonh/miniconda3/envs/arch/bin/python
B=/localhome/jiyeonh/local_install/usr/share/blender-5.1.0-linux-x64/blender
mkdir -p "$RUN/front"
for T in $(python3 -c "import json;print(' '.join(json.load(open('$IDS'))))"); do
  D=$RUN/$T; [ -f "$D/facade_dsl.json" ] || continue
  WALL=$($P -c "
import json,sys; sys.path.insert(0,'/localhome/jiyeonh/Research/arch/datasets/testset/tools'); import sam_iou_testset as m
print(m.photo_wall_rgb(json.load(open('$D/facade_dsl.json'))))" 2>/dev/null | tail -1)
  export FACADE_DSL_FRAME_RGB=0.85,0.85,0.83 FACADE_DSL_GLASS_RGB=0.03,0.04,0.05 FACADE_DSL_GLASS_TRANSMISSION=0.0 FACADE_DSL_RECESS=0.15 \
    FACADE_DSL_SUN_ROT=0.6,0.0,0.5 FACADE_DSL_SUN_ENERGY=4.0 FACADE_DSL_DOOR_RGB=0.06,0.035,0.02 FACADE_DSL_WALL_RGB=$WALL
  [ -f "$RUN/front/${T}_ours.png" ] || $B -b -P $E/scripts/front_real.py -- "$D/facade_dsl.json" "$RUN/front/${T}_ours.png" 1200 > /dev/null 2>&1
  BD=$D/balcony/facade_dsl_with_balconies.json
  [ -f "$BD" ] && { [ -f "$RUN/front/${T}_balc.png" ] || $B -b -P $E/scripts/front_real.py -- "$BD" "$RUN/front/${T}_balc.png" 1200 > /dev/null 2>&1; }
done
