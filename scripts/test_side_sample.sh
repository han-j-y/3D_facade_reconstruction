#!/usr/bin/env bash
# Sample of test100: same window DSLs with side windows off, then front and oblique renders of both versions.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."; E=$PWD; P=/localhome/jiyeonh/miniconda3/envs/arch/bin/python; V=$E/../../datasets/testset; O=$E/runs_test_side; F=$O/off
BL=/localhome/jiyeonh/local_install/usr/share/blender-5.1.0-linux-x64/blender; mkdir -p $F/logs
IDS="t013 t014 t109 t110 t074 t100 t104 t105 t021 t067 t103 t031 t005 t027 t032 t041 t017 t026 t075"
python3 -c "import json;print(json.dumps('$IDS'.split()))" > $O/sample_ids.json
for T in $IDS; do
  S=$(awk -v t=$T '$1==t{print $2}' $O/list.txt); mkdir -p $F/$T; cp $O/$T/facade_dsl.json $F/$T/
  [ -f $F/$T/balcony/manifest_done ] || { $P -c "
import sys; sys.argv = ['run.py']; sys.path.insert(0, '.')
import run
from pathlib import Path
run.run_balconies(Path('$V/photos/$S.jpg'), Path('$F/$T'), 'cuda')" > $F/logs/$T.balcony.log 2>&1 && touch $F/$T/balcony/manifest_done || echo "fail balcony $T"; }
done
echo "[$(date +%T)] off done"
bash scripts/val_render_sheet.sh $O $O/sample_ids.json &
bash scripts/val_render_sheet.sh $F $O/sample_ids.json &
wait
for R in $O $F; do mkdir -p $R/oblique; for T in $IDS; do
  D=$R/$T/balcony/facade_dsl_with_balconies.json; [ -f $D ] || D=$R/$T/facade_dsl.json
  [ -f $R/oblique/$T.png ] || $BL -b -P $E/scripts/oblique_ours.py -- $D $R/oblique/$T.png 1000 > /dev/null 2>&1 || echo "fail oblique $T"
done; done
echo ALLDONE
