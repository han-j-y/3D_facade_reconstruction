#!/usr/bin/env bash
# test100 with the current pipeline, side windows on, to find facades with bay windows for a before/after sheet.
# Windows in parallel, balconies one by one (they share a SAM3 cache file). Resumes where it stopped.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."; E=$PWD; P=/localhome/jiyeonh/miniconda3/envs/arch/bin/python; V=$E/../../datasets/testset; O=$E/runs_test_side; mkdir -p $O/logs
$P -c "import json
for t,s in json.load(open('$V/labels/ids.json')).items(): print(t,s)" > $O/list.txt
win() { T=$1; S=$2; [ -f $O/$T/facade_dsl.json ] || $P run.py --image $V/photos/$S.jpg --windows sam --out-dir $O/$T --device cuda > $O/logs/$T.log 2>&1 || echo "fail windows $T"; }
export -f win; export O P V
xargs -a $O/list.txt -P 3 -L 1 bash -c 'win "$0" "$1"'
echo "[$(date +%T)] windows done"
while read T S; do
  [ -f $O/$T/facade_dsl.json ] || continue
  [ -f $O/$T/balcony/manifest_done ] || { $P -c "
import sys; sys.argv = ['run.py']; sys.path.insert(0, '.')
import run
from pathlib import Path
run.run_balconies(Path('$V/photos/$S.jpg'), Path('$O/$T'), 'cuda', side_windows=True)" > $O/logs/$T.balcony.log 2>&1 && touch $O/$T/balcony/manifest_done || echo "fail balcony $T"; }
done < $O/list.txt
echo "[$(date +%T)] balconies done"; echo ALLDONE
