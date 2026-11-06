#!/usr/bin/env bash
# More samples for the balcony and side-window work: the validation candidate pool without val30 (no test overlap by
# construction). Windows in parallel, balconies one by one (they share a SAM3 cache file). Resumes where it stopped.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."; E=$PWD; P=/localhome/jiyeonh/miniconda3/envs/arch/bin/python; V=$E/../../datasets/valset; O=$E/runs_cand; mkdir -p $O/logs
$P - > $O/list.txt <<PY
import json
c=json.load(open("$V/candidates_ids.json")); v=set(json.load(open("$V/val30_ids.json")).values())
for k,s in c.items():
    if s not in v: print(k,s)
PY
win() { T=$1; S=$2; [ -f $O/$T/facade_dsl.json ] || $P run.py --image $V/candidates/$S.jpg --windows sam --out-dir $O/$T --device cuda > $O/logs/$T.log 2>&1 || echo "fail windows $T"; }
export -f win; export O P V
xargs -a $O/list.txt -P 3 -L 1 bash -c 'win "$0" "$1"'
echo "[$(date +%T)] windows done"
while read T S; do
  [ -f $O/$T/facade_dsl.json ] || continue
  [ -f $O/$T/balcony/facade_dsl_with_balconies.json ] || $P -c "
import sys; sys.argv = ['run.py']; sys.path.insert(0, '.')
import run
from pathlib import Path
run.run_balconies(Path('$V/candidates/$S.jpg'), Path('$O/$T'), 'cuda')" > $O/logs/$T.balcony.log 2>&1 || echo "fail balcony $T"
done < $O/list.txt
echo "[$(date +%T)] balconies done"; echo ALLDONE
