#!/usr/bin/env bash
# Score one finished tool condition.  Called automatically by run_single.sh /
# run_collab.sh; run it by hand to re-score or to resume an interrupted judge.
#
#   ./judge.sh B0
#
# Resumes rather than restarting: a judge run can die part-way (the judge
# occasionally returns null content), and re-judging from scratch would waste
# every verdict already paid for.
set -uo pipefail
source "$(dirname "$0")/repro_env_local.sh"
cd "$TA_ROOT"
export OPENAI_DEFAULT_MODEL=$JUDGE_MODEL

C=${1:?usage: ./judge.sh <condition>}
GEN=outputs/harness_tool/${C}.json
JUD=outputs/harness_tool/${C}_judge.json
[ -f "$GEN" ] || { echo "no generation output at $GEN"; exit 1; }

RESUME=""
[ -f "$JUD" ] && RESUME="--continue_run"
$TA_PY evaluation.py -api "$GEN" -out "$JUD" \
  --judge_model "$JUDGE_MODEL" --num_workers "${JUDGE_WORKERS:-16}" $RESUME

$TA_PY -c "
import json
s = json.load(open('$JUD'))['statistics']; n = s['num']
print('%s  Procedure %.1f  Response %.1f  T_T %.1f  (errors %d, n %d)'
      % ('$C', 100*s['process']['Yes']/n, 100*s['response']['Yes']/n,
         100*s['both']/n, s['error_num'], n))"
