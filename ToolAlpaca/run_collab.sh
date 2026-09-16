#!/usr/bin/env bash
# Planner-actor collaboration: the 13B writes reasoning, the 7B performs every
# action.  See EXPERIMENT.md for the mechanism.
#
#   ./run_collab.sh tool [H1|H2]   ToolAlpaca, 100 instructions
#   ./run_collab.sh general        BBH-Hard,   54 items
#
#   H1  the planner's role is implicit; the boundary is truncation alone
#   H2  the planner is additionally told it is advising and the 7B decides
#
# Both models are resident at once (7B on GPU1, 13B sharded 20GiB/6GiB across
# both cards), so this needs the whole box -- run one at a time.
set -uo pipefail
source "$(dirname "$0")/repro_env_local.sh"
cd "$TA_ROOT"
export PYTHONPATH=$TA_EXTRA_PKGS:$TA_ROOT/data/general:$TA_ROOT
export NLTK_DATA=$TA_ROOT/data/general/nltk_data
export CUDA_VISIBLE_DEVICES=0,1
export ACTOR_DEVICE=1                     # 7B on GPU1; the 13B fills GPU0 first

case "${1:-}" in
  tool)
    COND=${2:-H1}
    OPENAI_DEFAULT_MODEL=$SIMULATOR_MODEL $TA_PY harness/run_toolalpaca.py \
      --condition "$COND" --server_url "$SIM_URL" \
      --offset 0 --length -1 --instruction_offset 0 --instruction_length -1 \
      -out ./outputs/harness_tool || exit 1
    "$TA_ROOT/judge.sh" "$COND"
    ;;
  general)
    $TA_PY harness/run_general.py --condition H_plan13 \
      --bbh_per_task 2 --ifeval_n 30 --out_dir ./outputs/harness_general
    ;;
  *) echo "usage: ./run_collab.sh <tool|general> [H1|H2]"; exit 1 ;;
esac
