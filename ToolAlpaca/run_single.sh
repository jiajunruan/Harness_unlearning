#!/usr/bin/env bash
# Single-agent baseline: one model runs the whole ReAct loop by itself.
#
#   ./run_single.sh tool    [gpu]   ToolAlpaca, 100 instructions -> condition B0
#   ./run_single.sh general [gpu]   BBH-Hard,   54 items         -> condition H0
#
# Only the 7B actor is loaded, so one card is enough.  The tool run is judged
# straight after generation.
set -uo pipefail
source "$(dirname "$0")/repro_env_local.sh"
cd "$TA_ROOT"
export PYTHONPATH=$TA_EXTRA_PKGS:$TA_ROOT/data/general:$TA_ROOT
export NLTK_DATA=$TA_ROOT/data/general/nltk_data
export CUDA_VISIBLE_DEVICES=${2:-0}
export ACTOR_DEVICE=0                     # index *within* CUDA_VISIBLE_DEVICES

case "${1:-}" in
  tool)
    OPENAI_DEFAULT_MODEL=$SIMULATOR_MODEL $TA_PY harness/run_toolalpaca.py \
      --condition B0 --server_url "$SIM_URL" \
      --offset 0 --length -1 --instruction_offset 0 --instruction_length -1 \
      -out ./outputs/harness_tool || exit 1
    "$TA_ROOT/judge.sh" B0
    ;;
  general)
    $TA_PY harness/run_general.py --condition H0 \
      --bbh_per_task 2 --ifeval_n 30 --out_dir ./outputs/harness_general
    ;;
  *) echo "usage: ./run_single.sh <tool|general> [gpu]"; exit 1 ;;
esac
