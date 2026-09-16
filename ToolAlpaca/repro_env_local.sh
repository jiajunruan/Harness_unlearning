# Planner-Actor experiment on ToolAlpaca -- LOCAL copy (jiajunr machine).
#   source repro_env_local.sh
#
# Differs from the upstream repro_env.sh (which pointed at /home/wenxian/...):
#   * Python comes from a dedicated conda env (toolalpaca), NOT a venv and NOT
#     the shared 'unlearning' env.  See RUNBOOK.md for how it was built.
#   * No TA_EXTRA_PKGS -- every dependency is installed inside the conda env.
#   * ACTOR_MODEL points at the local download of TangQiaoYu/ToolAlpaca-7B.
export TA_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---------------------------------------------------------------- interpreter
# Dedicated conda env.  Nothing here touches the 'unlearning' env.
export TA_PY=/users/2/jruan/miniconda3/envs/toolalpaca/bin/python
export TA_EXTRA_PKGS=
export PYTHONPATH=$TA_ROOT/data/general:$TA_ROOT
export NLTK_DATA=$TA_ROOT/data/general/nltk_data

# --------------------------------------------------------------- local models
# ${VAR:-default} so a caller can override without editing this file, e.g.
#   ACTOR_MODEL=tool_unlearn/ckpt/npo_gdr ./run_single.sh tool 0
export ACTOR_MODEL=${ACTOR_MODEL:-/projects/standard/mhong/shared/jiajunr/ToolAlpaca-7B}   # 7B, ~13GB fp16
# 13B planner not downloaded yet.  Point at a local copy once you have one.
export PLANNER_MODEL=${PLANNER_MODEL:-/projects/standard/mhong/shared/jiajunr/ToolAlpaca-13B}
# GPU layout for the 7B actor (single card, index *within* CUDA_VISIBLE_DEVICES).
export ACTOR_DEVICE=${ACTOR_DEVICE:-0}
# Collaboration layout (7B on GPU1, 13B sharded across both cards).  Only used
# by run_collab.sh; overridden there.
export PLANNER_MEM0=20GiB
export PLANNER_MEM1=6GiB

# -------------------------------------------------------------- remote models
# OpenAI-compatible endpoint: the API simulator invents the tool responses, the
# judge scores the finished trajectories.  The credential belongs only in the
# ignored repository-root .env file (see .env.example).
export OPENAI_API_BASE=${OPENAI_API_BASE:-https://api.openai.com/v1}
if [ -f "$TA_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$TA_ROOT/.env"
  set +a
fi
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in $TA_ROOT/.env or in the environment}"
export SIMULATOR_MODEL=${SIMULATOR_MODEL:-gpt-3.5-turbo}
export JUDGE_MODEL=${JUDGE_MODEL:-gpt-4-0613}
export JUDGE_WORKERS=16

# ---------------------------------------------------------------------- proxy
# REQUIRED: the endpoint is not reachable through a proxy and the simulator is
# on localhost.  Leaving these set makes both fail confusingly.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy='*'
export NO_PROXY='*'

# ------------------------------------------------------------------ simulator
export SIM_PORT=5678
export SIM_URL=http://127.0.0.1:${SIM_PORT}
