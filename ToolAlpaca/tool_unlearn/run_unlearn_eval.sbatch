#!/usr/bin/env bash
#SBATCH --job-name=unlearn_eval
#SBATCH --partition=mhong
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --output=/users/2/jruan/Stable_evolving/logs/unlearn_eval_%j.out
#SBATCH --error=/users/2/jruan/Stable_evolving/logs/unlearn_eval_%j.err

# ============================================================================
# Eval sweep for the ToolAlpaca NPO+GDR unlearning run (tool_unlearn).
#
#   Model configs (4):   unlearned-7B alone | original-7B alone |
#                        original-7B + 13B planner | unlearned-7B + 13B planner
#   Metrics     (2):     T_f (forget tools, held-out) and T_r (retain tools)
#   Question:            does the 13B planner make the unlearned 7B "remember"
#                        the forgotten tools again (i.e. push T_f back up)?
#
# Usage:
#   sbatch tool_unlearn/run_unlearn_eval.sbatch            # full 8-eval sweep
#   sbatch tool_unlearn/run_unlearn_eval.sbatch smoke      # 4-instance sanity
#
# NOTE on GPUs: the sbatch reserves 2 A100s.  single-mode evals only touch
# CUDA 0; collab-mode evals need both (the 13B is ~26GB fp16 and is sharded
# 20GiB/6GiB across the two cards -- it cannot share one 40GB card with the
# 7B actor).  If you only want the single-mode numbers, edit --gres to
# gpu:a100:1 and comment out the collab block.
# ============================================================================
set -uo pipefail
RUN=${1:-full}

ROOT=/users/2/jruan/Stable_evolving
cd "$ROOT" || exit 1

# --- environment: conda env toolalpaca (the one training used) ----------------
export TA_PY=/users/2/jruan/miniconda3/envs/toolalpaca/bin/python
# evaluate.sh builds PYTHONPATH as $TA_EXTRA_PKGS:$ROOT; point it at the same
# dirs as the known-good repro_env_local.sh ($ROOT/data/general:$ROOT).
export TA_EXTRA_PKGS=$ROOT/data/general
export TRITON_CACHE_DIR=/tmp/triton_cd   # avoid NFS triton autotune warning
# DeepSeek is reachable directly; a leaked proxy breaks simulator + judge.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy='*' NO_PROXY='*'

# --- models (all under /projects/standard/mhong/shared/jiajunr) ---------------
ORIG=/projects/standard/mhong/shared/jiajunr/ToolAlpaca-7B
UNLEARN=/projects/standard/mhong/shared/jiajunr/npo_gdr_ckpt/npo_gdr_ep5
PLANNER=/projects/standard/mhong/shared/jiajunr/ToolAlpaca-13B
for m in "$ORIG" "$UNLEARN" "$PLANNER"; do
  [ -d "$m" ] || { echo "FATAL: model not found: $m"; exit 1; }
done

OUT=/projects/standard/mhong/shared/jiajunr/unlearn_eval_results
mkdir -p "$OUT/eval_out"
echo "[$(date '+%F %T')] eval sweep -> $OUT"

# --- smoke: verify the whole pipeline (simulator + rollout + judge) -----------
if [ "$RUN" = "smoke" ]; then
  echo "[$(date '+%F %T')] SMOKE: single | unlearn | T_f | first 4 instances"
  LENGTH=4 MODEL_UNDER_TEST="$UNLEARN" METRIC=T_f MODE=single SINGLE_GPU=0 \
    OUT_DIR="$OUT/eval_out" ./tool_unlearn/evaluate.sh || exit 1
  echo "[$(date '+%F %T')] smoke OK -> $OUT/eval_out"
  echo "    resubmit without 'smoke' for the full sweep."
  exit 0
fi
[ "$RUN" = "full" ] || { echo "usage: sbatch $(basename "$0") [full|smoke]"; exit 2; }

# --- the 8 full evals ---------------------------------------------------------
echo "[$(date '+%F %T')] === 1/4 unlearned-7B alone ==="
MODEL_UNDER_TEST="$UNLEARN" METRIC=T_f MODE=single SINGLE_GPU=0 \
  OUT_DIR="$OUT/eval_out" ./tool_unlearn/evaluate.sh || exit 1
MODEL_UNDER_TEST="$UNLEARN" METRIC=T_r MODE=single SINGLE_GPU=0 \
  OUT_DIR="$OUT/eval_out" ./tool_unlearn/evaluate.sh || exit 1

echo "[$(date '+%F %T')] === 2/4 original-7B alone (before) ==="
MODEL_UNDER_TEST="$ORIG" METRIC=T_f MODE=single SINGLE_GPU=0 \
  OUT_DIR="$OUT/eval_out" ./tool_unlearn/evaluate.sh || exit 1
MODEL_UNDER_TEST="$ORIG" METRIC=T_r MODE=single SINGLE_GPU=0 \
  OUT_DIR="$OUT/eval_out" ./tool_unlearn/evaluate.sh || exit 1

echo "[$(date '+%F %T')] === 3/4 original-7B + 13B planner (control) ==="
MODEL_UNDER_TEST="$ORIG" METRIC=T_f MODE=collab PLANNER_MODEL="$PLANNER" \
  COLLAB_GPUS=0,1 COLLAB_COND=H1 OUT_DIR="$OUT/eval_out" ./tool_unlearn/evaluate.sh || exit 1
MODEL_UNDER_TEST="$ORIG" METRIC=T_r MODE=collab PLANNER_MODEL="$PLANNER" \
  COLLAB_GPUS=0,1 COLLAB_COND=H1 OUT_DIR="$OUT/eval_out" ./tool_unlearn/evaluate.sh || exit 1

echo "[$(date '+%F %T')] === 4/4 unlearned-7B + 13B planner (the question) ==="
MODEL_UNDER_TEST="$UNLEARN" METRIC=T_f MODE=collab PLANNER_MODEL="$PLANNER" \
  COLLAB_GPUS=0,1 COLLAB_COND=H1 OUT_DIR="$OUT/eval_out" ./tool_unlearn/evaluate.sh || exit 1
MODEL_UNDER_TEST="$UNLEARN" METRIC=T_r MODE=collab PLANNER_MODEL="$PLANNER" \
  COLLAB_GPUS=0,1 COLLAB_COND=H1 OUT_DIR="$OUT/eval_out" ./tool_unlearn/evaluate.sh || exit 1

# --- recap --------------------------------------------------------------------
echo
echo "================================================================"
echo "overall (T_f = fraction of both Procedure+Response correct):"
echo "================================================================"
$TA_PY -c "
import json, glob
for f in sorted(glob.glob('$OUT/eval_out/*_judge.json')):
    tag = f.split('/')[-1].replace('_judge.json','')
    try:
        s = json.load(open(f))['statistics']; n = s['num']
        print('%-38s  T=%.1f  (errors %d, n %d)' % (tag, 100*s['both']/n, s['error_num'], n))
    except Exception as e:
        print('%-38s  ERROR %s' % (tag, e))
"
echo "[$(date '+%F %T')] done"
