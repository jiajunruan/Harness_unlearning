#!/usr/bin/env bash
# 4-experiment forget/retain eval on the user's reserved 2-GPU allocation
# (agc04: 2x A40 48GB).  Launch from the login node:
#
#   srun --jobid=<your-reserved-jobid> bash tool_unlearn/run_unlearn_eval_parallel.sh
#   srun --jobid=<your-reserved-jobid> bash tool_unlearn/run_unlearn_eval_parallel.sh smoke
#
# Fixed test set (built by build_eval_subset.py, seed 42):
#   forget_eval_100.json / retain_eval_100.json  -> exactly 100 instances each,
#   the SAME set used by all four experiments below.
#
# Schedule on the 2 GPUs:
#   Phase 1 (in parallel): GPU0 = unlearned-7B alone (T_f, T_r)
#                          GPU1 = original-7B alone   (T_f, T_r)
#   Phase 2 (serial):      both GPUs = original-7B + 13B   (T_f, T_r)   [control]
#                          both GPUs = unlearned-7B + 13B  (T_f, T_r)   [the question]
#   Collab needs BOTH cards because the 13B is 26GB fp16, sharded 20GiB/6GiB;
#   the A40's 48GB makes quantization unnecessary.  No bitsandbytes here.
set -uo pipefail
RUN=${1:-full}
ROOT=/users/2/jruan/Stable_evolving
cd "$ROOT" || exit 1

# --- environment: conda env toolalpaca (the one training used) ----------------
export TA_PY=/users/2/jruan/miniconda3/envs/toolalpaca/bin/python
export TA_EXTRA_PKGS=$ROOT/data/general   # -> PYTHONPATH=$ROOT/data/general:$ROOT
export TRITON_CACHE_DIR=/tmp/triton_cd    # avoid NFS triton autotune warning
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy='*' NO_PROXY='*'

# --- models + fixed subset (all under /projects/.../jiajunr) -------------------
ORIG=/projects/standard/mhong/shared/jiajunr/ToolAlpaca-7B
UNLEARN=/projects/standard/mhong/shared/jiajunr/npo_gdr_ckpt/npo_gdr_ep5
PLANNER=/projects/standard/mhong/shared/jiajunr/ToolAlpaca-13B
F_FILE=./tool_unlearn/data/forget_eval_100.json
R_FILE=./tool_unlearn/data/retain_eval_100.json
OUT=/projects/standard/mhong/shared/jiajunr/unlearn_eval_results
mkdir -p "$OUT" logs
for m in "$ORIG" "$UNLEARN" "$PLANNER"; do
  [ -d "$m" ] || { echo "FATAL missing model: $m"; exit 1; }
done
for f in "$F_FILE" "$R_FILE"; do
  [ -f "$f" ] || { echo "FATAL missing subset file: $f"; exit 1; }
done
echo "[$(date '+%F %T')] models + subset OK, results -> $OUT"

# --- smoke gate: 4 APIs (~8 instances) to prove GPU+DeepSeek+judge all work ----
smoke() {
  echo "[$(date '+%F %T')] SMOKE: single | unlearn | T_f | 4 APIs on GPU 0"
  LENGTH=4 MODEL_UNDER_TEST="$UNLEARN" METRIC=T_f MODE=single SINGLE_GPU=0 \
    SIM_PORT=5678 API_FILE="$F_FILE" OUT_DIR="$OUT/smoke" ./tool_unlearn/evaluate.sh
}
if [ "$RUN" = "smoke" ]; then
  smoke
  exit $?
fi
[ "$RUN" = "full" ] || { echo "usage: $(basename "$0") [full|smoke]"; exit 2; }
smoke || { echo "[$(date '+%F %T')] smoke failed, aborting"; exit 1; }
echo "[$(date '+%F %T')] smoke OK"

# ============================ PHASE 1 (parallel) ==============================
echo "[$(date '+%F %T')] ===== PHASE 1: two single-agent evals in parallel ====="
(
  MODEL_UNDER_TEST="$UNLEARN" METRIC=T_f MODE=single SINGLE_GPU=0 SIM_PORT=5678 \
    API_FILE="$F_FILE" OUT_DIR="$OUT/single_unlearn" ./tool_unlearn/evaluate.sh || exit 1
  MODEL_UNDER_TEST="$UNLEARN" METRIC=T_r MODE=single SINGLE_GPU=0 SIM_PORT=5678 \
    API_FILE="$R_FILE" OUT_DIR="$OUT/single_unlearn" ./tool_unlearn/evaluate.sh || exit 1
) > "$OUT/log_phase1_unlearn.txt" 2>&1 &
P1A=$!
(
  MODEL_UNDER_TEST="$ORIG" METRIC=T_f MODE=single SINGLE_GPU=1 SIM_PORT=5679 \
    API_FILE="$F_FILE" OUT_DIR="$OUT/single_orig" ./tool_unlearn/evaluate.sh || exit 1
  MODEL_UNDER_TEST="$ORIG" METRIC=T_r MODE=single SINGLE_GPU=1 SIM_PORT=5679 \
    API_FILE="$R_FILE" OUT_DIR="$OUT/single_orig" ./tool_unlearn/evaluate.sh || exit 1
) > "$OUT/log_phase1_orig.txt" 2>&1 &
P1B=$!

wait "$P1A"; A=$?
wait "$P1B"; B=$?
echo "[$(date '+%F %T')] phase 1 done (unlearn=$A orig=$B) -- logs: $OUT/log_phase1_*.txt"
[ "$A" -eq 0 ] && [ "$B" -eq 0 ] || echo "WARNING: a phase-1 run failed, see its log"

# ============================ PHASE 2 (serial, both GPUs) =====================
echo "[$(date '+%F %T')] ===== PHASE 2: collab (both GPUs), serial ====="
echo "[$(date '+%F %T')] 3/4 original-7B + 13B planner (control)"
MODEL_UNDER_TEST="$ORIG" METRIC=T_f MODE=collab PLANNER_MODEL="$PLANNER" \
  COLLAB_GPUS=0,1 COLLAB_COND=H1 SIM_PORT=5678 API_FILE="$F_FILE" \
  OUT_DIR="$OUT/collab_orig" ./tool_unlearn/evaluate.sh > "$OUT/log_collab_orig_tf.txt" 2>&1 \
  || echo "FAILED collab orig T_f (see $OUT/log_collab_orig_tf.txt)"
MODEL_UNDER_TEST="$ORIG" METRIC=T_r MODE=collab PLANNER_MODEL="$PLANNER" \
  COLLAB_GPUS=0,1 COLLAB_COND=H1 SIM_PORT=5678 API_FILE="$R_FILE" \
  OUT_DIR="$OUT/collab_orig" ./tool_unlearn/evaluate.sh > "$OUT/log_collab_orig_tr.txt" 2>&1 \
  || echo "FAILED collab orig T_r (see $OUT/log_collab_orig_tr.txt)"

echo "[$(date '+%F %T')] 4/4 unlearned-7B + 13B planner (the question)"
MODEL_UNDER_TEST="$UNLEARN" METRIC=T_f MODE=collab PLANNER_MODEL="$PLANNER" \
  COLLAB_GPUS=0,1 COLLAB_COND=H1 SIM_PORT=5678 API_FILE="$F_FILE" \
  OUT_DIR="$OUT/collab_unlearn" ./tool_unlearn/evaluate.sh > "$OUT/log_collab_unlearn_tf.txt" 2>&1 \
  || echo "FAILED collab unlearn T_f (see $OUT/log_collab_unlearn_tf.txt)"
MODEL_UNDER_TEST="$UNLEARN" METRIC=T_r MODE=collab PLANNER_MODEL="$PLANNER" \
  COLLAB_GPUS=0,1 COLLAB_COND=H1 SIM_PORT=5678 API_FILE="$R_FILE" \
  OUT_DIR="$OUT/collab_unlearn" ./tool_unlearn/evaluate.sh > "$OUT/log_collab_unlearn_tr.txt" 2>&1 \
  || echo "FAILED collab unlearn T_r (see $OUT/log_collab_unlearn_tr.txt)"

# ============================ recap ===========================================
echo
echo "=============================================================="
echo "OVERALL (T = fraction of Procedure+Response both correct):"
echo "=============================================================="
$TA_PY -c "
import json, glob, os
rows=[]
for f in sorted(glob.glob('$OUT/*/*_judge.json')):
    tag = os.path.basename(f).replace('_judge.json','')
    try:
        s=json.load(open(f))['statistics']; n=s['num']
        rows.append((tag, 100*s['both']/n, s['error_num'], n))
    except Exception as e:
        rows.append((tag, None, 0, 0)); print('  (bad judge file %s: %s)' % (f,e))
for tag,t,err,n in rows:
    if t is None: continue
    print('%-40s T=%.1f  (errors %d, n %d)' % (tag,t,err,n))
"
echo "[$(date '+%F %T')] ALL DONE.  results: $OUT/eval_out  judge JSON: $OUT/*/*_judge.json"
