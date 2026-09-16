# ============================================================================
#  Evaluation settings -- THIS IS THE ONLY FILE YOU EDIT BEFORE EVALUATING.
#  Used by ./tool_unlearn/evaluate.sh
# ============================================================================

# --- 1. Which model is being tested -----------------------------------------
# The unlearned checkpoint, or the original model for a "before" number.
# The real 5-epoch NPO+GDR run lives on /projects; checkpoints are
#   /projects/standard/mhong/shared/jiajunr/npo_gdr_ckpt/npo_gdr_ep5/checkpoint-{10,20,30,40,50}
# and the final unlearned model is .../npo_gdr_ep5/ itself.
MODEL_UNDER_TEST=/projects/standard/mhong/shared/jiajunr/ToolAlpaca-7B   # the "before" baseline
# MODEL_UNDER_TEST=/projects/standard/mhong/shared/jiajunr/npo_gdr_ckpt/npo_gdr_ep5


# --- 2. Which metric ---------------------------------------------------------
#   T_T  general tool ability, on 10 APIs the model has NEVER seen
#        (data/eval_simulated.json).  Should stay HIGH.
#   T_f  the tools you asked it to forget, on held-out demonstrations.
#        Should go DOWN.  <- the whole point
#   T_r  the tools it should still know, on held-out demonstrations.
#        Should stay HIGH.
METRIC=T_f


# --- 3. Single agent or planner-actor collaboration --------------------------
#   single  one model runs the whole ReAct loop.  This is what you want for
#           measuring unlearning: the number is about MODEL_UNDER_TEST alone.
#   collab  a 13B planner writes the reasoning, MODEL_UNDER_TEST executes.
#           Only meaningful if you want to ask whether a planner can paper over
#           the forgetting -- it needs BOTH GPUs.
MODE=single

# collab only: the planner checkpoint, and how its role is enforced.
#   H1  the planner's role is implicit; the boundary is truncation alone
#   H2  the planner is additionally told it is advising and the actor decides
PLANNER_MODEL=/home/wenxian/wenxian/models/ToolAlpaca-13B
COLLAB_COND=H1


# --- 4. GPUs -----------------------------------------------------------------
# single: one card is enough (the 7B is ~13.5GB in bf16).
# collab: both are needed -- the actor sits on the second visible card and the
#         13B planner is sharded 20GiB/6GiB across the two.
SINGLE_GPU=0
COLLAB_GPUS=0,1


# --- 5. The two remote models ------------------------------------------------
# Nothing here touches the real internet: every tool call is answered by
# SIMULATOR_MODEL pretending to be the API, and every trajectory is scored by
# JUDGE_MODEL.  Both go through one OpenAI-compatible endpoint.
# DeepSeek (OpenAI-compatible).  deepseek-v4-flash is a reasoning model, so it
# spends tokens on reasoning_content before content -- keep generation limits
# generous or the simulator/judge return empty answers.
API_BASE=https://api.deepseek.com
# Never commit a real key.  evaluate.sh exports this as OPENAI_API_KEY, so set
# it in the environment (or .env), or override inline: API_KEY=... ./evaluate.sh
API_KEY=${API_KEY:-${OPENAI_API_KEY:-}}

SIMULATOR_MODEL=deepseek-v4-flash     # invents the API responses
JUDGE_MODEL=deepseek-v4-flash         # scores Procedure / Response / T_T
JUDGE_WORKERS=16                      # concurrent judge requests

# Local port for the fake API server.  evaluate.sh starts and stops it for you;
# change this if 5678 is taken.
SIM_PORT=5678

# IMPORTANT: the endpoint is not reachable through a proxy and the simulator is
# on localhost, so any proxy variables must be cleared.  evaluate.sh does this.


# --- 6. Where results go -----------------------------------------------------
OUT_DIR=./tool_unlearn/eval_out
