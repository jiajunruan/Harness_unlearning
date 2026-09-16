# RUNBOOK — how to run this repo on the jiajunr machine

Everything in `repro_env_local.sh` is the single source of truth for paths and
credentials. `EXPERIMENT.md` explains the experiment itself (Planner-Actor
collaboration on ToolAlpaca) and its conditions.

**Status: verified working end-to-end on this machine.** A tiny single-agent run
(B0, 1 API × 1 instruction) completed the full loop — model generation → ReAct
agent → simulator → judge — with a **Procedure=Yes / Response=Yes (T_T=Yes)**
verdict (see §5).

---

## 1. Python environment

There is **no venv** and the shared `unlearning` conda env is **not used** —
the harness needs `openai==0.27.2` (pre-1.0 API), `pydantic` v1 and
`langchain==0.0.147`, which would force destructive downgrades in `unlearning`.
A dedicated conda env keeps everything isolated:

```bash
conda create -n toolalpaca python=3.10 -y
conda activate toolalpaca
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.29.2 accelerate safetensors sentencepiece protobuf numpy==1.26.4
pip install langchain==0.0.147 openai==0.27.2 pydantic==1.10.8
pip install fastapi==0.95.0 uvicorn==0.21.1 httpx==0.23.3 jsonref==1.1.0 \
            openapi_spec_validator==0.5.6 tenacity==8.2.2 tqdm requests \
            nltk absl-py immutabledict langdetect psutil
```

Key versions (all verified on this box, driver 580 / CUDA 13.0):

| package | version | why it is pinned |
|---|---|---|
| `torch` | 2.1.2+cu121 | contemporary with transformers 4.29.x |
| `transformers` | 4.29.2 | the model config itself says `transformers_version: 4.29.2` |
| `langchain` | 0.0.147 | the agent loop in `agent/` is built on this exact API |
| `openai` | 0.27.2 | pre-1.0: `openai.ChatCompletion.create` / `.acreate` / `openai.error.*` |
| `pydantic` | 1.10.8 | langchain 0.0.147 requires v1 |
| `numpy` | 1.26.4 | `<2`; newer numpy breaks old transformers/torch |

The interpreter is `/users/2/jruan/miniconda3/envs/toolalpaca/bin/python`
(exported as `$TA_PY` by `repro_env_local.sh`).

## 2. Models

7B actor — **this is the only model needed for the single-agent baseline** — is
already downloaded to `/projects/standard/mhong/shared/jiajunr/ToolAlpaca-7B`
(~13 GB, fp16). Re-download anytime with:

```bash
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('TangQiaoYu/ToolAlpaca-7B', local_dir='/projects/standard/mhong/shared/jiajunr/ToolAlpaca-7B')"
```

13B planner is **not downloaded yet**. To run `run_collab.sh` you will need a
local copy (e.g. `TangQiaoYu/ToolAlpaca-13B`) and set `PLANNER_MODEL` in
`repro_env_local.sh`.

To score an unlearned/fine-tuned checkpoint instead of the stock model, override
without editing the file:

```bash
ACTOR_MODEL=tool_unlearn/ckpt/<run_dir> ./run_single.sh tool 0
```

## 3. API endpoint (simulator + judge)

Two remote roles call the OpenAI-compatible DeepSeek API: the **simulator**
(fabricates tool responses) and the **judge** (scores trajectories).

| variable | value |
|---|---|
| `OPENAI_API_BASE` | `https://api.deepseek.com` |
| `OPENAI_API_KEY` | *(from the environment -- never commit a real key)* |
| `SIMULATOR_MODEL` | `deepseek-v4-flash` |
| `JUDGE_MODEL` | `deepseek-v4-flash` |

> ⚠️ These are **exported unconditionally** in `repro_env_local.sh` on purpose:
> the login shell already exports an unrelated `OPENAI_API_KEY` (an `sk-proj-…`
> key), so a `${VAR:-default}` would silently keep the wrong credential and every
> API call would die with `AuthenticationError`. If you want to override, edit
> the file (or `unset OPENAI_API_KEY` first).

Two small changes were needed to make the code work with this stack — both are in
the repo now, documented in §6.

## 4. Single-agent baseline (7B alone) — B0

Only the 7B actor is loaded → one A100-40GB card is enough (~13 GiB VRAM).

```bash
cd /users/2/jruan/Stable_evolving
source repro_env_local.sh

# 1) start the API simulator once, leave it running (currently already running)
OPENAI_DEFAULT_MODEL=$SIMULATOR_MODEL $TA_PY instance_generation/simulator.py \
    -api ./data/eval_simulated.json --port 5678 > logs/simulator.log 2>&1 &
curl -s --noproxy '*' http://127.0.0.1:5678/docs -o /dev/null -w '%{http_code}\n'   # expect 200

# 2) full tool benchmark (10 APIs, 100 instructions), then judged automatically
./run_single.sh tool 0

#    quick harness-only smoke test (no judge):
#    OPENAI_DEFAULT_MODEL=$SIMULATOR_MODEL $TA_PY harness/run_toolalpaca.py \
#      --condition B0 --server_url "$SIM_URL" --offset 0 --length 1 \
#      --instruction_offset 0 --instruction_length 1 -out ./outputs/harness_tool

# 3) general reasoning benchmark (BBH-Hard), no simulator needed
./run_single.sh general 0
```

## 5. What a verified run looks like (tiny B0, 1 API × 1 instruction)

Simulator fabricated a realistic tool response, the actor completed the task, and
the judge marked both axes Yes:

```
> Entering new CustomAgentExecutor chain...
I can use the getRandomAxolotlImage tool to retrieve a random axolotl image.
ASSISTANT Action: getRandomAxolotlImage
ASSISTANT Action Input: {}
ASSISTANT Observation: Status Code: 200. Response: {"url":"https://theaxolotlapi.netlify.app/images/axolotl-001.jpg", ...}
ASSISTANT Thought: The response contains the URL of the image, which I can provide to the user.
ASSISTANT Response: Here is a random picture of an axolotl: https://theaxolotlapi.netlify.app/images/axolotl-001.jpg ...
> Finished chain.
[harness] B0 done: 1 tasks, 0 failed, 0 planner steps leaked a call

$ ./judge.sh B0
B0  Procedure 100.0  Response 100.0  T_T 100.0  (errors 0, n 1)
```

## 6. Changes made to the repo (vs. the version that referenced the old machine)

1. **`repro_env_local.sh`** (new) — local copy of the environment: conda python
   path, local model path, DeepSeek credentials. The upstream `repro_env.sh`
   (which hardcoded a real API key) has been removed; `run_single.sh` /
   `run_collab.sh` / `judge.sh` now `source "$(dirname "$0")/repro_env_local.sh"`.
2. **`instance_generation/simulator.py`** — status-code parsing made robust. The
   stand-in model sometimes writes `Status Code: 404 Not Found` (status text
   after the code); the old `int(...)` crashed on that and every tool call came
   back `500 Internal Server Error`. It now regex-extracts the first 3-digit
   code. This only affects the simulator's reading of the LLM output, not the
   experiment logic.

If you ever switch back to a strict `gpt-3.5-turbo`-style model, change 2 is
harmless — it accepts bare `Status Code: 404` too.

## 7. Collaboration (13B planner + 7B actor) — H1 / H2

Holds both models at once (7B on GPU1, 13B sharded 20 GiB / 6 GiB across both
cards) → needs the whole box; run one at a time. **Requires the 13B model.**

```bash
./run_collab.sh tool H1
./run_collab.sh tool H2
./run_collab.sh general        # BBH-Hard with the 13B planner
```

## 8. Scoring / tables

Tool runs are judged automatically at the end of the run scripts. To re-score or
resume an interrupted judge:

```bash
./judge.sh B0                  # resumes if a partial *_judge.json exists
$TA_PY summarize_harness.py    # prints the comparison tables
```

## 9. GPU / VRAM notes

- 8 × A100-40GB are available. `run_single.sh` uses one card
  (`./run_single.sh tool 0` → `CUDA_VISIBLE_DEVICES=0`).
- The 7B fp16 actor uses ~13 GiB → comfortable on a 40 GB card.
- The harness generates one instruction at a time (batch = 1), so there is no
  batch size to shrink on the inference side. If you later run the unlearning
  trainer in `tool_unlearn/`, keep `per_device_train_batch_size` small (see
  `tool_unlearn/eval_config.sh`) — that is where VRAM is the constraint.

## 10. Troubleshooting

- `ModuleNotFoundError: torch` → you are not using the conda env; re-check
  `$TA_PY` / `conda activate toolalpaca`.
- `openai.error.AuthenticationError` → the shell's own `OPENAI_API_KEY` is
  shadowing the DeepSeek key; run `unset OPENAI_API_KEY` or re-`source
  repro_env_local.sh` from a fresh shell.
- Simulator returns `500 Internal Server Error` on every tool call → the
  simulator log (`logs/simulator.log`) will show a parse error or API error;
  `simulator.py` must be the version with the regex status-code fix (§6).
- Tokenizer errors → make sure you load with `use_fast=False` (the harness does);
  the fast tokenizer path is not supported for this checkpoint.
