# Planner–Actor Collaboration on ToolAlpaca

Does a **13B** model, used only as a *planner*, make a **7B** *actor* better at
tool use and at reasoning?

Results are in `REPORT_FOR_ADVISOR.md`. `README.md` is the upstream ToolAlpaca
documentation and is unrelated to this experiment.

---

## 1. Repository layout

```
EXPERIMENT.md            this file -- everything you need to run the experiment
REPORT_FOR_ADVISOR.md    the results

repro_env_local.sh       endpoint, model paths, GPU wiring   <- edit this
run_single.sh            single-agent baseline   (generate + judge)
run_collab.sh            13B planner + 7B actor  (generate + judge)
judge.sh                 re-score / resume judging one condition
summarize_harness.py     print the comparison tables

harness/
  models.py              loads the checkpoints, shards the 13B across two GPUs
  planner_llm.py         the planner->actor wrapper; all four modes live here
  run_toolalpaca.py      tool benchmark runner
  run_general.py         BBH runner
evaluation.py            the LLM judge (parallel, resumable)
instance_generation/
  simulator.py           the fake API server

data/eval_simulated.json      10 APIs, 54 tools, 100 instructions
data/general/bbh/             27 BBH-Hard tasks
outputs/harness_tool/         <cond>.json, <cond>_trace.json, <cond>_judge.json
outputs/harness_general/      <cond>_records.json, <cond>_summary.json
```

---

## 2. Python environment

There is **no `pip install` step** — the interpreter already exists:

```bash
/home/public/wenxian/ToolAlpaca/.venv/bin/python      # exported as $TA_PY
```

Python 3.8.10, with pinned old versions that the code depends on:

| | | |
|---|---|---|
| `torch` | 2.4.1+cu121 | driver here is 535 / CUDA 12.2 |
| `transformers` | 4.29.2 | newer versions break the ToolAlpaca tokenizer |
| `langchain` | 0.0.147 | the agent loop is built on this exact API |
| `openai` | 0.27.2 | pre-1.0 (`openai.ChatCompletion.create`) |
| `pydantic` | 1.10.8 | langchain 0.0.147 requires v1 |

Two things this venv does **not** contain, both supplied through `PYTHONPATH`:

- **`.extra_pkgs/`** in this repo — `accelerate`, `nltk`, `safetensors`, `absl`,
  `immutabledict`, `langdetect`, `psutil`. The run scripts prepend it for you.
- **torch itself** lives in *another* user's venv and is pulled in by
  `toolalpaca_external_torch.pth` inside our venv, which appends
  `/home/public/yukun/blueprints_demo/.venv/lib/python3.8/site-packages`.

> ⚠️ That last one is the fragile part: if `/home/public/yukun/blueprints_demo/`
> is ever deleted or moved, every run dies with `ModuleNotFoundError: torch`.
> The fix is to install torch 2.4.1+cu121 into our own venv.

---

## 3. Credentials and endpoints

Everything lives in **`repro_env_local.sh`** — this is the only file to edit:

```bash
export OPENAI_API_BASE=https://antchat.alipay.com/v1
export OPENAI_API_KEY=<your key>
export SIMULATOR_MODEL=DeepSeek-V4-Flash    # plays the API server
export JUDGE_MODEL=GLM-5.2                  # scores the trajectories

export ACTOR_MODEL=/home/public/wenxian/models/ToolAlpaca-7B      # 13 GB fp16
export PLANNER_MODEL=/home/wenxian/wenxian/models/ToolAlpaca-13B  # 25 GB fp16

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY   # REQUIRED
export no_proxy='*' NO_PROXY='*'
```

> ⚠️ The proxy must be unset. The endpoint is not reachable through the local
> proxy, and the simulator is on localhost — leaving the proxy set makes both
> fail in confusing ways. `repro_env_local.sh` already does this; just remember to
> `source` it rather than exporting variables by hand.

Two remote models are called over the OpenAI-compatible API (simulator and judge);
both local models run on our own GPUs.

---

## 4. The API simulator

**Nothing reaches the real internet.** Each of the 10 APIs is served locally: the
simulator receives the agent's HTTP request, hands it plus that API's OpenAPI spec
to an LLM, and the LLM **fabricates** a spec-conformant JSON response and status
code. Invalid arguments come back as 4xx, so the agent has to notice and recover.
Responses are cached within one instruction and cleared between instructions so
items stay independent.

Start it once and leave it running for every tool-side run:

```bash
source repro_env_local.sh
OPENAI_DEFAULT_MODEL=$SIMULATOR_MODEL $TA_PY instance_generation/simulator.py \
    -api ./data/eval_simulated.json --port 5678 > logs/simulator.log 2>&1 &
curl -s --noproxy '*' http://127.0.0.1:5678/docs -o /dev/null -w '%{http_code}\n'   # 200
```

> The simulator is the biggest source of variance in this study — the same
> instruction run twice can get different responses and diverge. Only compare
> conditions produced by this same harness; never against historical numbers.

---

## 5. How an agent runs here

ReAct loop, at most 15 steps:

```
USER: Convert 1000 AUD to NZD, and give me AUD rates vs USD, EUR, GBP
ASSISTANT Thought:       the model's reasoning
ASSISTANT Action:        tool name
ASSISTANT Action Input:  JSON arguments
ASSISTANT Observation:   <- the executor fills this in from the simulator
... repeat ...
ASSISTANT Response:      final answer (this is what gets graded)
```

The model only produces `Thought / Action / Action Input / Response`. An output the
parser cannot read counts as a failure and still lands in the denominator.

An **API** is a service (1Forge, AbuseIPDB, Auth0…); a **tool** is one function of
it. The agent only ever sees the current API's 2–9 tools, so the task is *pick the
right function and fill its arguments*, not large-scale retrieval.

---

## 6. Single agent

One model runs the whole loop by itself. Only the 7B is loaded, so one card is
enough — you can run this on the spare GPU while something else occupies the other.

```bash
./run_single.sh tool    0      # ToolAlpaca, 100 instructions, then judges it
./run_single.sh general 0      # BBH-Hard, 54 items
```

---

## 7. Collaboration

**The 13B writes reasoning; the 7B performs every action.**

```
prompt ──> 13B planner ──> reasoning + the Action it would have called
                       ──> ✂  cut at the first "ASSISTANT Action" (discarded)
       ──> prompt + reasoning ──> 7B actor ──> Action / Action Input / Response
                                            ──> executor (sees only 7B output)
```

Both models receive a **byte-identical** prompt — the 13B is a ToolAlpaca
fine-tune, so a bespoke `PLANNER:` format would mostly measure prompt shock. The
final decision and the actual call always belong to the 7B, and that is enforced
by **truncation**, not by wording.

```bash
./run_collab.sh tool H1        # planner's role is implicit
./run_collab.sh tool H2        # planner is also told it is advising
./run_collab.sh general        # BBH-Hard with the 13B planner
```

`H2` inserts this before the Thought slot:

> SYSTEM: You are the planner advising the assistant. In one or two sentences, say
> what should be done next and why; naming the tool you think fits is fine. The
> assistant makes the final call and carries it out, not you.

The planner is deliberately **allowed** to name tools: the traces show tool
selection is exactly where the 13B adds value — on 1Forge it named
`convertCurrency`, which the 7B alone never found. An earlier directive that
*forbade* naming a function was ignored in 9/9 planner steps, because this model is
a ReAct tool-use fine-tune whose Thought is written as the preamble to an Action.
That is why the boundary is enforced structurally instead.

**GPU layout.** Collaboration holds both models at once: the 7B sits on GPU1 and
the 13B is sharded 20 GiB / 6 GiB across both cards with
`device_map="sequential"` — **not** `"auto"`, which routes through
`get_balanced_memory`, silently cuts GPU0's budget, and offloads layers to disk.
This needs the whole machine, so run one collaboration condition at a time.

---

## 8. Scoring

Tool runs are judged automatically at the end of `run_single.sh` / `run_collab.sh`.
To re-score, or to resume a judge run that died part-way:

```bash
./judge.sh H1                 # resumes if a partial *_judge.json exists
$TA_PY summarize_harness.py   # prints the comparison tables
```

An LLM judge labels two axes — **Procedure** (right functions, right arguments, no
missing or spurious steps) and **Response** (the final text is correct). The
headline metric **`T_T`** is the fraction where *both* are Yes. BBH is exact match
on the answer after `So the answer is`.

The judge is held fixed across every condition. It is systematically stricter than
the one used in the original paper, so **absolute numbers are not comparable to
published results** — only differences within one table are meaningful.

---

## 9. Conditions

| | Setup | Purpose |
|---|---|---|
| `B0` / `H0` | 7B alone | baseline |
| `B1` / `H_empty` | 7B + a constant, task-independent filler in the Thought slot | control: separates *"the wrapper changed the prompt"* from *"the 13B said something useful"* |
| `H1` / `H_plan13` | 7B + 13B planner (truncation) | the claim |
| `H2` | 7B + 13B planner (told it is advising) | does stating the role help? |
| `S13` | 13B alone | ceiling |

Data, simulator, cache policy, prompt template, decoding and judge are identical
across conditions, so they are strictly paired on the same task ids.
