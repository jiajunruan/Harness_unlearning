# Tool Unlearning for ToolAlpaca-7B

Make ToolAlpaca-7B forget a chosen set of tools while keeping its ability to use
the rest, with the gradient-based methods from
[MUSE](https://github.com/swj0419/muse_bench) (GA / NPO), on the data split
defined by [ToolDelete](https://arxiv.org/abs/2502.01083).

ToolDelete itself is **not** implemented — only the general unlearning baselines
it compares against.

```
prepare_data.py     builds the frozen split (already run; data/ is committed)
unlearn.py          training entry point
dataset.py          masked-label datasets
trainer.py          GA / NPO losses
eval_config.sh      >>> the only file you edit before evaluating <<<
evaluate.sh         runs one evaluation end to end
data/               the frozen split (see below)
```

---

# Part 1 — The data

## 1.1 Three metrics, three different files

ToolDelete reports three numbers and they do **not** come from the same rows:

| Metric | Meaning | Where it comes from | Want |
|---|---|---|---|
| **T_T** | general tool ability | `data/eval_simulated.json` — 10 APIs the model has **never seen** | stays high |
| **T_f** | the forgotten tools | `data/forget_eval_api.json` — **held-out** demos of the forget tools | **goes down** |
| **T_r** | the retained tools | `data/retain_eval_api.json` — **held-out** demos of the retain tools | stays high |

The paper's `Original` row is `T_T 60.0 / T_r 73.1 / T_f 75.7`. T_f is *higher*
than T_T precisely because T_f measures training tools the model has memorised,
while T_T measures APIs it never saw. That number is also this repo's own
baseline anchor for `eval_simulated.json`.

**The unlearning data and the evaluation data are disjoint.** Every tool's
demonstrations are split into a train half (used to unlearn) and an eval half
(never trained on). If T_f were measured on the exact rows gradient ascent ran
on, it would be reporting optimisation success, not forgetting.

Verified isolation:

```
forget  train 635  eval 173   row overlap 0
retain  train 2462 eval 667   row overlap 0
forget ∩ retain tools = 0
training tools ∩ T_T APIs = 0     (447 training APIs vs 10 test APIs, fully disjoint)
```

## 1.2 The split is frozen, not redrawn

`data/split.json` holds the tool partition and is **read back** on every later
run, so a finished experiment cannot have its forget set change underneath it:

```
[split] reusing the frozen partition in ./tool_unlearn/data/split.json (89 forget tools).
```

It is already generated and committed — **you do not need to run
`prepare_data.py`**. Only run it with `--refreeze` if you deliberately want a
different partition (a different `--forget_ratio`, or another seed).

## 1.3 How the partition follows the paper

ToolDelete §2:

```
D_f = {T_f, Q_f, Y_f}   k < N tools + ALL their demonstrations
D_r = D \ D_f           every remaining tool + its demonstrations
```

and on the baselines: *"we treat all data related to T_f as unlearning examples
and all data related to T_r as remaining examples."*

* **Tool granularity, not sample granularity.** A tool is wholly forgotten or
  wholly retained; picking random individual samples would be sample-level
  unlearning, which the paper is explicitly not doing.
* **A tool is one API**, matching the paper's count for ToolAlpaca
  (495 tools / 3975 examples). `data/train_data.json` holds 447 unique APIs and
  yields 3097 training + 840 eval demonstrations.
* `--forget_ratio 0.20` reproduces the paper's Table 1 setting (it sweeps 2–20 %).

## 1.4 Details that were easy to get wrong

* **Examples are not re-derived here.** `prepare_data.py` calls upstream
  `build_dataset.build_dataset`, and `dataset.py` tokenises through upstream
  `train.preprocess`. Unlearning must act on the exact strings and token
  positions the model was trained on, and those two functions are the only
  definition of that format.
* **The loss is masked like training.** MUSE unlearns a raw corpus and sets
  `labels = input_ids`, so every token counts. ToolAlpaca's loss only ever
  covered what the **model itself produced**. Carrying MUSE's behaviour over
  would push up the likelihood of the prompt and of the simulated API responses
  — not the knowledge being removed, and a fast way to wreck general ability:

  ```
  [   0: 844] masked      prompt + user question
  [ 844: 918] SUPERVISED  Thought + Action + Action Input
  [ 918:1073] masked      tool Observation
  [1073:1200] SUPERVISED  final Thought + Response
  ```
* **Queries come from the instance, not from `Instructions`.** 42 APIs disagree
  on the length of those two lists and 157 instances have an empty query, so the
  query is taken from `instance["input"]`, the way `build_dataset` does it.
* **19 API names occur twice**, and 15 of those pairs carry *different* OpenAPI
  specs. The simulator registers APIs as `{api["Name"]: api}`, so a duplicate
  would silently answer calls against the wrong spec. Eval entries are renamed
  `Name__2`, `Name__3`; the suffix only ever appears in the routing and cache
  key, never in a tool name or a prompt.

---

# Part 2 — Training

## 2.1 Choosing the loss

Two independent choices: what to do on the **forget** set, and how to hold the
**retain** set in place.

```bash
--algo {ga,npo}  --retain_loss {none,ce,kl}
```

| | Forget objective |
|---|---|
| `ga` | gradient **ascent**: `-CE(forget)` — push the likelihood down |
| `npo` | negative preference optimisation: DPO with only a losing response |

| | Retain term |
|---|---|
| `none` | nothing holds the retain set in place |
| **`ce`** | **ordinary cross-entropy descent on retain — the same likelihood objective as forget, pushed the opposite way.** This is what you asked for; it is MUSE's `gdr` / "gradient difference". |
| `kl` | KL to the frozen reference model on retain (MUSE's `klr`) |

**So the retain loss you want is `--retain_loss ce`, not `kl`.** With `ce` the
total objective is literally `-CE(forget) + CE(retain)`: one term pushes the
forget tools' likelihood down, the other pulls the retain tools' likelihood up,
in the same units. `kl` instead constrains the whole output *distribution* on
retain to stay near the original model, which is a different (stronger, more
expensive) constraint.

MUSE's compound names still work if you prefer them — `--algo ga_gdr` is
identical to `--algo ga --retain_loss ce`, and passing both with a conflict is
rejected rather than silently resolved.

## 2.2 Running it

```bash
source repro_env_local.sh

# gradient ascent + CE retain  (the "opposite direction on the same loss" setup)
$TA_PY tool_unlearn/unlearn.py \
    --algo ga --retain_loss ce \
    --model_dir $ACTOR_MODEL \
    --out_dir tool_unlearn/ckpt/ga_ce \
    --epochs 5 --lr 1e-5

# NPO + CE retain  (the strongest general baseline in ToolDelete Table 1)
$TA_PY tool_unlearn/unlearn.py \
    --algo npo --retain_loss ce \
    --model_dir $ACTOR_MODEL \
    --out_dir tool_unlearn/ckpt/npo_ce \
    --epochs 5 --lr 1e-5 --beta 0.1
```

`--forget_file` and `--retain_file` already default to
`tool_unlearn/data/forget_train.json` / `retain_train.json`.

Other flags: `--per_device_batch_size` (default 1), `--grad_accum` (8),
`--optim` (`adamw_torch` or `adafactor`), `--no_gradient_checkpointing`,
`--max_steps`, `--save_strategy epoch`, `--resume_from_checkpoint`, `--seed`.
`--deepspeed <config.json>` is passed straight through to HF Trainer if you ever
want it; nothing here requires it.

`lr 1e-5` is the paper's setting (Appendix C: "We use a learning rate of 1e-5
across all experiments"). `--epochs 5` comes from MUSE and is **not** tuned for
this dataset.

## 2.3 Memory

Full fine-tuning of a 7B, bf16 + gradient checkpointing:

| Optimiser | Params | Grads | Optimiser state | Total |
|---|---|---|---|---|
| `adamw_torch` | 13.5 GB | 13.5 GB | 81 GB (fp32 master + m + v) | **~108 GB** |
| `adafactor` | 13.5 GB | 13.5 GB | negligible (factored) | **~30 GB** + activations |

`npo` and `--retain_loss kl` additionally hold a frozen reference copy of the
model: **+13.5 GB**.

Without ZeRO/FSDP offload, `adamw_torch` needs a single ~120 GB device or
sharding; `--optim adafactor` fits comfortably on one 48 GB card and easily on
an 80 GB one. Note that plain multi-GPU DDP replicates the optimiser on every
rank — it buys throughput, not memory.

## 2.4 A warning about `ga`

Gradient ascent is **unbounded below** — nothing stops `-CE` from running to
minus infinity, and the model will happily destroy itself if trained too long.
This is why ToolDelete's Table 1 shows GradAscent forgetting well (T_f 75.7 →
34.6) while wrecking everything else (T_T 60.0 → 33.3, T_r 73.1 → 51.4).

Save every epoch (`--save_strategy epoch`, the default) and evaluate T_r at each
checkpoint rather than trusting the last one.

## 2.5 Running the real 8-GPU NPO job

The verified production command (what the 5-epoch run used), **not** via
`repro_env_local.sh` — training uses the `toolalpaca` conda env, not the eval venv:

```bash
# conda env toolalpaca: python 3.10, torch 2.1.2+cu121, transformers 4.29.2,
# deepspeed 0.12.6.  The `unlearning` env's deepspeed fails to import on this
# box (CUDA_HOME not set for op_builder), so toolalpaca is the one to use.
export TRITON_CACHE_DIR=/tmp/triton_cd   # avoid NFS triton autotune warning

/users/2/jruan/miniconda3/envs/toolalpaca/bin/deepspeed --num_gpus 8 \
    tool_unlearn/unlearn.py \
    --algo npo_gdr \
    --model_dir /projects/standard/mhong/shared/jiajunr/ToolAlpaca-7B \
    --out_dir /projects/standard/mhong/shared/jiajunr/npo_gdr_ckpt/npo_gdr_ep5 \
    --epochs 5 --lr 1e-5 --beta 0.1 \
    --per_device_batch_size 1 --grad_accum 8 \
    --optim adafactor --gradient_checkpointing \
    --deepspeed tool_unlearn/ds_zero3_adafactor.json \
    --save_strategy epoch --save_total_limit 5 --seed 42
```

Why it looks the way it does:

* **ZeRO-3, not ZeRO-2.** ZeRO-2's `deepspeed.initialize` moved every rank's
  parameters to the host to flatten them (`move_to_cpu` + CPU `flatten_*`), a
  ~28 GB/rank host transient that blew the SLURM job's 64 GB memory cgroup cap
  with 8 ranks (host anon-rss jumped 6→58 GB in 9 s, SIGKILL -9). ZeRO-3 shards
  on the GPU and never moves params to CPU; host anon-rss stays ~0.5 GB/rank.
  The config `ds_zero3_adafactor.json` sets `stage3_gather_16bit_weights_on_model_save: true`
  so every checkpoint contains a **full** `pytorch_model.bin`, not shards.
* **NPO's reference model is eliminated.** log π_ref(y|x) does not depend on
  θ, so `unlearn.py` computes it once on rank 0, caches it as
  `ref_seq_logp.pt` in `--out_dir`, and frees the 7B reference — the 14 GB
  frozen copy never has to share a 40 GB card with the trainable policy.
* **out_dir on /projects, not $HOME.** Each epoch checkpoint is ~63 GB (13.4 GB
  full model + 51 GB ZeRO shards); the home dir has a ~200 GB quota and hit
  EDQUOT. `/projects/standard/mhong/shared/jiajunr` has 1.3 PB free.
* **GPU 0 must fit the ref-model precompute.** Rank 0 briefly holds the 7B
  reference (~34 GB peak on a 40 GB card) to build the cache, then frees it;
  steady-state training runs ~36–39 GB/card at seq 2048.

Result of the actual run (635 forget / 2462 retain, 10 steps/epoch, 50 steps):

```
checkpoint-10/20/30/40/50   full pytorch_model.bin (13.4 GB) each
loss: 0.516 → 0.173 over 5 epochs (NPO loss falls as forget likelihood drops)
final unlearned model: <out_dir>/pytorch_model.bin
```

The `deepspeed` binary drops `--local_rank` automatically; `--epochs 5`
produces one checkpoint per epoch via `--save_strategy epoch`.

---

# Part 3 — Evaluation

## 3.1 What you configure

Everything lives in **`tool_unlearn/eval_config.sh`** — that is the only file to
edit. It is grouped into six blocks:

| Block | Setting | What it means |
|---|---|---|
| **1. Model** | `MODEL_UNDER_TEST` | the unlearned checkpoint, or the original model for a "before" number |
| **2. Metric** | `METRIC` | `T_T` \| `T_f` \| `T_r` — picks the data file automatically |
| **3. Mode** | `MODE` | `single` = the model runs the whole ReAct loop alone. **Use this for unlearning numbers** — the score is about `MODEL_UNDER_TEST` and nothing else. `collab` = a 13B planner writes the reasoning and `MODEL_UNDER_TEST` executes; only useful if you want to ask whether a planner can paper over the forgetting |
| | `PLANNER_MODEL` | `collab` only: the 13B checkpoint |
| | `COLLAB_COND` | `collab` only: `H1` (role enforced by truncation alone) or `H2` (planner is also told it is advising and the actor decides) |
| **4. GPUs** | `SINGLE_GPU` | `single` needs one card (~13.5 GB bf16) |
| | `COLLAB_GPUS` | `collab` needs both: the actor sits on the second visible card, the 13B planner is sharded 20 GiB / 6 GiB across the two |
| **5. Remote models** | `API_BASE`, `API_KEY` | one OpenAI-compatible endpoint serves both roles below |
| | `SIMULATOR_MODEL` | **DeepSeek-V4-Flash** — plays the API server. No tool call reaches the real internet: this model reads the OpenAPI spec and *invents* a conforming JSON response and status code. Bad arguments come back as 4xx so the agent has to recover |
| | `JUDGE_MODEL` | **GLM-5.2** — scores each trajectory on Procedure and Response |
| | `JUDGE_WORKERS` | concurrent judge requests (16) |
| | `SIM_PORT` | local port for the fake API server; `evaluate.sh` starts and stops it |
| **6. Output** | `OUT_DIR` | where trajectories and judgements land |

The proxy is cleared automatically — the endpoint is not reachable through it
and the simulator is on localhost.

## 3.2 Running it

```bash
./tool_unlearn/evaluate.sh                              # uses eval_config.sh as-is
METRIC=T_r ./tool_unlearn/evaluate.sh                   # override one knob
MODEL_UNDER_TEST=$ACTOR_MODEL METRIC=T_f ./tool_unlearn/evaluate.sh   # "before"
```

Output:

```
=== T_f | single | npo_ce ===
  Procedure 31.2   Response 40.5   Overall 30.1   (errors 3, n 173)
```

`Overall` is the metric — the fraction where the judge marked *both* the call
sequence and the final response correct. It is what the paper calls T_f / T_r /
T_T depending on which data you ran.

## 3.3 The full before/after sweep

```bash
for M in T_T T_f T_r; do
  MODEL_UNDER_TEST=/home/public/wenxian/models/ToolAlpaca-7B \
    METRIC=$M ./tool_unlearn/evaluate.sh
  MODEL_UNDER_TEST=./tool_unlearn/ckpt/npo_ce \
    METRIC=$M ./tool_unlearn/evaluate.sh
done
```

Success looks like: **T_f drops a lot, T_r barely moves, T_T barely moves.**

Judge scores are not comparable to the paper's absolute numbers — a different
judge model is systematically stricter. Compare before/after within your own
runs only.

---

# Part 4 — What was verified, and what wasn't

All on CPU, before spending a single GPU hour.

| Check | Result |
|---|---|
| `_seq_logp` vs a hand-computed sum of label log-probs | max error 3e-07 |
| NPO at θ = ref hits its analytic fixed point (2/β)·log 2 | 13.8629 vs 13.8629 |
| NPO falls when forget likelihood is pushed down | 13.86 → 8.33 (right direction) |
| `KL(θ = ref)` is 0, `KL(random)` is positive | 0.0 / 0.93 |
| fully-masked sequence produces no NaN | 0.0, not NaN |
| label masking on real rows | prompt and Observations masked; only Thought/Action/Response supervised |
| all six loss combinations end-to-end through `Trainer` | each updates 21/23 tensors — the 2 skipped are non-trainable `rotary_emb.inv_freq` buffers |
| forget NLL after 3 steps | rises for every combination |
| the retain term actually pulls back | `ga+ce` 11.00 < `ga` 11.14; `npo+ce` 10.82 < `npo` 11.15 |
| `surviving_instances` matches upstream `build_dataset`'s filters | asserted per API at build time |
| train/eval row overlap, forget/retain tool overlap, training ∩ T_T | all 0 |
| eval files parse through the harness's OpenAPI checks | 87/87 and 338/338 APIs |
| eval files' key set matches `data/eval_simulated.json` | identical |

**Since 2026-08-28:** the 5-epoch NPO+GDR run has been executed at 7B scale on
8×40 GB GPUs (ZeRO-3, §2.5) and completed: 5 checkpoints, loss 0.516 → 0.173.
`--epochs 5` is still inherited from MUSE and not tuned here, and `ga` in
particular is expected to over-forget — watch T_r at each epoch checkpoint.

## Three bugs fixed relative to MUSE

1. **NPO was computed on raw logits.** MUSE has
   `neg_log_ratio = outputs_f_ref.logits - outputs_f.logits`, averaging a
   difference over the whole 32000-way vocabulary. NPO is defined on the
   *sequence log-likelihood of the labels*:
   `L = -(2/β)·E[log σ(β·(log π_ref(y|x) − log π_θ(y|x)))]`.
2. **KL was computed on raw logits too.** `F.kl_div` expects log-probabilities;
   MUSE passes logits, so what it minimises is not a KL. Both sides now go
   through `log_softmax`, restricted to supervised positions.
3. **Retain pairing starved the retain set.** MUSE pairs forget sample *i* with
   retain sample `i % len(retain)`, so with 635 forget and 2462 retain examples
   three quarters of the retain set is never seen. The partner is now drawn
   uniformly at random.

MUSE is also Python 3.10+ (`str | None`); this repo runs 3.8.10, so everything
here uses `Optional[...]`.
