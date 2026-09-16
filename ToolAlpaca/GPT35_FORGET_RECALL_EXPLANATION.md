# Why can GPT-3.5 restore forgotten tool use?

Same protocol for all rows: GPT-3.5 simulator, GPT-4-0613 judge, 100 forget and
100 retain tasks. **Process** is the primary metric. Higher retain is better;
higher forget means worse unlearning.

| Condition | Forget Process | Retain Process |
|---|---:|---:|
| Unlearned actor (B0) | 33% | 53% |
| Unlearned + 13B Thought (H1) | 53% | 62% |
| Unlearned + GPT-3.5 Thought (G35) | 52% | 42% |
| ToolAlpaca-13B alone (S13) | 82% | 76% |
| GPT-3.5 alone (GPT35) | 28% | 23% |

## Interpretation

GPT-3.5 does not execute tools in G35. It supplies a short Thought; the
unlearned local actor still emits every Action, argument, and final response.
The Thought can name the relevant tool, turning a difficult tool-retrieval step
into a strong next-token cue. Unlearning may reduce the actor's chance of
selecting a forgotten tool from the task alone, without removing its ability to
call that tool once it is named. This explains the forget increase (33% -> 52%).

The low GPT-3.5-alone score is mainly a **protocol mismatch**, not a direct
model-quality comparison: 80/100 forget and 75/100 retain rollouts had a legacy
ReAct parsing exception. GPT-3.5 often did not emit the exact local
`Thought -> Action -> Action Input` continuation syntax.

## Trace example (forget)

Task: retrieve AI / Machine Learning-tagged items from a specified RSS feed.
Gold tool: `getFeedItemsByTag`.

- **B0:** its Thought said it had already retrieved and filtered the feed, then
  emitted a final Response. It produced no Action or Action Input. This was not
  a parser failure (`failed=false`, with empty action and intermediate-step
  records); it was a hallucinated direct completion. GPT-4 Process: **No**.
- **G35:** GPT-3.5's Thought said: “I should use the
  `getFeedItemsByTag` function ...”. The unlearned actor then emitted
  `Action: getFeedItemsByTag` with the correct feed URL and tags. GPT-4 Process:
  **Yes**.

The harness truncates GPT-3.5 before its attempted Action, so GPT-3.5 cannot
execute the tool directly. The trace supports a cue-induced tool recall
mechanism, not boundary leakage. Results are single stochastic rollouts, so this
is mechanism evidence rather than a causal significance test.

## Where the results are

All result files for the five rows above are tracked in the repository; every
other run under `outputs/` is superseded and gitignored. Each condition has one
forget directory and one retain directory, holding `<COND>.json` (the 100
rollouts, grouped by tool/API row), `<COND>_trace.json` (the per-turn ReAct
trace behind the trace example above), `judge.json`, the run `manifest.json`,
and the simulator, rollout and judge `.log` files. The B0/H1 run adds
`summary.json` and `summary.md` at its top level with the per-row table.

Traces, by row and folder:

| Row | Forget trace | Retain trace |
|---|---|---|
| B0 | `outputs/paper_aligned_gpt35_gpt4_20260913_key2/B0_forget/B0_trace.json` | `outputs/paper_aligned_gpt35_gpt4_20260913_key2/B0_retain/B0_trace.json` |
| H1 | `outputs/paper_aligned_gpt35_gpt4_20260913_key2/H1_forget/H1_trace.json` | `outputs/paper_aligned_gpt35_gpt4_20260913_key2/H1_retain/H1_trace.json` |
| G35 | `outputs/requested_comparators_gpt35_gpt4_20260913/G35_forget/G35_trace.json` | `outputs/requested_comparators_gpt35_gpt4_20260913/G35_retain/G35_trace.json` |
| S13 | `outputs/requested_comparators_gpt35_gpt4_20260913/S13_forget/S13_trace.json` | `outputs/requested_comparators_gpt35_gpt4_20260913/S13_retain/S13_trace.json` |
| GPT35 | `outputs/requested_comparators_gpt35_gpt4_20260913/GPT35_forget/GPT35_trace.json` | `outputs/requested_comparators_gpt35_gpt4_20260913/GPT35_retain/GPT35_trace.json` |

`requested_comparators_gpt35_gpt4_20260913/invalid_s13_adapter_attempt/` is a
discarded S13 attempt kept only for provenance; it is excluded from every table
above.

## Finding a single case

The trace files are the easiest entry point for a case study. `{COND}_trace.json`
is a flat list of 100 records, one per instruction, each holding:

```text
condition, api, instruction_id (0-99), instruction, golden_answers,
steps, output, failed, execution_record
```

`steps` holds the ReAct turns. Under a harness condition each step carries
`planner_raw` (the GPT-3.5 Thought) alongside `actor_output`; a B0 step instead
has `mode: "vanilla"` and only `actor_output`. `failed` is the parser-exception
flag and lines up with `statistics.error_num`. `execution_record` holds what was
actually executed (`intermediate_steps`, `action_attempts`).

The trace example above is `instruction_id = 0`, `api = "RSS feed to JSON"`,
which is **index 13** in both `B0_forget/B0_trace.json` and
`G35_forget/G35_trace.json`:

- `B0` index 13 has `failed = false`, a single `vanilla` step, and
  `execution_record = {"intermediate_steps": [], "action_attempts": []}` — no
  Action was ever emitted, matching the hallucinated completion described above,
  so this is not a parser failure.
- `G35` index 13 has `steps[0].planner_raw` carrying the quoted Thought that
  names `getFeedItemsByTag`, and `execution_record.intermediate_steps` records
  the resulting call with the feed URL and tags.

To go deeper on the same case:

- `{COND}.json` — the full trajectory. Find the row whose `Name` equals the
  `api` value, then the instruction's index; `Instances[i]` holds `input`,
  `output`, `Final Thought` and `intermediate_steps` (an empty list means no tool
  call).
- `judge.json` — the GPT-4 verdict. Use the same row `Name` as the key; each
  record carries `process_correctness`, `final_response_correctness` and the
  `output` rationale. Records for empty trajectories (the ones counted by
  `error_num`) carry only `id`, `input` and `output` and have no verdict.

## How to run it

Use the `toolalpaca` environment
(`/users/2/jruan/miniconda3/envs/toolalpaca`) with `OPENAI_API_KEY` set. Both
scripts locate the unlearned actor and the 13B planner under
`/projects/standard/mhong/shared/jiajunr/` (`npo_gdr_ckpt/npo_gdr_ep5` and
`ToolAlpaca-13B`); they boot a GPT-3.5-turbo simulator, roll out through
`harness/run_toolalpaca.py`, then score with `evaluation.py` under `gpt-4-0613`.

```bash
# B0 and H1, both splits (uses two GPU pairs, default --gpu-groups 0,1;2,3)
python tool_unlearn/run_paper_aligned.py \
    --output-dir outputs/paper_aligned_gpt35_gpt4_20260913_key2

# G35 / S13 / GPT35: one condition and one split per invocation
python tool_unlearn/run_requested_comparators.py --condition G35 --split forget \
    --output-dir outputs/requested_comparators_gpt35_gpt4_20260913/G35_forget \
    --port 5720 --gpu 0
```

Give every invocation a free `--port`. `--resume` continues an interrupted run.
The exact commands used for the recorded results are saved in each
`manifest.json`.

## How to read it

The primary metric is process accuracy:

```text
statistics.process.Yes / statistics.num
```

`statistics.num` is 100 for every row and empty trajectories remain in the
denominator. `statistics.response.Yes` and `statistics.both` are the ancillary
response and joint counts. In `judge.json` the top-level keys are tool/API names
holding per-instruction records (`input`, `output` judge rationale,
`process_correctness`, `final_response_correctness`), followed by `statistics`,
which also carries the template, evaluator and input hashes.

`statistics.error_num` counts rollouts that raised a parser or infrastructure
exception, and it is what makes the **GPT35** row a protocol artifact rather
than a model-quality number: it is 53 (forget) and 54 (retain) for GPT35, versus
1 and 9 for G35, whose actions are emitted by the local unlearned actor. Check
`error_num` before reading any row's score.
