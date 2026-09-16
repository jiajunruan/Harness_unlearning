# ToolAlpaca forget / retain results summary

This file keeps **only like-for-like comparisons in the same table**.  A higher
retain score is better.  On forget tasks, a higher task/process score means the
model has recovered more of the capability meant to be removed, i.e. forgetting
is worse.

## Current paper-aligned run: GPT-3.5 simulator + GPT-4-0613 judge

All rows use the same unlearned checkpoint
`npo_gdr_ckpt/npo_gdr_ep5`, the original 100 forget and 100 retain tasks, and
the original ToolAlpaca process-evaluation template.  The primary metric is
`Process Correctness = Yes / 100`; `Overall` requires both process and final
response to be correct.

| Configuration | Forget process | Retain process | Forget overall | Retain overall |
| --- | ---: | ---: | ---: | ---: |
| Unlearned alone (B0) | 33% | 53% | 28% | 50% |
| Unlearned + ToolAlpaca-13B Thought harness (H1) | 53% | 62% | 47% | 54% |
| Change, H1 - B0 | **+20 pp** | **+9 pp** | +19 pp | +4 pp |

Thus, under this metric the 13B harness improves retain process accuracy by 9
points, but also increases forget process accuracy by 20 points; it therefore
partly restores the forgotten capability rather than improving forgetting.
The result files are in
`outputs/paper_aligned_gpt35_gpt4_20260913_key2/`.

## Earlier GPT-3.5 planner run: Luna simulator + Luna judge

This is a valid historical within-run comparison but **must not be merged
numerically** with the GPT-4 table above: it uses a different simulator, judge,
and success metric.  GPT-3.5 supplied only Thoughts; the unlearned local actor
made every tool call and final answer.

| Configuration | Forget target-tool invocation | Retain target-tool invocation | Forget Luna task success | Retain Luna task success |
| --- | ---: | ---: | ---: | ---: |
| Unlearned alone | 70% | 89% | 13% | 27% |
| Unlearned + GPT-3.5 Thought harness | 88% | 80% | 27% | 27% |
| Change, harness - alone | **+18 pp** | **-9 pp** | **+14 pp** | 0 pp |

On this historical run, GPT-3.5 did not improve retain task success and it made
forgetting worse (forget task success and forgotten-tool invocation both rose).
See `outputs/gpt35_harness_20260909/RESULTS.md`.

## Earlier three-way Luna run

This uses the same two 100-task splits but a separate run and Luna judge.  Here
`original` is the stock **ToolAlpaca-7B**, not ToolAlpaca-13B; `hybrid` is stock
7B Thought plus unlearned actor Action/Response.  It is included to avoid the
common mislabelling of this row as a 13B-alone result.

| Condition | Forget Luna task success | Retain Luna task success |
| --- | ---: | ---: |
| Stock ToolAlpaca-7B (original) | 41% | 42% |
| Stock-7B Thought + unlearned actor (hybrid) | 21% | 33% |
| Unlearned actor | 11% | 29% |

See `outputs/full_react_3way_luna_p0/final_tables.md`.

## Newly completed comparators: same GPT-3.5 simulator + GPT-4-0613 judge

All six new split-level runs completed: exactly 100 trajectories per row, with
complete GPT-4 judge records and no unresolved judge/API failures.  They use
the same data, simulator template, judge template, decoding limits, and judge
as the current B0/H1 table.  The newly generated simulator observations are
independent samples, as they were in the earlier B0/H1 run.

| Configuration | Forget process | Retain process | Forget overall | Retain overall |
| --- | ---: | ---: | ---: | ---: |
| Unlearned alone (existing B0) | 33% | 53% | 28% | 50% |
| Unlearned + 13B Thought (existing H1) | 53% | 62% | 47% | 54% |
| **Unlearned + GPT-3.5 Thought (new G35)** | **52%** | **42%** | **47%** | **40%** |
| **ToolAlpaca-13B alone (new S13)** | **82%** | **76%** | **76%** | **72%** |
| **GPT-3.5 alone (new GPT35)** | **28%** | **23%** | **14%** | **12%** |

Relative to B0, G35 changes process accuracy by **+19 pp on forget** and
**-11 pp on retain** (overall: +19 pp / -10 pp).  Thus GPT-3.5 Thought partly
restores forgotten tool-use ability but does not improve retained capability in
this run.  Its forget result is essentially the same as the 13B harness (52%
vs. 53%), but its retain result is 20 points lower (42% vs. 62%).

S13 is the valid end-to-end local-model ceiling: 82% forget / 76% retain process
accuracy.  High forget performance is expected for a stock ToolAlpaca model and
means it has not been unlearned; it is not a desirable forgetting score.

The GPT35 end-to-end row is **not an intrinsic GPT-3.5 quality estimate**.  The
remote chat actor often failed to emit the legacy continuation parser's exact
`Thought -> Action -> Action Input -> Response` syntax: 80/100 forget and
75/100 retain trajectories recorded a parse exception (many still retained a
partial tool trace, hence judge `error_num` is 53 and 54 rather than 80 and 75).
Its low score therefore measures chat-format/protocol incompatibility in this
local ReAct harness.  G35 avoids that issue because the local unlearned actor
still emits every executable action and response.

New raw trajectories, traces, GPT-4 judgments, and manifests are in
`outputs/requested_comparators_gpt35_gpt4_20260913/`.  The initial invalid S13
adapter attempt was preserved separately under `invalid_s13_adapter_attempt/`
and excluded; the reported S13 rows were rerun after the generation-interface
adapter was corrected.
