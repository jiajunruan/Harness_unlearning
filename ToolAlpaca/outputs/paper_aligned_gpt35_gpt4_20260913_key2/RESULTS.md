# Paper-aligned simulator and process evaluation

Completed all four evaluations on 2026-09-13: 400 trajectories and their evaluation records, 100 instructions per condition/split.

| Configuration | Forget process accuracy | Retain process accuracy | Forget Overall | Retain Overall |
| --- | ---: | ---: | ---: | ---: |
| Unlearned model alone | 33% | 53% | 28% | 50% |
| Unlearned model + ToolAlpaca-13B harness | 53% | 62% | 47% | 54% |

The primary metric is the original judge's `Process Correctness: Yes` count divided by all 100 instructions. It evaluates tool selection, arguments, and sequence; it is not a tool-name occurrence rate. Empty trajectories and `Uncertain` verdicts do not enter the numerator. Overall is joint process-and-response correctness and is provided separately. Adding the harness increased process accuracy by 20 percentage points on forget and 9 on retain in this run.

Configuration:

- Simulator: `gpt-3.5-turbo`, resolved by the endpoint to `gpt-3.5-turbo-0125`; temperature 0.5.
- Judge: `gpt-4-0613`; temperature 0.2.
- Simulator and judge templates: unchanged files from `ToolAlpaca_original/prompts`.
- Actor: `/projects/standard/mhong/shared/jiajunr/npo_gdr_ckpt/npo_gdr_ep5`.
- H1 planner: `/projects/standard/mhong/shared/jiajunr/ToolAlpaca-13B`. The planner supplies Thought; the unlearned actor supplies executable actions, arguments, and final responses.
- Existing fp16 greedy rollout settings retained: 256 new tokens per generation, configured early stopping, and the existing ReAct iteration limit.
- Original `forget_eval_100.json` and `retain_eval_100.json` remain unchanged. These are not the paper's unseen-tool evaluation set, so this is not a direct reproduction of the paper's 60% Overall result.

The initial forget/retain jobs ran concurrently and judging overlapped later rollouts. Once retain finished, the remaining forget harness work was scheduled across both GPU pairs: 45 completed instructions were preserved, and disjoint batches of 28 and 27 supplied the remaining 55. The planned interruption and continuation are recorded in the manifest and snapshots.

Validation passed: each condition contains all 100 original instructions exactly once; gold answers and dataset hashes are unchanged; saved judge prompts match the original template; all judge records are complete with no judge or infrastructure errors. The 15 empty baseline-forget trajectories remain in its denominator. All other conditions have zero empty trajectories. The 45 preserved harness trajectories are unchanged after merging. All 25 implementation/integrity tests pass. Evaluation processes have exited and GPU memory is released.

Artifacts: [machine-readable summary](summary.json), [run manifest](manifest.json), and each condition's `judge.json`, trajectory JSON, and trace JSON. The manifest records model configuration, dataset/template/code hashes, commands, and continuation provenance. API credentials are not stored in these artifacts.
