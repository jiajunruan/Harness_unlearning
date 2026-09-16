# Invocation versus task success: audit of 600 saved trajectories

This audit reads the three conditions on 100 retain and 100 forget tasks each from `/projects/standard/mhong/shared/jiajunr/full_react_3way_luna_p0`. It does not rerun inference or judging and does not change published counts or evaluation code. Run `python evaluation_audit/audit_rates.py` to reproduce the counts.

## What the metrics actually measure

`judge_full_react_luna.py:61–79` takes only `golden_answers[0]['Action']` as the target. Invocation is membership of that name in recorded intermediate steps. It checks neither arguments nor returned status nor subsequent required calls. A tool wrapper returning a local validation error also creates an intermediate step (`agent/custom_agent_executor.py:65–93`, `agent/tools.py:35–67`). Thus invocation is broader than a valid API request or successful execution.

Task success requires both parsed judge verdicts to equal `Yes`. `prompts/Evaluation.txt` asks for all standard-answer calls, consistent necessary parameters, no unrelated calls, and a correct final response grounded in the observations. This is a joint process-and-response criterion, stricter than response correctness alone. `Uncertain` and judge errors count as unsuccessful.

## Exact decomposition

Each row has 100 tasks. The last three columns partition the invocation-without-success group; they are saved verdicts, not independently verified ground truth about failure causes.

| Model | Split | Invocation | Success | Gap | Process No | Process Yes / Response No | Uncertain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Original | Retain | 91 | 42 | 49 | 23 | 17 | 9 |
| Original | Forget | 93 | 41 | 52 | 23 | 23 | 6 |
| Unlearned | Retain | 86 | 29 | 57 | 37 | 12 | 8 |
| Unlearned | Forget | 68 | 11 | 57 | 37 | 11 | 9 |
| Harness | Retain | 88 | 33 | 55 | 36 | 13 | 6 |
| Harness | Forget | 87 | 21 | 66 | 44 | 14 | 8 |

Most gaps remain after excluding explicit rollout exceptions: only 3/57 unlearned-retain gaps and 2/55 harness-retain gaps have an explicit exception. A successful HTTP status alone also does not solve the gap: respectively 56/57 and 54/55 have at least one target-tool 2xx observation.

## Confirmed implementation defects

1. **Missing parameters crash the rollout.** `agent/convert_request.py:78,95–98` initializes body-required parameters as `None`, iterates it when a required query/path parameter is absent, and also attempts string-plus-list concatenation. `Tool.func` catches `ValueError` but not this `TypeError`. A recoverable parameter error becomes a terminal exception. Replaying the actual last attempted calls, with HTTP forbidden, reproduces all 18 observed occurrences: 8 unlearned-forget and 10 harness-forget. Five in each condition occur after the target tool has already been recorded, directly contributing to the invocation-success gap. The other eight can also affect the invocation metric by preventing later calls. Correcting the exception does not imply these tasks would become successful.

2. **Iteration-limit output schema mismatch.** `get_agent.py` requires both `output` and `Final Thought`. The installed LangChain `Agent.return_stopped_response` returns only `output` for forced stopping. `Doge-Meme::1` reaches 15 calls in both unlearned-forget and harness-forget, then raises the recorded output-key exception. This is a bookkeeping defect on an already exhausted rollout; fixing it should not automatically turn the case into success.

3. **Parameter validation is incomplete.** `convert_request.py:75` passes the OpenAPI Parameter Object to `type_check` instead of its nested `schema`. Type/enum checks under `schema` are therefore skipped for ordinary query/path parameters. Values are copied to the outgoing request before any conversion. This is a code-level defect; its numerical impact has not been isolated.

4. **Clarification switch is inverted.** `get_agent.py:34` creates `GetDetailsTool` when `enable_getDetails` is false. The default true flag omits it. This contradicts the option name and matters for underspecified instructions; its impact on these runs has not been isolated.

## Judge reliability and setup limitations

- `judge_collab_curve.py:104–120` caps completions at 512 tokens, discards `finish_reason`, and maps absent regex matches to `Uncertain`. `judge_full_react_luna.py` discards the raw judgment returned by `judge_full`. There are 46 uncertain cases within the six invocation-success gaps, including one recorded API exception on original-retain. The other uncertainties cannot be separated retrospectively into genuine judge uncertainty, truncation, or format parsing failure. A zero `n_judge_errors` does not certify successful verdict parsing. Save raw output, finish reason, and explicit parse status, then rejudge unresolved cases before treating all of them as model failures.
- The simulator receives the API documentation and request, not the full task or reference answer (`instance_generation/simulator.py:49–68`). It samples at temperature 0.5. `run_full_react_3way.sh` does not enable `--use_cache`; even that cache is conversation history, not a fixed response fixture shared across conditions. Stateful consistency and identical tool observations across conditions are not guaranteed. Fixing a shared deterministic fixture would be a new evaluation protocol and requires rerunning all conditions.
- Gold traces can contain unstated dates, placeholder identifiers, and a particular call sequence. For example, the aviationstack task says “next month” but gold fixes 2020-08-15; the agent chooses 2020-08-01. The judge permits unspecified parameters to differ, but literal enforcement of all gold calls may still be stricter than semantic task completion. This warrants judgment review rather than automatic relabeling.
- Actor generation defaults to 256 new tokens per step. Saved responses include repetitive text cut mid-sentence. The trace does not record finish reasons, so cap-induced failures are plausible but not proven for individual cases. Save generated token counts and termination reasons before changing this setting.

## Concrete observed failures

- **Unlearned / retain / Domainsdb.info__2::0:** both gold calls execute and the judge marks the process `Yes`. The observation gives US domain share 41.47%; the final response instead states 2.84% and 79.9%, repeats itself, and ends mid-sentence. Response is `No` despite calling the correct tools.
- **Unlearned / retain / apilayer aviationstack::1:** the target flight-status tool executes, but the rollout omits the gold airline-info call and the final response omits requested airport information. The date also differs from the demonstration. This is more than target-tool selection.
- **Harness / forget / Quarantine::0:** `getCountryTimeline` returns March 17–19 observations although the task asks for June 1. The model relabels March 19's 193 cases, 3 deaths, and 1 recovery as June 1. The judge marks process `Yes`, response `No`. Tool-response coverage and incorrect summarization both matter.
- **Harness / forget / ipapi.com::0:** the target wrapper runs with the literal placeholder `User's IP address` and receives HTTP 400. The final answer repeats a tool call instead of explaining the failure. Invocation still counts.
- **Unlearned / forget / Quarantine::1:** a global-statistics call completes; the subsequent `getCountryStats` call uses `{"india":704753890}` instead of the required parameter. The request converter crashes with `NoneType`, eliminating the chance to recover. This exact crash was reproduced offline.

## Interpretation and next steps

The large gap is partly expected: target-name occurrence is a weak event, whereas task success requires a complete valid process and final answer. It is also contaminated by confirmed runtime defects and unresolved judge outcomes. The present success rates should be treated as provisional measurements of this implementation, not clean estimates of capability alone.

Prioritize (1) request-validation fixes and regression checks, (2) explicit judge parse/termination logging and rejudging unresolved records, and (3) paired deterministic simulator responses for a fresh comparison. Preserve the current artifacts and rerun under a new output directory; do not silently overwrite the paper's rates or count error-status calls as successful execution. Report invocation, valid target-call execution, process correctness, response correctness, and joint success separately during diagnosis. No corrected success rate can be inferred from this audit alone.
