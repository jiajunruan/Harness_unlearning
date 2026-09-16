# Implemented fixes

The manuscript now reports invocation only in the harness comparison (retain 86%/88%; forget 68%/87%). Task-success claims were removed, including the illustrative caption. The sequence log-likelihood diagnostic remains, with one concise statement that the evaluation concerns the initial decision. Its scores have not been relabeled as full-rollout measurements.

Changes in `/users/2/jruan/Stable_evolving`:

- `agent/convert_request.py`: missing query/path/body parameters raise actionable `ValueError` observations; non-object action inputs are rejected; query/path schemas are inspected; normalized values reach outgoing requests; boolean strings are converted correctly and booleans are rejected as integers.
- `agent/get_agent.py`: `enable_getDetails=True` now includes the clarification tool.
- `agent/custom_agent_executor.py`: synchronous and asynchronous completion fill the required `Final Thought` field, including forced iteration-limit termination. The stopped response remains a stopped response.
- `tool_unlearn/judge_collab_curve.py`: default completion budget is 2048; full judgments retry incomplete/malformed outputs or request errors up to three attempts, increasing the budget after length truncation. Valid Yes/No/Uncertain judgments are accepted without outcome-based retry. Full verdict parsing requires the explicit Results section. Legacy callers raise on unresolved judgments instead of silently treating malformed output as a genuine uncertain verdict.
- `tool_unlearn/judge_full_react_luna.py`: saves raw full-judge output, all attempt records, finish reason, usage, parse status, error status, valid-judgment count, valid-uncertain count, and `evaluation_complete`. The all-task denominator is retained and the treatment of unresolved judgments is explicit.
- `tool_unlearn/summarize_full_react_3way.py`: refuses task-success table generation when a new evaluation explicitly reports incomplete judgments.
- `tool_unlearn/run_reasoning_action_transfer.py`: explicitly omits getDetails when reconstructing historical B0 prompts, preserving the old probability evaluation after fixing the inverted switch.

Validation: 15 offline regression/integration tests passed in the pinned ToolAlpaca environment. They cover all 18 historical request crashes, recovery through the actual agent/tool wrapper, iteration-limit schema, clarification toggle, judge retry/parse/metadata behavior, and byte-identical reconstruction of all 100 saved probability prompts. Log: `fix_tests.log`.

Reproduce tests from Stable_evolving:

```bash
/users/2/jruan/miniconda3/envs/toolalpaca/bin/python -m unittest discover -s tests -v
```

No live inference, simulator, or judge rerun was performed. Existing CSVs, judgments, and numerical paper results remain historical observations. Request and clarification fixes can change both invocation and success rates in new rollouts. Future evaluations must use a new output directory. Simulator randomness/state consistency and generation-budget sensitivity remain experimental limitations; they are not resolved by these runtime fixes. The earlier audit report and reproduced-crash JSON document the pre-fix state; its standalone crash reproducer expects the old behavior and is superseded for validation by the regression tests.
