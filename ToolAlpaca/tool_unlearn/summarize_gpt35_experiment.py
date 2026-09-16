"""Validate paired coverage and report only new GPT-3.5 experiment results."""
import json
import hashlib
from collections import Counter
from pathlib import Path


def summarize(out):
    out = Path(out)
    manifest = json.loads((out / 'manifest.json').read_text())
    rows = {}
    paired = {}
    audits = []
    planner = manifest['planner']
    lines = [f'# Agent alone versus {planner} Thought harness', '',
        f'Actor: `{manifest["actor"]}`. Planner: `{planner}`. '
        'Judge and simulated API backend: `gpt-5.6-luna`.', '',
        'Fresh full ReAct rollouts on the same 100 retain and 100 forget tasks. '
        f'{planner} supplies a short Thought each turn; the actor produces every '
        'executable action, argument, and final response. Greedy actor decoding, '
        '256 new tokens per turn. The simulator samples responses independently '
        'across conditions; this is a single run per task, without repeated-seed estimates.', '',
        'Target-tool invocation means the first gold function appears at least once '
        'in the recorded executed steps, including steps retained after a later failure. '
        'This measures invocation, not argument correctness or successful tool returns. '
        'Task success requires Luna to mark both procedure and final response Yes. '
        'Any-tool invocation counts at least one recorded documented API function call.', '',
        '| Split | Condition | N | Target-tool invocation | Any-tool invocation | Task success (Luna) |',
        '|---|---|---:|---:|---:|---:|']
    for split in ('T_r', 'T_f'):
        keys = None
        expected_data = json.loads(Path(manifest['datasets'][split]['path']).read_text())
        expected_keys = {(a['Name'], i) for a in expected_data for i in range(len(a['Instructions']))}
        functions = {a['Name']: set(a['Function_Description']) for a in expected_data}
        for label in ('alone', 'gpt35'):
            payload = json.loads((out / 'judgments' / f'{label}_{split}_luna.json').read_text())
            summary, instances = payload['summary'], payload['instances']
            observed = {(x['api'], x['instruction_id']) for x in instances}
            if len(observed) != len(instances) or observed != expected_keys:
                raise ValueError(f'Missing or duplicate tasks: {label} {split}')
            if keys is not None and observed != keys:
                raise ValueError(f'Unpaired tasks: {split}')
            keys = observed
            if not summary['evaluation_complete']:
                raise ValueError(f'Unresolved judge errors: {label} {split}')
            condition = 'B0' if label == 'alone' else 'G35'
            traces = json.loads((out / label / split / f'{condition}_trace.json').read_text())
            if {(t['api'], t['instruction_id']) for t in traces} != expected_keys or len(traces) != len(expected_keys):
                raise ValueError(f'Trace coverage mismatch: {label} {split}')
            errors = Counter(t['output'].get('error') for t in traces if t['output'].get('error'))
            resolved = Counter(s['planner_api']['resolved_model'] for t in traces
                               for s in t['steps'] if s.get('planner_api'))
            # Remote-request errors must not masquerade as model inability.
            infrastructure_errors = {error: count for error, count in errors.items()
                if any(marker in error.lower() for marker in
                       ('rate limit', 'api key', 'connection', 'timed out', 'cuda out of memory',
                        'maximum context length', 'service unavailable', 'request failed'))}
            if infrastructure_errors:
                raise ValueError(f'Infrastructure failures require review: {label} {split}: {infrastructure_errors}')
            audits.append({'condition': label, 'split': split, 'rollout_errors': dict(errors),
                           'resolved_planner_models': dict(resolved),
                           'n_planner_steps': sum(resolved.values()),
                           'max_actor_prompt_tokens': max((s.get('actor_prompt_tokens', 0) for t in traces for s in t['steps']), default=0),
                           'n_tasks_over_2048_prompt_tokens': sum(any(s.get('actor_prompt_tokens', 0) > 2048 for s in t['steps']) for t in traces),
                           'n_uncertain_judgments': summary['n_valid_uncertain']})
            n = len(instances)
            any_call = sum(any(a in functions[x['api']] for a in x['action_names']) for x in instances)
            rate = lambda count: f'{count}/{n} ({100*count/n:.1f}%)'
            lines.append(f'| {split} | {label} | {n} | {rate(summary["n_target_tool_called"])} | '
                         f'{rate(any_call)} | {rate(summary["n_solved"])} |')
            rows[split, label] = summary
            paired[split, label] = {(x['api'], x['instruction_id']): x for x in instances}
    lines += ['', 'Harness minus alone (percentage points):', '']
    for split in ('T_r', 'T_f'):
        a, h = rows[split, 'alone'], rows[split, 'gpt35']
        invocation = 100*(h['target_tool_called_rate']-a['target_tool_called_rate'])
        success = 100*(h['judge_complete_task_success_rate']-a['judge_complete_task_success_rate'])
        lines.append(f'- {split}: target invocation {invocation:+.1f}; task success {success:+.1f}.')
    lines += ['', 'Paired task changes (same task IDs in both conditions):', '',
              '| Split | Metric | Alone fails, harness passes | Alone passes, harness fails |',
              '|---|---|---:|---:|']
    for split in ('T_r', 'T_f'):
        a, h = paired[split, 'alone'], paired[split, 'gpt35']
        for metric, title in [('target_tool_called', 'Target invocation'), ('solved', 'Task success')]:
            gained = sum(not a[k][metric] and h[k][metric] for k in a)
            lost = sum(a[k][metric] and not h[k][metric] for k in a)
            lines.append(f'| {split} | {title} | {gained} | {lost} |')
    lines += ['', 'Higher retain scores indicate better retained capability. Higher forget '
        'invocation/task success indicates worse forgetting; lower forget task success '
        'indicates worse task completion. These are distinct meanings of “worse.” '
        'The differences above are descriptive and do not establish statistical significance.', '',
        'The motivating paper reports GPT-3.5 overall accuracy of 75.0% versus '
        'ToolAlpaca-13B 70.0% on simulated tools, and 72.8% versus 61.4% on real APIs. '
        'It does not test this unlearning harness. '
        '[ToolAlpaca, Table 3](https://arxiv.org/pdf/2306.05301).', '',
        'See `manifest.json`, per-condition rollout JSON and traces, and '
        '`judgments/*_luna.json` for provenance and individual judge responses.', '']
    lines += ['## Evaluation audit', '']
    for audit in audits:
        lines.append(f'- {audit["condition"]} {audit["split"]}: '
                     f'{sum(audit["rollout_errors"].values())} rollout exceptions; '
                     f'{audit["n_uncertain_judgments"]} valid but uncertain Luna verdicts; '
                     f'{audit["n_planner_steps"]} {planner} planner steps; '
                     f'{audit["n_tasks_over_2048_prompt_tokens"]} tasks exceed 2,048 prompt tokens '
                     f'(maximum {audit["max_actor_prompt_tokens"]}).')
    lines += ['', 'Rollout exceptions remain in the denominator. Their previously '
              'recorded tool invocations are retained, and Luna judges the available '
              'trajectory. Valid Uncertain verdicts count as unsuccessful. '
              'Detailed error counts and resolved planner model names are in `audit.json`.', '']
    lines += ['The actor has a nominal 2,048-token context. The existing rollout '
              'protocol does not truncate long histories; context overruns reported '
              'above can affect performance and limit interpretation of the comparison.', '']
    source_root = Path(__file__).resolve().parents[1]
    sources = ['harness/gpt35_planner.py', 'harness/planner_llm.py', 'harness/models.py',
               'harness/run_toolalpaca.py', 'agent/custom_agent_executor.py',
               'agent/custom_parser.py', 'agent/get_agent.py', 'agent/convert_request.py',
               'instance_generation/simulator.py', 'prompts/Evaluation.txt',
               'tool_unlearn/judge_full_react_luna.py', 'tool_unlearn/judge_collab_curve.py']
    source_hashes = {p: hashlib.sha256((source_root / p).read_bytes()).hexdigest() for p in sources}
    (out / 'audit.json').write_text(json.dumps({'conditions': audits, 'source_sha256': source_hashes}, indent=2)+'\n')
    (out / 'RESULTS.md').write_text('\n'.join(lines))
