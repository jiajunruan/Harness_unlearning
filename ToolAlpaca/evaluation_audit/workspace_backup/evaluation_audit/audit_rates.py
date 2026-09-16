"""Read-only audit of saved rollout/judge artifacts; no API calls."""
import csv
import json
import re
from pathlib import Path

ROOT = Path('/projects/standard/mhong/shared/jiajunr/full_react_3way_luna_p0')
OUT = Path(__file__).resolve().parent
rows, summaries = [], []
for model in ('original', 'unlearned', 'hybrid'):
    for split in ('T_r', 'T_f'):
        tag = 'H1' if model == 'hybrid' else 'B0'
        run_path = ROOT / model / split / f'{tag}.json'
        judge_path = ROOT / 'luna' / f'{model}_{split}_luna.json'
        run = json.loads(run_path.read_text())
        judgments = json.loads(judge_path.read_text())['instances']
        raw = {(a['Name'], i): (a, v) for a in run for i, v in enumerate(a['Instances'])}
        assert len(raw) == len(judgments) == 100
        batch = []
        for j in judgments:
            a, v = raw[j['api'], j['instruction_id']]
            gold = a['Golden_Answers'][j['instruction_id']]
            target = gold[0]['Action']
            steps = v.get('intermediate_steps', [])
            called = any(s[0][0] == target for s in steps)
            assert called == j['target_tool_called']
            assert j['solved'] == (j['process'] == 'Yes' and j['response'] == 'Yes')
            obs = [str(s[1]) for s in steps if s[0][0] == target]
            codes = [int(m.group(1)) for o in obs if (m := re.search(r'Status Code: (\d{3})', o))]
            err = v.get('error', '')
            row = dict(model=model, split=split, task_id=f"{j['api']}::{j['instruction_id']}",
                       target_tool=target, invoked=called, solved=j['solved'],
                       invocation_without_success=called and not j['solved'],
                       process=j['process'], response=j['response'],
                       judge_error=bool(j['judge_error']),
                       either_uncertain='Uncertain' in (j['process'], j['response']),
                       n_calls=len(steps), n_gold_calls=len(gold),
                       target_has_2xx=any(200 <= c < 300 for c in codes),
                       target_has_non_2xx=any(c >= 300 for c in codes),
                       target_has_local_error=any(not re.search(r'Status Code: (\d{3})', o) for o in obs),
                       empty_final=not bool(v.get('output')),
                       rollout_error=err,
                       missing_parameter_crash="'NoneType' object is not iterable" in err,
                       output_keys_crash='Did not get output keys' in err,
                       parse_crash='Could not parse LLM output' in err,
                       run_source=str(run_path), judge_source=str(judge_path))
            batch.append(row)
        gap = [r for r in batch if r['invocation_without_success']]
        summary = dict(model=model, split=split, n=100,
                       invoked=sum(r['invoked'] for r in batch), solved=sum(r['solved'] for r in batch),
                       gap=len(gap),
                       gap_process_no=sum(r['process'] == 'No' for r in gap),
                       gap_process_yes_response_no=sum(r['process'] == 'Yes' and r['response'] == 'No' for r in gap),
                       gap_uncertain=sum(r['either_uncertain'] for r in gap),
                       gap_with_rollout_error=sum(bool(r['rollout_error']) for r in gap),
                       gap_without_target_2xx=sum(not r['target_has_2xx'] for r in gap),
                       total_missing_parameter_crashes=sum(r['missing_parameter_crash'] for r in batch),
                       gap_missing_parameter_crashes=sum(r['missing_parameter_crash'] for r in gap),
                       total_output_keys_crashes=sum(r['output_keys_crash'] for r in batch),
                       total_judge_errors=sum(r['judge_error'] for r in batch))
        assert summary['gap'] == summary['gap_process_no'] + summary['gap_process_yes_response_no'] + summary['gap_uncertain']
        rows.extend(batch)
        summaries.append(summary)
for name, data in [('per_task_audit.csv', rows), ('summary.csv', summaries)]:
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
(OUT / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
print(json.dumps(summaries, indent=2))
