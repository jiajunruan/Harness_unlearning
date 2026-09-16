"""Summarize sequence log probabilities saved by the local GPU evaluation."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'probability_results/local_gpu_verified'
SOURCE = ROOT / 'per_task_probabilities.csv'


def main():
    with SOURCE.open() as stream:
        source_rows = list(csv.DictReader(stream))
    assert len(source_rows) == 200
    rows, summaries = [], []
    paired = {}
    for model in ('original', 'unlearned'):
        selected = [r for r in source_rows if r['model'] == model]
        assert len(selected) == 100
        paired[model] = {r['task_id']: tuple(int(r[k + '_tokens']) for k in ('y', 'z', 'a'))
                         for r in selected}
        assert len(paired[model]) == 100
        for r in selected:
            row = {k: r[k] for k in ('model', 'model_path', 'task_id', 'target_tool')}
            for k in ('y', 'z', 'a'):
                row[k + '_tokens'] = int(r[k + '_tokens'])
                row[k + '_log_probability'] = float(r[k + '_log_probability'])
                assert math.isfinite(row[k + '_log_probability'])
                assert row[k + '_log_probability'] <= 0
            assert row['y_tokens'] == row['z_tokens'] + row['a_tokens']
            row['chain_rule_residual'] = (row['y_log_probability']
                                          - row['z_log_probability'] - row['a_log_probability'])
            assert abs(row['chain_rule_residual']) < 1e-8
            rows.append(row)
        summary = dict(model=model, n_tasks=100)
        for k in ('y', 'z', 'a'):
            total = math.fsum(float(r[k + '_log_probability']) for r in selected)
            summary[k + '_sum_log_probability_over_tasks'] = total
            summary[k + '_mean_sequence_log_probability'] = total / 100
            summary[k + '_mean_tokens'] = sum(int(r[k + '_tokens']) for r in selected) / 100
        summaries.append(summary)
    assert paired['original'] == paired['unlearned']
    for name, data in [('per_task_log_probabilities.csv', rows),
                       ('log_probability_summary.csv', summaries)]:
        with (ROOT / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    payload = dict(log_base='e', units='nats',
                   within_task='sum of reference-token log probabilities, without length normalization',
                   table_aggregate='equal arithmetic mean of sequence log probabilities over 100 tasks',
                   reference_unit='first reference thought and first tool call including arguments and markers',
                   source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                   max_abs_chain_rule_residual=max(abs(r['chain_rule_residual']) for r in rows),
                   summary=summaries)
    (ROOT / 'log_probability_summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
