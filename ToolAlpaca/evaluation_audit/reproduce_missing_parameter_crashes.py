"""Reproduce observed request conversion crashes offline; forbid HTTP requests."""
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'outputs/full_react_3way_luna_p0'
spec = importlib.util.spec_from_file_location('request_audit', REPO / 'agent/convert_request.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
def no_network(**kwargs):
    raise AssertionError('Audit must not send network requests')
module.requests.request = no_network
results = []
for model in ('original', 'unlearned', 'hybrid'):
    for split in ('T_r', 'T_f'):
        path = ROOT / model / split / ('H1.json' if model == 'hybrid' else 'B0.json')
        for api in json.loads(path.read_text()):
            for i, out in enumerate(api['Instances']):
                if "'NoneType' object is not iterable" not in out.get('error', ''):
                    continue
                attempt = out['action_attempts'][-1]
                route, method = api['Function_Projection'][attempt['tool']]
                try:
                    module.call_api_function(json.loads(attempt['tool_input']),
                                             json.loads(api['Documentation']), route, method,
                                             base_url='http://audit.invalid')
                except TypeError as exc:
                    assert str(exc) == "'NoneType' object is not iterable"
                    results.append(dict(model=model, split=split, task_id=f"{api['Name']}::{i}",
                                        tool=attempt['tool'], reproduced_error=str(exc)))
                else:
                    raise AssertionError('Expected recorded error')
assert len(results) == 18
Path(__file__).with_name('reproduced_crashes.json').write_text(json.dumps(results, indent=2) + '\n')
print('Reproduced all 18 observed missing-parameter crashes without network access.')
