"""Teacher-force the first reference thought/action for the fixed 100 forget tasks.

Run with the Stable_evolving toolalpaca Python environment on one GPU.
Outputs are generated experiment artifacts; reference text is never generated
by either evaluated model. Arithmetic token means and sequence log likelihoods
are both retained so the chain-rule identity can be checked independently.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path('/users/2/jruan/Stable_evolving')
MODELS = {
    'original': '/projects/standard/mhong/shared/jiajunr/ToolAlpaca-7B',
    'unlearned': '/projects/standard/mhong/shared/jiajunr/npo_gdr_ckpt/npo_gdr_ep5',
}
TRACE_ROOT = Path('/projects/standard/mhong/shared/jiajunr/full_react_3way_luna_p0')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def references():
    from tool_unlearn.run_reasoning_action_transfer import initial_prompt

    source = read(REPO / 'data/train_data.json')
    benchmark = read(REPO / 'tool_unlearn/data/forget_eval_100.json')
    traces = {(v['api'], v['instruction_id']): v
              for v in read(TRACE_ROOT / 'original/T_f/B0_trace.json')}
    index = defaultdict(list)
    for api_idx, api in enumerate(source):
        for instance_idx, inst in enumerate(api.get('Instances', [])):
            question = inst.get('input', '').rsplit('\nHint: ', 1)[0]
            index[(api['Name'], question)].append((api_idx, instance_idx, api, inst))
    result = []
    for api in benchmark:
        base_name = re.sub(r'__\d+$', '', api['Name'])
        for instruction_id, question in enumerate(api['Instructions']):
            gold = api['Golden_Answers'][instruction_id]
            matches = [v for v in index[(base_name, question)]
                       if v[2]['Documentation'] == api['Documentation']
                       and v[2]['Function_Projection'] == api['Function_Projection']
                       and [{'Action': s[0][0], 'Action_Input': s[0][1]}
                            for s in v[3]['intermediate_steps']] == gold]
            assert matches, (api['Name'], instruction_id)
            # Two source entries duplicate the Vancouver demonstrations.
            # Accept copies only when all gold calls and the scored first
            # thought/action agree (later simulator observations may differ).
            assert all(v[3]['intermediate_steps'][0][0] == matches[0][3]['intermediate_steps'][0][0]
                       for v in matches), (api['Name'], instruction_id, 'ambiguous reference')
            api_idx, instance_idx, _, inst = matches[0]
            steps = inst['intermediate_steps']
            gold = api['Golden_Answers'][instruction_id]
            assert [{'Action': s[0][0], 'Action_Input': s[0][1]} for s in steps] == gold
            tool, arguments, log = steps[0][0]
            thought, sep, action_text = log.rpartition('\nAction:')
            assert sep and thought.strip(), (api['Name'], instruction_id)
            expected = f' {tool}\nAction Input: {arguments}'
            assert action_text == expected, (api['Name'], instruction_id, action_text)
            # Keep the reference thought's leading space, as in training.
            action = f'\nASSISTANT Action: {tool}\nASSISTANT Action Input: {arguments}'
            trace = traces[(api['Name'], instruction_id)]
            prompt = initial_prompt(api, trace['instruction'])
            assert prompt.endswith(trace['steps'][0]['actor_prompt_tail'])
            result.append(dict(task_id=f"{api['Name']}::{instruction_id}",
                               api=api['Name'], instruction_id=instruction_id,
                               source_api_index=api_idx, source_instance_index=instance_idx,
                               equivalent_source_indices=[[v[0], v[1]] for v in matches],
                               prompt=prompt, thought=thought, action=action,
                               target_tool=tool))
    assert len(result) == 100 and len({v['task_id'] for v in result}) == 100
    return result


def score(model, tokenizer, ref):
    import torch

    prompt, thought, action = (ref[k] for k in ('prompt', 'thought', 'action'))
    # Tokenize the concatenation once. Exact prefix checks prevent a token
    # spanning a text boundary from being silently assigned to the wrong term.
    x = tokenizer.encode(prompt)
    xz = tokenizer.encode(prompt + thought)
    xyz = tokenizer.encode(prompt + thought + action)
    assert xz[:len(x)] == x and xyz[:len(xz)] == xz, ref['task_id']
    assert len(x) < len(xz) < len(xyz)
    assert len(xyz) <= model.config.max_position_embeddings, (ref['task_id'], len(xyz))
    ids = torch.tensor([xyz], device=model.device)
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False).logits[0, len(x)-1:-1].float()
        targets = ids[0, len(x):]
        logp = logits.gather(-1, targets[:, None]).squeeze(-1) - logits.logsumexp(-1)
        probs = logp.exp().double()
        nz = len(xz) - len(x)
        slices = {'y': slice(None), 'z': slice(0, nz), 'a': slice(nz, None)}
        stats = {}
        for term, span in slices.items():
            stats[f'{term}_tokens'] = len(probs[span])
            stats[f'{term}_mean_token_probability'] = probs[span].mean().item()
            stats[f'{term}_log_probability'] = logp[span].double().sum().item()
        assert abs(stats['y_log_probability'] - stats['z_log_probability']
                   - stats['a_log_probability']) < 1e-8
    return dict(prompt_tokens=len(x), **stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'probability_results')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(REPO))
    refs = references()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'references.json').write_text(json.dumps(refs, indent=2) + '\n')
    print(f'Validated {len(refs)} unique source references and rollout prompts.', flush=True)
    if args.prepare_only:
        return
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available(), 'Run this scoring job on an allocated GPU.'
    torch.set_num_threads(4)
    rows = []
    reference_ids = None
    for name, path in MODELS.items():
        print(f'Loading {name}: {path}', flush=True)
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, local_files_only=True)
        token_ids = [tokenizer.encode(v['prompt'] + v['thought'] + v['action']) for v in refs]
        if reference_ids is None:
            reference_ids = token_ids
        assert token_ids == reference_ids, 'Both checkpoints must use identical reference tokenization.'
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            local_files_only=True).to('cuda:0').eval()
        with (args.output_dir / f'{name}_per_task.csv').open('w', newline='') as stream:
            writer = None
            for i, ref in enumerate(refs):
                row = dict(model=name, model_path=path,
                           **{k: ref[k] for k in ('task_id', 'api', 'instruction_id',
                              'source_api_index', 'source_instance_index', 'target_tool')},
                           **score(model, tokenizer, ref))
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                rows.append(row)
                if (i+1) % 10 == 0:
                    print(f'{name}: {i+1}/100 scored', flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    summary = {}
    for name in MODELS:
        selected = [v for v in rows if v['model'] == name]
        assert len(selected) == 100
        summary[name] = {'n_tasks': len(selected), **{
            f'{term}_mean_token_probability': sum(v[f'{term}_mean_token_probability'] for v in selected) / 100
            for term in ('y', 'z', 'a')}}
    with (args.output_dir / 'per_task_probabilities.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = dict(summary=summary, models=MODELS, torch_version=torch.__version__,
                   dtype='float16', probability_reduction='float32 log-softmax; float64 arithmetic means',
                   reference_unit='first gold thought and first gold tool call including arguments and action markers',
                   aggregate='arithmetic mean over tokens within each term, then equal mean over 100 tasks',
                   source_sha256=digest(REPO / 'data/train_data.json'),
                   benchmark_sha256=digest(REPO / 'tool_unlearn/data/forget_eval_100.json'),
                   script_sha256=digest(__file__))
    (args.output_dir / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
