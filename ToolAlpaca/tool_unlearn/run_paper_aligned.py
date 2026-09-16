#!/usr/bin/env python3
"""Fresh B0/H1 forget/retain evaluation with the original simulator and judge."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT.parent / 'ToolAlpaca_original'
SHARED = Path('/projects/standard/mhong/shared/jiajunr')
SIMULATOR = 'gpt-3.5-turbo'
JUDGE = 'gpt-4-0613'


def fingerprint(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--actor-model', type=Path, default=SHARED / 'npo_gdr_ckpt/npo_gdr_ep5')
    parser.add_argument('--planner-model', type=Path, default=SHARED / 'ToolAlpaca-13B')
    parser.add_argument('--length', type=int, default=-1,
                        help='Smoke test: first N instructions per split; -1 runs all 100')
    parser.add_argument('--workers', type=int, default=8, help='Judge workers per condition/split')
    parser.add_argument('--port', type=int, default=5710, help='First of four isolated simulator ports')
    parser.add_argument('--gpu-groups', default='0,1;2,3', help='Two GPU pairs, separated by semicolon')
    args = parser.parse_args()
    if args.length == 0 or args.length < -1 or args.workers < 1:
        parser.error('--length must be -1 or positive; --workers must be positive')
    groups = [g.split(',') for g in args.gpu_groups.split(';')]
    if len(groups) != 2 or any(len(g) != 2 for g in groups) or len(set(sum(groups, []))) != 4:
        parser.error('--gpu-groups must specify two disjoint pairs')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    lock = threading.RLock()
    manifest = {'started_utc': utc(), 'status': 'preflight', 'simulator': SIMULATOR,
                'judge': JUDGE, 'actor': str(args.actor_model.resolve()),
                'planner': str(args.planner_model.resolve()), 'runs': {}, 'datasets': {},
                'decoding': {'dtype': 'float16', 'do_sample': False, 'max_new_tokens': 256,
                             'early_stop': True, 'stop_after_action_input': True},
                'primary_metric': 'process.Yes / statistics.num (empty traces included)',
                'length': args.length, 'gpu_groups': groups,
                'data_limitation': 'Existing forget/retain instructions retained; not the paper unseen-tool set.'}

    def save():
        with lock:
            (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def status(label, state, **extra):
        with lock:
            manifest['runs'].setdefault(label, {}).update(status=state, updated_utc=utc(), **extra)
            save()

    env = dict(os.environ)
    env.update(PYTHONPATH=f'{ROOT}/data/general:{ROOT}',
               OPENAI_API_BASE=os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
               ACTOR_MODEL=str(args.actor_model.resolve()), PLANNER_MODEL=str(args.planner_model.resolve()),
               ACTOR_BACKEND='transformers', ACTOR_MAX_NEW_TOKENS='256',
               TOOLALPACA_EARLY_STOP='1', TOOLALPACA_STOP_AFTER_ACTION_INPUT='1',
               COLLAB_LAYOUT='split', PLANNER_MEM0='30GiB', PLANNER_MEM1='1GiB',
               SIMULATOR_MODEL=SIMULATOR, OPENAI_DEFAULT_MODEL=SIMULATOR, JUDGE_MODEL=JUDGE,
               TOKENIZERS_PARALLELISM='false')
    env['NO_PROXY'] = env['no_proxy'] = ','.join(filter(None, [env.get('NO_PROXY'), '127.0.0.1', 'localhost']))

    def command(label, cmd, logfile, jobenv):
        with lock:
            manifest['runs'][label].setdefault('commands', []).append(cmd)
            save()
        print(f'[{label}] {logfile.name}', flush=True)
        with logfile.open('w') as log:
            subprocess.run(cmd, cwd=ROOT, env=jobenv, stdout=log, stderr=subprocess.STDOUT, check=True)

    def judge(label, condition):
        status(label, 'judging')
        try:
            trajectory = out / label / f'{condition}.json'
            judgments = out / label / 'judge.json'
            command(label, [sys.executable, 'evaluation.py', '-api', str(trajectory),
                           '-temp', str(ORIGINAL / 'prompts/Evaluation.txt'),
                           '-out', str(judgments), '--judge_model', JUDGE,
                           '--num_workers', str(args.workers)], out / f'{label}_judge.log', env)
            stats = json.loads(judgments.read_text())['statistics']
            if stats.get('complete') is not True:
                raise RuntimeError(f'{label} judge did not complete successfully')
            expected = manifest['runs'][label]['n']
            if stats['num'] != expected:
                raise RuntimeError(f'Judge denominator {stats["num"]} differs from expected {expected}')
            status(label, 'complete', statistics=stats,
                   process_accuracy=stats['process']['Yes'] / expected,
                   response_accuracy=stats['response']['Yes'] / expected,
                   both_accuracy=stats['both'] / expected)
        except Exception as exc:
            status(label, 'failed', error=str(exc))
            raise

    def rollout(split, condition, group, index):
        label = f'{condition}_{split}'
        jobenv = dict(env, CUDA_VISIBLE_DEVICES=','.join(group), ACTOR_DEVICE='1' if condition == 'H1' else '0')
        data = Path(manifest['datasets'][split]['run_path'])
        port = args.port + index
        status(label, 'rollout', n=manifest['datasets'][split]['n'], gpu_group=group, port=port)
        # Fail on occupied ports instead of connecting to another run's simulator.
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', port))
        simcmd = [sys.executable, 'instance_generation/simulator.py', '-api', str(data),
                  '-temp', str(ORIGINAL / 'prompts/Simulator.txt'), '--port', str(port), '--model', SIMULATOR]
        with lock:
            manifest['runs'][label]['simulator_command'] = simcmd
            save()
        with (out / f'{label}_simulator.log').open('w') as log:
            sim = subprocess.Popen(simcmd, cwd=ROOT, env=jobenv, stdout=log, stderr=subprocess.STDOUT)
            try:
                session = requests.Session()
                session.trust_env = False
                url = f'http://127.0.0.1:{port}'
                for _ in range(90):
                    if sim.poll() is not None:
                        raise RuntimeError(f'{label} simulator exited; inspect its log')
                    try:
                        if session.get(url + '/docs', timeout=2).ok:
                            break
                    except requests.RequestException:
                        pass
                    time.sleep(2)
                else:
                    raise RuntimeError(f'{label} simulator startup timed out')
                command(label, [sys.executable, 'harness/run_toolalpaca.py', '--condition', condition,
                               '-api', str(data), '--server_url', url, '--instruction_length', '-1',
                               '-out', str(out / label)], out / f'{label}_rollout.log', jobenv)
                simulator_status = session.get(url + '/__simulator_status__', timeout=5)
                simulator_status.raise_for_status()
                simulator_status = simulator_status.json()
                if simulator_status.get('model') != SIMULATOR or simulator_status.get('failures', 0):
                    raise RuntimeError(f'{label} simulator invalid or had API failures: {simulator_status}')
                rows = json.loads((out / label / f'{condition}.json').read_text())
                count = sum(len(row.get('Instances', [])) for row in rows)
                if count != manifest['datasets'][split]['n']:
                    raise RuntimeError(f'{label} produced {count} instances; expected {manifest["datasets"][split]["n"]}')
                status(label, 'rollout_complete')
            except Exception as exc:
                status(label, 'failed', error=str(exc))
                raise
            finally:
                sim.terminate()
                try:
                    sim.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    sim.kill()
                    sim.wait()

    save()
    try:
        if not env.get('OPENAI_API_KEY'):
            raise RuntimeError('OPENAI_API_KEY is required')
        for path in (args.actor_model, args.planner_model):
            if not (path / 'config.json').is_file() or not (list(path.glob('*.bin')) + list(path.glob('*.safetensors'))):
                raise RuntimeError(f'Missing model config or weights: {path}')
        manifest['model_configs'] = [fingerprint(p / 'config.json') for p in (args.actor_model, args.planner_model)]
        manifest['prompts'] = [fingerprint(ORIGINAL / 'prompts' / p) for p in ('Simulator.txt', 'Evaluation.txt')]
        manifest['code'] = [fingerprint(ROOT / p) for p in (
            'tool_unlearn/run_paper_aligned.py', 'harness/run_toolalpaca.py', 'harness/planner_llm.py',
            'agent/agent_prompts.py', 'evaluation.py', 'instance_generation/simulator.py', 'utils.py')]
        for split in ('forget', 'retain'):
            source = ROOT / f'tool_unlearn/data/{split}_eval_100.json'
            rows = json.loads(source.read_text())
            selected = rows
            run_path = source
            if args.length != -1:
                selected, remaining = [], args.length
                for row in rows:
                    if remaining <= 0:
                        break
                    item = deepcopy(row)
                    take = min(remaining, len(item['Instructions']))
                    item['Instructions'] = item['Instructions'][:take]
                    item['Golden_Answers'] = item['Golden_Answers'][:take]
                    selected.append(item)
                    remaining -= take
                run_path = out / f'{split}_smoke.json'
                run_path.write_text(json.dumps(selected, indent=2) + '\n')
            n = sum(len(row['Instructions']) for row in selected)
            if n == 0:
                raise RuntimeError(f'Empty {split} evaluation set')
            manifest['datasets'][split] = dict(fingerprint(source), run_path=str(run_path),
                                             run_sha256=fingerprint(run_path)['sha256'], n=n)
        save()
        # Check actual completions: model-list visibility does not prove usable credit/access.
        manifest['preflight'] = {}
        def preflight(model):
            reply = requests.post(env['OPENAI_API_BASE'].rstrip('/') + '/chat/completions',
                                  headers={'Authorization': 'Bearer ' + env['OPENAI_API_KEY']},
                                  json={'model': model, 'messages': [{'role': 'user', 'content': 'Reply OK.'}],
                                        'temperature': 0, 'max_tokens': 4}, timeout=(10, 60))
            details = {'http_status': reply.status_code}
            if not reply.ok:
                try:
                    error = reply.json().get('error', {})
                    for field in ('code', 'message'):
                        details[field] = str(error.get(field, '')).replace(env['OPENAI_API_KEY'], '[REDACTED]')[:500]
                except (ValueError, AttributeError):
                    pass
                return model, details
            result = reply.json()
            details.update(returned_model=result.get('model'), completion_received=bool(result.get('choices')))
            return model, details

        with ThreadPoolExecutor(max_workers=2) as checks:
            for future in as_completed([checks.submit(preflight, model) for model in (SIMULATOR, JUDGE)]):
                model, details = future.result()
                manifest['preflight'][model] = details
                save()
        if any(not item.get('completion_received') for item in manifest['preflight'].values()):
            raise RuntimeError('API preflight failed; see manifest preflight for both required models. No model substitution performed.')
        manifest['status'] = 'running'
        save()
        # Each GPU pair advances independently; CPU/network judges overlap GPU rollouts.
        with ThreadPoolExecutor(max_workers=4) as judges:
            judge_futures = []

            def group_worker(split, group, group_index):
                for phase, condition in enumerate(('B0', 'H1')):
                    rollout(split, condition, group, group_index + phase * 2)
                    with lock:
                        judge_futures.append(judges.submit(judge, f'{condition}_{split}', condition))

            with ThreadPoolExecutor(max_workers=2) as rollouts:
                futures = [rollouts.submit(group_worker, split, groups[i], i)
                           for i, split in enumerate(('forget', 'retain'))]
                for future in as_completed(futures):
                    future.result()
            for future in as_completed(judge_futures):
                future.result()
        summary = {k: {field: value[field] for field in ('n', 'process_accuracy', 'response_accuracy', 'both_accuracy', 'statistics')}
                   for k, value in manifest['runs'].items()}
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        lines = ['Primary metric: GPT-4 Process Correctness (Yes / all instructions).', '',
                 '| Run | N | Process | Response (ancillary) | Both (ancillary) |',
                 '| --- | ---: | ---: | ---: | ---: |']
        for label, value in sorted(summary.items()):
            lines.append(f'| {label} | {value["n"]} | {value["process_accuracy"]:.1%} | '
                         f'{value["response_accuracy"]:.1%} | {value["both_accuracy"]:.1%} |')
        lines += ['', manifest['data_limitation'], '']
        (out / 'summary.md').write_text('\n'.join(lines))
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        manifest['finished_utc'] = utc()
        save()


if __name__ == '__main__':
    main()
