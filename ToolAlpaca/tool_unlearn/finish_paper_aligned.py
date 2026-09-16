#!/usr/bin/env python3
"""Finish a stopped paper-aligned H1 forget run using two independent GPU pairs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT.parent / 'ToolAlpaca_original'
SIMULATOR = 'gpt-3.5-turbo'
JUDGE = 'gpt-4-0613'


def digest(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def metadata(row):
    return {k: v for k, v in row.items() if k != 'Instances'}


def completed_groups(source, partial):
    original = {row['Name']: row for row in source}
    if len(original) != len(source):
        raise ValueError('Duplicate API names in source')
    done = {}
    seen = set()
    for row in partial:
        name = row['Name']
        if name in seen or name not in original or metadata(row) != metadata(original[name]):
            raise ValueError(f'Partial source metadata mismatch or duplicate: {name}')
        seen.add(name)
        instances = row.get('Instances', [])
        if instances:
            if len(instances) != len(original[name]['Instructions']):
                raise ValueError(f'Incomplete saved API group: {name}')
            done[name] = row
    return done


def split_remaining(source, done):
    shards, sizes = [[], []], [0, 0]
    for row in source:
        if row['Name'] in done:
            continue
        index = min(range(2), key=lambda i: sizes[i])
        clean = deepcopy(row)
        clean.pop('Instances', None)
        shards[index].append(clean)
        sizes[index] += len(row['Instructions'])
    return shards


def validate_traces(source, rows):
    lookup = {row['Name']: row for row in source}
    expected = {(row['Name'], i) for row in source for i in range(len(row['Instructions']))}
    actual = set()
    for row in rows:
        key = (row['api'], row['instruction_id'])
        if key not in expected or key in actual:
            raise ValueError(f'Unexpected or duplicate trace: {key}')
        actual.add(key)
        original = lookup[key[0]]
        instruction = original['Instructions'][key[1]]
        if original.get('Authentication'):
            instruction += '\nAuthentication information: ' + ' '.join(
                f'{k}={v}' for k, v in original['Authentication'].items())
        if row['instruction'] != instruction or row['golden_answers'] != original['Golden_Answers'][key[1]]:
            raise ValueError(f'Trace instruction/gold mismatch: {key}')
    if actual != expected:
        raise ValueError(f'Trace coverage mismatch: {len(actual)} / {len(expected)}')
    order = {row['Name']: i for i, row in enumerate(source)}
    return sorted(rows, key=lambda row: (order[row['api']], row['instruction_id']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    try:
        relative_out = str(out.relative_to(ROOT))
    except ValueError:
        relative_out = str(out)
    # Refuse to race the original coordinator or any rollout writing this run.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().decode().split('\0')
        except (OSError, UnicodeError):
            continue
        if any(Path(arg).name in ('run_paper_aligned.py', 'run_toolalpaca.py', 'finish_paper_aligned.py') for arg in argv):
            if any(str(out) in arg or relative_out == arg for arg in argv):
                raise RuntimeError(f'Original run or continuation still active: PID {proc.name}')
    manifest_path = out / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for label in ('B0_forget', 'B0_retain', 'H1_retain'):
        if manifest['runs'][label]['status'] != 'complete':
            raise RuntimeError(f'{label} must already be complete')
    if manifest['simulator'] != SIMULATOR or manifest['judge'] != JUDGE:
        raise RuntimeError('Required simulator/judge mismatch')
    if not os.environ.get('OPENAI_API_KEY'):
        raise RuntimeError('OPENAI_API_KEY required')
    source_path = Path(manifest['datasets']['forget']['run_path'])
    if digest(source_path)['sha256'] != manifest['datasets']['forget']['run_sha256']:
        raise RuntimeError('Original forget dataset hash changed')
    source = json.loads(source_path.read_text())
    n = sum(len(row['Instructions']) for row in source)
    if n != 100:
        raise RuntimeError(f'Expected 100 forget instructions, got {n}')
    trajectory = out / 'H1_forget/H1.json'
    trace_path = out / 'H1_forget/H1_trace.json'
    done = completed_groups(source, json.loads(trajectory.read_text()))
    saved_traces = [row for row in json.loads(trace_path.read_text()) if row['api'] in done]
    validate_traces([row for row in source if row['Name'] in done], saved_traces)
    continuation = out / 'continuation'
    continuation.mkdir(exist_ok=False)
    for path in (manifest_path, trajectory, trace_path):
        shutil.copy2(path, continuation / ('original_' + path.name))
    shards = split_remaining(source, done)
    lock = threading.RLock()
    record = {'started_utc': utc(), 'previous_status': manifest['status'],
              'reason': 'Rescheduled remaining disjoint API groups onto newly free GPU pair; completed trajectories preserved',
              'previous_run': deepcopy(manifest['runs']['H1_forget']),
              'preserved_instructions': sum(len(row['Instances']) for row in done.values()),
              'snapshots': [digest(path) for path in sorted(continuation.glob('original_*'))],
              'code': [digest(ROOT / p) for p in ('tool_unlearn/finish_paper_aligned.py', 'evaluation.py',
                       'utils.py', 'harness/run_toolalpaca.py', 'harness/planner_llm.py',
                       'instance_generation/simulator.py')],
              'prompts': [digest(ORIGINAL / 'prompts' / p) for p in ('Simulator.txt', 'Evaluation.txt')],
              'shards': {}, 'commands': [], 'status': 'running'}
    manifest.setdefault('continuations', []).append(record)
    manifest['status'] = 'running'
    manifest['runs']['H1_forget']['status'] = 'continuing'
    for field in ('statistics', 'process_accuracy', 'response_accuracy', 'both_accuracy', 'error'):
        manifest['runs']['H1_forget'].pop(field, None)
    manifest.pop('error', None)

    def save():
        with lock:
            temporary = manifest_path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(manifest, indent=2) + '\n')
            temporary.replace(manifest_path)

    env = dict(os.environ)
    env.update(PYTHONPATH=f'{ROOT}/data/general:{ROOT}',
               OPENAI_API_BASE=os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
               ACTOR_MODEL=manifest['actor'], PLANNER_MODEL=manifest['planner'],
               ACTOR_BACKEND='transformers', ACTOR_DEVICE='1', ACTOR_MAX_NEW_TOKENS='256',
               COLLAB_LAYOUT='split', PLANNER_MEM0='30GiB', PLANNER_MEM1='1GiB',
               TOOLALPACA_EARLY_STOP='1', TOOLALPACA_STOP_AFTER_ACTION_INPUT='1',
               SIMULATOR_MODEL=SIMULATOR, OPENAI_DEFAULT_MODEL=SIMULATOR, JUDGE_MODEL=JUDGE,
               TOKENIZERS_PARALLELISM='false')
    env['NO_PROXY'] = env['no_proxy'] = ','.join(filter(None, [env.get('NO_PROXY'), '127.0.0.1', 'localhost']))

    def run_command(cmd, logfile, jobenv):
        with lock:
            record['commands'].append(cmd)
            save()
        with logfile.open('w') as log:
            subprocess.run(cmd, cwd=ROOT, env=jobenv, stdout=log, stderr=subprocess.STDOUT, check=True)

    def shard_run(index, rows):
        label = str(index)
        destination = continuation / f'shard{index}'
        destination.mkdir()
        dataset = destination / 'dataset.json'
        dataset.write_text(json.dumps(rows, indent=2) + '\n')
        count = sum(len(row['Instructions']) for row in rows)
        jobenv = dict(env, CUDA_VISIBLE_DEVICES='0,1' if index == 0 else '2,3')
        port = 5720 + index
        with lock:
            record['shards'][label] = {'dataset': digest(dataset), 'n': count,
                                       'gpu_group': jobenv['CUDA_VISIBLE_DEVICES'], 'port': port, 'status': 'running'}
            save()
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', port))
        cmd = [sys.executable, 'instance_generation/simulator.py', '-api', str(dataset),
               '-temp', str(ORIGINAL / 'prompts/Simulator.txt'), '--port', str(port), '--model', SIMULATOR]
        with lock:
            record['commands'].append(cmd)
            save()
        with (destination / 'simulator.log').open('w') as log:
            simulator = subprocess.Popen(cmd, cwd=ROOT, env=jobenv, stdout=log, stderr=subprocess.STDOUT)
            try:
                session = requests.Session()
                session.trust_env = False
                url = f'http://127.0.0.1:{port}'
                for _ in range(90):
                    if simulator.poll() is not None:
                        raise RuntimeError(f'Shard {index} simulator exited')
                    try:
                        if session.get(url + '/docs', timeout=2).ok:
                            break
                    except requests.RequestException:
                        pass
                    time.sleep(2)
                else:
                    raise RuntimeError(f'Shard {index} simulator startup timeout')
                run_command([sys.executable, 'harness/run_toolalpaca.py', '--condition', 'H1',
                             '-api', str(dataset), '--server_url', url, '--instruction_length', '-1',
                             '-out', str(destination)], destination / 'rollout.log', jobenv)
                health = session.get(url + '/__simulator_status__', timeout=5)
                health.raise_for_status()
                health = health.json()
                if health.get('model') != SIMULATOR or health.get('failures', 0):
                    raise RuntimeError(f'Shard {index} simulator failed: {health}')
                result = completed_groups(rows, json.loads((destination / 'H1.json').read_text()))
                if set(result) != {row['Name'] for row in rows}:
                    raise RuntimeError(f'Shard {index} output incomplete')
                traces = validate_traces(rows, json.loads((destination / 'H1_trace.json').read_text()))
                with lock:
                    record['shards'][label].update(status='complete', simulator_status=health)
                    save()
                return result, traces
            except BaseException as exc:
                with lock:
                    record['shards'][label].update(status='failed', error=str(exc))
                    save()
                raise
            finally:
                simulator.terminate()
                try:
                    simulator.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    simulator.kill()
                    simulator.wait()

    save()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(shard_run, index, rows) for index, rows in enumerate(shards) if rows]
            for future in as_completed(futures):
                result, traces = future.result()
                if set(done) & set(result):
                    raise RuntimeError('Duplicate merged API groups')
                done.update(result)
                saved_traces.extend(traces)
        merged = [done[row['Name']] for row in source]
        if sum(len(row['Instances']) for row in merged) != n:
            raise RuntimeError('Merged instruction count mismatch')
        traces = validate_traces(source, saved_traces)
        # Preserve raw snapshots; replace the output only after full validation.
        for path, rows in ((trajectory, merged), (trace_path, traces)):
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(rows, indent=2) + '\n')
            temporary.replace(path)
        record['merged_outputs'] = [digest(trajectory), digest(trace_path)]
        manifest['runs']['H1_forget']['status'] = 'judging'
        save()
        judge_path = out / 'H1_forget/judge.json'
        if judge_path.exists():
            shutil.copy2(judge_path, continuation / 'original_judge.json')
        run_command([sys.executable, 'evaluation.py', '-api', str(trajectory),
                     '-temp', str(ORIGINAL / 'prompts/Evaluation.txt'), '-out', str(judge_path),
                     '--judge_model', JUDGE, '--num_workers', '8'], continuation / 'judge.log', env)
        stats = json.loads(judge_path.read_text())['statistics']
        if stats.get('complete') is not True or stats['num'] != n:
            raise RuntimeError('Incomplete GPT-4 judge or denominator mismatch')
        manifest['runs']['H1_forget'].update(status='complete', updated_utc=utc(), n=n, statistics=stats,
            process_accuracy=stats['process']['Yes'] / n, response_accuracy=stats['response']['Yes'] / n,
            both_accuracy=stats['both'] / n)
        summary = {label: {field: item[field] for field in ('n', 'process_accuracy', 'response_accuracy',
                    'both_accuracy', 'statistics')} for label, item in manifest['runs'].items()}
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        lines = ['Primary metric: GPT-4 Process Correctness (Yes / all instructions).', '',
                 '| Run | N | Process | Response (ancillary) | Both (ancillary) |',
                 '| --- | ---: | ---: | ---: | ---: |']
        for label, item in sorted(summary.items()):
            lines.append(f'| {label} | {item["n"]} | {item["process_accuracy"]:.1%} | '
                         f'{item["response_accuracy"]:.1%} | {item["both_accuracy"]:.1%} |')
        lines += ['', manifest['data_limitation'], '']
        (out / 'summary.md').write_text('\n'.join(lines))
        record['status'] = manifest['status'] = 'complete'
    except BaseException as exc:
        record.update(status='failed', error=str(exc))
        manifest.update(status='failed', error=str(exc))
        manifest['runs']['H1_forget'].update(status='failed', error=str(exc))
        raise
    finally:
        record['finished_utc'] = manifest['finished_utc'] = utc()
        save()


if __name__ == '__main__':
    main()
