#!/usr/bin/env python3
"""Fresh paired B0/G35 rollouts, Luna judgments, and a Markdown report."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actor-model', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--port', type=int, default=5693)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--planner-model', default='gpt-3.5-turbo',
                        help='Remote model used only for the Thought slot')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not (args.actor_model / 'config.json').is_file():
        parser.error(f'Missing actor checkpoint: {args.actor_model}')
    if not any(args.actor_model.glob('*.bin')) and not any(args.actor_model.glob('*.safetensors')):
        parser.error(f'No actor weights found: {args.actor_model}')
    if not os.getenv('OPENAI_API_KEY'):
        parser.error('OPENAI_API_KEY is required')
    out = args.output_dir.resolve()
    # Never mix prior rollouts or judgments with this fresh experiment.
    out.mkdir(parents=True, exist_ok=args.resume)
    env = dict(os.environ)
    env.update(PYTHONPATH=f'{ROOT}/data/general:{ROOT}',
               OPENAI_API_BASE=os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
               ACTOR_MODEL=str(args.actor_model.resolve()), ACTOR_DEVICE='0',
               ACTOR_BACKEND='transformers', ACTOR_MAX_NEW_TOKENS='256',
               TOOLALPACA_EARLY_STOP='1', TOOLALPACA_STOP_AFTER_ACTION_INPUT='1',
               REMOTE_PLANNER_MODEL=args.planner_model,
               OPENAI_DEFAULT_MODEL='gpt-5.6-luna', JUDGE_MODEL='gpt-5.6-luna')
    env['NO_PROXY'] = env['no_proxy'] = '127.0.0.1,localhost'
    manifest = {'actor': str(args.actor_model.resolve()),
                'planner': args.planner_model, 'judge': 'gpt-5.6-luna',
                'simulator': 'gpt-5.6-luna', 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'actor_decoding': 'greedy fp16; 256 new tokens per turn',
                'datasets': {}, 'commands': [], 'status': 'running'}
    if args.resume:
        previous = json.loads((out / 'manifest.json').read_text())
        for field in ('actor', 'planner', 'judge', 'simulator', 'actor_decoding'):
            if previous[field] != manifest[field]:
                raise ValueError(f'Resume configuration mismatch: {field}')
        manifest = previous
        manifest['status'] = 'running'
        manifest.pop('error', None)
        manifest.setdefault('resumed_utc', []).append(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    def save_manifest():
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

    def run(command, log):
        manifest['commands'].append(command)
        save_manifest()
        print('Running:', ' '.join(command), flush=True)
        with log.open('a' if args.resume else 'w') as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)

    try:
        for split, name in [('T_r', 'retain'), ('T_f', 'forget')]:
            dataset = ROOT / f'tool_unlearn/data/{name}_eval_100.json'
            data = json.loads(dataset.read_text())
            dataset_info = {'path': str(dataset),
                'sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
                'n': sum(len(a['Instructions']) for a in data)}
            if split in manifest['datasets'] and manifest['datasets'][split] != dataset_info:
                raise ValueError(f'Resume dataset mismatch: {split}')
            manifest['datasets'][split] = dataset_info
            save_manifest()
            with (out / f'simulator_{split}.log').open('a' if args.resume else 'w') as log:
                sim = subprocess.Popen([sys.executable, 'instance_generation/simulator.py',
                    '-api', str(dataset), '--port', str(args.port)],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    session = requests.Session()
                    session.trust_env = False
                    url = f'http://127.0.0.1:{args.port}'
                    for _ in range(90):
                        if sim.poll() is not None:
                            raise RuntimeError('Simulator exited; inspect its log')
                        try:
                            if session.get(url+'/docs', timeout=2).ok:
                                break
                        except requests.RequestException:
                            pass
                        time.sleep(2)
                    else:
                        raise RuntimeError('Simulator startup timed out')
                    for label, condition in [('alone', 'B0'), ('gpt35', 'G35')]:
                        destination = out / label / split
                        run([sys.executable, 'harness/run_toolalpaca.py', '--condition', condition,
                             '-api', str(dataset), '--server_url', url,
                             '--instruction_length', '-1', '-out', str(destination)] + (['--resume'] if args.resume else []),
                            out / f'{label}_{split}.log')
                finally:
                    sim.terminate()
                    try:
                        sim.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        sim.kill()
                        sim.wait()
            for label, condition in [('alone', 'B0'), ('gpt35', 'G35')]:
                run([sys.executable, 'tool_unlearn/judge_full_react_luna.py',
                     '--run', f'{label}:{split}:{out / label / split / (condition+".json")}',
                     '--output-dir', str(out / 'judgments'), '--max-workers', str(args.workers)],
                    out / f'judge_{label}_{split}.log')
        from tool_unlearn.summarize_gpt35_experiment import summarize
        summarize(out)
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['error'] = str(exc)
        raise
    finally:
        save_manifest()


if __name__ == '__main__':
    main()
