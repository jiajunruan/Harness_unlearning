#!/usr/bin/env python3
"""Run the missing forget/retain comparators under the paper-aligned protocol.

Conditions: G35 (unlearned actor + GPT-3.5 Thought), S13 (ToolAlpaca-13B
alone), and GPT35 (GPT-3.5 end-to-end actor).  Each is evaluated on the fixed
100-task forget and retain splits with GPT-3.5-turbo as simulator and
gpt-4-0613 as the original process judge.  No credential is persisted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT.parent / "ToolAlpaca_original"
SHARED = Path("/projects/standard/mhong/shared/jiajunr")
SIMULATOR, JUDGE = "gpt-3.5-turbo", "gpt-4-0613"


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", required=True, choices=("G35", "S13", "GPT35"))
    p.add_argument("--split", required=True, choices=("forget", "retain"))
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--port", required=True, type=int)
    p.add_argument("--gpu", default="0", help="CUDA device for local conditions")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        p.error("OPENAI_API_KEY is required")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    data = ROOT / f"tool_unlearn/data/{args.split}_eval_100.json"
    expected = sum(len(row["Instructions"]) for row in json.loads(data.read_text()))
    tag = f"{args.condition}_{args.split}"
    manifest_path = out / "manifest.json"
    manifest = {
        "started_utc": utc(), "status": "running", "condition": args.condition,
        "split": args.split, "simulator": SIMULATOR, "judge": JUDGE,
        "dataset": {"path": str(data), "sha256": sha(data), "n": expected},
        "output": str(out), "port": args.port, "gpu": args.gpu,
        "remote_actor_model": os.getenv("REMOTE_ACTOR_MODEL", "gpt-3.5-turbo"),
        "remote_planner_model": os.getenv("REMOTE_PLANNER_MODEL", "gpt-3.5-turbo"),
    }
    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["condition"] != args.condition or manifest["split"] != args.split:
            raise RuntimeError("resume configuration mismatch")

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    env = dict(os.environ)
    env.update(PYTHONPATH=f"{ROOT}/data/general:{ROOT}",
               OPENAI_API_BASE=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
               ACTOR_MODEL=str(SHARED / "npo_gdr_ckpt/npo_gdr_ep5"),
               PLANNER_MODEL=str(SHARED / "ToolAlpaca-13B"),
               REMOTE_PLANNER_MODEL=os.getenv("REMOTE_PLANNER_MODEL", "gpt-3.5-turbo"),
               REMOTE_ACTOR_MODEL=os.getenv("REMOTE_ACTOR_MODEL", "gpt-3.5-turbo"),
               SIMULATOR_MODEL=SIMULATOR, JUDGE_MODEL=JUDGE,
               ACTOR_MAX_NEW_TOKENS="256", TOOLALPACA_EARLY_STOP="1",
               TOOLALPACA_STOP_AFTER_ACTION_INPUT="1", TOKENIZERS_PARALLELISM="false")
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    if args.condition == "S13":
        env.update(CUDA_VISIBLE_DEVICES=args.gpu, COLLAB_LAYOUT="single", PLANNER_MEM0="40GiB")
    elif args.condition == "G35":
        env.update(CUDA_VISIBLE_DEVICES=args.gpu, ACTOR_DEVICE="0", ACTOR_BACKEND="transformers")

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    simcmd = [sys.executable, "instance_generation/simulator.py", "-api", str(data),
              "-temp", str(ORIGINAL / "prompts/Simulator.txt"), "--port", str(args.port),
              "--model", SIMULATOR]
    manifest["simulator_command"] = simcmd
    save()
    with (out / "simulator.log").open("a" if args.resume else "w") as log:
        sim = subprocess.Popen(simcmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        session = requests.Session(); session.trust_env = False
        url = f"http://127.0.0.1:{args.port}"
        for _ in range(90):
            if sim.poll() is not None:
                raise RuntimeError("simulator exited; inspect simulator.log")
            try:
                if session.get(url + "/docs", timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            raise RuntimeError("simulator startup timed out")
        rollout = [sys.executable, "harness/run_toolalpaca.py", "--condition", args.condition,
                   "-api", str(data), "--server_url", url, "--instruction_length", "-1",
                   "-out", str(out)] + (["--resume"] if args.resume else [])
        manifest["rollout_command"] = rollout; save()
        with (out / "rollout.log").open("a" if args.resume else "w") as log:
            subprocess.run(rollout, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = json.loads((out / f"{args.condition}.json").read_text())
        n = sum(len(row.get("Instances", [])) for row in rows)
        if n != expected:
            raise RuntimeError(f"rollout has {n} tasks, expected {expected}")
        status = session.get(url + "/__simulator_status__", timeout=10).json()
        if status.get("model") != SIMULATOR or status.get("failures", 0):
            raise RuntimeError(f"simulator failures: {status}")
    finally:
        sim.terminate()
        try: sim.wait(timeout=15)
        except subprocess.TimeoutExpired: sim.kill(); sim.wait()
    judge = [sys.executable, "evaluation.py", "-api", str(out / f"{args.condition}.json"),
             "-temp", str(ORIGINAL / "prompts/Evaluation.txt"), "-out", str(out / "judge.json"),
             "--judge_model", JUDGE, "--num_workers", str(args.workers)]
    if args.resume and (out / "judge.json").exists(): judge.append("--continue_run")
    manifest["judge_command"] = judge; manifest["status"] = "judging"; save()
    with (out / "judge.log").open("a" if args.resume else "w") as log:
        subprocess.run(judge, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    stats = json.loads((out / "judge.json").read_text())["statistics"]
    if not stats.get("complete") or stats["num"] != expected:
        raise RuntimeError("judge incomplete")
    manifest.update(status="complete", finished_utc=utc(), statistics=stats,
                    process_accuracy=stats["process"]["Yes"] / expected,
                    response_accuracy=stats["response"]["Yes"] / expected,
                    overall_accuracy=stats["both"] / expected)
    save()
    print(f"{tag}: process={manifest['process_accuracy']:.1%}, overall={manifest['overall_accuracy']:.1%}")


if __name__ == "__main__":
    main()
