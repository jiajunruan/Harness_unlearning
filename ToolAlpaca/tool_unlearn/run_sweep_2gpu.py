#!/usr/bin/env python
"""Run the 8-cell unlearning eval matrix on 2 GPUs, as fast as possible.

Two workers (one per A40) pull jobs from a shared queue, so the schedule
self-balances against the very different per-instruction speeds of the base
vs the unlearned checkpoint.  Each job is one `evaluate.sh` invocation
(simulator + rollout + parallel judge), fully configured by env.

No smoke test: this goes straight to the full 100-instruction sets.
The 256-token generation cap is the default in harness/planner_llm.py
(ACTOR_MAX_NEW_TOKENS); orig is unaffected (it stops at EOS ~124 tokens).
Collab runs use COLLAB_LAYOUT=single (whole 13B fp16 + 7B on one 48GB card).

orig single T_f was already completed earlier (single_orig/T_f_*_judge.json).
"""
import os
import queue
import subprocess
import sys
import threading
import time

ROOT = "/users/2/jruan/Stable_evolving"
os.chdir(ROOT)

ORIG = "/projects/standard/mhong/shared/jiajunr/ToolAlpaca-7B"
UNLEARN = "/projects/standard/mhong/shared/jiajunr/npo_gdr_ckpt/npo_gdr_ep5"
PLANNER = "/projects/standard/mhong/shared/jiajunr/ToolAlpaca-13B"
F_FILE = os.path.join(ROOT, "tool_unlearn/data/forget_eval_100.json")
R_FILE = os.path.join(ROOT, "tool_unlearn/data/retain_eval_100.json")
OUT = "/projects/standard/mhong/shared/jiajunr/unlearn_eval_results"

TA_PY = "/users/2/jruan/miniconda3/envs/toolalpaca/bin/python"
for d in (ORIG, UNLEARN, PLANNER):
    assert os.path.isdir(d), "missing model: %s" % d
for f in (F_FILE, R_FILE):
    assert os.path.isfile(f), "missing subset: %s" % f

# (label, metric, mode, model, api_file, out_dir)
# IMPORTANT: every job needs its OWN out_dir -- two jobs sharing a dir would
# clobber the same B0.json/B0_trace.json from different harness processes.
JOBS = [
    ("unlearn_single_Tf", "T_f", "single", UNLEARN, F_FILE, "single_unlearn_Tf"),
    ("unlearn_single_Tr", "T_r", "single", UNLEARN, R_FILE, "single_unlearn_Tr"),
    ("unlearn_collab_Tf", "T_f", "collab", UNLEARN, F_FILE, "collab_unlearn_Tf"),
    ("unlearn_collab_Tr", "T_r", "collab", UNLEARN, R_FILE, "collab_unlearn_Tr"),
    ("orig_single_Tr",    "T_r", "single", ORIG,   R_FILE, "single_orig_Tr"),
    ("orig_collab_Tf",    "T_f", "collab", ORIG,   F_FILE, "collab_orig_Tf"),
    ("orig_collab_Tr",    "T_r", "collab", ORIG,   R_FILE, "collab_orig_Tr"),
]

base_env = os.environ.copy()
base_env.update({
    "TA_PY": TA_PY,
    "TA_EXTRA_PKGS": os.path.join(ROOT, "data/general"),
    "TRITON_CACHE_DIR": "/tmp/triton_cd",
    "ACTOR_MAX_NEW_TOKENS": "256",
    "COLLAB_LAYOUT": "single",
    "PLANNER_MODEL": PLANNER,
    "PYTHONPATH": "%s:%s" % (os.path.join(ROOT, "data/general"), ROOT),
    "OPENAI_API_BASE": "https://api.deepseek.com",
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
    "SIMULATOR_MODEL": "deepseek-v4-flash",
    "JUDGE_MODEL": "deepseek-v4-flash",
    "JUDGE_WORKERS": "16",
    "no_proxy": "*", "NO_PROXY": "*",
})
for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    base_env.pop(k, None)

results = {}
job_lock = threading.Lock()


def run_job(label, metric, mode, model, api_file, out_dir, gpu, sim_port):
    env = base_env.copy()
    env.update({
        "MODEL_UNDER_TEST": model,
        "METRIC": metric,
        "MODE": mode,
        "SINGLE_GPU": str(gpu),
        "SIM_PORT": str(sim_port),
        "API_FILE": api_file,
        "OUT_DIR": os.path.join(OUT, out_dir),
    })
    log = os.path.join(OUT, "log_%s_gpu%d.txt" % (label, gpu))
    cmd = ["./tool_unlearn/evaluate.sh"]
    t0 = time.time()
    with open(log, "wb") as fh:
        p = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    dt = (time.time() - t0) / 60.0
    with job_lock:
        results[label] = (p.returncode, dt)
    print("[driver] %-18s gpu%d rc=%d %.1fmin  -> %s" % (label, gpu, p.returncode, dt, log),
          flush=True)


def worker(gpu, sim_port, jobs):
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        label, metric, mode, model, api_file, out_dir = job
        try:
            run_job(label, metric, mode, model, api_file, out_dir, gpu, sim_port)
        except Exception as e:
            print("[driver] %s FAILED on gpu%d: %r" % (label, gpu, e), flush=True)
            with job_lock:
                results[label] = (-1, 0)
        jobs.task_done()


q = queue.Queue()
for j in JOBS:
    q.put(j)

print("[driver] launching %d jobs on 2 GPUs (256-token cap, collab single-card)" % len(JOBS), flush=True)
workers = [
    threading.Thread(target=worker, args=(0, 5678, q), daemon=True),
    threading.Thread(target=worker, args=(1, 5679, q), daemon=True),
]
for w in workers:
    w.start()
t0 = time.time()
for w in workers:
    w.join()
wall = (time.time() - t0) / 60.0

print("\n[driver] all jobs done in %.1f min wall" % wall, flush=True)
fails = [lbl for lbl, (rc, _) in results.items() if rc != 0]
if fails:
    print("[driver] FAILED jobs: %s" % fails, flush=True)
else:
    print("[driver] all jobs rc=0", flush=True)

# recap table using the judge JSONs
import glob
import json
print("\n==============================================================")
print("SUMMARY (T = fraction Procedure+Response both correct):")
print("==============================================================")
for f in sorted(glob.glob(os.path.join(OUT, "*/*_judge.json"))):
    tag = os.path.basename(f).replace("_judge.json", "")
    try:
        s = json.load(open(f))["statistics"]
        n = s["num"]
        t = 100.0 * s["both"] / n if n else 0.0
        print("  %-34s T=%.1f  (errors %d, n %d)" % (tag, t, s["error_num"], n))
    except Exception as e:
        print("  %-34s ERROR %s" % (tag, e))
