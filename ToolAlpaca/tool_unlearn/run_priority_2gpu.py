#!/usr/bin/env python
"""Priority re-queue on 2 GPUs, per user decision 2026-08-31.

Order (each phase runs BOTH cards in parallel for max speed):
  1. orig_single_Tr  -- split retain set into 2 API halves (50/50 instructions),
                        GPU0+GPU1 in parallel, then combine the two judge JSONs.
  2. unlearn_collab_Tf  (collab forget)  -- GPU0
  3. unlearn_collab_Tr  (collab retain)  -- GPU1   [last in queue]

The remaining orig_collab_* control jobs are SKIPPED for now.

Replaces run_sweep_2gpu.py for this run; keep it around (it does the 8-cell matrix).
"""
import json
import os
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

# ------------------------------------------------------------ split retain
HALF0 = os.path.join(ROOT, "tool_unlearn/data/retain_orig_half0.json")
HALF1 = os.path.join(ROOT, "tool_unlearn/data/retain_orig_half1.json")


def build_halves():
    if os.path.isfile(HALF0) and os.path.isfile(HALF1):
        return
    data = json.load(open(R_FILE))
    n = len(data)
    half0, half1 = data[:n // 2], data[n // 2:]
    json.dump(half0, open(HALF0, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(half1, open(HALF1, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    t0 = sum(len(a["Instructions"]) for a in half0)
    t1 = sum(len(a["Instructions"]) for a in half1)
    print("[driver] retain split: half0 %d APIs/%d instr, half1 %d APIs/%d instr"
          % (len(half0), t0, len(half1), t1), flush=True)


build_halves()

base_env = os.environ.copy()
base_env.update({
    "TA_PY": TA_PY,
    "TA_EXTRA_PKGS": os.path.join(ROOT, "data/general"),
    "TRITON_CACHE_DIR": "/tmp/triton_cd",
    "ACTOR_MAX_NEW_TOKENS": "256",
    "COLLAB_LAYOUT": "single",
    "COLLAB_COND": "H1",
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
    results[label] = (p.returncode, dt)
    print("[driver] %-20s gpu%d rc=%d %.1fmin  -> %s" % (label, gpu, p.returncode, dt, log),
          flush=True)


def worker(gpu, sim_port, jobs):
    while True:
        try:
            job = jobs.get_nowait()
        except Exception:
            return
        label, metric, mode, model, api_file, out_dir = job
        try:
            run_job(label, metric, mode, model, api_file, out_dir, gpu, sim_port)
        except Exception as e:
            print("[driver] %s FAILED on gpu%d: %r" % (label, gpu, e), flush=True)
            results[label] = (-1, 0)
        jobs.task_done()


def combine_orig_tr():
    """Merge the two half judge JSONs into one orig T_r result."""
    j0 = os.path.join(OUT, "single_orig_Tr_h0/T_r_single_ToolAlpaca-7B_judge.json")
    j1 = os.path.join(OUT, "single_orig_Tr_h1/T_r_single_ToolAlpaca-7B_judge.json")
    if not (os.path.isfile(j0) and os.path.isfile(j1)):
        print("[driver] WARNING: missing half judge file(s): %s %s" % (j0, j1), flush=True)
        return None
    d0 = json.load(open(j0))
    d1 = json.load(open(j1))
    merged = {"statistics": {
        "num": 0, "error_num": 0,
        "process": {"Yes": 0, "No": 0, "Uncertain": 0},
        "response": {"Yes": 0, "No": 0, "Uncertain": 0},
        "both": 0,
    }}
    for d in (d0, d1):
        s = d["statistics"]
        for k in ("num", "error_num"):
            merged["statistics"][k] += s[k]
        for v in ("process", "response"):
            for y in ("Yes", "No", "Uncertain"):
                merged["statistics"][v][y] += s[v][y]
        merged["statistics"]["both"] += s["both"]
        for api, recs in d.items():
            if api == "statistics":
                continue
            merged.setdefault(api, []).extend(recs)
    jout = os.path.join(OUT, "single_orig_Tr/T_r_single_ToolAlpaca-7B_judge.json")
    os.makedirs(os.path.dirname(jout), exist_ok=True)
    json.dump(merged, open(jout, "w"), indent=4, ensure_ascii=False)
    return merged


import queue
q = queue.Queue()
for j in [
    ("orig_single_Tr_h0", "T_r", "single", ORIG, HALF0, "single_orig_Tr_h0"),
    ("orig_single_Tr_h1", "T_r", "single", ORIG, HALF1, "single_orig_Tr_h1"),
]:
    q.put(j)

print("[driver] ===== PHASE A: orig_single_Tr split on 2 GPUs (parallel) =====", flush=True)
workers = [
    threading.Thread(target=worker, args=(0, 5678, q), daemon=True),
    threading.Thread(target=worker, args=(1, 5679, q), daemon=True),
]
t0 = time.time()
for w in workers:
    w.start()
for w in workers:
    w.join()
wall = (time.time() - t0) / 60.0
print("[driver] phase A (orig retain, split) done in %.1f min wall" % wall, flush=True)

comb = combine_orig_tr()
if comb:
    s = comb["statistics"]; n = s["num"]
    print("[driver] COMBINED orig T_r: n=%d  T=%.1f  Proc=%.1f  Resp=%.1f  err=%d"
          % (n, 100.0 * s["both"] / n, 100.0 * s["process"]["Yes"] / n,
             100.0 * s["response"]["Yes"] / n, s["error_num"]), flush=True)

# ----------------------------------------------------------- phase B: collab
q2 = queue.Queue()
for j in [
    ("unlearn_collab_Tf", "T_f", "collab", UNLEARN, F_FILE, "collab_unlearn_Tf"),
    ("unlearn_collab_Tr", "T_r", "collab", UNLEARN, R_FILE, "collab_unlearn_Tr"),
]:
    q2.put(j)

print("[driver] ===== PHASE B: unlearn collab forget/retain on 2 GPUs (parallel) =====", flush=True)
workers = [
    threading.Thread(target=worker, args=(0, 5678, q2), daemon=True),
    threading.Thread(target=worker, args=(1, 5679, q2), daemon=True),
]
t0 = time.time()
for w in workers:
    w.start()
for w in workers:
    w.join()
wall = (time.time() - t0) / 60.0
print("[driver] phase B (collab) done in %.1f min wall" % wall, flush=True)

# ------------------------------------------------------------ recap
import glob
print("\n==============================================================")
print("SUMMARY (T = fraction Procedure+Response both correct):")
print("==============================================================")
for f in sorted(glob.glob(os.path.join(OUT, "*/*_judge.json"))):
    tag = os.path.basename(f).replace("_judge.json", "")
    try:
        s = json.load(open(f))["statistics"]; n = s["num"]
        t = 100.0 * s["both"] / n if n else 0.0
        print("  %-38s T=%.1f  (errors %d, n %d)" % (tag, t, s["error_num"], n))
    except Exception as e:
        print("  %-38s ERROR %s" % (tag, e))
print("\n[driver] ALL DONE.")
