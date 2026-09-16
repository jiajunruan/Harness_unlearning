#!/usr/bin/env python
"""Build a FIXED evaluation subset: 100 forget + 100 retain instances.

Draws a deterministic sample (seed 42) of API entries until exactly N
instructions are selected; the last API's Instructions are truncated to hit N
exactly.  The harness runs one instance per instruction (verified: every API in
these files passes the OpenAPI checks), so these files yield exactly 100
instances each.  All four eval experiments use these same two files, so the
test set is fixed across runs.

Usage:  python tool_unlearn/build_eval_subset.py
Writes: tool_unlearn/data/forget_eval_100.json
        tool_unlearn/data/retain_eval_100.json
        tool_unlearn/data/eval_subset_manifest.json
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
N = 100
SEED = 42


def build(src_name, out_name, manifest_out):
    src = os.path.join(DATA, src_name)
    data = json.load(open(src, encoding="utf-8"))
    rng = random.Random(SEED)
    apis = data[:]
    rng.shuffle(apis)

    chosen, manifest, total = [], [], 0
    for api in apis:
        inst = api.get("Instructions") or []
        if not inst:
            continue
        if total + len(inst) <= N:
            chosen.append(api)
            manifest.append({"api": api["Name"], "n_instructions": len(inst), "truncated": False})
            total += len(inst)
            if total == N:
                break
        else:
            need = N - total
            if need <= 0:
                break
            kept = dict(api)
            kept["Instructions"] = inst[:need]
            chosen.append(kept)
            manifest.append({"api": api["Name"], "n_instructions": len(inst),
                             "truncated": True, "kept": need})
            total += need
            break

    if total != N:
        raise SystemExit(f"{src_name}: only {total} instructions available, need {N}")

    out = os.path.join(DATA, out_name)
    json.dump(chosen, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"{out_name}: {len(chosen)} APIs, {total} instructions (seed {SEED})")
    return manifest


manifests = {}
manifests["forget"] = build("forget_eval_api.json", "forget_eval_100.json", None)
manifests["retain"] = build("retain_eval_api.json", "retain_eval_100.json", None)
json.dump({"seed": SEED, "target": N, **manifests},
          open(os.path.join(DATA, "eval_subset_manifest.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("manifest -> tool_unlearn/data/eval_subset_manifest.json")
