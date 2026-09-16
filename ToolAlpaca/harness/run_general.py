"""General-capability side of the harness experiment: BBH-Hard and IFEval.

Conditions (toolalpaca_harness_experiment_plan.md section 8):

  H0        actor alone, standard protocol
  H_empty   wrapper present, guidance is a constant task-independent filler
  H_plan13  guidance written by the 13B planner
  S13       13B alone

The planner is never allowed to emit the graded artefact:
  * BBH   -- planner output is truncated at "So the answer is", so the final
             answer always comes from the actor;
  * IFEval-- the planner only writes a constraint checklist; the response that
             gets scored is always the actor's.
Whatever the planner emitted past the cut is kept as a leak marker.
"""
import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "general"))

from harness import models

BBH_DIR = "./data/general/bbh"
IFEVAL_PATH = "./data/general/instruction_following_eval/input_data.jsonl"

ANSWER_MARK = "So the answer is"
BBH_FILLER = "I will work through this carefully, one step at a time, checking each part before deciding."
IFEVAL_FILLER = "- Follow every explicit instruction stated in the request."

VICUNA_PREFIX = ("A chat between a curious user and an artificial intelligence assistant. "
                 "The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n")

IFEVAL_PLANNER_TMPL = (
    VICUNA_PREFIX +
    "USER: Read the request below and list, as a short checklist, every explicit "
    "formatting and content constraint it imposes (length, punctuation, casing, "
    "required sections, wrapping, language, and so on). Do not write the response "
    "to the request itself.\n\nRequest: {prompt}\nASSISTANT:"
)


# ------------------------------------------------------------------ context budget

# ToolAlpaca is a LLaMA-1 fine-tune: max_position_embeddings = 2048.  Several of
# the official 3-shot BBH CoT prompts do not fit (geometric_shapes alone is 2559
# tokens), and H_plan13 additionally has to fit the planner's rationale.  Shots
# are therefore dropped until prompt + RESERVE fits, using ONE reserve for every
# condition so that the same item is graded on the same prompt everywhere -- an
# item-dependent shot count would silently break the pairing.
CTX_LIMIT = 2048


def fit_shots(cot_prompt, question, tok, reserve):
    """Return (prompt, n_shots_used); drops trailing exemplars until it fits."""
    parts = cot_prompt.split("\n\nQ: ")
    head, shots = parts[0], parts[1:]
    while True:
        body = head + "".join("\n\nQ: " + s for s in shots)
        prompt = "%s\n\nQ: %s\nA: Let's think step by step." % (body, question)
        if len(tok(prompt)["input_ids"]) + reserve <= CTX_LIMIT or not shots:
            return prompt, len(shots)
        shots = shots[:-1]


# --------------------------------------------------------------------------- data

def load_bbh(per_task):
    """Stratified, deterministic: the first `per_task` examples of every task."""
    items = []
    for f in sorted(glob.glob(os.path.join(BBH_DIR, "bbh", "*.json"))):
        task = os.path.basename(f)[:-5]
        cot = open(os.path.join(BBH_DIR, "cot-prompts", task + ".txt")).read()
        cot = cot.split("-----", 1)[-1].strip()      # drop the canary header
        for i, ex in enumerate(json.load(open(f))["examples"][:per_task]):
            items.append({"task": task, "idx": i, "input": ex["input"],
                          "target": ex["target"], "cot_prompt": cot})
    return items


def load_ifeval(n, seed=0):
    rows = [json.loads(l) for l in open(IFEVAL_PATH)]
    rng = random.Random(seed)
    return rng.sample(rows, n) if n < len(rows) else rows


# ------------------------------------------------------------------------ scoring

def bbh_extract(text):
    if ANSWER_MARK not in text:
        return None
    ans = text.split(ANSWER_MARK, 1)[1].strip()
    ans = ans.split("\n")[0].strip()
    return ans[:-1].strip() if ans.endswith(".") else ans


# -------------------------------------------------------------------------- runs

def run_bbh(gen, items, mode, planner_gen, out, tok, reserve):
    correct = 0
    for it in items:
        base, n_shots = fit_shots(it["cot_prompt"], it["input"], tok, reserve)
        rec = {"task": it["task"], "idx": it["idx"], "target": it["target"],
               "mode": mode, "n_shots": n_shots}

        if mode == "H0":
            body = gen(base)
        else:
            if mode == "H_empty":
                guidance, leaked = BBH_FILLER, ""
            else:
                raw = planner_gen(base)
                if ANSWER_MARK in raw:
                    guidance, leaked = raw.split(ANSWER_MARK, 1)[0].strip(), ANSWER_MARK + raw.split(ANSWER_MARK, 1)[1]
                else:
                    guidance, leaked = raw.strip(), ""
                if not guidance:
                    guidance = BBH_FILLER
                rec["planner_raw"] = raw
            rec.update(planner_guidance=guidance, planner_leaked=bool(leaked),
                       planner_leaked_answer=leaked)
            body = guidance + "\n" + gen(base + " " + guidance + "\n")

        pred = bbh_extract(body)
        ok = pred is not None and pred == it["target"]
        correct += ok
        rec.update(actor_output=body, pred=pred, correct=bool(ok))
        out.append(rec)
    return correct, len(items)


def run_ifeval(gen, rows, mode, planner_gen, out):
    responses = {}
    for r in rows:
        rec = {"key": r["key"], "mode": mode}
        if mode == "H0":
            prompt = VICUNA_PREFIX + "USER: " + r["prompt"] + "\nASSISTANT:"
        else:
            if mode == "H_empty":
                checklist = IFEVAL_FILLER
            else:
                checklist = planner_gen(IFEVAL_PLANNER_TMPL.format(prompt=r["prompt"])).strip()
                rec["planner_raw"] = checklist
                if not checklist:
                    checklist = IFEVAL_FILLER
            rec["planner_guidance"] = checklist
            prompt = (VICUNA_PREFIX + "USER: " + r["prompt"] +
                      "\n\nBefore answering, note these constraints:\n" + checklist +
                      "\nASSISTANT:")
        resp = gen(prompt).strip()
        responses[r["prompt"]] = resp
        rec["response"] = resp
        out.append(rec)
    return responses


def score_ifeval(rows, responses):
    from instruction_following_eval import evaluation_lib as L
    inputs = [L.InputExample(key=r["key"], instruction_id_list=r["instruction_id_list"],
                             prompt=r["prompt"], kwargs=r["kwargs"]) for r in rows]
    outs = [L.test_instruction_following_strict(i, responses) for i in inputs]
    return sum(o.follow_all_instructions for o in outs), len(outs)


# -------------------------------------------------------------------------- main

CONDITIONS = {
    "H0":       {"actor": "7b",  "mode": "H0"},
    "H_empty":  {"actor": "7b",  "mode": "H_empty"},
    "H_plan13": {"actor": "7b",  "mode": "H_plan13"},
    "S13":      {"actor": "13b", "mode": "H0"},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    p.add_argument("--bbh_per_task", type=int, default=2)
    p.add_argument("--ifeval_n", type=int, default=30)
    p.add_argument("--out_dir", default="./outputs/harness_general")
    p.add_argument("--bbh_max_new", type=int, default=384)
    p.add_argument("--ifeval_max_new", type=int, default=768)
    p.add_argument("--planner_max_new", type=int, default=320)
    args = p.parse_args()

    cfg = CONDITIONS[args.condition]
    os.makedirs(args.out_dir, exist_ok=True)
    print("[general] condition=%s actor=%s mode=%s" % (args.condition, cfg["actor"], cfg["mode"]))

    if cfg["actor"] == "7b":
        a_tok, a_model = models.load_actor()
    else:
        a_tok, a_model = models.load_planner()
    p_tok = p_model = None
    if cfg["mode"] == "H_plan13":
        p_tok, p_model = models.load_planner()

    def make_gen(tok, model, max_new, stop):
        return lambda prompt: models.greedy_generate(tok, model, prompt,
                                                     max_new_tokens=max_new, stop=stop)

    bbh_items = load_bbh(args.bbh_per_task)
    ifeval_rows = load_ifeval(args.ifeval_n)
    records, summary = [], {"condition": args.condition}

    # one reserve for every condition, sized for planner rationale + answer
    reserve = args.bbh_max_new + args.planner_max_new
    c, n = run_bbh(make_gen(a_tok, a_model, args.bbh_max_new, ["\n\nQ:"]),
                   bbh_items, cfg["mode"],
                   make_gen(p_tok, p_model, args.planner_max_new, ["\n\nQ:"]) if p_model else None,
                   records, a_tok, reserve)
    summary["bbh_reserve_tokens"] = reserve
    summary["bbh_shot_reduced"] = sum(1 for r in records if r.get("n_shots", 3) < 3)
    summary["bbh"] = {"correct": c, "n": n, "exact_match": round(100.0 * c / n, 2)}
    print("[general] BBH-Hard exact match: %d/%d = %.1f" % (c, n, 100.0 * c / n))

    resp = run_ifeval(make_gen(a_tok, a_model, args.ifeval_max_new, ["\nUSER:"]),
                      ifeval_rows, cfg["mode"],
                      make_gen(p_tok, p_model, args.planner_max_new, ["\nUSER:"]) if p_model else None,
                      records)

    n_leak = sum(1 for r in records if r.get("planner_leaked"))
    summary["planner_leaked_steps"] = n_leak

    def dump():
        json.dump(records, open(os.path.join(args.out_dir, "%s_records.json" % args.condition), "w"),
                  ensure_ascii=False, indent=2)
        json.dump(resp, open(os.path.join(args.out_dir, "%s_ifeval_responses.json" % args.condition), "w"),
                  ensure_ascii=False, indent=2)
        json.dump(summary, open(os.path.join(args.out_dir, "%s_summary.json" % args.condition), "w"),
                  ensure_ascii=False, indent=2)

    # Generations are expensive; persist them before the scorer runs so a
    # scoring bug never costs a re-run.
    dump()
    # IFEval's scorer pulls in nltk and its punkt tables, which are not shipped
    # with this repo.  BBH is the headline general-capability number here, so a
    # missing scorer must not throw away a finished generation run -- the
    # responses are already on disk from the dump() above.
    try:
        c2, n2 = score_ifeval(ifeval_rows, resp)
    except (LookupError, ImportError) as e:
        print("[general] IFEval scoring skipped (%s: %s). Responses are saved; "
              "to score them install nltk and run "
              "`python -m nltk.downloader punkt punkt_tab`."
              % (type(e).__name__, str(e).strip().splitlines()[0]))
    else:
        summary["ifeval"] = {"correct": c2, "n": n2,
                             "prompt_strict_acc": round(100.0 * c2 / n2, 2)}
        print("[general] IFEval prompt-level strict acc: %d/%d = %.1f"
              % (c2, n2, 100.0 * c2 / n2))
    dump()
    print("[general] planner leaked the graded artefact in %d items" % n_leak)


if __name__ == "__main__":
    main()
