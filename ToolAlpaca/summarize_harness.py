"""Collect every finished harness condition into the two comparison tables.

Tool side  : Procedure / Response / Overall (=ToolDelete T_T), judge held fixed.
General side: BBH-Hard exact match / IFEval prompt-level strict acc.

The claim under test is the *paired delta*:
  tool     H1       - B1       (13B plan vs. constant-filler plan, same wrapper)
  general  H_plan13 - H_empty  (same)
B1/H_empty, not B0/H0, is the honest control: it holds the wrapper fixed so the
delta cannot be explained by "the prompt got an extra sentence".
"""
import json
import os

TOOL_DIR = "outputs/harness_tool"
GEN_DIR = "outputs/harness_general"

TOOL_ORDER = [("B0", "7B alone (baseline)"),
              ("B1", "7B + constant filler (wrapper control)"),
              ("H1", "7B + 13B planner  <-- claim"),
              ("S13", "13B alone (ceiling)")]
GEN_ORDER = [("H0", "7B alone (baseline)"),
             ("H_empty", "7B + constant filler (wrapper control)"),
             ("H_plan13", "7B + 13B planner  <-- claim"),
             ("S13", "13B alone (ceiling)")]


def tool_row(cond):
    p = os.path.join(TOOL_DIR, "%s_judge.json" % cond)
    if not os.path.exists(p):
        return None
    s = json.load(open(p))["statistics"]
    n = s["num"]
    return {"n": n,
            "procedure": 100.0 * s["process"]["Yes"] / n,
            "response": 100.0 * s["response"]["Yes"] / n,
            "overall": 100.0 * s["both"] / n,
            "errors": s["error_num"]}


def gen_row(cond):
    p = os.path.join(GEN_DIR, "%s_summary.json" % cond)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {"bbh": d.get("bbh", {}).get("exact_match"),
            "bbh_n": d.get("bbh", {}).get("n"),
            "ifeval": d.get("ifeval", {}).get("prompt_strict_acc"),
            "ifeval_n": d.get("ifeval", {}).get("n"),
            "leaked": d.get("planner_leaked_steps")}


def fmt(v, w=6):
    return " " * w if v is None else ("%*.1f" % (w, v))


print("=== TOOL (ToolAlpaca simulated, judge=GLM-5.2) ===")
print("    S13 is a reduced-n ceiling reference (30 instr, first 3 APIs), not a paired row")
print("%-32s %6s %6s %6s %5s %5s" % ("condition", "Proc", "Resp", "T_T", "err", "n"))
tool = {}
for c, label in TOOL_ORDER:
    r = tool_row(c)
    tool[c] = r
    if r is None:
        print("%-32s %s" % ("%s  %s" % (c, label), "-- not finished --"))
    else:
        print("%-32s %s %s %s %5d %5d" % ("%s  %s" % (c, label),
              fmt(r["procedure"]), fmt(r["response"]), fmt(r["overall"]),
              r["errors"], r["n"]))
EXPECTED_N = {"B0": 100, "B1": 100, "H1": 100, "S13": 30}


def _full(r, cond="H1"):
    return r is not None and r["n"] >= EXPECTED_N.get(cond, 100)

if _full(tool.get("H1"), "H1") and _full(tool.get("B1"), "B1"):
    print("\n  paired delta  H1 - B1 (T_T): %+.1f" % (tool["H1"]["overall"] - tool["B1"]["overall"]))
if _full(tool.get("H1"), "H1") and _full(tool.get("B0"), "B0"):
    print("  vs raw baseline H1 - B0 (T_T): %+.1f" % (tool["H1"]["overall"] - tool["B0"]["overall"]))

print("\n=== GENERAL (BBH-Hard + IFEval) ===")
print("%-32s %6s %6s %6s" % ("condition", "BBH", "IFEval", "leak"))
gen = {}
for c, label in GEN_ORDER:
    r = gen_row(c)
    gen[c] = r
    if r is None:
        print("%-32s %s" % ("%s  %s" % (c, label), "-- not finished --"))
    else:
        print("%-32s %s %s %6s" % ("%s  %s" % (c, label),
              fmt(r["bbh"]), fmt(r["ifeval"]), r["leaked"]))
if gen.get("H_plan13") and gen.get("H_empty"):
    a, b = gen["H_plan13"], gen["H_empty"]
    print("\n  paired delta  H_plan13 - H_empty:  BBH %+.1f   IFEval %+.1f"
          % (a["bbh"] - b["bbh"], a["ifeval"] - b["ifeval"]))
if gen.get("H_plan13") and gen.get("H0"):
    a, b = gen["H_plan13"], gen["H0"]
    print("  vs raw baseline H_plan13 - H0:     BBH %+.1f   IFEval %+.1f"
          % (a["bbh"] - b["bbh"], a["ifeval"] - b["ifeval"]))


# ------------------------------------------------- reduced-n ceiling comparison
# S13 only ran the first 3 APIs, so its 30-instruction score cannot be read against
# the 100-instruction rows above.  Rescore every condition on exactly those APIs.

def subset_row(cond, api_names):
    p = os.path.join(TOOL_DIR, "%s_judge.json" % cond)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    n = proc = resp = both = 0
    for a in api_names:
        for item in d.get(a, []):
            n += 1
            pc = item.get("process_correctness") == "Yes"
            rc = item.get("final_response_correctness") == "Yes"
            proc += pc
            resp += rc
            both += pc and rc
    if n == 0:
        return None
    return {"n": n, "procedure": 100.0 * proc / n, "response": 100.0 * resp / n,
            "overall": 100.0 * both / n}


S13_APIS = [a["Name"] for a in json.load(open("data/eval_simulated.json"))[:3]]
sub = {c: subset_row(c, S13_APIS) for c, _ in TOOL_ORDER}
if sub.get("S13"):
    print("\n=== TOOL, rescored on the S13 slice only (%s) ===" % ", ".join(S13_APIS))
    print("%-32s %6s %6s %6s %5s" % ("condition", "Proc", "Resp", "T_T", "n"))
    for c, label in TOOL_ORDER:
        r = sub.get(c)
        if r is None:
            print("%-32s %s" % ("%s  %s" % (c, label), "-- not available --"))
        else:
            print("%-32s %s %s %s %5d" % ("%s  %s" % (c, label),
                  fmt(r["procedure"]), fmt(r["response"]), fmt(r["overall"]), r["n"]))
