#!/usr/bin/env python3
"""Test whether forgotten tool calls recover when the original Thought is supplied.

This is the two-stage experiment proposed for trajectory-level tool unlearning:

  Setting 1 (loaded from saved traces)
      original model:  x -> Thought -> Action(target tool)
      unlearned model: x -> Thought -> Action(not target tool)

  Setting 2 (run by this script)
      original model on GPU 0:  x -> original Thought
      unlearned model on GPU 1: x + original Thought -> Action

The candidate set is intentionally strict.  A row is eligible only when the
original model's *first* saved action is the gold first action and the
unlearned model never calls that gold tool anywhere in its saved Setting-1
trajectory.  Thus a Setting-2 call of that tool is a genuine recovery, not a
tool that the unlearned model already reached later in Setting 1.

The script does not call the simulated API and evaluates only the first action.
It is therefore a direct test of p_unlearned(action | instruction, Thought),
not an end-to-end task-success evaluation.

Example (from the repository root):

  /users/2/jruan/miniconda3/envs/toolalpaca/bin/python \
    tool_unlearn/run_reasoning_action_transfer.py

Results are checkpointed after every case, so the same command can safely be
re-run after an interruption.  Use --overwrite to start a new run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

# Running ``python tool_unlearn/run_reasoning_action_transfer.py`` sets
# sys.path[0] to tool_unlearn/.  Add the repository root so the benchmark's
# agent and prompt modules resolve without requiring PYTHONPATH.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agent.agent_prompts import prompt_proj
from agent.get_agent import get_agent


DEFAULT_RESULTS_ROOT = Path("/projects/standard/mhong/shared/jiajunr/unlearn_eval_results")
DEFAULT_ORIGINAL_MODEL = "/projects/standard/mhong/shared/jiajunr/ToolAlpaca-7B"
DEFAULT_UNLEARNED_MODEL = "/projects/standard/mhong/shared/jiajunr/npo_gdr_ckpt/npo_gdr_ep5"
ACTION_RE = re.compile(r"ASSISTANT\s+Action\s*:\s*([^\n\r]+)", re.IGNORECASE)
ACTION_OR_RESPONSE_RE = re.compile(r"ASSISTANT\s+(?:Action|Response)\s*:", re.IGNORECASE)
# Complete Action line (rather than just its marker).  The action name must be
# present before generation stops, otherwise it cannot be scored.
# Do not accept end-of-current-generation as a completed line: action names can
# span SentencePiece tokens (e.g. ``search`` + ``Items``).  A following newline
# unambiguously means the entire action name has been emitted.
COMPLETE_ACTION_LINE_RE = re.compile(r"ASSISTANT\s+Action\s*:\s*[^\n\r]+\r?\n", re.IGNORECASE)
RESPONSE_MARK_RE = re.compile(r"ASSISTANT\s+Response\s*:", re.IGNORECASE)


class PromptOnlyLLM(LLM):
    """Minimal LLM used only to reuse ToolAlpaca's exact prompt builder."""

    @property
    def _llm_type(self) -> str:
        return "prompt_only"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        raise RuntimeError("PromptOnlyLLM must never be used for generation")


@dataclass(frozen=True)
class Candidate:
    api: str
    instruction_id: int
    instruction: str
    target_tool: str
    saved_original_actions: List[str]
    saved_unlearned_actions: List[str]

    @property
    def key(self) -> str:
        return f"{self.api}::{self.instruction_id}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--original-trace",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "single_orig/T_f_single_ToolAlpaca-7B_trace.json",
        help="Saved greedy Setting-1 trace for the original model.",
    )
    p.add_argument(
        "--unlearned-trace",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "single_unlearn_Tf/T_f_single_npo_gdr_ep5_trace.json",
        help="Saved greedy Setting-1 trace for the unlearned model.",
    )
    p.add_argument("--eval-data", type=Path, default=Path("tool_unlearn/data/forget_eval_100.json"))
    p.add_argument("--original-model", default=DEFAULT_ORIGINAL_MODEL)
    p.add_argument("--unlearned-model", default=DEFAULT_UNLEARNED_MODEL)
    p.add_argument("--original-device", default="cuda:0")
    p.add_argument("--unlearned-device", default="cuda:1")
    p.add_argument("--thought-max-new-tokens", type=int, default=256)
    p.add_argument("--action-max-new-tokens", type=int, default=256)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("tool_unlearn/setting2_out/reasoning_action_transfer_npo_gdr_ep5.json"),
    )
    p.add_argument("--overwrite", action="store_true", help="Ignore an existing output checkpoint.")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N selected cases (useful as a smoke test).",
    )
    return p.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def actions_in_trace(trace: Dict[str, Any]) -> List[str]:
    actions: List[str] = []
    for step in trace.get("steps", []):
        match = ACTION_RE.search(step.get("actor_output", ""))
        if match:
            actions.append(match.group(1).strip())
    return actions


def index_gold_first_actions(eval_data: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, int], str]:
    gold: Dict[Tuple[str, int], str] = {}
    for api in eval_data:
        name = api["Name"]
        for instruction_id, answers in enumerate(api.get("Golden_Answers", [])):
            if answers:
                gold[(name, instruction_id)] = answers[0]["Action"]
    return gold


def select_candidates(
    original_trace: Sequence[Dict[str, Any]],
    unlearned_trace: Sequence[Dict[str, Any]],
    gold_actions: Dict[Tuple[str, int], str],
) -> List[Candidate]:
    """Find original-success / unlearned-tool-absent Setting-1 cases."""
    original_by_key = {(x["api"], x["instruction_id"]): x for x in original_trace}
    unlearned_by_key = {(x["api"], x["instruction_id"]): x for x in unlearned_trace}
    if set(original_by_key) != set(unlearned_by_key):
        only_o = sorted(set(original_by_key) - set(unlearned_by_key))
        only_u = sorted(set(unlearned_by_key) - set(original_by_key))
        raise ValueError(f"Saved traces are not paired (only original={only_o[:3]}, only unlearned={only_u[:3]})")

    candidates: List[Candidate] = []
    for key in sorted(original_by_key):
        target_tool = gold_actions.get(key)
        if target_tool is None:
            continue
        original = original_by_key[key]
        unlearned = unlearned_by_key[key]
        original_actions = actions_in_trace(original)
        unlearned_actions = actions_in_trace(unlearned)

        # The original must select the gold forgotten tool immediately.  The
        # unlearned trace must never reach it, including after an initial error.
        if original_actions and original_actions[0] == target_tool and target_tool not in unlearned_actions:
            candidates.append(
                Candidate(
                    api=key[0],
                    instruction_id=key[1],
                    instruction=original["instruction"],
                    target_tool=target_tool,
                    saved_original_actions=original_actions,
                    saved_unlearned_actions=unlearned_actions,
                )
            )
    return candidates


def initial_prompt(api: Dict[str, Any], instruction: str) -> str:
    """Build the same initial test_v1 prompt used by the saved B0 traces."""
    executor = get_agent(
        llm=PromptOnlyLLM(),
        api_data=api,
        server_url=None,
        agent_prompt=prompt_proj["test_v1"],
        # Historical B0 traces omitted getDetails because the old enable flag
        # was inverted. Preserve their exact prompt after fixing that flag.
        enable_getDetails=False,
    )
    return executor.agent.llm_chain.prompt.format(input=instruction, agent_scratchpad="")


def load_model_and_tokenizer(path: str, device: str):
    print(f"[load] {path} -> {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return tokenizer, model


def model_context_limit(model: Any) -> Optional[int]:
    # LLaMA/ToolAlpaca exposes max_position_embeddings.  Keeping this generic
    # makes any future checkpoint with a different config fail safely.
    value = getattr(model.config, "max_position_embeddings", None)
    return int(value) if value else None


class StopAfterFirstAction(StoppingCriteria):
    """Stop a single greedy completion once its first tool decision is complete.

    ToolAlpaca normally keeps generating a synthetic Observation and Response in
    one completion.  That is needed by the full agent harness, but this causal
    experiment only needs ``Thought -> first Action``.  Stopping here makes the
    full 100-example evaluations substantially faster without changing the
    generated Thought or first Action under greedy decoding.
    """

    def __init__(self, tokenizer: Any, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        completion = self.tokenizer.decode(
            input_ids[0, self.prompt_length:], skip_special_tokens=True
        )
        return bool(COMPLETE_ACTION_LINE_RE.search(completion) or RESPONSE_MARK_RE.search(completion))


@torch.inference_mode()
def greedy_completion(
    tokenizer: Any,
    model: Any,
    prompt: str,
    max_new_tokens: int,
    stop_after_first_action: bool = False,
) -> str:
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    available = max_new_tokens
    context_limit = model_context_limit(model)
    if context_limit is not None:
        available = min(available, context_limit - input_ids.shape[1])
    if available <= 0:
        raise ValueError(
            f"Prompt is {input_ids.shape[1]} tokens, leaving no room in the model context "
            f"(limit={context_limit})."
        )
    encoded = {name: value.to(device) for name, value in encoded.items()}
    generation_kwargs: Dict[str, Any] = {}
    if stop_after_first_action:
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [StopAfterFirstAction(tokenizer, input_ids.shape[1])]
        )
    output_ids = model.generate(
        **encoded,
        max_new_tokens=available,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        **generation_kwargs,
    )
    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)


def thought_before_action_or_response(completion: str) -> str:
    """Strip the original model's attempted Action; retain only its Thought."""
    match = ACTION_OR_RESPONSE_RE.search(completion)
    return completion[:match.start()].strip() if match else completion.strip()


def first_action(completion: str) -> Optional[str]:
    match = ACTION_RE.search(completion)
    return match.group(1).strip() if match else None


def summary(records: Iterable[Dict[str, Any]], n_candidates: int) -> Dict[str, Any]:
    records = list(records)
    completed = len(records)
    valid_original = [r for r in records if r["original_regenerated_action"] == r["target_tool"]]
    recovered = [r for r in records if r["unlearned_action"] == r["target_tool"]]
    recovered_valid = [r for r in valid_original if r["unlearned_action"] == r["target_tool"]]
    return {
        "n_candidates": n_candidates,
        "n_completed": completed,
        "n_original_regenerated_target": len(valid_original),
        "n_target_recovered_all_candidates": len(recovered),
        "target_recovery_rate_all_candidates": len(recovered) / completed if completed else None,
        "n_target_recovered_given_regenerated_original_target": len(recovered_valid),
        "target_recovery_rate_given_regenerated_original_target": (
            len(recovered_valid) / len(valid_original) if valid_original else None
        ),
        "unlearned_action_counts": dict(Counter(r["unlearned_action"] or "<no action>" for r in records)),
    }


def output_document(args: argparse.Namespace, candidates: Sequence[Candidate], records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "experiment": "original_thought_to_unlearned_action_setting2",
        "protocol": {
            "setting_1_candidate_rule": (
                "saved original first action equals the gold first forgotten tool, and saved "
                "unlearned trajectory never calls that tool"
            ),
            "setting_2": (
                "original model greedily generates from the initial prompt; its completion is "
                "truncated before ASSISTANT Action/Response and supplied as the Thought prefix "
                "to the unlearned model, which greedily generates the Action"
            ),
            "prompt": "ToolAlpaca test_v1, reconstructed through the benchmark prompt builder",
        },
        "config": {
            "original_trace": str(args.original_trace),
            "unlearned_trace": str(args.unlearned_trace),
            "eval_data": str(args.eval_data),
            "original_model": args.original_model,
            "unlearned_model": args.unlearned_model,
            "original_device": args.original_device,
            "unlearned_device": args.unlearned_device,
            "thought_max_new_tokens": args.thought_max_new_tokens,
            "action_max_new_tokens": args.action_max_new_tokens,
        },
        "candidate_keys": [c.key for c in candidates],
        "summary": summary(records, len(candidates)),
        "records": list(records),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this two-model experiment")
    for device in (args.original_device, args.unlearned_device):
        if not device.startswith("cuda:"):
            raise ValueError(f"Expected a CUDA device, got {device!r}")
        index = int(device.split(":", 1)[1])
        if index >= torch.cuda.device_count():
            raise ValueError(f"{device} is unavailable; only {torch.cuda.device_count()} CUDA devices are visible")
    if args.original_device == args.unlearned_device:
        raise ValueError("Use separate GPUs for original and unlearned models")

    original_trace = read_json(args.original_trace)
    unlearned_trace = read_json(args.unlearned_trace)
    eval_data = read_json(args.eval_data)
    candidates = select_candidates(original_trace, unlearned_trace, index_gold_first_actions(eval_data))
    if args.limit is not None:
        candidates = candidates[:args.limit]
    if not candidates:
        raise RuntimeError("No Setting-1 candidates found; check traces and evaluation data")
    print(f"[select] {len(candidates)} strict Setting-1 candidates", flush=True)

    api_by_name = {api["Name"]: api for api in eval_data}
    prompts = {candidate.key: initial_prompt(api_by_name[candidate.api], candidate.instruction) for candidate in candidates}

    records: List[Dict[str, Any]] = []
    if args.output.exists() and not args.overwrite:
        previous = read_json(args.output)
        previous_keys = previous.get("candidate_keys")
        if previous_keys != [c.key for c in candidates]:
            raise ValueError(
                f"Existing output {args.output} has a different candidate set. "
                "Use --overwrite or choose a different --output path."
            )
        records = previous.get("records", [])
        print(f"[resume] {len(records)}/{len(candidates)} cases already completed", flush=True)
    elif args.overwrite and args.output.exists():
        print(f"[overwrite] replacing {args.output}", flush=True)

    done_keys = {record["key"] for record in records}
    if len(done_keys) != len(records):
        raise ValueError(f"Existing output {args.output} contains duplicate record keys")

    original_tok, original_model = load_model_and_tokenizer(args.original_model, args.original_device)
    unlearned_tok, unlearned_model = load_model_and_tokenizer(args.unlearned_model, args.unlearned_device)

    for number, candidate in enumerate(candidates, start=1):
        if candidate.key in done_keys:
            continue
        prompt = prompts[candidate.key]
        original_completion = greedy_completion(
            original_tok, original_model, prompt, args.thought_max_new_tokens,
            stop_after_first_action=True,
        )
        generated_thought = thought_before_action_or_response(original_completion)
        # This is precisely the Hadi Setting-2 prefix: x + z_f, then start the
        # unlearned model at the action position.  The added space/newline matches
        # PlannerAssistedLLM's existing actor hand-off convention.
        action_prompt = prompt + " " + generated_thought + "\n"
        unlearned_completion = greedy_completion(
            unlearned_tok, unlearned_model, action_prompt, args.action_max_new_tokens,
            stop_after_first_action=True,
        )
        record = {
            "key": candidate.key,
            "api": candidate.api,
            "instruction_id": candidate.instruction_id,
            "instruction": candidate.instruction,
            "target_tool": candidate.target_tool,
            "saved_original_actions": candidate.saved_original_actions,
            "saved_unlearned_actions": candidate.saved_unlearned_actions,
            "original_generated_thought": generated_thought,
            "original_completion": original_completion,
            "original_regenerated_action": first_action(original_completion),
            "unlearned_action_prompt_tokens": len(unlearned_tok(action_prompt)["input_ids"]),
            "unlearned_completion": unlearned_completion,
            "unlearned_action": first_action(unlearned_completion),
        }
        records.append(record)
        done_keys.add(candidate.key)
        document = output_document(args, candidates, records)
        atomic_write_json(args.output, document)
        recovered = record["unlearned_action"] == candidate.target_tool
        print(
            f"[case {number}/{len(candidates)}] {candidate.key}: target={candidate.target_tool}; "
            f"original_regenerated={record['original_regenerated_action']!r}; "
            f"unlearned={record['unlearned_action']!r}; recovered={recovered}",
            flush=True,
        )

    document = output_document(args, candidates, records)
    atomic_write_json(args.output, document)
    result = document["summary"]
    rate = result["target_recovery_rate_given_regenerated_original_target"]
    rate_text = "n/a" if rate is None else f"{rate:.1%}"
    print(
        "[done] "
        f"Setting-1 candidates={result['n_candidates']}; "
        f"original re-generated target={result['n_original_regenerated_target']}; "
        f"Setting-2 recovered target={result['n_target_recovered_given_regenerated_original_target']} "
        f"({rate_text} among valid original re-generations).\n"
        f"[done] wrote {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
