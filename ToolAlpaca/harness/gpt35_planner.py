"""A remote chat model supplies the Thought slot; the local actor executes it."""
import os
import re

from harness.planner_llm import PlannerAssistedLLM, split_planner_output
from utils import openai_chat_completions


class GPT35PlannerLLM(PlannerAssistedLLM):
    # Kept as a Pydantic field so every instantiated harness records the exact
    # requested/resolved remote model in its trace.  The old G35 condition name
    # is intentionally retained for backwards-compatible output paths.
    planner_model_name: str = os.getenv("REMOTE_PLANNER_MODEL", "gpt-3.5-turbo")

    def _plan(self, prompt, stop, instructed=False):
        completion = openai_chat_completions(
            model=self.planner_model_name,
            messages=[
                {"role": "system", "content": (
                    "You supply the next Thought for a tool-using assistant. "
                    "Read the tool documentation, user request, and actual prior "
                    "observations. Return only one or two sentences describing "
                    "what the assistant should do next and why. You may name "
                    "the appropriate function and relevant parameters. Do not "
                    "emit an Action, Action Input, Observation, or final Response. "
                    "The local assistant will choose and execute the action and "
                    "write the final response. Do not invent tool results."
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0, max_tokens=self.planner_max_new_tokens,
            request_timeout=60,
        )
        raw = completion["choices"][0]["message"]["content"] or ""
        # Normalize optional headings, and enforce the execution boundary even
        # if the remote model ignores the instruction about output formatting.
        normalized = re.sub(r"^\s*(?:ASSISTANT\s+)?Thought\s*:\s*", "", raw,
                            flags=re.IGNORECASE)
        normalized = re.sub(
            r"(?im)^\s*(?:ASSISTANT\s+)?(Action(?: Input)?|Response|Observation)\s*:",
            r"ASSISTANT \1:", normalized,
        )
        guidance, hint, leaked = split_planner_output(normalized)
        observation = re.search(r"ASSISTANT\s+Observation\s*:", guidance, re.I)
        if observation:
            leaked = guidance[observation.start():] + leaked
            guidance = guidance[:observation.start()].strip()
        self._remote_metadata = {
            "requested_model": self.planner_model_name,
            "resolved_model": completion.get("model"),
            "completion_id": completion.get("id"),
            "usage": dict(completion.get("usage") or {}),
        }
        return raw, guidance, hint, leaked

    def _call(self, prompt, stop=None, **kwargs):
        result = super()._call(prompt, stop=stop, **kwargs)
        self.trace[-1]["planner_api"] = self._remote_metadata
        return result

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    _remote_metadata: dict = {}
