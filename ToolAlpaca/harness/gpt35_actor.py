"""OpenAI chat model used as a complete ToolAlpaca ReAct actor.

The local ToolAlpaca checkpoints are continuation models: they receive a prompt
whose last line is ``ASSISTANT Thought:`` and return the continuation.  This
adapter gives a chat model that same transcript and asks it for the continuation
only, so the existing parser/executor and the original GPT-4 process judge can
be used unchanged.
"""
import os
from typing import List, Optional

from langchain.llms.base import LLM

from utils import openai_chat_completions


class GPT35ActorLLM(LLM):
    model_name: str = os.getenv("REMOTE_ACTOR_MODEL", "gpt-3.5-turbo")
    max_new_tokens: int = int(os.getenv("REMOTE_ACTOR_MAX_TOKENS", "256"))
    trace: List[dict] = []

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "remote_react_actor"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        completion = openai_chat_completions(
            model=self.model_name,
            messages=[
                {"role": "system", "content": (
                    "Continue the supplied ToolAlpaca ReAct transcript exactly at "
                    "its final `ASSISTANT Thought:` cursor. Return only the next "
                    "assistant continuation, without repeating the prompt or adding "
                    "a chat preamble. Follow its Thought/Action/Action Input/Response "
                    "format and use only tools documented in the transcript."
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0, max_tokens=self.max_new_tokens, stop=stop,
            request_timeout=90,
        )
        raw = completion["choices"][0]["message"]["content"] or ""
        # Some chat models echo the cursor despite the instruction.  The local
        # continuation model does not, and duplicating it breaks the parser.
        if raw.lstrip().startswith("ASSISTANT Thought:"):
            raw = raw.lstrip()[len("ASSISTANT Thought:"):].lstrip()
        self.trace.append({
            "mode": "remote_actor",
            "actor_prompt_tail": prompt[-400:],
            "actor_output": raw,
            "actor_api": {
                "requested_model": self.model_name,
                "resolved_model": completion.get("model"),
                "completion_id": completion.get("id"),
                "usage": dict(completion.get("usage") or {}),
            },
        })
        return raw
