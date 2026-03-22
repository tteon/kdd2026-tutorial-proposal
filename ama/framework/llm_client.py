"""
OpenAI client wrapper for structured completions.

Thin wrapper — no retry logic, no caching, no clever abstraction.
Each call is one OpenAI chat completion with a Pydantic response model.
This keeps each LLM call inspectable in traces.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class CompletionResult(Generic[T]):
    """Wrapper for structured completion results with token usage."""
    parsed: T
    usage: dict[str, int] = field(default_factory=dict)


def get_openai_client() -> OpenAI:
    """Create an OpenAI client from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    return OpenAI(api_key=api_key)


def structured_completion(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
) -> CompletionResult[T]:
    """Call OpenAI with structured output, returning parsed model + token usage.

    Uses OpenAI's `response_format` with `json_schema` for guaranteed structure.
    """
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=response_model,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError(f"OpenAI returned no parsed content for {response_model.__name__}")

    usage: dict[str, int] = {}
    if completion.usage:
        usage = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens,
        }

    return CompletionResult(parsed=parsed, usage=usage)
