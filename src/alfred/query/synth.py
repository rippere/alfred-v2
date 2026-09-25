"""Synthesis backend: the configured local LLM (cfg.llm — Ollama by default).

This used to be a three-link chain (Anthropic → OpenRouter → Ollama). It was
removed rather than repaired: the chain degraded silently. Anthropic 400'd on
an exhausted credit balance and OpenRouter 404'd on a slug xAI had deprecated,
and because `_warn` printed only `e.__class__.__name__` the operator saw
"BadRequestError" instead of the message naming the cause. Every request still
returned 200 while answers quietly dropped to the local model.

One declared backend, and a hard failure when it is gone, beats a fallback
ladder nobody can see sliding down.
"""
from __future__ import annotations

from urllib.parse import urlparse

from alfred.core.local_llm import LocalLLMRequestTooLarge, LocalLLMUnavailable, complete

SYSTEM_PROMPT = (
    "You are Alfred, a personal knowledge assistant with access to a private vault "
    "of notes, projects, decisions, assumptions, conversations, and learnings. "
    "Answer the query using ONLY the provided vault context. Be specific and "
    "cite which vault documents support each point. If context is insufficient, say so."
)

__all__ = ["SYSTEM_PROMPT", "LocalLLMRequestTooLarge", "LocalLLMUnavailable", "synthesize"]


def synthesize(
    query: str,
    context: str,
    base_url: str,
    model: str,
    preamble: str = "",
    *,
    api: str = "ollama",
    api_key_env: str | None = None,
) -> tuple[str, str, str]:
    """Returns (answer, backend_label, model_label). Pass **cfg.llm.

    Raises LocalLLMUnavailable if the backend cannot be reached, and
    LocalLLMRequestTooLarge if it refused the request's size — callers must
    surface either to the user rather than presenting an empty answer as
    though the vault had nothing to say.
    """
    system = (preamble + "\n\n" + SYSTEM_PROMPT).strip() if preamble else SYSTEM_PROMPT
    answer = complete(
        system,
        f"Query: {query}\n\nVault context:\n\n{context}",
        base_url=base_url,
        model=model,
        api=api,
        api_key_env=api_key_env,
        max_tokens=2048,
    )
    if api == "ollama":
        return answer, "Ollama (local)", model
    return answer, f"OpenAI-compatible at {urlparse(base_url).netloc}", model
