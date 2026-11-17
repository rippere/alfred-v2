"""Synthesis backend: local Ollama.

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

from alfred.core.local_llm import LocalLLMUnavailable, complete

SYSTEM_PROMPT = (
    "You are Alfred, a personal knowledge assistant with access to a private vault "
    "of notes, projects, decisions, assumptions, conversations, and learnings. "
    "Answer the query using ONLY the provided vault context. Be specific and "
    "cite which vault documents support each point. If context is insufficient, say so."
)

__all__ = ["SYSTEM_PROMPT", "LocalLLMUnavailable", "synthesize"]


def synthesize(
    query: str,
    context: str,
    ollama_base_url: str,
    ollama_model: str,
    preamble: str = "",
) -> tuple[str, str, str]:
    """Returns (answer, backend_label, model_label).

    Raises LocalLLMUnavailable if Ollama cannot be reached — callers must
    surface that to the user rather than presenting an empty answer as though
    the vault had nothing to say.
    """
    system = (preamble + "\n\n" + SYSTEM_PROMPT).strip() if preamble else SYSTEM_PROMPT
    answer = complete(
        system,
        f"Query: {query}\n\nVault context:\n\n{context}",
        base_url=ollama_base_url,
        model=ollama_model,
        max_tokens=2048,
    )
    return answer, "Ollama (local)", ollama_model
