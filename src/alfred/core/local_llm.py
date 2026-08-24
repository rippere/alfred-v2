"""Local LLM backend (Ollama) — the single completion path for every daemon.

This replaces the former Anthropic → OpenRouter → Ollama chain. That chain
failed in the worst possible way: Anthropic returned 400 ("credit balance is
too low"), OpenRouter returned 404 (deprecated model slug), and every caller
caught the exception and returned None. Callers then read None as "nothing to
classify" rather than "the backend is gone", so the curator stalled its whole
inbox while every request still reported success.

Hence the one rule this module exists to enforce:

    a backend that cannot be reached raises LocalLLMUnavailable.
    It never returns an empty result that a caller can mistake for an answer.

`alfred.core.anthropic_client` is retained only for callers that still want a
cloud path explicitly; nothing in the daemon fleet uses it.
"""
from __future__ import annotations

import json as _json

import httpx


class LocalLLMUnavailable(RuntimeError):
    """The local backend could not be reached, or returned an unusable response.

    Callers MUST treat this as "the work is undone", not "the work produced
    nothing". Anything that consumes a completion (classify, distill,
    consolidate, summarise) has to leave its input in place and retry later.
    """


def complete(
    system: str,
    user: str,
    *,
    base_url: str,
    model: str,
    json_mode: bool = False,
    max_tokens: int = 2048,
    timeout: float = 180.0,
) -> str:
    """Run one completion against Ollama and return the assistant's text.

    json_mode asks Ollama to constrain decoding to valid JSON, which is what
    the classifier paths want — it removes the need to strip ``` fences off a
    model's prose and then hope json.loads survives it.

    Raises LocalLLMUnavailable on any transport error, any non-2xx status, or a
    malformed response body. Never returns "" to signal failure.
    """
    payload: dict = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"num_predict": max_tokens},
    }
    if json_mode:
        payload["format"] = "json"

    try:
        resp = httpx.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise LocalLLMUnavailable(
            f"Ollama at {base_url} returned {e.response.status_code} for model {model!r}"
        ) from e
    except httpx.HTTPError as e:
        # Connection refused is the common one: ollama-game-guard stops the
        # service while a game is running. That is a pause, not a failure to
        # classify, and callers must be able to tell the difference.
        raise LocalLLMUnavailable(f"Ollama at {base_url} unreachable: {e}") from e

    try:
        content = resp.json()["message"]["content"]
    except (ValueError, KeyError, TypeError) as e:
        raise LocalLLMUnavailable(
            f"Ollama at {base_url} returned an unparseable response: {e}"
        ) from e

    return content or ""


def complete_json(
    system: str,
    user: str,
    *,
    base_url: str,
    model: str,
    max_tokens: int = 1024,
    timeout: float = 180.0,
) -> dict:
    """complete() in JSON mode, parsed to a dict.

    Returns {} when the model emitted syntactically valid but non-object JSON,
    or JSON that will not parse. That is a genuine "the model had nothing
    useful to say" and is distinct from LocalLLMUnavailable, which still
    propagates — the caller needs those two outcomes separated.
    """
    raw = complete(
        system, user,
        base_url=base_url, model=model,
        json_mode=True, max_tokens=max_tokens, timeout=timeout,
    ).strip()

    # Belt and braces: format=json should make fences impossible, but a model
    # that ignores it should degrade to "no classification", never to a crash.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        parsed = _json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
