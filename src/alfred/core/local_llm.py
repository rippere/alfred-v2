"""Local LLM backend — the single completion path for every daemon.

Two wire formats sit behind it, chosen by the config's `llm.api`: Ollama's
native /api/chat (the default, and what every call used before) and an
OpenAI-compatible /chat/completions (the DGX Spark's vLLM). Callers pass
`**cfg.llm` and never branch on which one answered.

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
import re

import httpx
import structlog

from alfred.config import LocalOnlyViolation, spark_env
from alfred.core.failures import record_failure
from alfred.core.ollama_guard import ModelCheckFailed, check_model_runs_here

log = structlog.get_logger()


class LocalLLMUnavailable(RuntimeError):
    """The local backend could not be reached, or returned an unusable response.

    Callers MUST treat this as "the work is undone", not "the work produced
    nothing". Anything that consumes a completion (classify, distill,
    consolidate, summarise) has to leave its input in place and retry later.
    """


class LocalLLMBadRequest(LocalLLMUnavailable):
    """The OpenAI-compatible server answered 400 for a reason other than size.

    A rejected response_format or json_schema, an unsupported
    chat_template_kwargs, a malformed request: the setup is at fault, not this
    input, so it would fail for every input alike. It IS a LocalLLMUnavailable,
    so every caller treats it as "retry later": nothing is marked processed,
    stamped or dropped. It is also logged at error level and counted in
    error_counts (local_llm.bad_request) when raised, so a fleet stalled on it
    is loud. Before this, every 400 was LocalLLMRequestTooLarge, and a setup
    mistake would have silently skipped the whole inbox.
    """


class LocalLLMRemoteModel(LocalLLMUnavailable, LocalOnlyViolation):
    """A local-only call named a model the local Ollama would run remotely.

    Nothing was sent. It is a LocalLLMUnavailable, so callers leave the work
    undone and retry later, and a LocalOnlyViolation, so code that asks "is
    this the carve-out refusing?" can tell. Logged at error level and counted
    (local_llm.remote_model_refused) every time, so a stalled vault is loud.
    """


class LocalLLMRequestTooLarge(RuntimeError):
    """The backend refused or cut this request because of its size.

    HTTP 400 from the OpenAI-compatible path when the server says the prompt
    plus max_tokens is over the context window, which is what vLLM answers
    (Ollama silently truncated instead); the server's message is kept. Other
    400s are LocalLLMBadRequest. finish_reason "length" is the subclass
    LocalLLMOutputTruncated.

    Deliberately NOT a LocalLLMUnavailable. Sending the same request again gets
    the same answer, so a caller that deferred on it would defer forever.
    Callers skip the item (or shrink it and try once more) and log it.
    """


class LocalLLMOutputTruncated(LocalLLMRequestTooLarge):
    """finish_reason "length": the answer ran out of max_tokens and is cut off.

    A LocalLLMRequestTooLarge, so callers that skip on that keep doing so. A
    caller whose answer is short and bounded (the curator's classification)
    can tell it apart: a sampled answer that ran long once need not run long
    again, so it retries later instead of skipping the input for good.
    """


# What vLLM (and OpenAI) say when prompt + max_tokens is over the window:
#   "This model's maximum context length is 32768 tokens. However, you requested ..."
#   "'max_tokens' ... is too large: ... This model's maximum context length is ..."
#   "The prompt (40014 tokens) is longer than the model's context length ..."
#   "Input prompt (N tokens) is too long and exceeds limit of M" / max_model_len
#   code "context_length_exceeded"
# Deliberately narrow: a schema complaint about a `maxLength` keyword must not
# read as "too large", or it would skip every input for good.
_CONTEXT_LENGTH_RE = re.compile(
    r"maximum context length|context[ _]length|max_model_len"
    r"|\b(?:prompt|input)\b[^.]{0,60}\btoo long",
    re.IGNORECASE,
)


def _is_context_length_error(e) -> bool:
    return bool(_CONTEXT_LENGTH_RE.search(f"{getattr(e, 'message', '')} {getattr(e, 'body', '')}"))


# Qwen3 reasons inside <think> and vLLM has no reasoning parser configured, so
# any reasoning lands in the content. An unclosed tag means it ran out mid-way.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def complete(
    system: str,
    user: str,
    *,
    base_url: str,
    model: str,
    api: str = "ollama",
    api_key_env: str | None = None,
    json_mode: bool = False,
    schema: dict | None = None,
    max_tokens: int = 2048,
    timeout: float = 180.0,
    local_only: bool = False,
) -> str:
    """Run one completion and return the assistant's text.

    json_mode asks the backend to constrain decoding to valid JSON, which is
    what the classifier paths want — it removes the need to strip ``` fences
    off a model's prose and then hope json.loads survives it. schema narrows
    that to an exact shape on the OpenAI path; Ollama keeps format=json, so its
    requests are byte-for-byte what they were before the flag existed.

    An empty system prompt sends the user message alone, as the consolidator's
    old /api/generate calls did.

    local_only (a local-only vault's cfg.llm sets it) first asks the local
    Ollama whether the model runs there; see alfred.core.ollama_guard.

    Raises LocalLLMUnavailable on any transport error, any non-2xx status
    (bar the OpenAI path's context-length 400), or a malformed response body;
    LocalLLMBadRequest (a LocalLLMUnavailable) on any other OpenAI-path 400;
    LocalLLMRemoteModel (a LocalLLMUnavailable) when local_only and the model
    would run remotely; and LocalLLMRequestTooLarge / LocalLLMOutputTruncated
    as described there. Never returns "" to signal failure.
    """
    messages = [{"role": "user", "content": user}]
    if system:
        messages.insert(0, {"role": "system", "content": system})

    if local_only:
        if api != "ollama":
            raise LocalOnlyViolation(f"a local-only call asked for llm api {api!r}")
        _ensure_model_runs_here(base_url, model)

    if api == "openai":
        return _complete_openai(
            messages,
            base_url=base_url, model=model, api_key_env=api_key_env,
            json_mode=json_mode, schema=schema, max_tokens=max_tokens, timeout=timeout,
        )
    if api != "ollama":
        raise ValueError(f"unknown llm api {api!r}")

    payload: dict = {
        "model": model,
        "stream": False,
        # Qwen3.x is a thinking model; suppress reasoning tokens so structured
        # extraction stays clean and fast. Harmless no-op for non-thinking models.
        "think": False,
        "messages": messages,
        "options": {"num_predict": max_tokens},
    }
    if json_mode:
        payload["format"] = "json"

    try:
        # trust_env=False: Ollama is on loopback, and an HTTP(S)_PROXY or
        # ALL_PROXY in the environment must not carry vault text elsewhere.
        resp = httpx.post(f"{base_url}/api/chat", json=payload, timeout=timeout, trust_env=False)
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


def _ensure_model_runs_here(base_url: str, model: str) -> None:
    """Refuse, before anything is sent, a model the local Ollama runs remotely."""
    try:
        check_model_runs_here(base_url, model)
    except LocalOnlyViolation as e:
        log.error("local_llm.remote_model_refused", base_url=base_url, model=model, error=str(e))
        record_failure("local_llm.remote_model_refused")
        raise LocalLLMRemoteModel(str(e)) from e
    except ModelCheckFailed as e:
        raise LocalLLMUnavailable(str(e)) from e


def _openai_client(base_url: str, api_key: str, timeout: float):
    """One OpenAI SDK client per call, closed after it. The seam tests replace."""
    import openai  # only the OpenAI path needs the SDK

    # Contract §7: every call has a timeout, and at most one retry.
    return openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=1)


def _complete_openai(
    messages: list[dict],
    *,
    base_url: str,
    model: str,
    api_key_env: str | None,
    json_mode: bool,
    schema: dict | None,
    max_tokens: int,
    timeout: float,
) -> str:
    """complete() against an OpenAI-compatible server (contract §2-4, §7)."""
    import openai

    # Read per call, so a rotated key needs no restart. "dummy" is what an
    # unauthenticated vLLM accepts; once it has a key, a missing one is a 401.
    api_key = (spark_env(api_key_env) if api_key_env else None) or "dummy"
    request: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        # §3: without this Qwen3 spends the budget reasoning in the content.
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if schema is not None:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "reply", "schema": schema},
        }
    elif json_mode:
        request["response_format"] = {"type": "json_object"}

    try:
        with _openai_client(base_url, api_key, timeout) as client:
            resp = client.chat.completions.create(**request)
    except openai.BadRequestError as e:
        if _is_context_length_error(e):
            raise LocalLLMRequestTooLarge(
                f"{base_url} rejected the request for model {model!r} (400): {e.message}"
            ) from e
        # Not about size, so not this input's fault: retry later, loudly.
        log.error(
            "local_llm.bad_request",
            base_url=base_url,
            model=model,
            response_format=(request.get("response_format") or {}).get("type"),
            error=e.message,
        )
        record_failure("local_llm.bad_request")
        raise LocalLLMBadRequest(
            f"{base_url} rejected the request for model {model!r} (400, not a size "
            f"error; retrying later): {e.message}"
        ) from e
    except openai.APIStatusError as e:
        # 401/403 (key missing or wrong), 404 (model or path), 429, 5xx: the
        # server or its setup is at fault, not this request. Defer and retry.
        raise LocalLLMUnavailable(
            f"{base_url} returned {e.status_code} for model {model!r}"
        ) from e
    except openai.APIConnectionError as e:  # includes APITimeoutError
        raise LocalLLMUnavailable(f"{base_url} unreachable: {e}") from e
    except openai.OpenAIError as e:
        raise LocalLLMUnavailable(f"{base_url} returned an unusable response: {e}") from e

    try:
        choice = resp.choices[0]
        finish_reason = choice.finish_reason
        content = choice.message.content or ""
    except (IndexError, AttributeError, TypeError) as e:
        raise LocalLLMUnavailable(
            f"{base_url} returned an unparseable response: {e}"
        ) from e

    if finish_reason == "length":
        raise LocalLLMOutputTruncated(
            f"{base_url}: the answer hit max_tokens={max_tokens} for model {model!r} and is cut off"
        )
    if finish_reason != "stop":
        # "abort" is vLLM dropping the request (a restart, say): try later.
        raise LocalLLMUnavailable(f"{base_url}: finish_reason={finish_reason!r} for model {model!r}")

    content = _THINK_RE.sub("", content)
    return content.split("<think>", 1)[0].strip()


def complete_json(
    system: str,
    user: str,
    *,
    base_url: str,
    model: str,
    api: str = "ollama",
    api_key_env: str | None = None,
    schema: dict | None = None,
    max_tokens: int = 1024,
    timeout: float = 180.0,
    local_only: bool = False,
) -> dict:
    """complete() in JSON mode, parsed to a dict.

    Returns {} when the model emitted syntactically valid but non-object JSON,
    or JSON that will not parse. That is a genuine "the model had nothing
    useful to say" and is distinct from LocalLLMUnavailable, which still
    propagates — the caller needs those two outcomes separated.
    """
    raw = complete(
        system, user,
        base_url=base_url, model=model, api=api, api_key_env=api_key_env,
        json_mode=True, schema=schema, max_tokens=max_tokens, timeout=timeout,
        local_only=local_only,
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
