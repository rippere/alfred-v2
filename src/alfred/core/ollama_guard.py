"""Does this Ollama model run on benderman?

A loopback URL keeps a request on this machine only if the model it names runs
here too. Ollama also serves cloud models: the local server accepts the request
and forwards the prompt to ollama.com. AlfredConfig.check_local_only refuses
cloud model *names*, but an alias can carry any name (`ollama cp
gpt-oss:120b-cloud qwen:local`). /api/show reports such a model's remote_host /
remote_model, so a local-only vault asks before it sends.

A confirmed-local answer is cached for VERIFY_TTL_S per (url, model): one
loopback call per model every few minutes (under 0.4 s for the 27B model). A
refused or failed check is never cached.
"""
from __future__ import annotations

import time

import httpx

from alfred.config import LocalOnlyViolation, is_ollama_cloud_model

VERIFY_TTL_S = 300.0
_REMOTE_FIELDS = ("remote_host", "remote_model")
_verified: dict[tuple[str, str], float] = {}


class ModelCheckFailed(RuntimeError):
    """Ollama could not say where the model runs: down, 404, or a bad reply.

    Nothing was sent. Callers treat it like an unreachable backend: the work
    stays undone and is retried later.
    """


def reset() -> None:
    """Forget every cached answer (tests, and a model swap under a live process)."""
    _verified.clear()


def _key(base_url: str, model: str) -> tuple[str, str]:
    return (base_url.rstrip("/"), model)


def _is_fresh(key: tuple[str, str]) -> bool:
    stamp = _verified.get(key)
    return stamp is not None and time.monotonic() - stamp < VERIFY_TTL_S


def _refuse_cloud_name(base_url: str, model: str) -> None:
    if is_ollama_cloud_model(model):
        raise LocalOnlyViolation(
            f"Ollama model {model!r} at {base_url} is a cloud model (runs on ollama.com); "
            "a local-only vault never sends to it"
        )


def _judge(base_url: str, model: str, body) -> None:
    if not isinstance(body, dict):
        raise ModelCheckFailed(f"Ollama at {base_url} sent an unreadable /api/show reply for {model!r}")
    remote = {f: body[f] for f in _REMOTE_FIELDS if body.get(f)}
    if remote:
        where = ", ".join(f"{k}={v}" for k, v in remote.items())
        raise LocalOnlyViolation(
            f"Ollama model {model!r} at {base_url} runs remotely ({where}); "
            "a local-only vault never sends to it"
        )


def check_model_runs_here(base_url: str, model: str, *, timeout: float = 30.0) -> None:
    """Return only if Ollama at base_url runs `model` itself.

    Raises LocalOnlyViolation for a cloud model (by name, or by what /api/show
    reports) and ModelCheckFailed when Ollama cannot be asked.
    """
    _refuse_cloud_name(base_url, model)
    key = _key(base_url, model)
    if _is_fresh(key):
        return
    try:
        # trust_env=False: a proxy variable must not carry this, or the send
        # after it, anywhere but the loopback address.
        resp = httpx.post(
            f"{key[0]}/api/show", json={"model": model}, timeout=timeout, trust_env=False,
        )
        resp.raise_for_status()
        body = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        raise ModelCheckFailed(f"could not ask Ollama at {base_url} where {model!r} runs: {e}") from e
    _judge(base_url, model, body)
    _verified[key] = time.monotonic()


async def acheck_model_runs_here(client: httpx.AsyncClient, base_url: str, model: str) -> None:
    """check_model_runs_here on the caller's async client (built with trust_env=False)."""
    _refuse_cloud_name(base_url, model)
    key = _key(base_url, model)
    if _is_fresh(key):
        return
    try:
        resp = await client.post(f"{key[0]}/api/show", json={"model": model})
        resp.raise_for_status()
        body = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        raise ModelCheckFailed(f"could not ask Ollama at {base_url} where {model!r} runs: {e}") from e
    _judge(base_url, model, body)
    _verified[key] = time.monotonic()
