"""Async Ollama embedding client with connection pooling and retry."""
from __future__ import annotations

import asyncio

import httpx
import structlog

from alfred.config import LocalOnlyViolation
from alfred.core.failures import record_failure
from alfred.core.ollama_guard import ModelCheckFailed, acheck_model_runs_here

log = structlog.get_logger()

MAX_RETRIES = 5
RETRY_BASE = 2.0
THROTTLE = 0.15   # seconds between sequential embed calls


class EmbeddingBackendUnavailable(RuntimeError):
    """The embedding backend could not be reached.

    Distinct from embed() returning None, which means "skip this one chunk, the
    text is too long" — a permanent property of the input. Callers must treat
    this as "the work is undone" and leave their state untouched, exactly as
    alfred.core.local_llm.LocalLLMUnavailable does for completions.
    """


class EmbeddingModelRefused(EmbeddingBackendUnavailable, LocalOnlyViolation):
    """A local-only vault's embed model runs remotely (an Ollama cloud model).

    Nothing was sent. An EmbeddingBackendUnavailable, so the surveyor leaves
    the file's state untouched and retries later; logged at error and counted.
    """


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, *, local_only: bool = False) -> None:
        self.base_url = base_url
        self.url = f"{base_url}/api/embeddings"
        self.model = model
        # A local-only vault asks Ollama, before it sends, whether the model
        # runs here (alfred.core.ollama_guard).
        self.local_only = local_only
        self._http: httpx.AsyncClient | None = None

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            # trust_env=False: Ollama is on loopback, and a proxy variable in
            # the environment must not carry chunk text anywhere else.
            self._http = httpx.AsyncClient(timeout=60.0, trust_env=False)
        return self._http

    async def close(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    async def embed(self, text: str) -> list[float] | None:
        client = await self._client()
        for attempt in range(MAX_RETRIES):
            try:
                if self.local_only:
                    await acheck_model_runs_here(client, self.base_url, self.model)
                resp = await client.post(self.url, json={"model": self.model, "prompt": text})
                resp.raise_for_status()
                return resp.json()["embedding"]
            except httpx.HTTPStatusError as e:
                detail = e.response.text[:200]
                if "input length exceeds" in detail:
                    log.warning("ollama.embed_skip_too_long", chars=len(text))
                    return None
                delay = RETRY_BASE * (2 ** attempt)
                log.warning("ollama.embed_retry", attempt=attempt + 1, error=str(e), delay=delay)
                await asyncio.sleep(delay)
            except LocalOnlyViolation as e:
                log.error("ollama.embed_remote_model_refused", model=self.model, error=str(e))
                record_failure("ollama.embed_remote_model_refused")
                raise EmbeddingModelRefused(str(e)) from e
            except (httpx.ConnectError, httpx.TimeoutException, ModelCheckFailed) as e:
                delay = RETRY_BASE * (2 ** attempt)
                log.warning("ollama.embed_retry", attempt=attempt + 1, error=str(e), delay=delay)
                await asyncio.sleep(delay)
        log.error("ollama.embed_failed", retries=MAX_RETRIES)
        # Raise rather than return None. None already means "skip this chunk,
        # it is too long" — a legitimate, permanent property of the text. A
        # dead backend is neither, and conflating the two let the surveyor
        # treat an outage as "this file has no embeddable content": it deleted
        # the file's existing vectors as stale, wrote FileState(chunk_ids=[]),
        # and because the md5 then matched, _compute_diff never looked at the
        # file again. Silent, permanent loss of that file from search.
        raise EmbeddingBackendUnavailable(
            f"Ollama embeddings at {self.url} unreachable after {MAX_RETRIES} retries"
        )

    async def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        results = []
        for text in texts:
            vec = await self.embed(text)
            results.append(vec)
            await asyncio.sleep(THROTTLE)
        return results
