"""Async Ollama embedding client with connection pooling and retry."""
from __future__ import annotations

import asyncio

import httpx
import structlog

log = structlog.get_logger()

MAX_RETRIES = 5
RETRY_BASE = 2.0
THROTTLE = 0.15   # seconds between sequential embed calls


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str) -> None:
        self.url = f"{base_url}/api/embeddings"
        self.model = model
        self._http: httpx.AsyncClient | None = None

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=60.0)
        return self._http

    async def close(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    async def embed(self, text: str) -> list[float] | None:
        client = await self._client()
        for attempt in range(MAX_RETRIES):
            try:
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
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                delay = RETRY_BASE * (2 ** attempt)
                log.warning("ollama.embed_retry", attempt=attempt + 1, error=str(e), delay=delay)
                await asyncio.sleep(delay)
        log.error("ollama.embed_failed", retries=MAX_RETRIES)
        return None

    async def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        results = []
        for text in texts:
            vec = await self.embed(text)
            results.append(vec)
            await asyncio.sleep(THROTTLE)
        return results
