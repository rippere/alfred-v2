"""Coverage for alfred.embed.ollama.OllamaEmbedder: retry/backoff logic and
batch sequencing. Mocking strategy: stub the private async `_client()`
accessor to return a fake httpx-shaped client (post/aclose/is_closed) and
stub asyncio.sleep to make retry-loop tests instant — no real network I/O,
consistent with how test_query_engine.py stubs out the store/embedder at
the boundary rather than mocking deep inside httpx. Async coroutines are
driven with asyncio.run(), matching the existing pattern in
tests/test_surveyor.py and tests/test_meta_server.py (no pytest-asyncio
plugin is installed in this project)."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from alfred.embed.ollama import MAX_RETRIES, OllamaEmbedder


class _FakeResponse:
    def __init__(self, json_data=None, status_code=200, text=""):
        self._json_data = json_data
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://fake")
            response = httpx.Response(self.status_code, request=request, text=self.text)
            raise httpx.HTTPStatusError("boom", request=request, response=response)

    def json(self):
        return self._json_data


class _FakeAsyncClient:
    def __init__(self, responses=None, exceptions=None):
        # responses/exceptions: list consumed in order across successive post() calls
        self._responses = list(responses or [])
        self._exceptions = list(exceptions or [])
        self.post_calls: list[dict] = []
        self.is_closed = False

    async def post(self, url, json):
        self.post_calls.append({"url": url, "json": json})
        if self._exceptions:
            exc = self._exceptions.pop(0)
            if exc is not None:
                raise exc
        return self._responses.pop(0)

    async def aclose(self):
        self.is_closed = True


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Retry backoff uses real seconds (RETRY_BASE * 2**attempt) — patch
    asyncio.sleep in the ollama module so retry tests run instantly."""
    async def _instant_sleep(_delay):
        return None
    monkeypatch.setattr("alfred.embed.ollama.asyncio.sleep", _instant_sleep)


def test_embed_success_on_first_attempt_returns_vector(monkeypatch):
    embedder = OllamaEmbedder("http://localhost:11434", "nomic-embed-text")
    fake_client = _FakeAsyncClient(responses=[_FakeResponse({"embedding": [0.1, 0.2, 0.3]})])
    embedder._http = fake_client

    result = asyncio.run(embedder.embed("hello world"))

    assert result == [0.1, 0.2, 0.3]
    assert len(fake_client.post_calls) == 1
    assert fake_client.post_calls[0]["json"] == {"model": "nomic-embed-text", "prompt": "hello world"}


def test_embed_too_long_input_short_circuits_without_retry():
    embedder = OllamaEmbedder("http://localhost:11434", "nomic-embed-text")
    fake_client = _FakeAsyncClient(
        responses=[_FakeResponse(status_code=400, text="input length exceeds maximum")],
    )
    embedder._http = fake_client

    result = asyncio.run(embedder.embed("a very long text"))

    assert result is None
    assert len(fake_client.post_calls) == 1, "must not retry on the too-long error"


def test_embed_retries_on_connect_error_then_succeeds():
    embedder = OllamaEmbedder("http://localhost:11434", "nomic-embed-text")
    connect_err = httpx.ConnectError("refused")
    fake_client = _FakeAsyncClient(
        exceptions=[connect_err, None],
        responses=[_FakeResponse({"embedding": [1.0]})],
    )
    embedder._http = fake_client

    result = asyncio.run(embedder.embed("hi"))

    assert result == [1.0]
    assert len(fake_client.post_calls) == 2


def test_embed_exhausts_retries_and_returns_none():
    embedder = OllamaEmbedder("http://localhost:11434", "nomic-embed-text")
    fake_client = _FakeAsyncClient(
        exceptions=[httpx.TimeoutException("slow")] * MAX_RETRIES,
        responses=[],
    )
    embedder._http = fake_client

    result = asyncio.run(embedder.embed("hi"))

    assert result is None
    assert len(fake_client.post_calls) == MAX_RETRIES


def test_embed_retries_on_generic_http_status_error_then_succeeds():
    embedder = OllamaEmbedder("http://localhost:11434", "nomic-embed-text")
    fake_client = _FakeAsyncClient(
        responses=[
            _FakeResponse(status_code=500, text="internal error"),
            _FakeResponse({"embedding": [2.0]}),
        ],
    )
    embedder._http = fake_client

    result = asyncio.run(embedder.embed("hi"))

    assert result == [2.0]
    assert len(fake_client.post_calls) == 2


def test_embed_batch_calls_embed_sequentially_and_preserves_order():
    embedder = OllamaEmbedder("http://localhost:11434", "nomic-embed-text")
    fake_client = _FakeAsyncClient(responses=[
        _FakeResponse({"embedding": [1.0]}),
        _FakeResponse({"embedding": [2.0]}),
        _FakeResponse({"embedding": [3.0]}),
    ])
    embedder._http = fake_client

    results = asyncio.run(embedder.embed_batch(["a", "b", "c"]))

    assert results == [[1.0], [2.0], [3.0]]
    assert len(fake_client.post_calls) == 3


def test_embed_batch_preserves_none_for_failed_items():
    embedder = OllamaEmbedder("http://localhost:11434", "nomic-embed-text")
    fake_client = _FakeAsyncClient(responses=[
        _FakeResponse({"embedding": [1.0]}),
        _FakeResponse(status_code=400, text="input length exceeds maximum"),
    ])
    embedder._http = fake_client

    results = asyncio.run(embedder.embed_batch(["a", "b"]))

    assert results == [[1.0], None]


def test_close_closes_open_client():
    embedder = OllamaEmbedder("http://localhost:11434", "nomic-embed-text")
    fake_client = _FakeAsyncClient()
    embedder._http = fake_client

    asyncio.run(embedder.close())

    assert fake_client.is_closed is True


def test_close_is_noop_when_no_client_ever_created():
    embedder = OllamaEmbedder("http://localhost:11434", "nomic-embed-text")
    asyncio.run(embedder.close())  # must not raise


def test_close_is_noop_when_client_already_closed():
    embedder = OllamaEmbedder("http://localhost:11434", "nomic-embed-text")
    fake_client = _FakeAsyncClient()
    fake_client.is_closed = True
    embedder._http = fake_client

    asyncio.run(embedder.close())  # must not attempt to re-close


def test_url_and_model_set_from_constructor():
    embedder = OllamaEmbedder("http://localhost:11434", "nomic-embed-text")
    assert embedder.url == "http://localhost:11434/api/embeddings"
    assert embedder.model == "nomic-embed-text"
