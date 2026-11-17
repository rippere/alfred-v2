"""A dead embedder must not be mistaken for a file with no embeddable content.

`OllamaEmbedder.embed()` used to return None for two unrelated things: "this
chunk is too long, skip it" (permanent, a property of the text) and "the backend
is gone" (transient). The surveyor read the second as the first:

    rows == []            -> "no embeddable content"
    -> delete every existing chunk_id as stale
    -> FileState(md5=current, chunk_ids=[])
    -> md5 now matches, so _compute_diff never revisits the file

which removed the file from search permanently, with no error anywhere. The
retry budget is 2+4+8+16+32 = 62s, so any Ollama outage longer than that while
the daemon runs was enough.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from alfred.config import AlfredConfig
from alfred.core.models import FileState
from alfred.daemons.surveyor import SurveyorDaemon
from alfred.embed.ollama import EmbeddingBackendUnavailable, OllamaEmbedder
from alfred.store.state import StateStore


class _RecordingStore:
    """Minimal LanceDBStore stand-in that records destructive calls."""

    def __init__(self) -> None:
        self.deleted: list[tuple[str, list | None]] = []
        self.upserted: list[list[dict]] = []

    def delete_file(self, rel_path, chunk_ids=None):
        self.deleted.append((rel_path, chunk_ids))

    def upsert_many(self, rows):
        self.upserted.append(rows)


def _make(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    cfg = AlfredConfig(vault_path=vault, data_dir=tmp_path / "data")
    state = StateStore(tmp_path / "state.json")
    state.load()
    store = _RecordingStore()
    return SurveyorDaemon(cfg, state, asyncio.Queue(), store=store), store, vault


def test_embed_raises_when_backend_unreachable(monkeypatch):
    """Retry exhaustion must raise, not return None."""
    emb = OllamaEmbedder("http://localhost:11434", "nomic-embed-text")

    class _DeadClient:
        async def post(self, *a, **kw):
            raise httpx.ConnectError("connection refused")

    async def _client(self):
        return _DeadClient()

    monkeypatch.setattr(OllamaEmbedder, "_client", _client)
    monkeypatch.setattr("alfred.embed.ollama.RETRY_BASE", 0.0)   # don't sleep 62s

    with pytest.raises(EmbeddingBackendUnavailable):
        asyncio.run(emb.embed("some text"))


def test_over_long_chunk_still_returns_none(monkeypatch):
    """The legitimate skip must keep its old behaviour — it is not an outage."""
    emb = OllamaEmbedder("http://localhost:11434", "nomic-embed-text")

    class _TooLongClient:
        async def post(self, url, json=None):
            resp = httpx.Response(
                400, text="input length exceeds context window",
                request=httpx.Request("POST", url),
            )
            raise httpx.HTTPStatusError("400", request=resp.request, response=resp)

    async def _client(self):
        return _TooLongClient()

    monkeypatch.setattr(OllamaEmbedder, "_client", _client)
    assert asyncio.run(emb.embed("x" * 100_000)) is None


def test_outage_does_not_delete_vectors_or_mark_file_indexed(tmp_path, monkeypatch):
    """The regression itself, end to end through _process_diff."""
    daemon, store, vault = _make(tmp_path)

    rel = "decision/keep-me.md"
    (vault / "decision").mkdir()
    (vault / "decision" / "keep-me.md").write_text(
        "---\ntype: decision\n---\n" + ("Real content. " * 40), encoding="utf-8"
    )

    # The file is already indexed with real vectors.
    original = FileState(md5="oldmd5", chunk_ids=["decision/keep-me.md::chunk_00"])
    daemon.state.state.files[rel] = original

    class _DeadEmbedder:
        async def embed(self, text):
            raise EmbeddingBackendUnavailable("Ollama unreachable")

        async def close(self):
            pass

    monkeypatch.setattr(daemon, "_get_embedder", lambda: _DeadEmbedder())
    monkeypatch.setattr(daemon, "_get_bm25", lambda: type("B", (), {"is_fitted": False})())

    asyncio.run(daemon._process_diff({
        "new": [], "changed": [rel], "deleted": [],
        "current": {rel: "newmd5"},
    }))

    assert store.deleted == [], "an outage must not delete the file's vectors"
    assert daemon.state.state.files[rel] is original, (
        "state must be untouched so the next tick retries; overwriting it with "
        "md5=newmd5, chunk_ids=[] is what made the loss permanent"
    )
    assert daemon.state.state.files[rel].chunk_ids == ["decision/keep-me.md::chunk_00"]


def test_deleted_files_are_still_reaped_before_the_outage_check(tmp_path, monkeypatch):
    """Deletions run before any embedding, so an outage must not block them."""
    daemon, store, vault = _make(tmp_path)
    daemon.state.state.files["gone.md"] = FileState(md5="x", chunk_ids=["gone.md::chunk_00"])

    class _DeadEmbedder:
        async def embed(self, text):
            raise EmbeddingBackendUnavailable("Ollama unreachable")

        async def close(self):
            pass

    monkeypatch.setattr(daemon, "_get_embedder", lambda: _DeadEmbedder())
    monkeypatch.setattr(daemon, "_get_bm25", lambda: type("B", (), {"is_fitted": False})())

    asyncio.run(daemon._process_diff({
        "new": [], "changed": [], "deleted": ["gone.md"], "current": {},
    }))

    assert store.deleted == [("gone.md", ["gone.md::chunk_00"])]
    assert "gone.md" not in daemon.state.state.files
