"""Retrieval-mode pin for audit quick win #5 (structural #6d).

The LanceDB backend is dense-only: LanceDBStore.search() ignores its sparse
argument, so encoding a BM25 sparse vector on the query hot path was dead work
on every vault_query/vault_search across all 3 MCP surfaces. Quick win #5
removed that call — these tests pin the resolution so it cannot regress.
"""
from __future__ import annotations

import pytest

from alfred.config import AlfredConfig
from alfred.query.engine import QueryEngine, QueryOptions
from alfred.store.bm25 import BM25Store
from alfred.store.types import SearchHit


class _FakeStore:
    """Stands in for LanceDBStore: records search() calls, dense-only."""

    def __init__(self) -> None:
        self.search_calls: list[dict] = []

    def search(self, dense_vec, sparse_vec, top_k, include_inbox=False):
        self.search_calls.append({
            "dense_vec": dense_vec,
            "sparse_vec": sparse_vec,
            "top_k": top_k,
            "include_inbox": include_inbox,
        })
        return [SearchHit(
            chunk_id="notes/alpha.md::chunk_00",
            rel_path="notes/alpha.md",
            score=0.9,
            record_type="note",
            name="alpha",
        )]

    def count(self) -> int:
        return 0  # hopfield refinement early-exits on an empty store


class _FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.1] * 768


@pytest.fixture
def cfg(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "notes").mkdir()
    (vault / "notes" / "alpha.md").write_text("---\ntype: note\n---\nalpha body\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()  # AlfredConfig.load() creates this in production
    return AlfredConfig(vault_path=vault, data_dir=data_dir)


@pytest.fixture
def engine(cfg, monkeypatch):
    assert cfg.vector_store == "lancedb"  # the live backend under test
    eng = QueryEngine(cfg)
    eng._store = _FakeStore()
    eng._embedder = _FakeEmbedder()
    # Keep the test hermetic: identity rerank instead of loading FlashRank.
    monkeypatch.setattr(
        "alfred.embed.reranker.rerank",
        lambda query, hits, texts, top_n: hits[:top_n],
    )
    return eng


def test_query_hot_path_never_encodes_bm25(engine, monkeypatch):
    encode_calls: list[str] = []
    monkeypatch.setattr(
        BM25Store, "encode",
        lambda self, text: encode_calls.append(text) or {},
    )
    get_bm25_calls: list[bool] = []
    real_get_bm25 = engine._get_bm25
    engine._get_bm25 = lambda: get_bm25_calls.append(True) or real_get_bm25()

    result = engine.query("What is Alfred")

    assert encode_calls == [], "BM25Store.encode ran on the dense query hot path"
    assert get_bm25_calls == [], "_get_bm25 was invoked on the dense query hot path"
    assert engine._bm25 is None, "BM25 index was lazily loaded despite dense-only retrieval"
    assert result.hits, "dense retrieval produced no hits — pipeline broke"


def test_query_passes_empty_sparse_vector_to_store(engine):
    engine.query("What is Alfred", QueryOptions(top_k=4))

    calls = engine._store.search_calls
    assert len(calls) == 1, "vector search must run exactly once per query"
    assert calls[0]["sparse_vec"] == {}, (
        "a non-empty sparse vector reached the store — the dead BM25 encode "
        "has been reintroduced on the hot path"
    )
    assert calls[0]["top_k"] == 4


def test_bm25_timing_absent_from_dense_query_elapsed(engine):
    result = engine.query("What is Alfred")
    assert "bm25" not in result.elapsed, (
        "elapsed timings show a bm25 stage on the dense hot path"
    )
    assert "search" in result.elapsed and "embed" in result.elapsed
