"""Coverage for alfred.embed.reranker.rerank(): passage-building, no-text
passthrough, and top_n slicing. flashrank isn't installed in this
environment (and loading the real cross-encoder would defeat the point of a
deterministic unit test), so we stub `_get_ranker()` and inject a fake
`flashrank` module into sys.modules to satisfy rerank()'s lazy
`from flashrank import RerankRequest` import — mirroring how
test_query_engine.py stubs FlashRank at the reranker call site rather than
loading the real model."""
from __future__ import annotations

import sys
import types

import pytest

from alfred.embed import reranker
from alfred.store.types import SearchHit


class _FakeRerankRequest:
    def __init__(self, query, passages):
        self.query = query
        self.passages = passages


@pytest.fixture(autouse=True)
def _fake_flashrank_module(monkeypatch):
    fake_mod = types.ModuleType("flashrank")
    fake_mod.RerankRequest = _FakeRerankRequest
    monkeypatch.setitem(sys.modules, "flashrank", fake_mod)
    return fake_mod


class _FakeRanker:
    """Reverses passage order and assigns descending scores — deterministic
    and easy to assert on, without pulling in the real cross-encoder."""

    def __init__(self) -> None:
        self.rerank_calls: list = []

    def rerank(self, request):
        self.rerank_calls.append(request)
        n = len(request.passages)
        return [
            {"id": p["id"], "score": float(n - i)}
            for i, p in enumerate(reversed(request.passages))
        ]


def test_rerank_empty_hits_returns_immediately(monkeypatch):
    def _boom():
        raise AssertionError("must not load a ranker for empty hits")
    monkeypatch.setattr(reranker, "_get_ranker", _boom)

    assert reranker.rerank("q", [], {}, top_n=5) == []


def test_rerank_no_hits_have_text_skips_ranker_call_and_passes_through(monkeypatch):
    fake_ranker = _FakeRanker()
    monkeypatch.setattr(reranker, "_get_ranker", lambda: fake_ranker)

    hits = [SearchHit(chunk_id="a::chunk_00", rel_path="notes/a.md", score=0.5)]
    result = reranker.rerank("q", hits, texts={}, top_n=5)

    assert result == hits
    assert fake_ranker.rerank_calls == [], "ranker.rerank must not be invoked with zero passages"


def test_rerank_orders_hits_by_fake_ranker_scores_and_sets_rerank_score(monkeypatch):
    fake_ranker = _FakeRanker()
    monkeypatch.setattr(reranker, "_get_ranker", lambda: fake_ranker)

    hits = [
        SearchHit(chunk_id="a::chunk_00", rel_path="notes/a.md", score=0.1, name="a"),
        SearchHit(chunk_id="b::chunk_00", rel_path="notes/b.md", score=0.2, name="b"),
    ]
    texts = {"a::chunk_00": "text a", "b::chunk_00": "text b"}

    result = reranker.rerank("q", hits, texts, top_n=5)

    # _FakeRanker reverses order, so "b" (added second) should rank first
    assert [h.rel_path for h in result] == ["notes/b.md", "notes/a.md"]
    assert result[0].rerank_score == 2.0
    assert result[1].rerank_score == 1.0


def test_rerank_appends_no_text_hits_after_reranked_ones(monkeypatch):
    fake_ranker = _FakeRanker()
    monkeypatch.setattr(reranker, "_get_ranker", lambda: fake_ranker)

    hits = [
        SearchHit(chunk_id="a::chunk_00", rel_path="notes/a.md", score=0.1, name="a"),
        SearchHit(chunk_id="notxt::chunk_00", rel_path="notes/notext.md", score=0.9, name="notext"),
    ]
    texts = {"a::chunk_00": "text a"}  # no entry for notxt

    result = reranker.rerank("q", hits, texts, top_n=5)

    assert result[-1].rel_path == "notes/notext.md", "text-less hits must be appended, not ranked"
    assert result[0].rel_path == "notes/a.md"


def test_rerank_applies_top_n_slicing(monkeypatch):
    fake_ranker = _FakeRanker()
    monkeypatch.setattr(reranker, "_get_ranker", lambda: fake_ranker)

    hits = [
        SearchHit(chunk_id=f"{i}::chunk_00", rel_path=f"notes/{i}.md", score=0.1, name=str(i))
        for i in range(5)
    ]
    texts = {h.chunk_id: f"text {h.name}" for h in hits}

    result = reranker.rerank("q", hits, texts, top_n=2)

    assert len(result) == 2


def test_rerank_calls_get_ranker_lazily_once_per_call(monkeypatch):
    calls = []

    def _tracked():
        calls.append(True)
        return _FakeRanker()
    monkeypatch.setattr(reranker, "_get_ranker", _tracked)

    hits = [SearchHit(chunk_id="a::chunk_00", rel_path="notes/a.md", score=0.5)]
    reranker.rerank("q", hits, {"a::chunk_00": "text"}, top_n=5)

    assert len(calls) == 1
