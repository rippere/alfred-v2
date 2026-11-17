"""Regression test for audit:alfred-v2:performance:chunk-text-reparse-no-cache.

_chunk_text() used to call parse_file()+chunk_record() on every invocation with
no caching. It's called once per hit while building the reranker text map
(engine.py) and again per deduped rel_path inside assemble() (context.py), so
a source file contributing 2+ chunks to the rerank set was parsed 3+ times per
query. A per-query cache dict, shared between the two call sites, now bounds
each source file to a single parse per query — behavior (chunk text returned)
must stay identical.
"""
from __future__ import annotations

import pytest

from alfred.config import AlfredConfig
from alfred.query import context as context_module
from alfred.query.engine import QueryEngine
from alfred.store.types import SearchHit


class _FakeStore:
    """Returns 2 chunks from the same file plus 1 chunk from a second file."""

    def search(self, dense_vec, sparse_vec, top_k, include_inbox=False):
        return [
            SearchHit(chunk_id="notes/alpha.md::chunk_00", rel_path="notes/alpha.md", score=0.9, record_type="note", name="alpha"),
            SearchHit(chunk_id="notes/alpha.md::chunk_01", rel_path="notes/alpha.md", score=0.8, record_type="note", name="alpha"),
            SearchHit(chunk_id="notes/beta.md::chunk_00", rel_path="notes/beta.md", score=0.7, record_type="note", name="beta"),
        ]

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
    (vault / "notes" / "beta.md").write_text("---\ntype: note\n---\nbeta body\n")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return AlfredConfig(vault_path=vault, data_dir=data_dir)


@pytest.fixture
def engine(cfg, monkeypatch):
    eng = QueryEngine(cfg)
    eng._store = _FakeStore()
    eng._embedder = _FakeEmbedder()
    # Keep the test hermetic: identity rerank instead of loading FlashRank.
    monkeypatch.setattr(
        "alfred.embed.reranker.rerank",
        lambda query, hits, texts, top_n: hits[:top_n],
    )
    return eng


def test_source_file_parsed_once_per_query_despite_multiple_chunk_hits(engine, monkeypatch):
    calls: list[str] = []
    real_parse_file = context_module.parse_file

    def counting_parse_file(vault_path, rel_path):
        calls.append(rel_path)
        return real_parse_file(vault_path, rel_path)

    monkeypatch.setattr(context_module, "parse_file", counting_parse_file)

    result = engine.query("What is Alfred")

    assert calls.count("notes/alpha.md") == 1, (
        f"alpha.md contributes 2 chunk hits but was parsed {calls.count('notes/alpha.md')} "
        "times — expected exactly once (rerank text map + assemble() must share a cache)"
    )
    assert calls.count("notes/beta.md") == 1
    assert result.context, "query produced no context — pipeline broke"
