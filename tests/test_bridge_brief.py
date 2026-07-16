"""Unit tests for alfred.bridge.brief — vault query -> LLM brief synthesis.

The QueryEngine and the Anthropic client are both faked: these are unit
tests of the bridge's own glue code (query construction, empty-context
short-circuit, hash computation), not of retrieval or the model itself. No
real network/API call is made.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from alfred.bridge import brief
from alfred.bridge.resolve import EntityMatch


ENTITY = EntityMatch(
    rel_path="person/David Szabo-Stuban.md",
    entity_type="person",
    name="David Szabo-Stuban",
    frontmatter={"email": "dstuban@example.com"},
)


class _FakeQueryResult:
    def __init__(self, *, context="", sources=None, hits=None):
        self.context = context
        self.sources = sources or []
        self.hits = hits or []


class _FakeEngine:
    """Records .query() calls and returns a canned QueryResult."""

    def __init__(self, result):
        self.result = result
        self.calls: list[dict] = []

    def query(self, text, opts):
        self.calls.append({"text": text, "opts": opts})
        return self.result


class _FakeMessages:
    def __init__(self, text):
        self._text = text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text=self._text)])


class _FakeClient:
    def __init__(self, text):
        self.messages = _FakeMessages(text)


def test_build_query_text_mentions_entity_name():
    text = brief.build_query_text(ENTITY)
    assert "David Szabo-Stuban" in text


def test_query_vault_context_calls_engine_with_query_options_and_no_synthesis():
    from alfred.query.engine import QueryOptions

    engine = _FakeEngine(_FakeQueryResult(context="some context"))

    result = brief.query_vault_context(engine, ENTITY, top_k=4)

    assert result.context == "some context"
    assert len(engine.calls) == 1
    call = engine.calls[0]
    assert ENTITY.name in call["text"]
    assert isinstance(call["opts"], QueryOptions)
    assert call["opts"].top_k == 4
    assert call["opts"].include_synthesis is False


def test_compute_brief_hash_is_deterministic_and_content_sensitive():
    h1 = brief.compute_brief_hash("person/David Szabo-Stuban.md", "Some brief text.")
    h2 = brief.compute_brief_hash("person/David Szabo-Stuban.md", "Some brief text.")
    h3 = brief.compute_brief_hash("person/David Szabo-Stuban.md", "Different brief text.")
    h4 = brief.compute_brief_hash("person/Someone Else.md", "Some brief text.")

    assert h1 == h2
    assert h1 != h3
    assert h1 != h4
    assert isinstance(h1, str) and len(h1) > 0


def test_synthesize_entity_brief_returns_none_when_no_vault_context(monkeypatch):
    engine = _FakeEngine(_FakeQueryResult(context=""))

    def _boom():
        raise AssertionError("LLM should never be called when there is no vault context")

    monkeypatch.setattr("alfred.core.anthropic_client.get_client", _boom)

    result = brief.synthesize_entity_brief(engine, ENTITY)

    assert result is None


def test_synthesize_entity_brief_returns_none_when_context_is_whitespace_only(monkeypatch):
    engine = _FakeEngine(_FakeQueryResult(context="   \n  "))
    monkeypatch.setattr(
        "alfred.core.anthropic_client.get_client",
        lambda: (_ for _ in ()).throw(AssertionError("LLM should not be called")),
    )

    result = brief.synthesize_entity_brief(engine, ENTITY)

    assert result is None


def test_synthesize_entity_brief_calls_llm_and_builds_brief(monkeypatch):
    fake_source = SimpleNamespace(rel_path="note/meeting-2026-01.md")
    engine = _FakeEngine(
        _FakeQueryResult(context="David works on the WSU project.", sources=[fake_source])
    )
    fake_client = _FakeClient("David is a WSU collaborator focused on the platform migration.")
    monkeypatch.setattr("alfred.core.anthropic_client.get_client", lambda: fake_client)

    result = brief.synthesize_entity_brief(engine, ENTITY, top_k=3, model="test-model", max_tokens=256)

    assert result is not None
    assert result.entity_name == "David Szabo-Stuban"
    assert result.entity_rel_path == ENTITY.rel_path
    assert result.text == "David is a WSU collaborator focused on the platform migration."
    assert result.source_paths == ["note/meeting-2026-01.md"]
    assert result.note_hash == brief.compute_brief_hash(ENTITY.rel_path, result.text)

    # LLM was called exactly once, with the expected model/max_tokens/context.
    assert len(fake_client.messages.calls) == 1
    call = fake_client.messages.calls[0]
    assert call["model"] == "test-model"
    assert call["max_tokens"] == 256
    assert "David works on the WSU project." in call["messages"][0]["content"]


def test_synthesize_entity_brief_returns_none_when_llm_returns_blank_text(monkeypatch):
    engine = _FakeEngine(_FakeQueryResult(context="some context"))
    fake_client = _FakeClient("   ")
    monkeypatch.setattr("alfred.core.anthropic_client.get_client", lambda: fake_client)

    result = brief.synthesize_entity_brief(engine, ENTITY)

    assert result is None
    assert len(fake_client.messages.calls) == 1


def test_synthesize_entity_brief_uses_default_model_and_max_tokens_from_config(monkeypatch):
    from alfred.bridge import config as C

    engine = _FakeEngine(_FakeQueryResult(context="some context"))
    fake_client = _FakeClient("A short brief.")
    monkeypatch.setattr("alfred.core.anthropic_client.get_client", lambda: fake_client)

    brief.synthesize_entity_brief(engine, ENTITY)

    call = fake_client.messages.calls[0]
    assert call["model"] == C.BRIEF_SYNTHESIS_MODEL
    assert call["max_tokens"] == C.BRIEF_MAX_TOKENS
