"""Unit tests for alfred.bridge.brief — vault query -> LLM brief synthesis.

The QueryEngine and the local LLM backend are both faked: these are unit
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


LLM_URL = "http://127.0.0.1:11434"
LLM_MODEL = "local-test-model"
LLM = {"api": "ollama", "base_url": LLM_URL, "model": LLM_MODEL, "local_only": False}


class _FakeLocalLLM:
    """Stands in for alfred.core.local_llm.complete; records every call."""

    def __init__(self, text):
        self._text = text
        self.calls: list[dict] = []

    def __call__(self, system, user, **kwargs):
        self.calls.append({"system": system, "user": user, **kwargs})
        return self._text


def _no_cloud(monkeypatch):
    """Any cloud client construction fails the test."""
    monkeypatch.setattr(
        "alfred.core.anthropic_client.get_client",
        lambda: (_ for _ in ()).throw(AssertionError("bridge brief must never use a cloud LLM")),
    )


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
    llm = _FakeLocalLLM("should not be used")
    monkeypatch.setattr("alfred.core.local_llm.complete", llm)
    _no_cloud(monkeypatch)

    result = brief.synthesize_entity_brief(engine, ENTITY, llm=LLM)

    assert result is None
    assert llm.calls == []


def test_synthesize_entity_brief_returns_none_when_context_is_whitespace_only(monkeypatch):
    engine = _FakeEngine(_FakeQueryResult(context="   \n  "))
    llm = _FakeLocalLLM("should not be used")
    monkeypatch.setattr("alfred.core.local_llm.complete", llm)
    _no_cloud(monkeypatch)

    result = brief.synthesize_entity_brief(engine, ENTITY, llm=LLM)

    assert result is None
    assert llm.calls == []


def test_synthesize_entity_brief_calls_local_llm_and_builds_brief(monkeypatch):
    fake_source = SimpleNamespace(rel_path="note/meeting-2026-01.md")
    engine = _FakeEngine(
        _FakeQueryResult(context="David works on the WSU project.", sources=[fake_source])
    )
    llm = _FakeLocalLLM("  David is a WSU collaborator focused on the platform migration.\n")
    monkeypatch.setattr("alfred.core.local_llm.complete", llm)
    _no_cloud(monkeypatch)

    result = brief.synthesize_entity_brief(
        engine, ENTITY, top_k=3, llm=LLM, max_tokens=256
    )

    assert result is not None
    assert result.entity_name == "David Szabo-Stuban"
    assert result.entity_rel_path == ENTITY.rel_path
    assert result.text == "David is a WSU collaborator focused on the platform migration."
    assert result.source_paths == ["note/meeting-2026-01.md"]
    assert result.note_hash == brief.compute_brief_hash(ENTITY.rel_path, result.text)

    # Local LLM was called exactly once, with the caller's local settings.
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["api"] == "ollama"
    assert call["base_url"] == LLM_URL
    assert call["model"] == LLM_MODEL
    assert call["local_only"] is False
    assert call["max_tokens"] == 256
    assert call["system"] == brief.SYSTEM_PROMPT
    assert "David works on the WSU project." in call["user"]


def test_synthesize_entity_brief_returns_none_when_llm_returns_blank_text(monkeypatch):
    engine = _FakeEngine(_FakeQueryResult(context="some context"))
    llm = _FakeLocalLLM("   ")
    monkeypatch.setattr("alfred.core.local_llm.complete", llm)
    _no_cloud(monkeypatch)

    result = brief.synthesize_entity_brief(engine, ENTITY, llm=LLM)

    assert result is None
    assert len(llm.calls) == 1


def test_synthesize_entity_brief_uses_default_max_tokens_from_config(monkeypatch):
    from alfred.bridge import config as C

    engine = _FakeEngine(_FakeQueryResult(context="some context"))
    llm = _FakeLocalLLM("A short brief.")
    monkeypatch.setattr("alfred.core.local_llm.complete", llm)
    _no_cloud(monkeypatch)

    brief.synthesize_entity_brief(engine, ENTITY, llm=LLM)

    assert llm.calls[0]["max_tokens"] == C.BRIEF_MAX_TOKENS


def test_synthesize_entity_brief_propagates_local_llm_unavailable(monkeypatch):
    """Backend down is 'work undone', never a silent None or a cloud fallback."""
    from alfred.core.local_llm import LocalLLMUnavailable

    engine = _FakeEngine(_FakeQueryResult(context="some context"))

    def _down(*a, **k):
        raise LocalLLMUnavailable("ollama unreachable")

    monkeypatch.setattr("alfred.core.local_llm.complete", _down)
    _no_cloud(monkeypatch)

    with pytest.raises(LocalLLMUnavailable):
        brief.synthesize_entity_brief(engine, ENTITY, llm=LLM)


def test_synthesize_entity_brief_requires_local_llm_settings():
    """No implicit backend default: omitting llm is a TypeError."""
    engine = _FakeEngine(_FakeQueryResult(context="some context"))
    with pytest.raises(TypeError):
        brief.synthesize_entity_brief(engine, ENTITY)


def test_bridge_package_has_no_cloud_llm_reference():
    """The bridge must not import or name a cloud LLM client or model."""
    import pathlib

    import alfred.bridge as pkg

    for path in pathlib.Path(pkg.__file__).parent.glob("*.py"):
        src = path.read_text()
        assert "import anthropic" not in src and "anthropic_client import" not in src, path
        assert "claude-" not in src, path
