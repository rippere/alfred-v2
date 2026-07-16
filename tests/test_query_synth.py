"""Coverage for alfred.query.synth.synthesize(): backend fallback chain
(Anthropic -> OpenRouter -> Ollama). Mocking strategy mirrors
test_query_engine.py — monkeypatch the module's private backend callables
directly rather than mocking the underlying SDK/httpx clients, since the
chain's branching logic (env-gated selection + fallback-on-exception) is
what we're covering, not model output quality."""
from __future__ import annotations

import pytest

from alfred.query import synth


@pytest.fixture(autouse=True)
def _clear_api_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def _silence_warn(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(synth, "_warn", lambda msg: calls.append(msg))
    return calls


def test_no_keys_set_goes_straight_to_ollama(monkeypatch):
    monkeypatch.setattr(synth, "_ollama", lambda *a, **kw: "ollama answer")

    answer, backend, model = synth.synthesize(
        "q", "ctx", "claude-x", "or-model", "http://localhost:11434", "llama3",
    )

    assert (answer, backend, model) == ("ollama answer", "Ollama (local)", "llama3")


def test_anthropic_key_present_and_succeeds_short_circuits_chain(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(synth, "_anthropic", lambda *a, **kw: "anthropic answer")

    def _boom_ollama(*a, **kw):
        raise AssertionError("ollama must not be called when anthropic succeeds")
    monkeypatch.setattr(synth, "_ollama", _boom_ollama)

    answer, backend, model = synth.synthesize(
        "q", "ctx", "claude-x", "or-model", "http://localhost:11434", "llama3",
    )

    assert (answer, backend, model) == ("anthropic answer", "Anthropic", "claude-x")


def test_anthropic_failure_falls_back_to_ollama_when_no_openrouter_key(monkeypatch, _silence_warn):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    def _boom(*a, **kw):
        raise RuntimeError("anthropic down")
    monkeypatch.setattr(synth, "_anthropic", _boom)
    monkeypatch.setattr(synth, "_ollama", lambda *a, **kw: "ollama answer")

    answer, backend, model = synth.synthesize(
        "q", "ctx", "claude-x", "or-model", "http://localhost:11434", "llama3",
    )

    assert (answer, backend, model) == ("ollama answer", "Ollama (local)", "llama3")
    assert any("Anthropic" in msg for msg in _silence_warn)


def test_openrouter_key_present_and_succeeds_after_anthropic_absent(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    monkeypatch.setattr(synth, "_openrouter", lambda *a, **kw: "openrouter answer")

    answer, backend, model = synth.synthesize(
        "q", "ctx", "claude-x", "or-model", "http://localhost:11434", "llama3",
    )

    assert (answer, backend, model) == ("openrouter answer", "OpenRouter", "or-model")


def test_openrouter_402_failure_warns_credits_and_falls_back_to_ollama(monkeypatch, _silence_warn):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")

    def _boom(*a, **kw):
        raise RuntimeError("402 Payment Required: insufficient credits")
    monkeypatch.setattr(synth, "_openrouter", _boom)
    monkeypatch.setattr(synth, "_ollama", lambda *a, **kw: "ollama answer")

    answer, backend, model = synth.synthesize(
        "q", "ctx", "claude-x", "or-model", "http://localhost:11434", "llama3",
    )

    assert (answer, backend, model) == ("ollama answer", "Ollama (local)", "llama3")
    assert any("insufficient credits" in msg for msg in _silence_warn)


def test_openrouter_non_credit_failure_warns_with_exception_class(monkeypatch, _silence_warn):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")

    def _boom(*a, **kw):
        raise ValueError("some other failure")
    monkeypatch.setattr(synth, "_openrouter", _boom)
    monkeypatch.setattr(synth, "_ollama", lambda *a, **kw: "ollama answer")

    synth.synthesize(
        "q", "ctx", "claude-x", "or-model", "http://localhost:11434", "llama3",
    )

    assert any("ValueError" in msg for msg in _silence_warn)


def test_both_keys_present_prefers_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    monkeypatch.setattr(synth, "_anthropic", lambda *a, **kw: "anthropic answer")

    def _boom_openrouter(*a, **kw):
        raise AssertionError("openrouter must not be tried when anthropic succeeds")
    monkeypatch.setattr(synth, "_openrouter", _boom_openrouter)

    answer, backend, model = synth.synthesize(
        "q", "ctx", "claude-x", "or-model", "http://localhost:11434", "llama3",
    )

    assert (answer, backend, model) == ("anthropic answer", "Anthropic", "claude-x")


def test_preamble_is_prepended_to_system_prompt(monkeypatch):
    captured = {}

    def _capture(query, context, system, base_url, model):
        captured["system"] = system
        return "answer"
    monkeypatch.setattr(synth, "_ollama", _capture)

    synth.synthesize(
        "q", "ctx", "claude-x", "or-model", "http://localhost:11434", "llama3",
        preamble="CUSTOM PREAMBLE",
    )

    assert captured["system"].startswith("CUSTOM PREAMBLE")
    assert synth.SYSTEM_PROMPT in captured["system"]


def test_no_preamble_uses_bare_system_prompt(monkeypatch):
    captured = {}

    def _capture(query, context, system, base_url, model):
        captured["system"] = system
        return "answer"
    monkeypatch.setattr(synth, "_ollama", _capture)

    synth.synthesize(
        "q", "ctx", "claude-x", "or-model", "http://localhost:11434", "llama3",
    )

    assert captured["system"] == synth.SYSTEM_PROMPT
