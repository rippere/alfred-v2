"""Synthesis backend tests.

Rewritten when the Anthropic → OpenRouter → Ollama chain was removed. The old
suite proved the *ladder* worked (which key wins, what falls back to what).
There is no ladder now, so the thing worth proving changed: a reachable backend
returns an answer, and an unreachable one raises instead of returning "".

That last case is the whole point of the redesign. The chain's failure mode was
returning something plausible while the real backend was gone.
"""
from __future__ import annotations

import httpx
import pytest

from alfred.core.local_llm import LocalLLMUnavailable
from alfred.query import synth


def _ok(content: str):
    def _post(url, json=None, timeout=None):
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": content}},
            request=httpx.Request("POST", url),
        )
    return _post


def _capture(seen: dict):
    def _post(url, json=None, timeout=None):
        seen["system"] = json["messages"][0]["content"]
        seen["user"] = json["messages"][1]["content"]
        seen["model"] = json["model"]
        seen["url"] = url
        return httpx.Response(
            200, json={"message": {"content": "ok"}},
            request=httpx.Request("POST", url),
        )
    return _post


def test_synthesize_returns_answer_and_labels(monkeypatch):
    monkeypatch.setattr(httpx, "post", _ok("the answer"))

    answer, backend, model = synth.synthesize(
        query="q",
        context="ctx",
        ollama_base_url="http://localhost:11434",
        ollama_model="mistral:latest",
    )

    assert answer == "the answer"
    assert backend == "Ollama (local)"
    assert model == "mistral:latest"


def test_unreachable_backend_raises_rather_than_returning_empty(monkeypatch):
    """The regression this redesign exists to prevent.

    Previously a dead backend surfaced as a degraded answer behind a 200, so
    nothing downstream could tell "the vault had nothing" from "nobody answered".
    """
    def _boom(url, json=None, timeout=None):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", _boom)

    with pytest.raises(LocalLLMUnavailable):
        synth.synthesize(
            query="q", context="ctx",
            ollama_base_url="http://localhost:11434",
            ollama_model="mistral:latest",
        )


def test_http_error_status_raises_unavailable(monkeypatch):
    def _post(url, json=None, timeout=None):
        return httpx.Response(500, text="boom", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", _post)

    with pytest.raises(LocalLLMUnavailable):
        synth.synthesize(
            query="q", context="ctx",
            ollama_base_url="http://localhost:11434",
            ollama_model="mistral:latest",
        )


def test_preamble_is_prepended_to_system_prompt(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(httpx, "post", _capture(seen))

    synth.synthesize(
        query="q", context="ctx",
        ollama_base_url="http://localhost:11434", ollama_model="mistral:latest",
        preamble="CUSTOM PREAMBLE",
    )

    assert seen["system"].startswith("CUSTOM PREAMBLE")
    assert synth.SYSTEM_PROMPT in seen["system"]


def test_no_preamble_uses_bare_system_prompt(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(httpx, "post", _capture(seen))

    synth.synthesize(
        query="q", context="ctx",
        ollama_base_url="http://localhost:11434", ollama_model="mistral:latest",
    )

    assert seen["system"] == synth.SYSTEM_PROMPT


def test_query_and_context_both_reach_the_user_message(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(httpx, "post", _capture(seen))

    synth.synthesize(
        query="what did I decide", context="VAULT BODY",
        ollama_base_url="http://localhost:11434", ollama_model="mistral:latest",
    )

    assert "what did I decide" in seen["user"]
    assert "VAULT BODY" in seen["user"]


def test_configured_model_and_base_url_reach_the_request(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(httpx, "post", _capture(seen))

    synth.synthesize(
        query="q", context="ctx",
        ollama_base_url="http://elsewhere:11434", ollama_model="qwen2.5:1.5b",
    )

    assert seen["model"] == "qwen2.5:1.5b"
    assert seen["url"] == "http://elsewhere:11434/api/chat"
