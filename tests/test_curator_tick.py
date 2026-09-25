"""Behavioral coverage for CuratorDaemon's public tick() entry point.

tests/test_curator_dedup.py already covers _process_inbox()'s dedup-key
logic using a frontmatter-declared `type` (so classification short-circuits
the LLM branch entirely). This file covers the two things that leaves
uncovered: the LLM-classification branch itself (via a mocked Anthropic
client) driven through the real tick() -> _process_inbox() -> _ingest_file()
-> _classify() path, and tick()'s exception-handling contract.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from alfred.config import AlfredConfig
from alfred.daemons.curator import CuratorDaemon
from alfred.store.state import StateStore


def _make_daemon(tmp_path: Path) -> tuple[CuratorDaemon, StateStore]:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    (vault_path / "inbox").mkdir()
    cfg = AlfredConfig(vault_path=vault_path, data_dir=tmp_path / "data")
    state = StateStore(tmp_path / "state.json")
    state.load()
    events: asyncio.Queue = asyncio.Queue()
    daemon = CuratorDaemon(cfg, state, events)
    return daemon, state


class _FakeUsage:
    input_tokens = 12
    output_tokens = 6
    cache_read_input_tokens = 0


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.usage = _FakeUsage()
        self.content = [type("Block", (), {"text": json.dumps(payload)})()]


class _FakeMessages:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def create(self, **kwargs):
        return _FakeResponse(self._payload)


class _FakeClient:
    def __init__(self, payload: dict) -> None:
        self.messages = _FakeMessages(payload)


def test_tick_normal_path_classifies_and_ingests_via_mocked_local_llm(tmp_path, monkeypatch):
    """A note with no `type` frontmatter must be routed through the real LLM
    classification branch — mocked local backend only — and land as a
    real vault record via the actual tick() APScheduler entry point."""
    daemon, state_store = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path

    inbox_file = vault_path / "inbox" / "untyped.md"
    inbox_file.write_text("Just a raw dropped note with no frontmatter at all.\n", encoding="utf-8")

    classification = {
        "type": "note",
        "name": "llm-classified-note",
        "status": None,
        "tags": ["misc"],
    }
    monkeypatch.setattr(
        "alfred.daemons.curator.complete_json",
        lambda *a, **kw: classification,
    )

    asyncio.run(daemon.tick())

    assert not inbox_file.exists()
    assert (vault_path / "inbox" / "processed" / "untyped.md").exists()
    assert (vault_path / "note" / "llm-classified-note.md").exists()
    assert len(state_store.state.curator_processed) == 1


def test_tick_defers_whole_batch_when_backend_unavailable(tmp_path, monkeypatch):
    """The F1 regression guard.

    A down backend must leave inbox files exactly where they are. The old code
    flattened this into "classified as nothing" and moved on, which is how the
    inbox stalled silently behind an expired Anthropic credit balance.
    """
    from alfred.core.local_llm import LocalLLMUnavailable

    daemon, state_store = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path

    first = vault_path / "inbox" / "a-untyped.md"
    second = vault_path / "inbox" / "b-untyped.md"
    for f in (first, second):
        f.write_text("Raw dropped note, no frontmatter.\n", encoding="utf-8")

    calls = {"n": 0}

    def _down(*a, **kw):
        calls["n"] += 1
        raise LocalLLMUnavailable("connection refused")

    monkeypatch.setattr("alfred.daemons.curator.complete_json", _down)

    asyncio.run(daemon.tick())   # must not raise

    assert first.exists() and second.exists(), "files must stay in inbox for retry"
    assert not (vault_path / "inbox" / "processed" / "a-untyped.md").exists()
    assert state_store.state.curator_processed == {}
    # Stopped after the first failure rather than retrying a dead backend per file.
    assert calls["n"] == 1


def test_tick_exception_is_caught_and_logged_not_propagated(tmp_path, monkeypatch):
    """tick() must swallow any exception raised inside _process_inbox() and
    log it rather than letting it propagate — this is what makes it safe to
    register directly as an APScheduler job function."""
    daemon, _ = _make_daemon(tmp_path)

    async def _boom() -> None:
        raise RuntimeError("simulated curator inbox failure")

    monkeypatch.setattr(daemon, "_process_inbox", _boom)

    with capture_logs() as logs:
        asyncio.run(daemon.tick())  # must not raise

    errors = [e for e in logs if e.get("log_level") == "error"
              and e.get("event") == "curator.tick_error"]
    assert len(errors) == 1
    assert "simulated curator inbox failure" in errors[0]["error"]


def test_request_too_large_skips_that_file_and_not_the_tick(tmp_path, monkeypatch):
    """Unlike a down backend, an oversized request fails the same way every
    10 s: skip that file (left in inbox/ for a person), keep going with the
    rest, and do not send it again until its content changes."""
    from alfred.core.local_llm import LocalLLMRequestTooLarge

    daemon, state_store = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path
    big = vault_path / "inbox" / "a-too-big.md"
    fine = vault_path / "inbox" / "b-fine.md"
    big.write_text("BIG raw note, no frontmatter.\n", encoding="utf-8")
    fine.write_text("Fine raw note, no frontmatter.\n", encoding="utf-8")
    calls: list[str] = []

    def _complete_json(system, user, **kw):
        calls.append(user)
        if "BIG" in user:
            raise LocalLLMRequestTooLarge("400: over the window")
        return {"type": "note", "name": "fine-note"}

    monkeypatch.setattr("alfred.daemons.curator.complete_json", _complete_json)

    with capture_logs() as logs:
        asyncio.run(daemon.tick())

    assert big.exists(), "the skipped file must stay in inbox/"
    assert not fine.exists() and (vault_path / "note" / "fine-note.md").exists()
    assert len(state_store.state.curator_processed) == 2
    assert [e for e in logs if e.get("event") == "curator.request_too_large"]

    calls.clear()
    asyncio.run(daemon.tick())
    assert calls == [], "the oversized file was sent again"

    big.write_text("BIG raw note, edited.\n", encoding="utf-8")
    asyncio.run(daemon.tick())
    assert len(calls) == 1, "an edited file must be tried again"


def test_classification_request_per_backend(tmp_path, monkeypatch):
    """Flag off: the request Ollama always got (format=json, no schema).
    Flag on: a json_schema whose `type` is the known-type enum."""
    import httpx
    import openai

    from alfred.core import local_llm
    from alfred.core.schema import KNOWN_TYPES

    daemon, _ = _make_daemon(tmp_path)
    inbox = daemon.cfg.vault_path / "inbox"
    (inbox / "a.md").write_text("Raw note one.\n", encoding="utf-8")

    ollama: list[dict] = []

    def _ollama_post(url, json=None, timeout=None):
        ollama.append({"url": url, "json": json})
        return httpx.Response(
            200, json={"message": {"content": '{"type": "note", "name": "one"}'}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", _ollama_post)
    asyncio.run(daemon.tick())

    (request,) = ollama
    assert request["url"] == f"{daemon.cfg.ollama_base_url}/api/chat"
    assert request["json"]["model"] == daemon.cfg.ollama_llm_model
    assert request["json"]["format"] == "json"
    assert request["json"]["options"] == {"num_predict": 256}
    assert request["json"]["think"] is False

    daemon.cfg.llm_api = "openai"
    daemon.cfg.llm_base_url = "http://spark.test:8000/v1"
    daemon.cfg.llm_model = "qwen3-30b"
    monkeypatch.setattr("alfred.config.SPARK_ENV_PATH", tmp_path / "no-spark-env")
    monkeypatch.delenv("SPARK_API_KEY", raising=False)
    spark: list[httpx.Request] = []

    def _spark(request: httpx.Request) -> httpx.Response:
        spark.append(request)
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": "qwen3-30b",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": '{"type": "task", "name": "two"}'}}],
        })

    monkeypatch.setattr(local_llm, "_openai_client", lambda base_url, api_key, timeout: openai.OpenAI(
        base_url=base_url, api_key=api_key, max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(_spark)),
    ))
    (inbox / "b.md").write_text("Raw note two.\n", encoding="utf-8")
    asyncio.run(daemon.tick())

    assert len(ollama) == 1, "flag on, yet Ollama was called"
    (request,) = spark
    body = json.loads(request.content)
    assert body["model"] == "qwen3-30b"
    assert body["max_tokens"] == 256
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    schema = body["response_format"]["json_schema"]["schema"]
    assert body["response_format"]["type"] == "json_schema"
    assert schema["properties"]["type"]["enum"] == sorted(KNOWN_TYPES)
    assert (daemon.cfg.vault_path / "task" / "two.md").exists()
