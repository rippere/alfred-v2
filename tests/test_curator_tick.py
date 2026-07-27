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
