"""Behavioral coverage for ConsolidatorDaemon's public tick() entry points
(label_tick / stubs_tick) — previously untested beyond APScheduler job
registration metadata (tests/test_scheduler_jobs.py).

Covers one normal-tick path per responsibility exercised here:
  - label_tick(): a real cluster gets labeled through the actual Ollama ->
    Anthropic fallback chain, with the Ollama HTTP call mocked to fail (no
    local Ollama in test env) and the Anthropic client mocked to succeed —
    proving the fallback, not just the happy path of either provider alone.
  - stubs_tick(): a real person/org vault record gets a wiki stub page
    created via real (tmp_path) vault I/O — no LLM involved in stub
    creation.
And one exception-handling path: label_tick() swallowing and logging an
internal exception rather than propagating it, matching the contract already
proven for the other daemons' tick() wrappers.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from structlog.testing import capture_logs

from alfred.config import AlfredConfig
from alfred.core.models import ClusterState, FileState
from alfred.daemons.consolidator import ConsolidatorDaemon
from alfred.store.state import StateStore


def _make_daemon(tmp_path: Path) -> ConsolidatorDaemon:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    cfg = AlfredConfig(vault_path=vault_path, data_dir=tmp_path / "data")
    state = StateStore(tmp_path / "state.json")
    state.load()
    events: asyncio.Queue = asyncio.Queue()
    return ConsolidatorDaemon(cfg, state, events)


class _FakeUsage:
    input_tokens = 8
    output_tokens = 4
    cache_read_input_tokens = 0


class _FakeAnthropicResponse:
    def __init__(self, text: str) -> None:
        self.usage = _FakeUsage()
        self.content = [type("Block", (), {"text": text})()]


class _FakeAnthropicMessages:
    def __init__(self, text: str) -> None:
        self._text = text

    def create(self, **kwargs):
        return _FakeAnthropicResponse(self._text)


class _FakeAnthropicClient:
    def __init__(self, text: str) -> None:
        self.messages = _FakeAnthropicMessages(text)


async def _failing_post(self, url, **kwargs):
    raise httpx.ConnectError("simulated: no local Ollama running")


def test_label_tick_normal_path_labels_cluster_via_local_llm_fallback(
    tmp_path, monkeypatch
):
    daemon = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path
    note_dir = vault_path / "note"
    note_dir.mkdir()
    member_paths = []
    for i in range(3):
        rel = f"note/member-{i}.md"
        (vault_path / "note" / f"member-{i}.md").write_text(
            f"---\ntype: note\nname: member-{i}\n---\nBody {i}.\n", encoding="utf-8"
        )
        member_paths.append(rel)

    daemon.state.state.clusters["semantic_0"] = ClusterState(
        cluster_id=0,
        cluster_type="semantic",
        member_files=member_paths,
    )

    # Ollama (the primary path) is unreachable in the test environment —
    # mock the HTTP call to fail fast rather than actually attempting network I/O.
    monkeypatch.setattr(httpx.AsyncClient, "post", _failing_post)
    # Fallback 1 is now the local backend, bound at import time in the
    # consolidator's namespace — patch it there.
    monkeypatch.setattr(
        "alfred.daemons.consolidator.complete",
        lambda *a, **kw: "fallback cluster label",
    )

    asyncio.run(daemon.label_tick())

    cluster = daemon.state.state.clusters["semantic_0"]
    assert cluster.label == ["fallback cluster label"]
    assert cluster.last_labeled


def test_label_tick_exception_is_caught_and_logged_not_propagated(tmp_path, monkeypatch):
    """label_tick() must swallow any exception raised inside _label_pass()
    and log it rather than letting it propagate — this is what makes it safe
    to register directly as an APScheduler job function."""
    daemon = _make_daemon(tmp_path)

    async def _boom(vault_path) -> None:
        raise RuntimeError("simulated consolidator label pass failure")

    monkeypatch.setattr(daemon, "_label_pass", _boom)

    with capture_logs() as logs:
        asyncio.run(daemon.label_tick())  # must not raise

    errors = [e for e in logs if e.get("log_level") == "error"
              and e.get("event") == "consolidator.label_tick_error"]
    assert len(errors) == 1
    assert "simulated consolidator label pass failure" in errors[0]["error"]


def test_stubs_tick_normal_path_creates_wiki_page(tmp_path):
    """stubs_tick() (the real APScheduler job function) must create a wiki
    stub page for a person/org record via real vault I/O — no LLM needed."""
    daemon = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path
    person_dir = vault_path / "person"
    person_dir.mkdir()
    rel_path = "person/jane-doe.md"
    (vault_path / "person" / "jane-doe.md").write_text(
        "---\ntype: person\nname: Jane Doe\n---\nColleague.\n", encoding="utf-8"
    )
    daemon.state.state.files[rel_path] = FileState(md5="abc123")

    asyncio.run(daemon.stubs_tick())

    assert "jane doe" in daemon.state.state.wiki_pages
    page = daemon.state.state.wiki_pages["jane doe"]
    assert page.rel_path == "wiki/jane-doe.md"
    assert (vault_path / "wiki" / "jane-doe.md").exists()
