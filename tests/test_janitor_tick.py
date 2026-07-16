"""Behavioral coverage for JanitorDaemon's public tick() entry points
(structural_tick / deep_tick), which are what APScheduler actually calls —
as opposed to tests/test_janitor.py (embedding cleanup via internal
_archive_sessions/_dedup_sweep) and tests/test_janitor_autofix.py (the
deterministic _autofix stage called through a _check_file/_autofix helper).

Covers one normal-tick path (structural_tick doing a real autofix sweep;
deep_tick doing real LLM-enrichment with the Anthropic client mocked) and one
exception-handling path (deep_tick swallowing and logging an internal
exception rather than propagating it, matching the contract asserted for the
other daemons' tick() wrappers).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from alfred.config import AlfredConfig
from alfred.core.models import FileState
from alfred.core.vault_ops import vault_read
from alfred.daemons.janitor import IssueCode, JanitorDaemon
from alfred.store.state import StateStore


def _make_daemon(tmp_path: Path) -> JanitorDaemon:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    cfg = AlfredConfig(vault_path=vault_path, data_dir=tmp_path / "data")
    state = StateStore(tmp_path / "state.json")
    state.load()
    events: asyncio.Queue = asyncio.Queue()
    return JanitorDaemon(cfg, state, events, store=None)


def test_structural_tick_normal_path_autofixes_and_records_sweep(tmp_path):
    """structural_tick() (the actual APScheduler job function) must run the
    full scan -> autofix -> save pipeline: a file missing type/created gets
    fixed on disk and a sweep summary is appended to state."""
    daemon = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path
    note_dir = vault_path / "note"
    note_dir.mkdir()
    fp = note_dir / "bare.md"
    fp.write_text("---\n---\nJust a body, no frontmatter fields.\n", encoding="utf-8")

    asyncio.run(daemon.structural_tick())

    rec = vault_read(vault_path, "note/bare.md")
    fm = rec["frontmatter"]
    assert fm["type"] == "note"
    assert "created" in fm and fm["created"]

    sweeps = daemon.state.state.janitor_sweeps
    assert len(sweeps) == 1
    assert sweeps[0]["files_with_issues"] >= 1
    assert sweeps[0]["autofixed"] >= 1


def test_deep_tick_normal_path_enriches_stub_via_mocked_anthropic(tmp_path, monkeypatch):
    """deep_tick() (the real APScheduler job function) must call through to
    _deep_sweep() -> _enrich_file(), with only the Anthropic client mocked,
    and clear the STUB_RECORD issue once the file has been enriched."""
    daemon = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    note_dir = vault_path / "note"
    note_dir.mkdir()
    rel_path = "note/stub.md"
    fp = note_dir / "stub.md"
    fp.write_text("---\ntype: note\ncreated: '2026-01-01'\n---\nToo short.\n", encoding="utf-8")

    daemon.state.state.files[rel_path] = FileState(
        md5="abc123",
        open_issues=[IssueCode.STUB_RECORD.value],
    )

    enriched_body = "A properly enriched body with plenty of descriptive detail about this note."

    class _FakeContentBlock:
        text = enriched_body

    class _FakeResponse:
        content = [_FakeContentBlock()]

    class _FakeMessages:
        def create(self, **kwargs):
            return _FakeResponse()

    class _FakeClient:
        messages = _FakeMessages()

    monkeypatch.setattr("alfred.daemons.janitor.get_client", lambda: _FakeClient())

    asyncio.run(daemon.deep_tick())

    rec = vault_read(vault_path, rel_path)
    assert rec["body"].strip() == enriched_body

    fs = daemon.state.state.files[rel_path]
    assert IssueCode.STUB_RECORD.value not in fs.open_issues


def test_deep_tick_exception_is_caught_and_logged_not_propagated(tmp_path, monkeypatch):
    """deep_tick() must swallow any exception raised inside _deep_sweep() and
    log it rather than letting it propagate — this is what makes it safe to
    register directly as an APScheduler job function."""
    daemon = _make_daemon(tmp_path)

    async def _boom() -> None:
        raise RuntimeError("simulated janitor deep sweep failure")

    monkeypatch.setattr(daemon, "_deep_sweep", _boom)

    with capture_logs() as logs:
        asyncio.run(daemon.deep_tick())  # must not raise

    errors = [e for e in logs if e.get("log_level") == "error"
              and e.get("event") == "janitor.deep_tick_error"]
    assert len(errors) == 1
    assert "simulated janitor deep sweep failure" in errors[0]["error"]
