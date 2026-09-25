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


def test_deep_tick_normal_path_enriches_stub_via_mocked_local_llm(tmp_path, monkeypatch):
    """deep_tick() (the real APScheduler job function) must call through to
    _deep_sweep() -> _enrich_file(), with only the local backend mocked,
    and clear the STUB_RECORD issue once the file has been enriched."""
    daemon = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path

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

    monkeypatch.setattr(
        "alfred.daemons.janitor.complete",
        lambda *a, **kw: enriched_body,
    )

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


def test_deep_sweep_request_too_large_drops_the_stub_and_goes_on(tmp_path, monkeypatch):
    """An oversized request gets the same answer every sweep: log it, stop
    asking, and carry on with the other stubs instead of stopping the sweep."""
    from alfred.core.local_llm import LocalLLMRequestTooLarge

    daemon = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path
    (vault_path / "note").mkdir()
    for name in ("a", "b"):
        (vault_path / "note" / f"{name}.md").write_text(
            f"---\ntype: note\ncreated: '2026-01-01'\n---\nStub {name}.\n", encoding="utf-8"
        )
        daemon.state.state.files[f"note/{name}.md"] = FileState(
            md5=name, open_issues=[IssueCode.STUB_RECORD.value]
        )

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("alfred.daemons.janitor.asyncio.sleep", _no_sleep)
    calls: list[str] = []

    def _complete(system, prompt, **kw):
        calls.append(prompt)
        if "note/a.md" in prompt:
            raise LocalLLMRequestTooLarge("finish_reason=length")
        return "An enriched body for b, long enough to be written to the vault."

    monkeypatch.setattr("alfred.daemons.janitor.complete", _complete)

    with capture_logs() as logs:
        asyncio.run(daemon._deep_sweep())

    assert len(calls) == 2
    files = daemon.state.state.files
    assert IssueCode.STUB_RECORD.value not in files["note/a.md"].open_issues
    assert IssueCode.STUB_RECORD.value not in files["note/b.md"].open_issues
    assert "enriched body for b" in vault_read(vault_path, "note/b.md")["body"]
    assert [e for e in logs if e.get("event") == "janitor.request_too_large"]
