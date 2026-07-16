"""JanitorDaemon must not orphan LanceDB embeddings for files it archives or
dedupes.

Regression coverage: `_archive_sessions()` and `_dedup_sweep()` used to only
`state.files.pop(rel_path, None)` when moving/deleting a file off disk — the
vector store was never told, so old embeddings for that file lingered in
LanceDB forever (searchable, unreachable, permanently orphaned). Surveyor's
own deletion path (`_process_diff`'s `diff["deleted"]` loop) already calls
`self.store.delete_file(rel_path, chunk_ids)` before dropping the state
entry; this test proves JanitorDaemon now mirrors that pattern via its own
`store` reference.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from alfred.config import AlfredConfig
from alfred.core.models import FileState
from alfred.daemons.janitor import JanitorDaemon
from alfred.store.state import StateStore


class _RecordingStore:
    """Stub vector store that records delete_file calls — mirrors the stub
    used in tests/test_surveyor.py so the janitor's use of the same
    `store.delete_file(rel_path, chunk_ids)` contract is exercised the same
    way."""

    def __init__(self) -> None:
        self.delete_calls: list[tuple[str, list[str] | None]] = []

    def delete_file(self, rel_path: str, chunk_ids: list[str] | None = None) -> None:
        self.delete_calls.append((rel_path, list(chunk_ids) if chunk_ids else chunk_ids))


def _make_daemon(tmp_path: Path, store: _RecordingStore) -> tuple[JanitorDaemon, StateStore]:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    cfg = AlfredConfig(vault_path=vault_path, data_dir=tmp_path / "data")
    state = StateStore(tmp_path / "state.json")
    state.load()
    events: asyncio.Queue = asyncio.Queue()
    daemon = JanitorDaemon(cfg, state, events, store=store)
    return daemon, state


def test_archive_sessions_deletes_embeddings_for_moved_file(tmp_path):
    """A session file old enough to archive must have its vector-store
    chunks deleted, not just its state.files entry popped."""
    daemon, state_store = _make_daemon(tmp_path, _RecordingStore())
    store: _RecordingStore = daemon.store
    vault_path = daemon.cfg.vault_path

    session_dir = vault_path / "session"
    session_dir.mkdir()
    session_file = session_dir / "old-session.md"
    session_file.write_text("---\ntype: session\nstatus: absorbed\n---\nBody.\n")

    # Make the file look old enough to archive (>90 days).
    old_ts = (datetime.now(timezone.utc) - timedelta(days=120)).timestamp()
    import os
    os.utime(session_file, (old_ts, old_ts))

    rel_path = "session/old-session.md"
    chunk_ids = ["session/old-session.md::chunk_00", "session/old-session.md::chunk_01"]
    state_store.state.files[rel_path] = FileState(md5="abc123", chunk_ids=chunk_ids)

    archived = asyncio.run(daemon._archive_sessions(vault_path))

    assert archived == 1
    # File physically moved, not left behind.
    assert not session_file.exists()
    assert (vault_path / "_archived" / "session" / "old-session.md").exists()

    # The vector store must have been told to drop exactly this file's chunks.
    assert store.delete_calls == [(rel_path, chunk_ids)], (
        "session archival moved the file without deleting its embeddings — "
        "this orphans LanceDB vectors for a file no longer at its indexed path"
    )
    # And state must no longer reference it.
    assert rel_path not in state_store.state.files


def test_dedup_sweep_deletes_embeddings_for_removed_duplicate(tmp_path):
    """When dedup merges and deletes a near-duplicate file, its vector-store
    chunks must be deleted too, not just its state.files entry popped."""
    daemon, state_store = _make_daemon(tmp_path, _RecordingStore())
    store: _RecordingStore = daemon.store
    vault_path = daemon.cfg.vault_path

    notes_dir = vault_path / "notes"
    notes_dir.mkdir()
    (vault_path / "inbox").mkdir()  # _dedup_sweep writes its report here

    body = "This is a fairly long duplicate paragraph of note content used to trigger dedup merging behavior reliably across both files without tripping the minimum length guard."
    keeper = notes_dir / "keeper.md"
    keeper.write_text(f"---\ntype: note\n---\n{body}\nExtra unique keeper line.\n")
    dupe = notes_dir / "dupe.md"
    dupe.write_text(f"---\ntype: note\n---\n{body}\n")

    keeper_rel = "notes/keeper.md"
    dupe_rel = "notes/dupe.md"
    dupe_chunk_ids = ["notes/dupe.md::chunk_00"]
    state_store.state.files[keeper_rel] = FileState(md5="keeper-md5", chunk_ids=["notes/keeper.md::chunk_00"])
    state_store.state.files[dupe_rel] = FileState(md5="dupe-md5", chunk_ids=dupe_chunk_ids)

    asyncio.run(daemon._dedup_sweep())

    # Exactly one of the two files should have been deleted (the shorter one).
    assert not dupe.exists()
    assert keeper.exists()

    assert store.delete_calls == [(dupe_rel, dupe_chunk_ids)], (
        "dedup sweep deleted the duplicate file without deleting its "
        "embeddings — this orphans LanceDB vectors for a file removed from "
        "the vault"
    )
    assert dupe_rel not in state_store.state.files
    assert keeper_rel in state_store.state.files


def test_janitor_constructor_requires_store(tmp_path):
    """JanitorDaemon must take a vector-store reference like SurveyorDaemon
    does, so both daemons share the same deletion contract."""
    import inspect

    params = inspect.signature(JanitorDaemon.__init__).parameters
    assert "store" in params, (
        "JanitorDaemon.__init__ has no `store` parameter — it cannot clean "
        "up vector-store embeddings for archived/deduped files"
    )
