"""Behavioral coverage for CuratorDaemon's dedup-key logic in _process_inbox.

The process key is `f"{rel_path}#{content_hash}"` rather than just `rel_path`,
specifically so a reused inbox filename (a new note dropped at a path an
older, already-archived note used) gets ingested instead of being silently
swallowed forever. See the inline comment in curator.py:

    "A path-only key marks that path 'done' forever and silently swallows
    every future drop at it — that's how curator lost ~10 re-dropped notes
    over 3 weeks before anyone noticed."

These tests exercise that logic directly against _process_inbox, using a
frontmatter-declared `type` so classification short-circuits the LLM branch
entirely (no ANTHROPIC_API_KEY / network dependency).
"""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

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


def _content_hash(text: bytes) -> str:
    return hashlib.sha256(text).hexdigest()[:16]


def test_process_inbox_ingests_and_records_content_hash_key(tmp_path):
    daemon, state_store = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path
    inbox_file = vault_path / "inbox" / "drop.md"
    content = "---\ntype: note\nname: hello-world\n---\nSome inbox note body.\n"
    inbox_file.write_text(content, encoding="utf-8")
    expected_hash = _content_hash(content.encode("utf-8"))

    asyncio.run(daemon._process_inbox())

    expected_key = f"inbox/drop.md#{expected_hash}"
    assert expected_key in state_store.state.curator_processed
    assert len(state_store.state.curator_processed) == 1

    # File archived out of inbox/, vault record created.
    assert not inbox_file.exists()
    assert (vault_path / "inbox" / "processed" / "drop.md").exists()
    assert (vault_path / "note" / "hello-world.md").exists()


def test_process_inbox_skips_when_exact_process_key_already_recorded(tmp_path):
    """If the same path+content-hash pair is already in curator_processed,
    the file must be left alone entirely — not re-ingested, not moved."""
    daemon, state_store = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path
    inbox_file = vault_path / "inbox" / "drop.md"
    content = "---\ntype: note\nname: hello-world\n---\nSome inbox note body.\n"
    inbox_file.write_text(content, encoding="utf-8")
    existing_hash = _content_hash(content.encode("utf-8"))
    state_store.state.curator_processed[f"inbox/drop.md#{existing_hash}"] = "2026-01-01T00:00:00+00:00"

    asyncio.run(daemon._process_inbox())

    # Nothing new happened: file untouched, no vault record, no new key.
    assert inbox_file.exists()
    assert not (vault_path / "note" / "hello-world.md").exists()
    assert len(state_store.state.curator_processed) == 1


def test_process_inbox_reingests_reused_path_with_new_content_despite_legacy_key(tmp_path):
    """Regression for the exact incident described in curator.py: a legacy
    (pre-fix) path-only key for this filename must NOT prevent a new drop
    with different content at the same path from being ingested — only an
    exact path+hash match should skip."""
    daemon, state_store = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path

    # Simulate a leftover legacy-format entry: bare path, no content hash.
    state_store.state.curator_processed["inbox/drop.md"] = "2020-01-01T00:00:00+00:00"

    inbox_file = vault_path / "inbox" / "drop.md"
    new_content = "---\ntype: note\nname: brand-new-drop\n---\nCompletely different content.\n"
    inbox_file.write_text(new_content, encoding="utf-8")
    new_hash = _content_hash(new_content.encode("utf-8"))

    asyncio.run(daemon._process_inbox())

    # The new content must have been ingested, not swallowed by the legacy key.
    assert not inbox_file.exists(), "new drop at a reused path was silently skipped"
    assert (vault_path / "inbox" / "processed" / "drop.md").exists()
    assert (vault_path / "note" / "brand-new-drop.md").exists()

    new_key = f"inbox/drop.md#{new_hash}"
    assert new_key in state_store.state.curator_processed
    # Legacy key is untouched/still present alongside the new one.
    assert "inbox/drop.md" in state_store.state.curator_processed


def test_process_inbox_same_path_different_content_gets_distinct_keys(tmp_path):
    """Two successive drops at the same inbox filename but with different
    content must each get ingested and each leave their own distinct
    path+hash key — proving the key is content-sensitive, not path-only."""
    daemon, state_store = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path
    inbox_file = vault_path / "inbox" / "recycled.md"

    first_content = "---\ntype: note\nname: first-drop\n---\nFirst note body.\n"
    inbox_file.write_text(first_content, encoding="utf-8")
    asyncio.run(daemon._process_inbox())

    # Re-use the same inbox filename with different content (e.g. a workflow
    # that always writes to inbox/recycled.md).
    second_content = "---\ntype: note\nname: second-drop\n---\nSecond, unrelated note body.\n"
    inbox_file.write_text(second_content, encoding="utf-8")
    asyncio.run(daemon._process_inbox())

    first_key = f"inbox/recycled.md#{_content_hash(first_content.encode('utf-8'))}"
    second_key = f"inbox/recycled.md#{_content_hash(second_content.encode('utf-8'))}"
    assert first_key in state_store.state.curator_processed
    assert second_key in state_store.state.curator_processed
    assert first_key != second_key

    assert (vault_path / "note" / "first-drop.md").exists()
    assert (vault_path / "note" / "second-drop.md").exists()
