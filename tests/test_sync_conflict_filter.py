"""Sync-conflict exclusion: indexing, wikilink targets, and the dedup sweep.

Syncthing writes `<stem>.sync-conflict-<YYYYMMDD>-<HHMMSS>-<ID><ext>`, so a
conflicted note ends in `.md` and matched every `rglob("*.md")` in the codebase.
`cfg.ignore_dirs` could not express it — that filter is directory-based.

Measured on the live vault before the fix: 162 conflict copies indexed, 129
alongside their still-present original, and a `vault_search "decision"` that
returned 11 conflicts in its 40 results.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from alfred.core.vault import is_sync_conflict


@pytest.mark.parametrize("name", [
    "Executive Mind Matrix.sync-conflict-20260622-145746-ZO3SA2G.md",
    "alfred-v2.sync-conflict-20260622-145838-ZO3SA2G.md",
    "notes.sync-conflict-20250101-000000-A1B2C3D.md",
])
def test_recognises_real_conflict_names(name):
    assert is_sync_conflict(name)
    assert is_sync_conflict(Path("/vault/decision") / name)


@pytest.mark.parametrize("name", [
    "decision.md",
    "how to resolve a sync-conflict.md",       # prose, not Syncthing's marker
    "sync-conflict-notes.md",                  # no dot, no timestamp
    "backup.sync-conflict.md",                 # marker without the timestamp
])
def test_does_not_flag_ordinary_notes(name):
    assert not is_sync_conflict(name)


def test_accepts_str_and_path_identically():
    n = "x.sync-conflict-20260622-145746-ZO3SA2G.md"
    assert is_sync_conflict(n) is is_sync_conflict(Path("/a/b") / n)


def test_dedup_sweep_never_pairs_a_conflict_copy(tmp_path):
    """The F4 guard.

    _dedup_sweep merges same-directory near-duplicates and DELETES the shorter,
    with keeper = longer body. A conflict copy sits in the same directory as its
    original and is near-identical to it, so it always clears the 0.85 threshold
    — and when the conflict is the longer of the two, the live note is what gets
    deleted. The sweep must not see conflicts at all.
    """
    import asyncio

    from alfred.config import AlfredConfig
    from alfred.daemons.janitor import JanitorDaemon
    from alfred.store.state import StateStore

    vault = tmp_path / "vault"
    (vault / "decision").mkdir(parents=True)
    (vault / "inbox").mkdir()          # the sweep writes its report here

    body = "---\ntype: decision\n---\n" + ("A real decision body. " * 20)
    original = vault / "decision" / "keep-me.md"
    original.write_text(body, encoding="utf-8")
    # Deliberately LONGER than the original: under "longer body wins" this is
    # the copy that would have survived and deleted the live note.
    conflict = vault / "decision" / "keep-me.sync-conflict-20260622-145746-ZO3SA2G.md"
    conflict.write_text(body + "\nOne extra stale line.\n", encoding="utf-8")

    cfg = AlfredConfig(vault_path=vault, data_dir=tmp_path / "data")
    state = StateStore(tmp_path / "state.json")
    state.load()
    daemon = JanitorDaemon(cfg, state, asyncio.Queue(), store=None)

    asyncio.run(daemon._dedup_sweep())

    assert original.exists(), "the live note must never be deleted"
    assert conflict.exists(), "reconciliation is reconcile_conflicts.py's job, not dedup's"
    assert "One extra stale line." not in original.read_text(encoding="utf-8"), (
        "conflict content must not be merged into the live note"
    )
