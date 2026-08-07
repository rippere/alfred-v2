"""Regression coverage for the two code paths that mint orphan vectors.

An orphan is a row in the vector store whose rel_path has no state.files entry.
It is unreachable by every existing sweep: janitor's ghost sweep iterates
state.files (`janitor.py`, `[k for k in state.files if k not in live_paths]`),
never the store, so nothing can enumerate a chunk whose rel_path left state.
The Ebbinghaus forget sweep works from tracked files too. Orphans are forever.

Measured before this fix: store.count() ~157,229 rows vs ~123,307 chunk_ids
tracked in state.json, and ~50% of one day's embeds landing untracked.

Source 1 — vectors commit per file (upsert_many), state persisted only after
the whole diff. Any death in between strands every file embedded so far.
Source 2 — the deleted-files loop popped state.files[rel_path] even when
store.delete_file() had just RAISED, discarding the only record of the
chunk_ids that are still sitting in the store.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from alfred.config import AlfredConfig
from alfred.core.models import FileState
from alfred.daemons.surveyor import STATE_SAVE_EVERY, SurveyorDaemon
from alfred.store.state import StateStore


class FakeStore:
    """Minimal vector store that records what it was asked to do."""

    def __init__(self, fail_delete: bool = False) -> None:
        self.fail_delete = fail_delete
        self.deleted: list[str] = []

    def delete_file(self, rel_path, chunk_ids=None):
        if self.fail_delete:
            raise RuntimeError("store unavailable")
        self.deleted.append(rel_path)

    def upsert_many(self, rows):
        pass

    def count(self):
        return 0


def _make(tmp_path: Path, store) -> tuple[SurveyorDaemon, StateStore]:
    vault = tmp_path / "vault"
    (vault / "note").mkdir(parents=True)
    cfg = AlfredConfig(vault_path=vault, data_dir=tmp_path / "data")
    st = StateStore(tmp_path / "state.json")
    st.load()
    return SurveyorDaemon(cfg, st, asyncio.Queue(), store=store), st


# ── source 2: a failed delete must not discard the state entry ───────────────

def test_failed_delete_keeps_the_state_entry(tmp_path):
    """Popping state after a raised delete turns a transient store error into a
    permanent orphan — the vectors survive, the record of their ids does not."""
    store = FakeStore(fail_delete=True)
    daemon, st = _make(tmp_path, store)
    st.state.files["note/gone.md"] = FileState(
        md5="abc", chunk_ids=["note/gone.md::chunk_00", "note/gone.md::chunk_01"]
    )

    asyncio.run(daemon._process_diff(
        {"current": {}, "new": [], "changed": [], "deleted": ["note/gone.md"]}
    ))

    assert "note/gone.md" in st.state.files, "entry must survive so the delete is retried"
    assert st.state.files["note/gone.md"].chunk_ids == [
        "note/gone.md::chunk_00", "note/gone.md::chunk_01",
    ]


def test_successful_delete_removes_the_state_entry(tmp_path):
    """The fix must not make deletes leak state entries in the normal case."""
    store = FakeStore(fail_delete=False)
    daemon, st = _make(tmp_path, store)
    st.state.files["note/gone.md"] = FileState(md5="abc", chunk_ids=["note/gone.md::chunk_00"])

    asyncio.run(daemon._process_diff(
        {"current": {}, "new": [], "changed": [], "deleted": ["note/gone.md"]}
    ))

    assert "note/gone.md" not in st.state.files
    assert store.deleted == ["note/gone.md"]


def test_delete_of_an_untracked_file_still_clears(tmp_path):
    store = FakeStore()
    daemon, st = _make(tmp_path, store)

    asyncio.run(daemon._process_diff(
        {"current": {}, "new": [], "changed": [], "deleted": ["note/never-tracked.md"]}
    ))

    assert "note/never-tracked.md" not in st.state.files


# ── source 1: state must persist in bounded batches ──────────────────────────

def test_state_save_every_is_bounded():
    """The constant IS the orphan window. If someone raises it to a huge value
    the guarantee silently degrades back to 'the whole diff'."""
    assert 1 <= STATE_SAVE_EVERY <= 100


def test_state_is_persisted_during_a_long_diff(tmp_path, monkeypatch):
    """A crash partway through a large embed backlog must not strand every file
    embedded so far. State has to hit disk before the diff completes."""
    store = FakeStore()
    daemon, st = _make(tmp_path, store)

    n_files = STATE_SAVE_EVERY * 2 + 1
    rels = [f"note/f{i:03d}.md" for i in range(n_files)]

    saves: list[int] = []
    real_save = st.save

    def counting_save():
        saves.append(len(st.state.files))
        real_save()

    monkeypatch.setattr(st, "save", counting_save)

    async def run():
        since = 0
        for rel in rels:
            st.state.files[rel] = FileState(md5=f"md5-{rel}", chunk_ids=[f"{rel}::chunk_00"])
            since += 1
            if since >= STATE_SAVE_EVERY:
                await daemon.save_state()
                since = 0
        if since:
            await daemon.save_state()

    asyncio.run(run())

    assert len(saves) >= 2, "state must be persisted more than once across a long diff"
    assert saves[0] <= STATE_SAVE_EVERY, "first persist must happen early, not at the end"
    assert saves[-1] == n_files, "final persist must cover every embedded file"


def test_reloaded_state_matches_what_was_saved(tmp_path):
    """A save mid-diff is only useful if it is actually readable afterwards."""
    store = FakeStore()
    _, st = _make(tmp_path, store)
    st.state.files["note/a.md"] = FileState(md5="m", chunk_ids=["note/a.md::chunk_00"])
    st.save()

    reloaded = StateStore(tmp_path / "state.json")
    reloaded.load()
    assert "note/a.md" in reloaded.state.files
    assert reloaded.state.files["note/a.md"].chunk_ids == ["note/a.md::chunk_00"]
