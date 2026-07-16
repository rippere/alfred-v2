"""SurveyorDaemon._tick() must not independently trigger `_recluster()`.

Regression coverage for the duplicate-recluster-trigger bug: `_tick()` used to
carry an internal `time.time() - self._last_cluster > CLUSTER_INTERVAL` check
that called `self._recluster()` directly, racing against the independently
scheduled APScheduler `surveyor_recluster` job (see runner.py) which also
calls `_recluster()`. Because APScheduler's `max_instances=1` is scoped per
job id, nothing prevented both triggers from running `_recluster()` — an
expensive vector-store `query_all` + HDBSCAN pass — concurrently, each
writing `state.clusters` / `state.files[...].semantic_cluster_id` from a
different, potentially stale snapshot.

The fix: the dedicated `surveyor_recluster` APScheduler job (-> `recluster()`
-> `_recluster()`) is now the sole trigger. `_tick()` (-> `tick()`, the
`surveyor_tick` job) must never call `_recluster()`, regardless of elapsed
time or how many times it runs.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from alfred.config import AlfredConfig
from alfred.daemons.surveyor import SurveyorDaemon
from alfred.store.state import StateStore


class _StubStore:
    """Stands in for LanceDBStore — never touches disk or lancedb."""


def _make_daemon(tmp_path: Path) -> SurveyorDaemon:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    cfg = AlfredConfig(vault_path=vault_path, data_dir=tmp_path / "data")
    state = StateStore(tmp_path / "state.json")
    state.load()
    events: asyncio.Queue = asyncio.Queue()
    return SurveyorDaemon(cfg, state, events, store=_StubStore())


def test_tick_never_calls_recluster_regardless_of_elapsed_time(tmp_path, monkeypatch):
    daemon = _make_daemon(tmp_path)

    recluster_calls: list[None] = []

    async def _fake_recluster() -> None:
        recluster_calls.append(None)

    monkeypatch.setattr(daemon, "_recluster", _fake_recluster)

    # Run many ticks — before the fix, whichever tick crossed CLUSTER_INTERVAL
    # (1800s) since the last internal recluster would have called
    # self._recluster() directly. With the vault empty, _compute_diff() finds
    # nothing to process each time, isolating the recluster-triggering logic.
    for _ in range(5):
        asyncio.run(daemon._tick())

    assert recluster_calls == [], (
        "_tick() invoked _recluster() directly; the internal recluster "
        "trigger should have been removed in favor of the dedicated "
        "APScheduler surveyor_recluster job"
    )


def test_last_cluster_and_cluster_interval_are_gone():
    """The pre-split tick-based recluster-scheduling state must not linger
    unused now that the APScheduler job is the sole trigger."""
    import alfred.daemons.surveyor as surveyor_module

    daemon = SurveyorDaemon.__new__(SurveyorDaemon)
    assert not hasattr(daemon, "_last_cluster")
    assert not hasattr(surveyor_module, "CLUSTER_INTERVAL")


def test_recluster_job_is_sole_scheduler_path_to_recluster(tmp_path, monkeypatch):
    """runner.py must register the `surveyor_tick` job against `tick()` and
    the `surveyor_recluster` job against `recluster()` on the *same* daemon
    instance — confirming `recluster()` (-> `_recluster()`) is reachable only
    via its own dedicated job, never via the tick job."""
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    from alfred.runner import run_daemons

    class _SchedulerCaptured(Exception):
        pass

    class _StubVectorStore:
        was_recreated = False

        def __init__(self, *args, **kwargs):
            pass

    (tmp_path / "vault").mkdir()
    cfg_path = tmp_path / "config-test.yaml"
    cfg_path.write_text(f"vault:\n  path: {tmp_path / 'vault'}\ndata_dir: ./data\n")
    cfg = AlfredConfig.load(cfg_path)

    captured = {}

    def _fake_start(self, *args, **kwargs):
        captured["scheduler"] = self
        raise _SchedulerCaptured

    monkeypatch.setattr(AsyncIOScheduler, "start", _fake_start)
    monkeypatch.setattr("alfred.store.lancedb_store.LanceDBStore", _StubVectorStore)

    with pytest.raises(_SchedulerCaptured):
        asyncio.run(run_daemons(cfg))

    jobs = {job.id: job for job in captured["scheduler"].get_jobs()}
    assert "surveyor_tick" in jobs
    assert "surveyor_recluster" in jobs

    tick_func = jobs["surveyor_tick"].func
    recluster_func = jobs["surveyor_recluster"].func

    assert tick_func.__func__ is SurveyorDaemon.tick
    assert recluster_func.__func__ is SurveyorDaemon.recluster
    # Same daemon instance backs both jobs — recluster() is not duplicated
    # elsewhere, and tick()'s bound instance is the one whose _tick() we
    # proved above never calls _recluster().
    assert tick_func.__self__ is recluster_func.__self__


class _RecordingStore:
    """Stub vector store that records delete_file/upsert_many calls and lets
    upsert_many be forced to fail, to exercise the changed-file error path."""

    def __init__(self, fail_upsert: bool = False) -> None:
        self.fail_upsert = fail_upsert
        self.delete_calls: list[tuple[str, list[str] | None]] = []
        self.upsert_calls: list[list[dict]] = []

    def delete_file(self, rel_path: str, chunk_ids: list[str] | None = None) -> None:
        self.delete_calls.append((rel_path, list(chunk_ids) if chunk_ids else chunk_ids))

    def upsert_many(self, rows: list[dict]) -> None:
        self.upsert_calls.append(rows)
        if self.fail_upsert:
            raise RuntimeError("simulated upsert_many failure")


class _StubEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _StubBM25:
    is_fitted = False

    def encode(self, text: str) -> dict:
        return {}


def test_upsert_failure_leaves_old_chunks_available_not_dangling(tmp_path, monkeypatch):
    """Regression coverage for the silent content-availability gap: a changed
    file whose re-embed upsert fails must NOT have its previously-indexed
    chunks deleted from the vector store, and state must keep claiming
    exactly the (still-valid) old chunk_ids — never the new ones, and never
    a dangling reference to chunks that no longer exist in the store.
    """
    from alfred.core.models import FileState

    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    rel_path = "note.md"
    (vault_path / rel_path).write_text(
        "---\ntype: note\n---\nUpdated body content for the note.\n"
    )

    cfg = AlfredConfig(vault_path=vault_path, data_dir=tmp_path / "data")
    state_store = StateStore(tmp_path / "state.json")
    state_store.load()

    old_chunk_ids = ["note.md::chunk_00"]
    state_store.state.files[rel_path] = FileState(
        md5="old-md5",
        last_embedded="2026-01-01T00:00:00+00:00",
        chunk_ids=old_chunk_ids,
    )

    events: asyncio.Queue = asyncio.Queue()
    store = _RecordingStore(fail_upsert=True)
    daemon = SurveyorDaemon(cfg, state_store, events, store=store)
    monkeypatch.setattr(daemon, "_get_embedder", lambda: _StubEmbedder())
    monkeypatch.setattr(daemon, "_get_bm25", lambda: _StubBM25())

    diff = {
        "new": [],
        "changed": [rel_path],
        "deleted": [],
        "current": {rel_path: "new-md5"},
    }

    asyncio.run(daemon._process_diff(diff))

    # upsert_many was attempted and failed — confirm the test actually
    # exercised the failure path.
    assert len(store.upsert_calls) == 1

    # The old chunks must never have been deleted from the vector store: a
    # failed upsert should not be able to orphan them.
    assert store.delete_calls == [], (
        "delete_file was called even though upsert_many failed — this "
        "orphans the old chunk_ids that state still (correctly) references"
    )

    # State must still point at the old, still-present chunk_ids — not the
    # new md5 (which would falsely mark the file as freshly indexed), and
    # not an empty/dangling chunk_ids list.
    fs = state_store.state.files[rel_path]
    assert fs.md5 == "old-md5"
    assert fs.chunk_ids == old_chunk_ids


def test_upsert_success_deletes_only_stale_chunks_after_write(tmp_path, monkeypatch):
    """On a successful re-embed, the old chunk_ids should be removed only
    *after* the new chunks are written, and only the ids no longer produced
    (e.g. the file got shorter) should be deleted — the shared chunk_00 id
    must not be deleted, since delete_file would otherwise race the
    just-completed upsert_many and strip the freshly-written row."""
    from alfred.core.models import FileState

    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    rel_path = "note.md"
    (vault_path / rel_path).write_text(
        "---\ntype: note\n---\nShort body now.\n"
    )

    cfg = AlfredConfig(vault_path=vault_path, data_dir=tmp_path / "data")
    state_store = StateStore(tmp_path / "state.json")
    state_store.load()

    # Previously this file produced two chunks; chunk_00 will be reproduced
    # again (deterministic id), chunk_01 is now stale.
    old_chunk_ids = ["note.md::chunk_00", "note.md::chunk_01"]
    state_store.state.files[rel_path] = FileState(
        md5="old-md5",
        last_embedded="2026-01-01T00:00:00+00:00",
        chunk_ids=old_chunk_ids,
    )

    events: asyncio.Queue = asyncio.Queue()
    store = _RecordingStore(fail_upsert=False)
    daemon = SurveyorDaemon(cfg, state_store, events, store=store)
    monkeypatch.setattr(daemon, "_get_embedder", lambda: _StubEmbedder())
    monkeypatch.setattr(daemon, "_get_bm25", lambda: _StubBM25())

    diff = {
        "new": [],
        "changed": [rel_path],
        "deleted": [],
        "current": {rel_path: "new-md5"},
    }

    asyncio.run(daemon._process_diff(diff))

    # upsert_many happened before any delete_file call.
    assert len(store.upsert_calls) == 1
    assert len(store.delete_calls) == 1
    assert store.delete_calls[0] == (rel_path, ["note.md::chunk_01"])

    fs = state_store.state.files[rel_path]
    assert fs.md5 == "new-md5"
    assert fs.chunk_ids == ["note.md::chunk_00"]
