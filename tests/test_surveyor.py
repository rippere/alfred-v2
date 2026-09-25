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
from structlog.testing import capture_logs

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


def test_tick_normal_path_embeds_new_file_end_to_end(tmp_path, monkeypatch):
    """Exercise the actual `tick()` APScheduler entry point (not `_tick()` or
    `_process_diff()` directly): a brand-new vault file should be discovered
    by `_compute_diff()`, embedded via the mocked embedder/BM25, and recorded
    in state with chunk_ids — with the vector store itself mocked out."""
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    (vault_path / "note.md").write_text(
        "---\ntype: note\n---\nFresh new note body content.\n", encoding="utf-8"
    )

    cfg = AlfredConfig(vault_path=vault_path, data_dir=tmp_path / "data")
    state_store = StateStore(tmp_path / "state.json")
    state_store.load()
    events: asyncio.Queue = asyncio.Queue()
    store = _RecordingStore()
    daemon = SurveyorDaemon(cfg, state_store, events, store=store)
    monkeypatch.setattr(daemon, "_get_embedder", lambda: _StubEmbedder())
    monkeypatch.setattr(daemon, "_get_bm25", lambda: _StubBM25())

    asyncio.run(daemon.tick())

    assert len(store.upsert_calls) == 1
    assert "note.md" in state_store.state.files
    assert state_store.state.files["note.md"].chunk_ids


def test_tick_exception_is_caught_and_logged_not_propagated(tmp_path, monkeypatch):
    """`tick()` must swallow any exception raised inside `_tick()` and log it
    rather than letting it propagate — this is what makes it safe to register
    directly as an APScheduler job function."""
    daemon = _make_daemon(tmp_path)

    async def _boom() -> None:
        raise RuntimeError("simulated surveyor tick failure")

    monkeypatch.setattr(daemon, "_tick", _boom)

    with capture_logs() as logs:
        asyncio.run(daemon.tick())  # must not raise

    errors = [e for e in logs if e.get("log_level") == "error"
              and e.get("event") == "surveyor.tick_error"]
    assert len(errors) == 1
    assert "simulated surveyor tick failure" in errors[0]["error"]


class _RowsStore:
    """query_all() returns canned rows, in whatever order the test chose."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def query_all(self, output_fields=None) -> list[dict]:
        return [dict(r) for r in self.rows]


class _PositionalHDBSCAN:
    """Stands in for HDBSCAN where its output depends on input order, as the
    real one's does on the live vectors (the neuro vault's own vectors,
    shuffled, kept 13 of 30 clusters). Pairs up consecutive rows."""

    seen_inputs: list = []

    def __init__(self, **kwargs) -> None:
        pass

    def fit_predict(self, vectors):
        import numpy as np

        _PositionalHDBSCAN.seen_inputs.append(vectors.copy())
        return np.array([i // 2 for i in range(len(vectors))])


def _rows() -> list[dict]:
    # Two chunks for a.md: its first chunk (value 1.0) is the one that must
    # represent it, whichever row the store happens to return first.
    return [
        {"id": "a.md::chunk_01", "embedding": [9.0, 0.0]},
        {"id": "a.md::chunk_00", "embedding": [1.0, 0.0]},
        {"id": "b.md::chunk_00", "embedding": [2.0, 0.0]},
        {"id": "c.md::chunk_00", "embedding": [3.0, 0.0]},
        {"id": "d.md::chunk_00", "embedding": [4.0, 0.0]},
    ]


def _clusters_for(tmp_path, monkeypatch, rows) -> dict[str, list[str]]:
    import sklearn.cluster

    monkeypatch.setattr(sklearn.cluster, "HDBSCAN", _PositionalHDBSCAN)
    tmp_path.mkdir()
    daemon = _make_daemon(tmp_path)
    daemon.cfg.data_dir.mkdir()  # graph.pkl lives there
    daemon.store = _RowsStore(rows)
    asyncio.run(daemon._recluster())
    return {k: sorted(c.member_files) for k, c in daemon.state.state.clusters.items()}


def test_recluster_does_not_depend_on_store_row_order(tmp_path, monkeypatch):
    """Re-embeds rewrite rows, so the store's row order drifts between passes.
    Clusters built from that order changed membership and key for files
    nobody touched, and the consolidator paid to relabel every one of them."""
    forward = _clusters_for(tmp_path / "f", monkeypatch, _rows())
    first_input = _PositionalHDBSCAN.seen_inputs[-1]
    backward = _clusters_for(tmp_path / "b", monkeypatch, list(reversed(_rows())))

    assert forward == backward
    assert forward == {"semantic_0": ["a.md", "b.md"], "semantic_1": ["c.md", "d.md"]}
    assert first_input[0][0] == 1.0, "a.md must be represented by its first chunk"


def test_recluster_keeps_the_cluster_object_and_its_bookkeeping(tmp_path, monkeypatch):
    """Reclustering updates an existing cluster in place: its label and
    synthesis bookkeeping ride along, and a consolidator pass holding the
    object keeps writing to the one that gets saved."""
    import sklearn.cluster

    from alfred.core.models import ClusterState

    (tmp_path / "x").mkdir()
    monkeypatch.setattr(sklearn.cluster, "HDBSCAN", _PositionalHDBSCAN)
    daemon = _make_daemon(tmp_path / "x")
    daemon.cfg.data_dir.mkdir()
    daemon.store = _RowsStore(_rows())
    held = ClusterState(
        cluster_id=0,
        label=["alpha"],
        member_files=["a.md", "b.md"],
        last_labeled="2026-09-24T00:00:00+00:00",
        consolidated_chunk_id="synthesis/alpha.md",
    )
    held.labeled_members = "fp-labeled"
    held.synthesized_members = "fp-synth"
    daemon.state.state.clusters["semantic_0"] = held

    asyncio.run(daemon._recluster())

    assert daemon.state.state.clusters["semantic_0"] is held
    assert held.label == ["alpha"]
    assert held.consolidated_chunk_id == "synthesis/alpha.md"
    assert (held.labeled_members, held.synthesized_members) == ("fp-labeled", "fp-synth")


def test_embed_that_outlasts_a_save_is_not_redone(tmp_path, monkeypatch):
    """The re-embed churn: on 2026-09-24 the main vault re-embedded ~517K
    chunks, nearly all from ~1.3 MB instinct snapshots (5.9K chunks each) that
    had not changed. Embedding one took longer than the 5-min periodic save,
    and save() used to swap out the state object _process_diff was writing
    into, so the new md5 never reached disk and every tick saw the file as
    changed again. (Fixed by StateStore._fold_into; this pins the symptom.)"""
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    (vault_path / "big.md").write_text(
        "---\ntype: note\n---\nA long feed snapshot body.\n", encoding="utf-8"
    )
    cfg = AlfredConfig(vault_path=vault_path, data_dir=tmp_path / "data")
    cfg.data_dir.mkdir()
    state_store = StateStore(tmp_path / "state.json")
    state_store.load()
    daemon = SurveyorDaemon(cfg, state_store, asyncio.Queue(), store=_RecordingStore())

    class _SlowEmbedder:
        async def embed(self, text: str) -> list[float]:
            state_store.save()  # the periodic save lands mid-embed
            return [0.1, 0.2, 0.3]

    monkeypatch.setattr(daemon, "_get_embedder", lambda: _SlowEmbedder())
    monkeypatch.setattr(daemon, "_get_bm25", lambda: _StubBM25())

    asyncio.run(daemon.tick())

    restarted = StateStore(tmp_path / "state.json")
    restarted.load()
    assert restarted.state.files.get("big.md") is not None, "embed not recorded"
    daemon.state = restarted
    assert daemon._compute_diff()["changed"] == [], "unchanged file queued for re-embed"
