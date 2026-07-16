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
