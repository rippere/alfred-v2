"""Off-loop StateStore save: race-freedom, fold-back, bounded final save, and
event-loop responsiveness.

Background: the synchronous save() blocked the daemon's event loop. Under
memory pressure that ran past systemd's 30s TimeoutStopSec, so every stop
since 2026-10-13 ended in SIGKILL with no alfred.final_save. The fix
snapshots on the loop, does the I/O in a worker thread, and folds the merge
back in place. These tests pin each of those properties down.
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import threading
import time
from dataclasses import asdict

import pytest
from structlog.testing import capture_logs

import alfred.store.state as state_mod
from alfred.core.failures import drain_failures, record_failure, reset_failures
from alfred.core.models import ClusterState, FileState, MemoryStrength, WikiPage
from alfred.daemons.base import BaseDaemon
from alfred.store.state import StateSaveTimeout, StateStore, _state_to_raw


@pytest.fixture(autouse=True)
def _clean_failures():
    reset_failures()
    yield
    reset_failures()


def _populated(tmp_path, n: int = 50) -> StateStore:
    st = StateStore(tmp_path / "state.json")
    st.load()
    for i in range(n):
        st.state.files[f"f{i}.md"] = FileState(
            md5=f"m{i}", chunk_ids=[f"f{i}::c0", f"f{i}::c1"], open_issues=["x"],
        )
    st.state.clusters["c1"] = ClusterState(cluster_id=1, label=["a"], member_files=["f1.md"])
    st.state.memory["f1.md"] = MemoryStrength(rel_path="f1.md", access_count=2)
    st.state.wiki_pages["e"] = WikiPage(
        entity_name="e", entity_type="t", rel_path="w.md", created="c", updated="u",
        sources=["s"],
    )
    st.state.curator_processed["a.md"] = "md5"
    st.state.distiller_runs.append({"timestamp": "t", "n": 1})
    st.state.error_counts["x"] = 1
    return st


def _slow_merge(monkeypatch, seconds: float) -> threading.Event:
    """Make the worker-thread half of a save take `seconds`. Returns an event
    that's set once the worker is inside the merge, i.e. after the snapshot."""
    real = state_mod._merge_pipeline_state
    entered = threading.Event()

    def slow(base, theirs, mine):
        entered.set()
        time.sleep(seconds)
        return real(base, theirs, mine)

    monkeypatch.setattr(state_mod, "_merge_pipeline_state", slow)
    return entered


async def _until(event: threading.Event, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not event.is_set():
        assert time.monotonic() < deadline, "worker never reached the merge"
        await asyncio.sleep(0.001)


def _disk(st: StateStore) -> dict:
    return json.loads(st.path.read_text())


# ── snapshot fidelity ─────────────────────────────────────────────────────────

def test_state_to_raw_matches_asdict(tmp_path):
    st = _populated(tmp_path)
    assert _state_to_raw(st.state) == asdict(st.state)


def test_snapshot_shares_no_mutable_objects(tmp_path):
    st = _populated(tmp_path)
    raw = _state_to_raw(st.state)
    st.state.files["f0.md"].chunk_ids.append("later")
    st.state.distiller_runs[0]["n"] = 99
    assert raw["files"]["f0.md"]["chunk_ids"] == ["f0::c0", "f0::c1"]
    assert raw["distiller_runs"][0]["n"] == 1


# ── concurrent mutation during an async save ─────────────────────────────────

def test_concurrent_mutation_during_async_save_does_not_crash_or_tear(tmp_path, monkeypatch):
    st = _populated(tmp_path)
    st.save()
    before = _state_to_raw(st.state)["files"]
    entered = _slow_merge(monkeypatch, 0.3)

    async def main():
        save = asyncio.create_task(st.async_save())
        await _until(entered)  # snapshot taken, worker is mid-merge
        for i in range(300):
            st.state.files[f"new{i}.md"] = FileState(md5="n", chunk_ids=[f"new{i}::c0"])
            st.state.files["f0.md"].chunk_ids.append(f"extra{i}")
            if i < 20:
                del st.state.files[f"f{i + 1}.md"]
            st.record_api_call(10, 10)
            await asyncio.sleep(0)
        await save
        assert entered.is_set()
        return _disk(st)

    first = asyncio.run(main())

    # The write is exactly the snapshot. No window mutation leaked in half-applied.
    assert first["files"] == before
    # Every window mutation survived the fold-back in memory...
    files = st.state.files
    assert all(f"new{i}.md" in files for i in range(300))
    assert len(files["f0.md"].chunk_ids) == 2 + 300
    assert not any(f"f{i}.md" in files for i in range(1, 21))
    assert st.state.api_calls_today == 300
    # ...and the next save persists all of it.
    monkeypatch.undo()  # drop the slow merge
    asyncio.run(st.async_save())
    on_disk = _disk(st)
    assert on_disk["files"] == _state_to_raw(st.state)["files"]
    assert on_disk["api_calls_today"] == 300


def test_held_state_reference_is_not_orphaned_by_saves(tmp_path):
    """The surveyor holds `state = self.state.state` for a whole diff and saves
    every 25 files. The old save() rebound _state, so files 26+ went into a
    dead object and never reached disk: the perpetual re-embed."""
    st = StateStore(tmp_path / "state.json")
    st.load()
    state = st.state

    async def surveyor_like():
        for i in range(60):
            state.files[f"f{i}.md"] = FileState(md5=str(i))
            if (i + 1) % 25 == 0:
                await st.async_save()
        await st.async_save()

    asyncio.run(surveyor_like())
    assert st.state is state
    assert len(_disk(st)["files"]) == 60

    # Same guarantee for the synchronous path.
    for i in range(60, 90):
        state.files[f"f{i}.md"] = FileState(md5=str(i))
        st.save()
    assert st.state is state
    assert len(_disk(st)["files"]) == 90


def test_other_writers_changes_fold_in_without_clobbering_window_edits(tmp_path, monkeypatch):
    p = tmp_path / "state.json"
    a = StateStore(p)
    a.load()
    for k in ("shared.md", "k2.md", "gone.md"):
        a.state.files[k] = FileState(md5="v0")
    a.save()
    held = a.state.files["k2.md"]  # a reference a daemon might be holding

    b = StateStore(p)
    b.load()
    b.state.files["shared.md"].md5 = "b"
    b.state.files["k2.md"].md5 = "b-k2"
    b.state.files["theirs.md"] = FileState(md5="t")
    del b.state.files["gone.md"]
    b.state.curator_processed["inbox.md"] = "h"
    b.record_api_call(1, 1)
    b.record_api_call(1, 1)
    b.save()

    a.record_api_call(1, 1)  # in a's snapshot
    entered = _slow_merge(monkeypatch, 0.2)

    async def main():
        save = asyncio.create_task(a.async_save())
        await _until(entered)
        a.state.files["shared.md"].md5 = "a-window"  # conflicts with b's edit
        a.state.files["window.md"] = FileState(md5="w")
        a.record_api_call(1, 1)
        await save

    asyncio.run(main())

    f = a.state.files
    assert f["theirs.md"].md5 == "t"            # b's addition folded in
    assert "gone.md" not in f                   # b's deletion folded in
    assert f["shared.md"].md5 == "a-window"     # fresher local edit kept
    assert f["window.md"].md5 == "w"
    assert a.state.files["k2.md"] is held and held.md5 == "b-k2"  # updated in place
    assert a.state.curator_processed["inbox.md"] == "h"
    assert a.state.api_calls_today == 4         # 2 (b) + 2 (a), additive

    a.save()
    d = _disk(a)
    assert d["files"]["shared.md"]["md5"] == "a-window"
    assert d["files"]["theirs.md"]["md5"] == "t"
    assert d["files"]["window.md"]["md5"] == "w"
    assert "gone.md" not in d["files"]
    assert d["api_calls_today"] == 4


# ── event loop responsiveness ─────────────────────────────────────────────────

def _ticks_during(coro_factory) -> int:
    async def main():
        ticks = 0
        done = False

        async def ticker():
            nonlocal ticks
            while not done:
                await asyncio.sleep(0.01)
                ticks += 1

        t = asyncio.create_task(ticker())
        await asyncio.sleep(0)
        await coro_factory()
        done = True
        await t
        return ticks

    return asyncio.run(main())


def test_event_loop_stays_responsive_during_slow_save(tmp_path, monkeypatch):
    st = _populated(tmp_path)
    _slow_merge(monkeypatch, 0.5)
    daemon = BaseDaemon(cfg=None, state=st, events=asyncio.Queue())

    ticks = _ticks_during(daemon.save_state)  # the path the surveyor uses
    assert ticks >= 20, f"timer starved during async save: {ticks} ticks in 0.5s"

    # Control: the same slow save done synchronously on the loop starves it,
    # which proves the assertion above actually measures something.
    async def blocking():
        st.save()

    assert _ticks_during(blocking) <= 2


# ── bounded saves: timeouts leave the previous state.json intact ──────────────

def _hold_lock(st: StateStore) -> int:
    fd = os.open(str(st._lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def test_lock_wait_timeout_leaves_prior_state_intact(tmp_path):
    st = _populated(tmp_path)
    st.save()
    prior = st.path.read_bytes()
    st.state.files["unsaved.md"] = FileState(md5="u")
    record_failure("test.swallowed")

    fd = _hold_lock(st)  # e.g. a frozen MCP query process mid-save
    try:
        t0 = time.monotonic()
        with pytest.raises(StateSaveTimeout):
            asyncio.run(st.async_save(timeout=0.3))
        assert time.monotonic() - t0 < 1.5
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    assert st.path.read_bytes() == prior
    assert not st.path.with_name("state.json.tmp").exists()
    # The swallowed failure is neither lost nor double counted.
    assert "test.swallowed" not in st.state.error_counts
    st.save()
    assert _disk(st)["error_counts"]["test.swallowed"] == 1
    assert "unsaved.md" in _disk(st)["files"]
    assert drain_failures() == {}


def test_deadline_inside_slow_merge_never_replaces_state(tmp_path, monkeypatch):
    st = _populated(tmp_path)
    st.save()
    prior = st.path.read_bytes()
    st.state.files["unsaved.md"] = FileState(md5="u")
    _slow_merge(monkeypatch, 0.5)

    with pytest.raises(StateSaveTimeout):
        asyncio.run(st.async_save(timeout=0.2))
    assert st.path.read_bytes() == prior
    assert not st.path.with_name("state.json.tmp").exists()

    with pytest.raises(StateSaveTimeout):  # the sync path honours it too
        st.save(timeout=0.2)
    assert st.path.read_bytes() == prior


# ── runner final save ─────────────────────────────────────────────────────────

def test_final_save_logs_success(tmp_path):
    from alfred.runner import final_save

    st = _populated(tmp_path)
    st.state.files["last.md"] = FileState(md5="l")
    with capture_logs() as logs:
        ok = asyncio.run(final_save(st, 2.0))
    assert ok
    assert [e["event"] for e in logs if e["event"].startswith("alfred.final_save")] == [
        "alfred.final_save"
    ]
    assert "last.md" in _disk(st)["files"]


def test_final_save_timeout_logs_and_keeps_prior_state(tmp_path):
    from alfred.runner import final_save

    st = _populated(tmp_path)
    st.save()
    prior = st.path.read_bytes()
    st.state.files["late.md"] = FileState(md5="l")

    fd = _hold_lock(st)
    try:
        t0 = time.monotonic()
        with capture_logs() as logs:
            ok = asyncio.run(final_save(st, 0.3))
        elapsed = time.monotonic() - t0
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    assert not ok
    events = [e["event"] for e in logs]
    assert "alfred.final_save_timeout" in events
    assert "alfred.final_save" not in events
    assert elapsed < 1.5
    assert st.path.read_bytes() == prior
    assert not st.path.with_name("state.json.tmp").exists()


def test_final_save_timeout_budget_fits_stop_timeout():
    from alfred import runner

    assert runner.final_save_timeout(None) == runner.FINAL_SAVE_MAX_S
    assert runner.final_save_timeout(100.0, now=100.0 + runner.SHUTDOWN_GRACE_S) == pytest.approx(
        min(runner.FINAL_SAVE_MAX_S, runner.SHUTDOWN_BUDGET_S - runner.SHUTDOWN_GRACE_S)
    )
    assert runner.final_save_timeout(0.0, now=1000.0) == runner.FINAL_SAVE_MIN_S
    # grace + final save + backstop must fit inside TimeoutStopSec=30.
    assert (
        runner.SHUTDOWN_GRACE_S + runner.FINAL_SAVE_MAX_S + runner.FINAL_SAVE_BACKSTOP_S
    ) < 30.0


# ── review follow-ups ─────────────────────────────────────────────────────────

def test_cancelled_caller_still_completes_and_folds(tmp_path, monkeypatch):
    """The scheduler cancels in-flight jobs on shutdown. The save they were
    awaiting must still commit and fold back, or the next save double counts."""
    p = tmp_path / "state.json"
    other = StateStore(p)
    other.load()
    other.record_api_call(1, 1)
    other.save()

    st = _populated(tmp_path)
    st.record_api_call(1, 1)
    entered = _slow_merge(monkeypatch, 0.2)

    async def main():
        job = asyncio.create_task(st.async_save())
        await _until(entered)
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
        async with st._lock:  # waits for the shielded save to finish
            pass

    asyncio.run(main())
    assert st.state.api_calls_today == 2  # folded: other's call + ours
    monkeypatch.undo()
    st.save()
    assert _disk(st)["api_calls_today"] == 2  # not 3


def test_log_list_fold_keeps_window_appends(tmp_path, monkeypatch):
    st = _populated(tmp_path)
    st.save()
    b = StateStore(st.path)
    b.load()
    b.state.janitor_sweeps.append({"ts": "other"})
    b.save()
    held = st.state.janitor_sweeps
    entered = _slow_merge(monkeypatch, 0.2)

    async def main():
        save = asyncio.create_task(st.async_save())
        await _until(entered)
        st.state.janitor_sweeps.append({"ts": "window"})
        await save

    asyncio.run(main())
    assert st.state.janitor_sweeps is held
    assert {"ts": "other"} in held and {"ts": "window"} in held
    monkeypatch.undo()
    st.save()
    assert {"ts": "window"} in _disk(st)["janitor_sweeps"]


def test_daemon_save_state_timeout_is_logged_not_raised(tmp_path, monkeypatch):
    import alfred.daemons.base as base_mod

    st = _populated(tmp_path)
    st.save()
    prior = st.path.read_bytes()
    monkeypatch.setattr(base_mod, "DAEMON_SAVE_TIMEOUT_S", 0.2)
    daemon = BaseDaemon(cfg=None, state=st, events=asyncio.Queue())
    fd = _hold_lock(st)
    try:
        with capture_logs() as logs:
            asyncio.run(daemon.save_state())  # must not raise
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    assert "daemon.save_state_timeout" in [e["event"] for e in logs]
    assert st.path.read_bytes() == prior
