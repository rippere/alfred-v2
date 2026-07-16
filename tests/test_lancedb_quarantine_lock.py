"""Concurrency test for LanceDBStore's quarantine-and-recreate race.

Two (or more) processes opening the same corrupted table each independently
pass the ``_looks_like_corruption`` check in ``LanceDBStore.__init__`` and
each decide to quarantine it. Without cross-process coordination, whichever
loses the ``mode="create"`` race crashes uninformatively (table already
exists) instead of healing, and the racing ``_MAX_QUARANTINES_PER_DAY``
reads can double-count the breaker window. This mirrors that shape with
threads racing construction of independent ``LanceDBStore`` instances
pointed at the same on-disk corrupted table, each getting its own lock-file
descriptor (so ``flock`` arbitrates between them exactly as it would across
separate OS processes).
"""
from __future__ import annotations

import threading
from pathlib import Path

import lancedb
import pytest

from alfred.store.lancedb_store import LanceDBStore

N_RACERS = 6


@pytest.fixture
def corrupted_uri(tmp_path):
    """Build a real LanceDB table, then truncate its latest manifest to
    reproduce the exact zero-byte-manifest corruption this module heals.
    """
    uri = str(tmp_path)
    seed = LanceDBStore(uri, collection="race_tbl", dims=4)
    seed.upsert("a", [0.1, 0.2, 0.3, 0.4], {}, "note", "a.md", 0)
    assert seed.was_recreated is False

    tbl_dir = tmp_path / "race_tbl.lance"
    manifests = sorted((tbl_dir / "_versions").glob("*.manifest"))
    latest = max(manifests, key=lambda p: p.stat().st_mtime)
    latest.write_bytes(b"")

    # Sanity: the corruption actually reproduces on a fresh connection, and
    # trips the same marker-based detection LanceDBStore uses.
    from alfred.store.lancedb_store import _looks_like_corruption

    db = lancedb.connect(uri)
    with pytest.raises(Exception) as excinfo:
        db.open_table("race_tbl")
    assert _looks_like_corruption(excinfo.value)

    return uri


def test_concurrent_quarantine_no_crash_single_winner(corrupted_uri):
    """N racing constructions of the same corrupted table must all heal —
    none may crash, only one may actually perform the destructive quarantine
    (move + mode="create"), and the rest must join its fresh table as
    readers instead.
    """
    results: list[LanceDBStore] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(N_RACERS)

    def racer() -> None:
        try:
            barrier.wait(timeout=10)
            store = LanceDBStore(corrupted_uri, collection="race_tbl", dims=4)
            results.append(store)
        except BaseException as e:  # noqa: BLE001 — collect everything for the assertion
            errors.append(e)

    threads = [threading.Thread(target=racer) for _ in range(N_RACERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"racing construction crashed instead of healing: {errors!r}"
    assert len(results) == N_RACERS

    # Every racer must have healed onto a working, empty table — the old
    # corrupt data is gone, and none of them silently kept the broken table.
    for store in results:
        assert store.count() == 0

    # Locking now spans the whole table_names()-then-branch decision (not just
    # the quarantine body), so only the single racer who actually holds the
    # lock while the table is still corrupt observes the corruption and
    # performs (and flags) the heal. Everyone else acquires the lock only
    # after the winner has already recreated the table, so they take the
    # plain open_table() success path against an already-healthy empty table
    # — never raising, never setting was_recreated themselves.
    recreated_flags = [store.was_recreated for store in results]
    assert recreated_flags.count(True) == 1, recreated_flags

    # Exactly one destructive quarantine happened, not N — the losers joined
    # the winner's already-recreated table instead of each performing (and
    # crashing on, or double-counting) their own move + create.
    base = Path(corrupted_uri)
    quarantine_dirs = [p for p in base.glob(".quarantine-corrupt-*") if p.is_dir()]
    assert len(quarantine_dirs) == 1, (
        f"expected exactly one quarantine dir, got {len(quarantine_dirs)}: {quarantine_dirs}"
    )

    # The healed table must still be fully usable afterward.
    winner = results[0]
    winner.upsert("post-heal", [0.5, 0.5, 0.5, 0.5], {}, "note", "post.md", 0)
    assert winner.count() == 1


def test_concurrent_fresh_create_no_crash_single_creator(tmp_path, monkeypatch):
    """N racing constructions of a collection that does not exist at all yet
    (no prior table, no corruption — distinct from ``corrupted_uri`` above)
    must not crash.

    Before the fix, ``table_names()`` was checked *outside* the flock in
    ``__init__``. Two racers both starting fresh would both see the
    collection missing from ``table_names()`` and both take the unlocked
    ``else: create_table(...)`` branch, racing each other directly — the
    loser crashing with ``ValueError("Table '...' already exists")``. Locking
    now spans the whole table_names()-then-branch decision, so only the
    winner ever calls ``create_table``; every other racer blocks on the lock
    and, once it wakes, ``table_names()`` already shows the winner's table —
    it opens that as a reader instead of racing a second ``create_table``.
    """
    uri = str(tmp_path)
    collection = "fresh_tbl"
    assert not (Path(uri) / f"{collection}.lance").exists()

    create_calls: list[int] = []
    calls_lock = threading.Lock()
    real_connect = lancedb.connect

    class _CountingDB:
        """Wraps a real connection, counting successful create_table calls."""

        def __init__(self, real_db) -> None:
            self._real = real_db

        def table_names(self):
            return self._real.table_names()

        def open_table(self, name):
            return self._real.open_table(name)

        def create_table(self, *args, **kwargs):
            tbl = self._real.create_table(*args, **kwargs)
            with calls_lock:
                create_calls.append(1)
            return tbl

    def fake_connect(u):
        return _CountingDB(real_connect(u))

    monkeypatch.setattr(lancedb, "connect", fake_connect)

    results: list[LanceDBStore] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(N_RACERS)

    def racer() -> None:
        try:
            barrier.wait(timeout=10)
            store = LanceDBStore(uri, collection=collection, dims=4)
            results.append(store)
        except BaseException as e:  # noqa: BLE001 — collect everything for the assertion
            errors.append(e)

    threads = [threading.Thread(target=racer) for _ in range(N_RACERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, (
        f"racing construction of a brand-new collection crashed instead of "
        f"joining as a reader: {errors!r}"
    )
    assert len(results) == N_RACERS

    # Exactly one create_table call ever succeeded — the rest joined as
    # readers instead of racing the else-branch directly.
    assert len(create_calls) == 1, (
        f"expected exactly one create_table call, got {len(create_calls)}"
    )

    for store in results:
        assert store.count() == 0

    # The table must be fully usable afterward, from any of the racers.
    winner = results[0]
    winner.upsert("post-create", [0.5, 0.5, 0.5, 0.5], {}, "note", "post.md", 0)
    assert winner.count() == 1
