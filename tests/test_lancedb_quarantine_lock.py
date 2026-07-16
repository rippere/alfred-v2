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
        assert store.was_recreated is True
        assert store.count() == 0

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
