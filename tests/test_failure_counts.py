"""Swallowed failures must leave a durable, countable trace.

~23 handlers in this tree have a body of `pass`/`continue`. Individually
defensible, collectively they made "zero errors" unfalsifiable: the failure
happened, nothing recorded it, and the next status read looked clean. These
tests hold the floor that alfred.core.failures puts under that — the counter
survives the process (via state.json), sums correctly across concurrent
writers, and is actually reached from a real swallowing code path rather
than only from a direct call to record_failure().
"""
from __future__ import annotations

import threading

import pytest

from alfred.core.failures import (
    drain_failures,
    peek_failures,
    record_failure,
    reset_failures,
    restore_failures,
)
from alfred.store.state import StateStore


@pytest.fixture(autouse=True)
def _clean_counters():
    """The counters are process-global by design; isolate each test."""
    reset_failures()
    yield
    reset_failures()


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / "state.json"


def test_record_failure_counts_and_drain_zeroes():
    record_failure("demo.thing_failed", error=ValueError("boom"))
    record_failure("demo.thing_failed", error=ValueError("boom again"))
    record_failure("demo.other_failed")

    assert peek_failures() == {"demo.thing_failed": 2, "demo.other_failed": 1}

    drained = drain_failures()
    assert drained == {"demo.thing_failed": 2, "demo.other_failed": 1}
    assert peek_failures() == {}, "drain must zero the counters, not just read them"


def test_save_persists_counts_to_state_json(state_path):
    """The point of the whole exercise: a failure swallowed in one process is
    still countable after that process is gone."""
    store = StateStore(state_path)
    store.load()

    record_failure("surveyor.file_read_failed", error=OSError("EIO"))
    record_failure("surveyor.file_read_failed", error=OSError("EIO"))
    store.save()

    reborn = StateStore(state_path)
    reborn.load()
    assert reborn.state.error_counts == {"surveyor.file_read_failed": 2}


def test_repeated_saves_do_not_double_count(state_path):
    """save() drains rather than reads, so saving three times after one
    failure must still report one — a wrong number is worse than none."""
    store = StateStore(state_path)
    store.load()

    record_failure("janitor.session_stat_failed", error=OSError("ENOENT"))
    store.save()
    store.save()
    store.save()

    reborn = StateStore(state_path)
    reborn.load()
    assert reborn.state.error_counts == {"janitor.session_stat_failed": 1}


def test_counts_accumulate_across_separate_store_instances(state_path):
    """Each CLI/MCP invocation constructs its own StateStore; the persisted
    total must keep climbing rather than being overwritten."""
    for _ in range(4):
        record_failure("graph.rebuild_read_failed", error=OSError("EIO"))
        store = StateStore(state_path)
        store.load()
        store.save()

    final = StateStore(state_path)
    final.load()
    assert final.state.error_counts == {"graph.rebuild_read_failed": 4}


def test_concurrent_savers_sum_rather_than_clobber(state_path):
    """Two processes each swallowing an error must total two. A plain
    dict-field merge would take whichever side changed the key and silently
    discard the other's increment — the same class of bug the API counters
    already had to solve additively."""
    seed = StateStore(state_path)
    seed.load()
    seed.save()

    n_writers = 8
    errors: list[BaseException] = []
    barrier = threading.Barrier(n_writers)

    def worker(_tid: int) -> None:
        try:
            store = StateStore(state_path)
            store.load()
            record_failure("vault.grep_read_failed", error=OSError("EIO"))
            barrier.wait(timeout=10)
            store.save()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"exceptions during concurrent hammering: {errors!r}"

    final = StateStore(state_path)
    final.load()
    assert final.state.error_counts.get("vault.grep_read_failed") == n_writers, (
        f"expected {n_writers} counted failures, got "
        f"{final.state.error_counts!r} — an increment was clobbered"
    )


def test_failed_save_puts_the_counts_back(state_path, monkeypatch):
    """A disk error during save must not also destroy the record of the
    errors that came before it — otherwise the failure that most needs
    reporting is the one that erases the evidence."""
    store = StateStore(state_path)
    store.load()
    record_failure("lancedb.diagnostic_scan_failed", error=OSError("EIO"))

    def _boom(*_a, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr("alfred.store.state.json.dumps", _boom)
    with pytest.raises(OSError):
        store.save()

    assert peek_failures() == {"lancedb.diagnostic_scan_failed": 1}, (
        "counts were drained and then lost when the save failed"
    )


def test_restore_failures_is_additive():
    record_failure("demo.thing_failed")
    restore_failures({"demo.thing_failed": 3, "demo.new_failed": 1})
    assert peek_failures() == {"demo.thing_failed": 4, "demo.new_failed": 1}


def test_unreadable_file_during_grep_is_counted(tmp_path):
    """End-to-end through a real swallowing code path, not a direct call:
    vault_search's grep filter drops unreadable files from its result set
    with `continue`. The dropped file must still be counted."""
    from alfred.core.vault_ops import vault_search

    vault = tmp_path / "vault"
    (vault / "learn").mkdir(parents=True)
    good = vault / "learn" / "good.md"
    good.write_text("---\ntype: learn\nname: good\n---\n\nneedle here\n")
    bad = vault / "learn" / "bad.md"
    bad.write_text("---\ntype: learn\nname: bad\n---\n\nneedle here too\n")

    # Make one file genuinely unreadable so the handler is reached for real.
    bad.chmod(0o000)
    try:
        results = vault_search(vault, grep_pattern="needle")
    finally:
        bad.chmod(0o644)

    paths = {r["path"] for r in results}
    assert "learn/good.md" in paths
    assert "learn/bad.md" not in paths, "unreadable file should drop out of results"
    assert peek_failures().get("vault.grep_read_failed") == 1, (
        f"the dropped file left no trace — counts were {peek_failures()!r}"
    )
