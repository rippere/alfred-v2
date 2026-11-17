"""Concurrency test for GraphStore's locking + atomic save (audit structural #1/#6a).

Threads hammer a single shared GraphStore (mutations, saves, traversals) while
reader threads concurrently unpickle the on-disk file. The lock must serialize
read-modify-write so no write is lost, and the tmp-file + os.replace() save
must guarantee readers never observe a truncated pickle.
"""
from __future__ import annotations

import pickle
import subprocess
import sys
import textwrap
import threading

import pytest

from alfred.store.graph import GraphStore

N_WRITERS = 8
OPS_PER_WRITER = 60


@pytest.fixture
def graph_path(tmp_path):
    return tmp_path / "graph.pkl"


def test_concurrent_save_load_no_lost_writes(graph_path):
    store = GraphStore(graph_path)
    store.add_edges_from_wikilinks("seed.md", ["seed-target.md"])
    store.save()  # readers always have a file to open

    errors: list[BaseException] = []
    stop_readers = threading.Event()

    def writer(tid: int) -> None:
        try:
            for i in range(OPS_PER_WRITER):
                src = f"t{tid}/note-{i:03d}.md"
                store.add_edges_from_wikilinks(src, [f"t{tid}/target-{i % 7}.md"])
                if i % 3 == 0:
                    store.save()
                if i % 5 == 0:
                    store.add_cluster_edges([f"t{tid}/note-{j:03d}.md" for j in range(max(0, i - 3), i + 1)])
                # Interleave reads that re-acquire the lock internally
                store.get_node_degree(src)
                store.spreading_activation([src], hops=2)
                store.node_count()
        except BaseException as e:  # noqa: BLE001 — collect everything for the assertion
            errors.append(e)

    def reader() -> None:
        # Validate on-disk file integrity while writers save concurrently:
        # os.replace() is atomic, so this must never see a truncated pickle.
        try:
            while not stop_readers.is_set():
                g = pickle.loads(graph_path.read_bytes())
                assert g.number_of_nodes() >= 1
                fresh = GraphStore(graph_path)
                assert fresh.load() is True
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    writers = [threading.Thread(target=writer, args=(tid,)) for tid in range(N_WRITERS)]
    readers = [threading.Thread(target=reader) for _ in range(2)]
    for t in readers + writers:
        t.start()
    for t in writers:
        t.join()
    stop_readers.set()
    for t in readers:
        t.join()

    assert not errors, f"exceptions during concurrent hammering: {errors!r}"

    # No lost writes: after a final save, every writer's every node must be present.
    store.save()
    final = GraphStore(graph_path)
    assert final.load() is True
    g = pickle.loads(graph_path.read_bytes())
    expected = {
        f"t{tid}/note-{i:03d}.md" for tid in range(N_WRITERS) for i in range(OPS_PER_WRITER)
    }
    missing = expected - set(g.nodes)
    assert not missing, f"lost writes — {len(missing)} nodes missing, e.g. {sorted(missing)[:5]}"

    # Atomic-save hygiene: no orphaned tmp file left behind.
    assert not list(graph_path.parent.glob(f".{graph_path.name}.*.tmp"))


def test_independent_instances_share_lock_no_lost_writes(graph_path):
    """Two independently-constructed GraphStore(path) instances (mirroring how
    surveyor.py / consolidator.py each build their own) must serialize against
    each other. Before the module-level path-keyed lock registry, each instance
    got its own private RLock, so concurrent load/mutate/save across instances
    was not mutually exclusive and silently dropped edges (last-writer-wins).
    """
    seed = GraphStore(graph_path)
    seed.add_edges_from_wikilinks("seed.md", ["seed-target.md"])
    seed.save()

    errors: list[BaseException] = []

    def worker(tid: int) -> None:
        # Each call site constructs its own GraphStore, exactly like
        # surveyor.py / consolidator.py do — this is the reported bug shape.
        try:
            for i in range(OPS_PER_WRITER):
                store = GraphStore(graph_path)
                with store.transaction():
                    store.load()
                    store.add_edges_from_wikilinks(
                        f"t{tid}/note-{i:03d}.md", [f"t{tid}/target-{i % 7}.md"]
                    )
                    store.save()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(N_WRITERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"exceptions during concurrent hammering: {errors!r}"

    g = pickle.loads(graph_path.read_bytes())
    expected = {
        f"t{tid}/note-{i:03d}.md" for tid in range(N_WRITERS) for i in range(OPS_PER_WRITER)
    }
    missing = expected - set(g.nodes)
    assert not missing, f"lost writes across independent instances — {len(missing)} missing, e.g. {sorted(missing)[:5]}"

    assert not list(graph_path.parent.glob(f".{graph_path.name}.*.tmp"))


def test_save_is_atomic_replacement(graph_path):
    """save() must go through tmp + os.replace, leaving a loadable file."""
    store = GraphStore(graph_path)
    store.add_edges_from_wikilinks("a.md", ["b.md"])
    store.save()
    first = graph_path.read_bytes()
    store.add_edges_from_wikilinks("c.md", ["d.md"])
    store.save()
    second = graph_path.read_bytes()
    assert first != second
    g = pickle.loads(second)
    assert set(g.nodes) >= {"a.md", "b.md", "c.md", "d.md"}
    assert not list(graph_path.parent.glob(f".{graph_path.name}.*.tmp"))


N_PROCS = 8
OPS_PER_PROC = 20

_WORKER_SCRIPT = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    from alfred.store.graph import GraphStore

    graph_path, tid, ops_per_proc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    for i in range(ops_per_proc):
        store = GraphStore(Path(graph_path))
        with store.transaction():
            store.load()
            store.add_edges_from_wikilinks(
                f"p{tid}/note-{i:03d}.md", [f"p{tid}/target-{i % 7}.md"]
            )
            store.save()
    """
)


def test_concurrent_real_subprocesses_no_lost_writes_no_crash(graph_path, tmp_path):
    """Reproduces the cross-process gap: `alfred up --only <daemon>` lets
    surveyor and consolidator run as genuinely separate OS processes against
    the same graph file, each independently constructing GraphStore(path)
    and calling transaction(): load(); mutate(); save(). threading.RLock is
    process-local and does nothing here — before the flock sidecar lock and
    unique per-call temp filename, this crashed with FileNotFoundError on
    os.replace() (processes collided on the fixed "<path>.tmp" name) and
    silently lost the majority of writes to last-writer-wins clobbering.
    """
    seed = GraphStore(graph_path)
    seed.add_edges_from_wikilinks("seed.md", ["seed-target.md"])
    seed.save()

    script_path = tmp_path / "graph_worker.py"
    script_path.write_text(_WORKER_SCRIPT)

    procs = [
        subprocess.Popen(
            [sys.executable, str(script_path), str(graph_path), str(tid), str(OPS_PER_PROC)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for tid in range(N_PROCS)
    ]

    failures = []
    for tid, p in enumerate(procs):
        out, _ = p.communicate(timeout=120)
        if p.returncode != 0:
            failures.append(f"process {tid} exited {p.returncode}:\n{out}")

    assert not failures, "subprocess crash(es) — the exact FileNotFoundError/os.replace race this test guards against:\n" + "\n\n".join(failures)

    g = pickle.loads(graph_path.read_bytes())
    expected = {
        f"p{tid}/note-{i:03d}.md" for tid in range(N_PROCS) for i in range(OPS_PER_PROC)
    }
    missing = expected - set(g.nodes)
    assert not missing, (
        f"lost writes across real OS processes — {len(missing)}/{len(expected)} nodes missing, "
        f"e.g. {sorted(missing)[:5]}"
    )

    assert not list(graph_path.parent.glob(f".{graph_path.name}.*.tmp"))


def test_nested_transaction_same_thread_same_instance_no_deadlock(graph_path):
    """Regression: transaction() opens a brand-new fd and flock()s it on
    every call. flock(2) is scoped to the open-file-description that took
    it, not the process or thread — so a naive re-entry (nested
    `with store.transaction():` on the same thread, same instance) opens a
    *second* fd whose flock() blocks forever on the lock the outer fd
    already holds, a self-deadlock. Runs the nested calls in a background
    daemon thread with a hard join timeout so a regression fails fast
    instead of hanging the whole test suite.
    """
    store = GraphStore(graph_path)
    completed = threading.Event()

    def body() -> None:
        with store.transaction():
            store.load()
            with store.transaction():  # nested — same thread, same instance
                store.add_edges_from_wikilinks("a.md", ["b.md"])
            store.save()
        completed.set()

    t = threading.Thread(target=body, daemon=True)
    t.start()
    t.join(timeout=10)
    assert completed.is_set(), (
        "nested transaction() on the same thread/instance hung — "
        "flock self-deadlock regression"
    )

    g = pickle.loads(graph_path.read_bytes())
    assert set(g.nodes) >= {"a.md", "b.md"}


def test_nested_transaction_same_thread_different_instances_no_deadlock(graph_path):
    """Same regression, but the nested call comes from a second,
    independently-constructed GraphStore(path) — the shape surveyor.py /
    consolidator.py actually use (each daemon builds its own instance). The
    module-level _path_locks registry already shares the RLock across
    instances for the same path; the flock reentry tracking must be shared
    the same way, keyed by resolved path rather than by instance.
    """
    outer = GraphStore(graph_path)
    completed = threading.Event()

    def body() -> None:
        with outer.transaction():
            outer.load()
            inner = GraphStore(graph_path)
            with inner.transaction():  # nested — same thread, different instance
                inner.load()
                inner.add_edges_from_wikilinks("c.md", ["d.md"])
                inner.save()
        completed.set()

    t = threading.Thread(target=body, daemon=True)
    t.start()
    t.join(timeout=10)
    assert completed.is_set(), (
        "nested transaction() across independently-constructed instances on "
        "the same thread hung — flock self-deadlock regression"
    )

    g = pickle.loads(graph_path.read_bytes())
    assert set(g.nodes) >= {"c.md", "d.md"}


def test_different_thread_still_blocks_not_a_reentrancy_bypass(graph_path):
    """Guard against an over-eager fix that makes flock reentry global
    instead of per-thread: a second, genuinely different thread must still
    be blocked out while the first thread's transaction() is open — it must
    NOT be treated as already holding the lock just because some thread does.
    The cross-process fix from c4a4a35 depends on this: real concurrent
    holders (other threads, other processes) must still serialize.
    """
    store = GraphStore(graph_path)
    order: list[str] = []
    first_in = threading.Event()
    let_first_finish = threading.Event()

    def first() -> None:
        with store.transaction():
            order.append("first-enter")
            first_in.set()
            let_first_finish.wait(timeout=10)
            order.append("first-exit")

    def second() -> None:
        first_in.wait(timeout=10)
        with store.transaction():
            order.append("second-enter")

    t1 = threading.Thread(target=first, daemon=True)
    t2 = threading.Thread(target=second, daemon=True)
    t1.start()
    first_in.wait(timeout=10)
    t2.start()

    # second() must still be blocked — first() hasn't released yet.
    t2.join(timeout=1)
    assert t2.is_alive(), (
        "second thread entered transaction() while the first thread still "
        "held it — flock reentrancy tracking leaked across threads"
    )

    let_first_finish.set()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert not t1.is_alive() and not t2.is_alive()
    assert order == ["first-enter", "first-exit", "second-enter"]
