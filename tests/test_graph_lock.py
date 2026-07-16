"""Concurrency test for GraphStore's locking + atomic save (audit structural #1/#6a).

Threads hammer a single shared GraphStore (mutations, saves, traversals) while
reader threads concurrently unpickle the on-disk file. The lock must serialize
read-modify-write so no write is lost, and the tmp-file + os.replace() save
must guarantee readers never observe a truncated pickle.
"""
from __future__ import annotations

import pickle
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
    assert not graph_path.with_name(graph_path.name + ".tmp").exists()


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
    assert not graph_path.with_name(graph_path.name + ".tmp").exists()
