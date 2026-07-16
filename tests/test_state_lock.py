"""Concurrency test for StateStore's cross-process file lock + merge-on-save.

state.json is constructed independently at up to six call sites (runner.py,
query/engine.py, cli.py, mcp/meta_server.py, mcp/server.py,
mcp/server_http.py) — the daemon and any long-lived MCP/CLI query process
each hold their own in-memory StateStore built from whatever state.json
looked like at their own load() time. Without a real cross-process lock plus
a merge (not just "last raw dict wins"), a long-lived query process that
loaded stale and later saves after a trivial mutation (e.g. an API-budget
counter bump) can silently overwrite newer daemon writes — new embeds,
cluster state, API budget counters.

These tests reproduce that shape with independently-constructed StateStore
instances racing save() and confirm no writes are silently lost.
"""
from __future__ import annotations

import threading

import pytest

from alfred.core.models import ClusterState, FileState
from alfred.store.state import StateStore


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / "state.json"


def test_concurrent_independent_instances_no_lost_writes(state_path):
    """Mirrors how runner.py / cli.py / the mcp servers each independently
    construct StateStore(cfg.state_path) for the same file. Many threads each
    build their own instance, load, add a distinct file entry, and save
    repeatedly — every writer's every file must survive on disk."""
    seed = StateStore(state_path)
    seed.load()
    seed.save()

    n_writers = 8
    ops_per_writer = 40
    errors: list[BaseException] = []

    def worker(tid: int) -> None:
        try:
            for i in range(ops_per_writer):
                store = StateStore(state_path)
                store.load()
                rel = f"t{tid}/note-{i:03d}.md"
                store.state.files[rel] = FileState(md5=f"m{tid}-{i}", last_embedded="2026-07-16")
                store.save()
        except BaseException as e:  # noqa: BLE001 — collect everything for the assertion
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(n_writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"exceptions during concurrent hammering: {errors!r}"

    final = StateStore(state_path)
    final.load()
    expected = {
        f"t{tid}/note-{i:03d}.md" for tid in range(n_writers) for i in range(ops_per_writer)
    }
    missing = expected - set(final.state.files.keys())
    assert not missing, f"lost writes — {len(missing)} files missing, e.g. {sorted(missing)[:5]}"

    assert not state_path.with_name(state_path.name + ".tmp").exists()
    assert state_path.with_name(state_path.name + ".lock").exists()


def test_stale_query_process_save_does_not_clobber_daemon_writes(state_path):
    """The exact reported bug shape: a long-lived query-server StateStore
    loads once (caching its snapshot), then — much later, after the daemon
    has since written new embeds and cluster state and saved — the query
    server saves too, merely because it recorded an API call. The query
    server's save must not wipe out the daemon's writes."""
    daemon = StateStore(state_path)
    daemon.load()
    daemon.save()

    # Query server "starts up" and caches its snapshot before the daemon does
    # any of the work below — this is the stale, long-lived cache the bug
    # report describes.
    query_server = StateStore(state_path)
    query_server.load()

    # Daemon does real work: new embeds + cluster state, then saves.
    daemon.state.files["docs/a.md"] = FileState(md5="aaa", last_embedded="2026-07-16T00:00:00Z")
    daemon.state.files["docs/b.md"] = FileState(md5="bbb", last_embedded="2026-07-16T00:00:01Z")
    daemon.state.clusters["42"] = ClusterState(cluster_id=42, label=["topic"])
    daemon.save()

    # Query server, oblivious to any of that, does its own trivial mutation
    # (an API-budget bump) and saves — using its stale in-memory snapshot.
    query_server.record_api_call(input_tokens=100, output_tokens=20)
    query_server.save()

    final = StateStore(state_path)
    final.load()
    assert "docs/a.md" in final.state.files, "daemon's embed lost to query server's save"
    assert "docs/b.md" in final.state.files, "daemon's embed lost to query server's save"
    assert "42" in final.state.clusters, "daemon's cluster state lost to query server's save"
    assert final.state.api_calls_today == 1, "query server's own counter bump must still land"


def test_concurrent_api_counter_increments_all_recorded(state_path):
    """Several independent short-lived StateStore instances (as a query
    server or CLI invocation would construct) each record one API call and
    save concurrently — the additive counter merge must land every increment,
    not just whichever process saved last."""
    seed = StateStore(state_path)
    seed.load()
    seed.save()

    n_writers = 10
    errors: list[BaseException] = []

    def worker(_tid: int) -> None:
        try:
            store = StateStore(state_path)
            store.load()
            store.record_api_call(input_tokens=10, output_tokens=5)
            store.save()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(n_writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"exceptions during concurrent hammering: {errors!r}"

    final = StateStore(state_path)
    final.load()
    assert final.state.api_calls_today == n_writers, (
        f"expected {n_writers} recorded calls, got {final.state.api_calls_today} — "
        "a concurrent save clobbered another writer's counter bump"
    )
