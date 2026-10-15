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


def test_clean_delete_with_no_concurrent_writer_actually_deletes(state_path):
    """Adversarial-verification regression: the three-way dict-field merge
    used to only overlay `mine`'s additions/changes onto `theirs`, never
    `mine`'s deletions relative to `base` — so a key removed by this
    instance (e.g. janitor ghost-file pruning) was silently resurrected on
    the very next save, even with zero concurrent writers. This is the exact
    reproduction from the bug report."""
    seed = StateStore(state_path)
    seed.load()
    seed.state.files["ghost.md"] = FileState(md5="dead", last_embedded="2026-07-16")
    seed.save()

    store2 = StateStore(state_path)
    store2.load()
    assert "ghost.md" in store2.state.files

    del store2.state.files["ghost.md"]
    store2.save()

    final = StateStore(state_path)
    final.load()
    assert final.state.files == {}, (
        f"deleted key wrongly resurrected — expected {{}}, got {final.state.files!r}"
    )


def test_long_lived_instance_second_save_persists_curator_processed(state_path):
    """Aliasing regression: save() decoded the merged dict back into memory
    while ALSO keeping that same dict as the next save's merge `base`.
    _decode_state passes `curator_processed` (and the two list fields)
    through by reference, so state.curator_processed and
    _base_raw["curator_processed"] were the SAME object — every later
    mutation retroactively rewrote the base it was about to be diffed
    against, so the merge saw "unchanged" and dropped the write.

    This is the daemon's shape, not the tests above: ONE long-lived store
    that saves more than once. The curator dedup guard reads
    curator_processed to decide whether a file was already handled, so
    dropping these writes means silent reprocessing."""
    store = StateStore(state_path)
    store.load()
    store.save()  # establishes _base_raw aliased to the in-memory state

    store.state.curator_processed["inbox/note.md"] = "2026-08-06T00:00:00Z"
    store.state.distiller_runs.append({"run": 1, "files": 3})
    store.save()

    final = StateStore(state_path)
    final.load()
    assert final.state.curator_processed == {"inbox/note.md": "2026-08-06T00:00:00Z"}, (
        f"curator_processed write silently dropped — got "
        f"{final.state.curator_processed!r}"
    )
    assert final.state.distiller_runs == [{"run": 1, "files": 3}], (
        f"distiller_runs append silently dropped — got {final.state.distiller_runs!r}"
    )


def test_long_lived_instance_repeated_saves_accumulate(state_path):
    """Same aliasing defect across many saves on one instance. files/clusters
    /memory/wiki_pages are rebuilt into fresh dataclasses by _decode_state so
    they were never aliased, but curator_processed was, and nothing tested
    it. Ten sequential saves must leave ten entries on disk."""
    store = StateStore(state_path)
    store.load()

    for i in range(10):
        store.state.curator_processed[f"inbox/n{i}.md"] = f"2026-08-06T00:00:{i:02d}Z"
        store.state.files[f"docs/n{i}.md"] = FileState(md5=f"m{i}", last_embedded="2026-08-06")
        store.save()

    final = StateStore(state_path)
    final.load()
    assert len(final.state.curator_processed) == 10, (
        f"expected 10 curator_processed entries, got "
        f"{len(final.state.curator_processed)}: {final.state.curator_processed!r}"
    )
    assert len(final.state.files) == 10, f"expected 10 files, got {len(final.state.files)}"


def test_concurrent_edit_wins_over_stale_delete(state_path):
    """Genuine concurrent conflict: instance A deletes key K while instance B
    independently modifies K's value and saves first. Per the documented
    resolution in _merge_dict_field, a live concurrent edit wins over a
    stale delete (the deleting instance's `base` no longer reflects reality
    for that key), so K must survive with B's value."""
    seed = StateStore(state_path)
    seed.load()
    seed.state.files["shared.md"] = FileState(md5="orig", last_embedded="2026-07-16")
    seed.save()

    instance_a = StateStore(state_path)
    instance_a.load()
    instance_b = StateStore(state_path)
    instance_b.load()

    # A deletes the key based on its (now stale) loaded snapshot.
    del instance_a.state.files["shared.md"]

    # B independently changes the key's value and saves first.
    instance_b.state.files["shared.md"] = FileState(md5="updated", last_embedded="2026-07-16T01:00:00Z")
    instance_b.save()

    # A's delete-based-on-stale-base then saves.
    instance_a.save()

    final = StateStore(state_path)
    final.load()
    assert "shared.md" in final.state.files, (
        "concurrent edit was wrongly discarded in favor of a stale delete"
    )
    assert final.state.files["shared.md"].md5 == "updated", (
        "expected theirs's (B's) concurrent edit to win over A's stale delete"
    )


def test_reference_held_across_save_keeps_persisting(state_path):
    """Detached-state regression: save() rebound self._state to a freshly
    decoded PipelineState, so every object a daemon picked up before an await
    — `state = self.state.state`, a FileState from state.files — stopped being
    the one that gets saved the moment ANY other job saved (the 5-min periodic
    save, a surveyor tick). The distiller's last_distilled stamps and its
    distiller_runs entry went into those orphans, which is why its runs were
    never recorded and runner.py's catch-up re-swept after every restart.

    The tests above never hit it: they all re-read `store.state` after a save."""
    store = StateStore(state_path)
    store.load()
    store.state.files["note.md"] = FileState(md5="m1")
    store.state.clusters["semantic_0"] = ClusterState(cluster_id=0, member_files=["note.md"])
    store.save()

    # What a daemon pass captures before its first await.
    held = store.state
    fs = held.files["note.md"]
    cluster = held.clusters["semantic_0"]

    store.save()  # another job saves while the pass is awaiting

    fs.last_distilled = "2026-09-24T09:00:00+00:00"
    cluster.label = ["held label"]
    held.distiller_runs.append({"timestamp": "2026-09-24T09:00:00+00:00"})
    store.save()

    assert held is store.state, "save() replaced the live PipelineState"

    final = StateStore(state_path)
    final.load()
    assert final.state.files["note.md"].last_distilled == "2026-09-24T09:00:00+00:00"
    assert final.state.clusters["semantic_0"].label == ["held label"]
    assert final.state.distiller_runs == [{"timestamp": "2026-09-24T09:00:00+00:00"}]


def test_save_still_folds_in_another_writers_changes(state_path):
    """Keeping object identity must not cost the merge: another process's new
    entry, edit and delete all show up in this instance's live objects."""
    seed = StateStore(state_path)
    seed.load()
    seed.state.files["kept.md"] = FileState(md5="old")
    seed.state.files["gone.md"] = FileState(md5="bye")
    seed.save()

    daemon = StateStore(state_path)
    daemon.load()
    held_kept = daemon.state.files["kept.md"]

    other = StateStore(state_path)
    other.load()
    other.state.files["kept.md"].md5 = "new"
    del other.state.files["gone.md"]
    other.state.files["added.md"] = FileState(md5="hi")
    other.save()

    daemon.save()

    assert daemon.state.files["kept.md"] is held_kept
    assert held_kept.md5 == "new"
    assert "gone.md" not in daemon.state.files
    assert daemon.state.files["added.md"].md5 == "hi"
