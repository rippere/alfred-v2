"""A long-lived LanceDBStore must survive another process compacting and
pruning old versions (scripts/compact_vault.py runs daily).

Regression for 2026-11-15: after the 12:56 alfred-compact run, every surveyor
upsert failed with "Not found: .../_versions/<n>.manifest" (567 in 30 min)
because the daemon's handle stayed pinned to the version it had opened at.
"""
from __future__ import annotations

import time
from datetime import timedelta

import lance

from alfred.store.lancedb_store import LanceDBStore


def _row(i: int) -> dict:
    return {"chunk_id": f"f.md::chunk_{i:02d}", "dense": [float(i)] * 4,
            "record_type": "note", "name": "f", "chunk_index": i}


def test_upsert_survives_external_compaction_and_version_cleanup(tmp_path):
    store = LanceDBStore(str(tmp_path / "lancedb"), collection="vault_v2", dims=4)
    for i in range(4):  # several versions so cleanup has something to prune
        store.upsert_many([_row(i)])

    path = str(tmp_path / "lancedb" / "vault_v2.lance")
    lance.dataset(path).optimize.compact_files()
    time.sleep(0.05)
    lance.dataset(path).cleanup_old_versions(older_than=timedelta(0))

    store.upsert_many([_row(9)])  # raised "Not found ... manifest" before the fix
    store.delete_file("f.md", ["f.md::chunk_00"])
    assert store.count() == 4
