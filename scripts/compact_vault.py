"""Scheduled Alfred vault LanceDB compaction (daily, via alfred-compact.timer).

Keeps the main store's fragment/version count bounded so query_all() and
searches stay fast and low-memory. Fixes the failure mode found 2026-09-02:
the store had grown to 25 GB / 66,079 versions / 5,414 fragments because the
surveyor commits one Lance version per file and nothing ever compacted, so
every query tried to materialize the whole thing and OOM-froze the desktop.

Safe by design: compaction preserves every current row's id/vector/metadata
exactly (verified: 266,564 rows and identical top-k before/after); a killed or
conflicted run is a no-op because Lance commits atomically.

Strategy:
  1. Compact live, retrying on Lance's *retryable* commit conflict (the surveyor
     may be writing at the same time).
  2. If live retries are exhausted, briefly stop the writer (alfred.service),
     compact once, and ALWAYS restart it (try/finally) so a failure can never
     leave Alfred down.
  3. cleanup_old_versions() to reclaim disk.
"""
import subprocess
import sys
import time
from datetime import timedelta

import lance

PATH = "/home/rippere/alfred-v2/data/lancedb/vault_v2.lance"
LIVE_ATTEMPTS = 6


def _compact_once() -> None:
    lance.dataset(PATH).optimize.compact_files()


def _compact_live() -> bool:
    """Return True if compaction committed; raise on a non-conflict error."""
    for i in range(LIVE_ATTEMPTS):
        try:
            _compact_once()
            return True
        except OSError as e:
            if "conflict" in str(e).lower() and i < LIVE_ATTEMPTS - 1:
                time.sleep(3 * (i + 1))
                continue
            raise
    return False


def _ensure_vector_index() -> None:
    """Keep an IVF_PQ index on the vector column (searches use it via
    SEARCH_NPROBES in alfred.store.lancedb_store).

    Without it every query flat-scans all ~300k vectors: ~1.3 GB peak per
    process, times one alfred MCP server per Claude session — a main driver of
    the 2026-09 global OOM freezes. Build it if missing (~80 s); otherwise fold
    rows written since the last run into it. Rows not yet indexed are still
    found (Lance flat-scans the unindexed tail), so this is purely a cost fix.
    """
    for i in range(LIVE_ATTEMPTS):
        try:
            ds = lance.dataset(PATH)
            if any("vector" in idx["fields"] for idx in ds.list_indices()):
                ds.optimize.optimize_indices()
                print("alfred-compact: vector index optimized", flush=True)
            else:
                ds.create_index(
                    "vector", index_type="IVF_PQ", metric="cosine",
                    num_partitions=512, num_sub_vectors=96,
                )
                print("alfred-compact: vector index created", flush=True)
            return
        except OSError as e:
            if "conflict" in str(e).lower() and i < LIVE_ATTEMPTS - 1:
                time.sleep(3 * (i + 1))
                continue
            raise


def main() -> int:
    committed = False
    try:
        committed = _compact_live()
    except OSError as e:
        print(f"alfred-compact: live retries failed ({e}); stop-writer fallback", flush=True)

    if not committed:
        # Guaranteed-quiet path. try/finally: alfred.service ALWAYS comes back,
        # even if compaction raises.
        subprocess.run(["systemctl", "--user", "stop", "alfred.service"], check=False)
        try:
            _compact_once()
        finally:
            subprocess.run(["systemctl", "--user", "start", "alfred.service"], check=False)

    try:
        _ensure_vector_index()
    except Exception as e:
        # Non-fatal: searches fall back to a flat scan (correct, just heavier).
        print(f"alfred-compact: index warning (non-fatal): {e!r}", flush=True)

    try:
        lance.dataset(PATH).cleanup_old_versions(older_than=timedelta(hours=6))
    except Exception as e:
        print(f"alfred-compact: cleanup warning (non-fatal): {e!r}", flush=True)

    d = lance.dataset(PATH)
    print(
        f"alfred-compact ok rows={d.count_rows()} "
        f"frags={len(d.get_fragments())} vers={len(d.versions())}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
