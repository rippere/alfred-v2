"""LanceDB vector store — drop-in replacement for MilvusStore.

LanceDB is an embedded columnar vector database that uses memory-mapped files
instead of a gRPC subprocess, so multiple readers can open the same database
simultaneously without file-lock contention.  This eliminates the daemon vs.
CLI lock conflict that existed with Milvus Lite.

Interface mirrors MilvusStore exactly so callers don't need to change.
"""
from __future__ import annotations

import ctypes
import fcntl
import gc
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from alfred.core.failures import record_failure

log = structlog.get_logger()

# glibc hands freed heap back to the process's per-thread arenas, not to the
# OS, so a long delete sweep ratchets RSS upward even when every individual
# call is small — this box has 16 cores, so up to 128 arenas. That ratchet, not
# any single delete, is what reached 23 GB and got OOM-killed on 2026-08-07:
# three deletes committed fine while climbing and the fourth crossed the
# ceiling. Measured on a scratch table: malloc_trim(0) after each delete held
# growth to +31 MB over 8 calls versus +1,764 MB without it — 57x.
# (MALLOC_ARENA_MAX=2 achieves the same, but only if set before the process
# starts, so it cannot be applied from inside a running daemon.)
def _release_freed_memory() -> None:
    gc.collect()
    if not sys.platform.startswith("linux"):
        return
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass          # non-glibc (musl) — the sweep is still correct, just heavier

# Re-export SearchHit from the shared types module so importers don't have to change.
from alfred.store.types import SearchHit  # noqa: F401

# ANN probe settings for the IVF_PQ index that scripts/compact_vault.py
# maintains on the vector column. Without an index every query flat-scans all
# ~300k vectors (~1.3 GB peak, ~7 s); with it, ~0.3 GB and ~0.2 s. Measured
# 2026-09-24 against flat search: nprobes=128/512 partitions + refine 20 gives
# recall@24 = 0.97. Both are ignored on a table with no index (flat scan).
SEARCH_NPROBES = 128
SEARCH_REFINE_FACTOR = 20

# Substrings that mark a *corrupt table* (interrupted-write damage: zero-byte
# manifests, truncated fragments) as opposed to a transient/operational error
# (permissions, disk full, schema mismatch).  We auto-quarantine only on these;
# anything else re-raises so genuine faults stay loud.
_CORRUPTION_MARKERS = (
    "invalid range",
    "lanceerror(io)",
    "of size 0 bytes",
    "corrupt",
    "manifest",
)

# Circuit breaker: if we have to quarantine more than this many tables within a
# rolling 24h window, the corruption cause is persistent (read-only FS, a Lance
# bug, a hardware fault).  Silently recreating would shred the store and lose
# data on every restart, so we re-raise instead and let the watchdog's tier-3
# inbox alert page a human.
_MAX_QUARANTINES_PER_DAY = 3


def _looks_like_corruption(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _CORRUPTION_MARKERS)


# Journal lines that look like a process kill (OOM-killer, SIGKILL escalation,
# unit killed).  Used by the best-effort kill-context scan below so a
# quarantine alert arrives with its probable cause attached.
_KILL_PATTERN = r"oom.kill|out of memory|oom_reaper|signal=KILL|SIGKILL|code=killed"


def _recent_kill_context(hours: int = 48, max_lines: int = 20) -> str:
    """Best-effort journal scan for kill/OOM events preceding a corruption.

    Corrupt Lance tables are caused by a process dying mid-commit (OOM-killer,
    SIGKILL, or an unclean host shutdown — the 2026-05-27 incident was a hard
    power-off).  This grabs any kill-shaped journal lines from the last
    ``hours`` plus the recent boot boundaries (an unclean shutdown shows up
    only as a boot with no shutdown sequence), so the quarantine alert can be
    *diagnosed*, not just recovered from.

    Never raises and is bounded by subprocess timeouts — diagnostics must not
    break or stall the self-heal path.
    """
    import subprocess

    sections: list[str] = []
    scans = (
        ("user journal", ["journalctl", "--user", "--since", f"-{hours}h",
                          "-o", "short-iso", "--no-pager", "-q", "-g", _KILL_PATTERN]),
        ("kernel journal", ["journalctl", "-k", "--since", f"-{hours}h",
                            "-o", "short-iso", "--no-pager", "-q", "-g", _KILL_PATTERN]),
    )
    for label, cmd in scans:
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        except Exception as e:
            # Diagnostic section silently missing from the crash report, which
            # then reads as "no evidence of a kill" rather than "couldn't look".
            record_failure("lancedb.diagnostic_scan_failed", error=e, scan=label)
            continue
        if out:
            lines = out.splitlines()[-max_lines:]
            sections.append(f"[{label}]\n" + "\n".join(lines))
    # Boot boundaries: a crash/power-loss kill leaves no journal line at all —
    # it is visible only as a boot whose predecessor ended without a shutdown.
    try:
        boots = subprocess.run(
            ["journalctl", "--list-boots", "--no-pager", "-q"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if boots:
            sections.append("[recent boots]\n" + "\n".join(boots.splitlines()[-3:]))
    except Exception as e:
        record_failure("lancedb.diagnostic_scan_failed", error=e, scan="list-boots")
    return "\n\n".join(sections)


class LanceDBStore:
    """LanceDB-backed vector store with the same public interface as MilvusStore.

    Parameters
    ----------
    uri:
        Path to the LanceDB *directory* (not a file).  LanceDB manages its own
        internal layout inside this directory.
    collection:
        Table name inside the database (default "vault_v2").
    dims:
        Dimensionality of the dense embedding vectors (must match embed model).
    """

    def __init__(
        self,
        uri: str,
        collection: str = "vault_v2",
        dims: int = 768,
    ) -> None:
        import lancedb
        import pyarrow as pa

        self.uri = uri
        self.collection = collection
        self.dims = dims

        Path(uri).mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(uri)

        # Schema: id + vector + lightweight metadata columns
        self._schema = pa.schema([
            pa.field("id",           pa.string()),
            pa.field("vector",       pa.list_(pa.float32(), dims)),
            pa.field("record_type",  pa.string()),
            pa.field("name",         pa.string()),
            pa.field("chunk_index",  pa.int32()),
        ])

        # True if a corrupt table had to be quarantined + recreated empty during
        # this open.  runner.py checks this to invalidate embed state and force
        # a full re-embed (otherwise the surveyor sees no md5 diff and search
        # stays permanently empty).
        self.was_recreated = False

        # Cross-process coordination: without a lock, two processes constructing
        # a store for the same collection can both observe an in-between state
        # and race each other's mode="create" call.
        #
        # This covers two distinct races under one lock:
        #   1. Corrupted table: both processes pass `_looks_like_corruption` and
        #      both try to quarantine + recreate.
        #   2. Fresh/mid-quarantine table: `table_names()` briefly returns empty
        #      while the winner has moved the corrupt dir aside but not yet
        #      recreated it (or the collection simply doesn't exist yet). A
        #      second process observing that gap would take the unlocked
        #      `else: create_table(...)` branch and crash with "table already
        #      exists" when the winner's create_table lands a moment later.
        #
        # Locking *before* the table_names() check — and holding the lock
        # through whichever branch is taken — closes both: only one process at
        # a time can decide open_table vs. create_table, and losers re-check
        # table state once they acquire the lock rather than acting on a stale
        # observation.
        base = Path(self.uri)
        lock_path = base / f".{collection}.quarantine.lock"
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                if collection in self._db.table_names():
                    try:
                        self._tbl = self._db.open_table(collection)
                    except Exception as e:
                        # Only auto-heal genuine corruption; re-raise transient/op
                        # errors so they surface loudly instead of nuking a
                        # recoverable table.
                        if not _looks_like_corruption(e):
                            raise
                        self._tbl = self._quarantine_and_recreate_locked(collection, e, base)
                else:
                    self._tbl = self._db.create_table(
                        collection,
                        schema=self._schema,
                        mode="create",
                    )
                    log.info("lancedb.table_created", name=collection)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)

    def _quarantine_and_recreate_locked(self, collection: str, exc: Exception, base: Path):
        """Move a corrupt table aside (reversible) and recreate it empty.

        Called from :meth:`__init__` while already holding the collection's
        flock sidecar, so this never acquires its own lock.

        Re-detects corruption first: another process may have already won this
        race and recreated the table (or fixed it some other way) while this
        one was blocked waiting for the lock, in which case there is nothing
        left to quarantine — just open the now-healthy table as a reader.
        """
        import lancedb

        self._db = lancedb.connect(self.uri)
        try:
            tbl = self._db.open_table(collection)
        except Exception:
            pass
        else:
            log.info("lancedb.quarantine_lost_race_joined_winner", name=collection)
            # A sibling process just recreated this table empty; treat it the
            # same as if *we* had recreated it so callers (runner.py) still
            # invalidate embed state and force a full re-embed.
            self.was_recreated = True
            return tbl

        cutoff = datetime.now().timestamp() - 24 * 3600
        recent = [
            p for p in base.glob(".quarantine-corrupt-*")
            if p.is_dir() and p.stat().st_mtime >= cutoff
        ]
        if len(recent) >= _MAX_QUARANTINES_PER_DAY:
            log.error(
                "lancedb.quarantine_circuit_open",
                name=collection,
                recent_quarantines=len(recent),
                error=str(exc),
            )
            raise exc

        table_dir = base / f"{collection}.lance"
        # Unique dest: a bare second-resolution stamp can collide (two
        # corruptions in the same second), and shutil.move would then nest the
        # table inside the existing dir and miscount the breaker.  Suffix until
        # the path is free.
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = base / f".quarantine-corrupt-{stamp}"
        n = 1
        while dest.exists():
            dest = base / f".quarantine-corrupt-{stamp}-{n}"
            n += 1
        if table_dir.exists():
            shutil.move(str(table_dir), str(dest))
            # shutil.move preserves the source's mtime (a Lance table dir keeps
            # its creation time), so the circuit-breaker's 24h mtime window
            # would never see freshly-quarantined dirs.  Stamp it to now so the
            # breaker can actually count recent quarantines.
            os.utime(dest, None)
        log.error(
            "lancedb.table_quarantined",
            name=collection,
            quarantine=str(dest),
            error=str(exc),
        )
        self._alert_corruption(collection, dest, exc)

        # Reconnect so the table_names cache reflects the move, then recreate.
        import lancedb
        self._db = lancedb.connect(self.uri)
        tbl = self._db.create_table(collection, schema=self._schema, mode="create")
        self.was_recreated = True
        log.info("lancedb.table_recreated", name=collection)
        return tbl

    def _alert_corruption(self, collection: str, dest: Path, exc: Exception) -> None:
        """Best-effort inbox drop so an auto-healed corruption stays visible.

        Without this, silent self-heal would hide a recurring data-loss bug.
        """
        # Attach kill-context so the alert arrives diagnosed, not just healed.
        # _recent_kill_context never raises; a failed scan just yields "".
        kill_ctx = _recent_kill_context()
        if kill_ctx:
            log.info("lancedb.quarantine_kill_context", name=collection, context=kill_ctx)
        try:
            inbox = Path("/mnt/external/obsidian-vault/inbox")
            if not inbox.is_dir():
                return
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            note = inbox / f"alfred-lancedb-corruption-{stamp}.md"
            kill_section = (
                f"## Kill context (journal scan, last 48h)\n\n```\n{kill_ctx}\n```\n\n"
                if kill_ctx
                else "## Kill context\n\nJournal scan returned nothing.\n\n"
            )
            note.write_text(
                "# Alfred LanceDB table auto-quarantined\n\n"
                f"- Table: `{collection}`\n"
                f"- Quarantined to: `{dest}`\n"
                f"- Error: `{exc}`\n"
                "- Action: recreated empty; surveyor will re-embed the corpus.\n"
                "- Follow up: interrupted-write cause (OOM / SIGKILL mid-commit / "
                "unclean shutdown) — see kill context below, and run "
                "`scripts/alfred-oom-correlate.sh` for a 7-day correlation.\n\n"
                f"{kill_section}"
                "<!-- alfred:source lancedb_quarantine -->\n"
            )
        except Exception as e:
            # Critical: self-heal happened but the operator was never told. Surface
            # it loudly so a recurring data-loss bug isn't masked by a failed alert.
            log.warning("lancedb.corruption_alert_failed", collection=collection, error=str(e))

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    def upsert(
        self,
        chunk_id: str,
        dense: list[float],
        sparse: dict[int, float],
        record_type: str,
        name: str,
        chunk_index: int = 0,
    ) -> None:
        """Insert or update a single chunk.

        ``sparse`` is accepted for interface compatibility but not stored —
        this backend does no sparse retrieval; BM25Store serves only the
        separate ``bm25_only`` offline path.
        """
        import pyarrow as pa

        row = pa.table({
            "id":          [chunk_id],
            "vector":      pa.array([dense], type=pa.list_(pa.float32(), self.dims)),
            "record_type": [record_type],
            "name":        [name],
            "chunk_index": pa.array([chunk_index], type=pa.int32()),
        })

        # LanceDB upsert: merge_insert matches on "id" and updates or inserts.
        (
            self._tbl.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(row)
        )

    def upsert_many(self, rows: list[dict]) -> None:
        """Batch upsert — ONE Lance commit for the whole batch.

        The surveyor previously called :meth:`upsert` once per chunk, producing
        one manifest version per chunk (~2,600 per full re-embed).  Each commit
        is a fragile write window; if the process dies mid-commit Lance can be
        left with a zero-byte manifest that crash-loops the next open.  Batching
        per file cuts that churn ~50-100x and shrinks the corruption window
        proportionally.

        Each row dict needs: ``chunk_id``, ``dense``, ``record_type``, ``name``,
        ``chunk_index``.  ``sparse`` is accepted and ignored — this backend
        stores no sparse vectors (retrieval is dense-only).
        """
        if not rows:
            return
        import pyarrow as pa

        batch = pa.table({
            "id":          [r["chunk_id"] for r in rows],
            "vector":      pa.array(
                [r["dense"] for r in rows],
                type=pa.list_(pa.float32(), self.dims),
            ),
            "record_type": [r.get("record_type", "") for r in rows],
            "name":        [r.get("name", "") for r in rows],
            "chunk_index": pa.array(
                [r.get("chunk_index", 0) for r in rows], type=pa.int32()
            ),
        })
        (
            self._tbl.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(batch)
        )

    def delete_file(self, rel_path: str, chunk_ids: list[str] | None = None) -> None:
        """Delete all chunks belonging to *rel_path*.

        Uses ``chunk_ids`` when available (fast explicit delete); falls back to
        a prefix filter on the ``id`` column otherwise.
        """
        if chunk_ids:
            # Build a SQL IN clause — safe because chunk_ids come from our own
            # state store and never contain user input.
            ids_sql = ", ".join(f"'{cid}'" for cid in chunk_ids)
            self._tbl.delete(f"id IN ({ids_sql})")
        else:
            # Prefix match: id starts with rel_path + "::"
            safe = rel_path.replace("'", "\\'")
            self._tbl.delete(f"starts_with(id, '{safe}::')")

    # Ceiling for one delete_ids() sweep, in MB. Not a per-call peak — the cost
    # that matters is the RUNNING SUM, because glibc returns freed arena space
    # to the process, not the OS. 3.0 GB leaves a wide margin under the 23 GB
    # that OOM-killed this box, and _release_freed_memory() should keep a
    # compacted store nowhere near it.
    RSS_CEILING_MB = 3072.0

    @staticmethod
    def _rss_mb() -> float:
        """Current RSS in MB, or 0.0 where /proc is unavailable."""
        try:
            with open("/proc/self/status", "r") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024.0
        except OSError:
            pass
        return 0.0

    def delete_ids(self, chunk_ids: list[str], batch: int = 500) -> int:
        """Delete explicit chunk ids in bounded batches.  Returns ids submitted.

        Why this exists instead of one big ``id IN (...)``: on 2026-08-07 a
        bulk delete reached 23 GB RSS and was OOM-killed.  Measured cause is
        that DataFusion plans the predicate against *every* fragment, so peak
        memory is proportional to ``fragment_count x predicate_terms`` — about
        3.6e-4 MB per pair.  At the live store's 10,274 fragments a single
        3,700-term IN clause budgets ~13.7 GB before glibc arena growth
        ratchets it higher across successive calls.  500 terms keeps one call
        under ~2 GB even uncompacted, which is why the batch is small and not
        tunable upward without re-measuring.

        Ids containing an apostrophe are skipped rather than interpolated:
        ``_safe_chunk_id`` (core/vault.py) strips ``'`` when minting ids, so
        such a row cannot legitimately exist, and quoting it into SQL would be
        an injection rather than a delete.
        """
        submitted = 0
        pending: list[str] = []

        def _flush() -> bool:
            """Delete one batch. Returns False when the sweep must stop."""
            nonlocal submitted, pending
            self._delete_id_batch(pending)
            submitted += len(pending)
            pending = []
            # Batching bounds each call's PEAK, but the sweep pays the SUM:
            # nothing is released between calls, so a long sweep climbs even
            # though every individual delete is small. _delete_id_batch calls
            # malloc_trim, but if RSS still crosses the ceiling something is
            # retaining memory we don't understand — stop rather than march
            # toward another OOM. The ids not yet submitted stay in the store
            # and the next sweep retries them.
            rss = self._rss_mb()
            if rss and rss > self.RSS_CEILING_MB:
                record_failure(
                    "lancedb.delete_ids_rss_ceiling",
                    error=MemoryError(f"RSS {rss:.0f}MB > ceiling {self.RSS_CEILING_MB:.0f}MB"),
                    submitted=submitted,
                )
                log.warning(
                    "lancedb.delete_ids_aborted_rss",
                    rss_mb=round(rss, 1), ceiling_mb=self.RSS_CEILING_MB, submitted=submitted,
                )
                return False
            return True

        for cid in chunk_ids:
            if "'" in cid:
                record_failure(
                    "lancedb.delete_ids_quote_in_id", error=ValueError(cid), chunk_id=cid
                )
                continue
            pending.append(cid)
            if len(pending) >= batch and not _flush():
                return submitted
        if pending:
            _flush()
        return submitted

    def _delete_id_batch(self, ids: list[str]) -> None:
        ids_sql = ", ".join(f"'{cid}'" for cid in ids)
        self._tbl.delete(f"id IN ({ids_sql})")
        _release_freed_memory()

    # ------------------------------------------------------------------
    # Read methods
    # ------------------------------------------------------------------

    def iter_ids(self, batch_size: int = 4096):
        """Stream every stored chunk id, one at a time, id column only.

        Peak memory is set by the projection, NOT by row count: measured on
        the live store (157,233 rows / 10,274 fragments) this is ~301 MB peak
        and 2.1 s, versus 4,181 MB and 145 s for ``query_all()``, which calls
        ``to_arrow()`` and therefore materialises all 483 MB of vectors before
        its ``select()`` can drop them.  Projection pushdown was confirmed by
        an IO counter, not inferred: 49 MB read for the id column vs 465 MB
        with the vector column included.

        ``limit(None)`` is already the builder default, but it is spelled out
        so a future refactor cannot silently reintroduce the default-10 limit
        that ``search()`` applies on other paths.

        ``batch_size`` is a ceiling, not a memory knob — Lance will not merge
        a batch across fragments, so on the live store the effective batch is
        ~15 rows.  Marginal memory tracks fragment count (~13 KB/fragment),
        which is the number to watch if this ever grows.
        """
        reader = (
            self._tbl.search()
            .select(["id"])
            .limit(None)
            .to_batches(batch_size=batch_size)
        )
        for batch in reader:
            yield from batch.column(0).to_pylist()

    def search(
        self,
        dense_vec: list[float],
        sparse_vec: dict[int, float],
        top_k: int = 8,
        include_inbox: bool = False,
    ) -> list[SearchHit]:
        """Dense cosine search — retrieval on this backend is dense-only.

        ``sparse_vec`` is accepted for interface parity with MilvusStore but
        ignored: no sparse/BM25 component contributes to this ranking.  BM25
        is used only by the QueryEngine's separate ``bm25_only`` offline path.
        """
        results = (
            self._tbl
            .search(dense_vec, vector_column_name="vector")
            .metric("cosine")
            .nprobes(SEARCH_NPROBES)
            .refine_factor(SEARCH_REFINE_FACTOR)
            .limit(top_k * 3)
            .select(["id", "record_type", "name", "_distance"])
            .to_list()
        )

        hits: list[SearchHit] = []
        seen: set[str] = set()
        for r in results:
            chunk_id = r["id"]
            rel_path = chunk_id.rsplit("::", 1)[0]
            if not include_inbox and rel_path.startswith("inbox/"):
                continue
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            # LanceDB returns _distance (0 = identical for cosine); convert to
            # similarity score matching Milvus COSINE convention (1 = identical).
            distance = r.get("_distance", 0.0)
            score = float(1.0 - distance)
            hits.append(SearchHit(
                chunk_id=chunk_id,
                rel_path=rel_path,
                score=score,
                record_type=r.get("record_type", "") or "",
                name=r.get("name", "") or "",
            ))
            if len(hits) >= top_k:
                break

        return hits

    def query_all(self, output_fields: list[str] | None = None) -> list[dict[str, Any]]:
        """Return all rows.  Used by Surveyor for HDBSCAN clustering.

        Returns dicts with at least ``id`` and ``embedding`` keys to match the
        shape that Surveyor expects from the Milvus version.
        """
        cols = output_fields or ["id", "embedding", "record_type", "name"]

        # Translate "embedding" → "vector" for LanceDB column naming, then
        # rename back in the output so callers see "embedding" as before.
        lancedb_cols = ["vector" if c == "embedding" else c for c in cols]
        # Always include "id"
        if "id" not in lancedb_cols:
            lancedb_cols.insert(0, "id")

        rows = self._tbl.to_arrow().select(
            [c for c in lancedb_cols if c in self._tbl.schema.names]
        ).to_pylist()

        # Rename "vector" back to "embedding" if the caller asked for "embedding"
        if "embedding" in cols:
            for row in rows:
                if "vector" in row:
                    row["embedding"] = list(row.pop("vector"))
        return rows

    def count(self) -> int:
        """Return total number of stored chunks."""
        return self._tbl.count_rows()

    def close(self) -> None:
        """No-op for LanceDB (connection is not stateful)."""
        pass
