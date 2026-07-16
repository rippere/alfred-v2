"""LanceDB vector store — drop-in replacement for MilvusStore.

LanceDB is an embedded columnar vector database that uses memory-mapped files
instead of a gRPC subprocess, so multiple readers can open the same database
simultaneously without file-lock contention.  This eliminates the daemon vs.
CLI lock conflict that existed with Milvus Lite.

Interface mirrors MilvusStore exactly so callers don't need to change.
"""
from __future__ import annotations

import fcntl
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

# Re-export SearchHit from the shared types module so importers don't have to change.
from alfred.store.types import SearchHit  # noqa: F401

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
        except Exception:
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
    except Exception:
        pass
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

        if collection in self._db.table_names():
            try:
                self._tbl = self._db.open_table(collection)
            except Exception as e:
                # Only auto-heal genuine corruption; re-raise transient/op errors
                # so they surface loudly instead of nuking a recoverable table.
                if not _looks_like_corruption(e):
                    raise
                self._tbl = self._quarantine_and_recreate(collection, e)
        else:
            self._tbl = self._db.create_table(
                collection,
                schema=self._schema,
                mode="create",
            )
            log.info("lancedb.table_created", name=collection)

    def _quarantine_and_recreate(self, collection: str, exc: Exception):
        """Move a corrupt table aside (reversible) and recreate it empty.

        Cross-process coordination: without a lock, two processes opening the
        same corrupted table both pass the corruption check and both decide to
        quarantine. Whichever loses the ``mode="create"`` race then crashes on
        "table already exists" instead of healing, and the two racing
        ``_MAX_QUARANTINES_PER_DAY`` reads/moves can double-count (or miscount)
        the breaker window. An flock on a sidecar lock file (alongside the
        table directory, not inside it — quarantine moves the table dir itself)
        serializes the *entire* detect-and-recreate sequence across processes.
        The loser blocks on the lock, and once it wakes up re-checks whether
        the table already opens cleanly (the winner having just recreated it)
        before doing anything destructive — if so it simply joins the winner's
        fresh table as a reader instead of re-quarantining or crashing.
        """
        base = Path(self.uri)
        lock_path = base / f".{collection}.quarantine.lock"
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                return self._quarantine_and_recreate_locked(collection, exc, base)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)

    def _quarantine_and_recreate_locked(self, collection: str, exc: Exception, base: Path):
        """Body of :meth:`_quarantine_and_recreate`, run while holding the lock.

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

    # ------------------------------------------------------------------
    # Read methods
    # ------------------------------------------------------------------

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
