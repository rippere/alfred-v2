"""SurveyorDaemon — watch → embed → cluster → graph → update state."""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from alfred.core.failures import record_failure
from alfred.core.provenance import is_daemon_generated_raw
from alfred.embed.ollama import EmbeddingBackendUnavailable
from alfred.core.vault import VaultRecord, chunk_record, is_sync_conflict, parse_file
from alfred.daemons.base import BaseDaemon, DaemonEvent

if TYPE_CHECKING:
    from alfred.config import AlfredConfig
    from alfred.store.lancedb_store import LanceDBStore
    from alfred.store.state import StateStore

WATCH_INTERVAL = 60.0      # seconds between filesystem polls

# Files embedded between state.json persists inside one diff. Vectors commit
# per file, so this bounds how many committed-but-unrecorded files a crash can
# strand as orphans. Small enough to keep the window tight, large enough that
# a 7,000-file catch-up doesn't rewrite state.json once per file.
STATE_SAVE_EVERY = 25


def _chunk_rank(chunk_id: str) -> tuple[int, str]:
    """Sort key for a chunk id ("<rel_path>::chunk_NN"): chunk number first."""
    suffix = chunk_id.rsplit("::", 1)[-1]
    digits = "".join(ch for ch in suffix if ch.isdigit())
    return (int(digits) if digits else 0, suffix)


class SurveyorDaemon(BaseDaemon):
    name = "surveyor"

    def __init__(self, cfg, state, events, store: LanceDBStore) -> None:
        super().__init__(cfg, state, events)
        self.store = store
        self._embedder = None
        self._bm25 = None

    def _get_embedder(self):
        if self._embedder is None:
            from alfred.embed.ollama import OllamaEmbedder
            self._embedder = OllamaEmbedder(self.cfg.ollama_base_url, self.cfg.ollama_embed_model)
        return self._embedder

    def _get_bm25(self):
        if self._bm25 is None:
            from alfred.store.bm25 import BM25Store
            store = BM25Store(self.cfg.bm25_path)
            store.load()
            self._bm25 = store
        return self._bm25

    async def run(self) -> None:
        self.log.info("surveyor.start")
        try:
            while not self._stop.is_set():
                await self._tick()
                await asyncio.sleep(WATCH_INTERVAL)
        finally:
            if self._embedder:
                await self._embedder.close()
            await self.save_state()
            self.log.info("surveyor.stopped")

    async def tick(self) -> None:
        """One-shot poll — called by APScheduler every WATCH_INTERVAL seconds."""
        try:
            await self._tick()
        except Exception as e:
            self.log.error("surveyor.tick_error", error=str(e))

    async def recluster(self) -> None:
        """One-shot recluster — the sole trigger for `_recluster()`, called by
        the dedicated `surveyor.recluster` APScheduler job (see runner.py)."""
        try:
            await self._recluster()
        except Exception as e:
            self.log.error("surveyor.recluster_error", error=str(e))

    async def teardown(self) -> None:
        """Clean shutdown: flush embedder and save state."""
        if self._embedder:
            await self._embedder.close()
        await self.save_state()

    async def _tick(self) -> None:
        diff = self._compute_diff()
        if diff["new"] or diff["changed"] or diff["deleted"]:
            await self._process_diff(diff)
            await self.save_state()

    def _compute_diff(self) -> dict[str, list[str]]:
        """Scan vault for new/changed/deleted files."""
        vault_path = self.cfg.vault_path
        ignore = set(self.cfg.ignore_dirs)
        current: dict[str, str] = {}

        for md_file in vault_path.rglob("*.md"):
            rel = md_file.relative_to(vault_path)
            if any(part in ignore for part in rel.parts):
                continue
            if is_sync_conflict(md_file):
                continue
            rel_str = str(rel).replace("\\", "/")
            try:
                raw = md_file.read_bytes()
                # Skip LLM-generated files — they must not feed back into the index
                if is_daemon_generated_raw(raw):
                    continue
                current[rel_str] = hashlib.md5(raw).hexdigest()
            except OSError as e:
                # An unreadable file drops silently out of `current`, which the
                # caller reads as "this file no longer exists" — a transient
                # read error is indistinguishable from a deletion.
                record_failure("surveyor.file_read_failed", error=e, path=rel_str)
                continue

        known = self.state.state.files
        new = [r for r in current if r not in known]
        changed = [r for r in current if r in known and current[r] != known[r].md5]
        deleted = [r for r in known if r not in current]
        return {"new": new, "changed": changed, "deleted": deleted, "current": current}

    async def _process_diff(self, diff: dict) -> None:
        current = diff["current"]
        bm25 = self._get_bm25()
        embedder = self._get_embedder()
        state = self.state.state
        # Vectors are committed to the store per file (upsert_many, below), but
        # state was persisted only once, after the entire diff. Any death in
        # between — OOM, SIGKILL, power loss — left committed vectors with no
        # state.files entry, and nothing can find them again: janitor's ghost
        # sweep iterates state.files, not the store, so an orphan whose
        # rel_path never reached state is invisible to every existing sweep.
        # Persisting in bounded batches makes that window STATE_SAVE_EVERY
        # files instead of the whole diff.
        since_save = 0

        # Delete removed files
        for rel_path in diff["deleted"]:
            fs = state.files.get(rel_path)
            chunk_ids = fs.chunk_ids if fs else None
            try:
                self.store.delete_file(rel_path, chunk_ids)
            except Exception as e:
                # Keep the state entry. Popping it after a FAILED delete is how
                # a transient store error becomes a permanent orphan: the
                # vectors are still in the store, but the only record of their
                # chunk_ids is gone, and janitor's ghost sweep iterates
                # state.files — so nothing can ever enumerate them again.
                # Retaining the entry means the next tick retries the delete.
                self.log.warning("surveyor.delete_failed", path=rel_path, error=str(e))
                continue
            state.files.pop(rel_path, None)
            self.log.info("surveyor.deleted", path=rel_path)

        # Embed new + changed
        from alfred.core.models import FileState
        from datetime import datetime, timezone

        for rel_path in diff["new"] + diff["changed"]:
            vault_file = self.cfg.vault_path / rel_path
            if not vault_file.exists():
                continue
            try:
                record = parse_file(self.cfg.vault_path, rel_path)
                chunks = chunk_record(record)
            except Exception as e:
                self.log.warning("surveyor.parse_failed", path=rel_path, error=str(e))
                continue

            chunk_ids: list[str] = []
            rows: list[dict] = []
            try:
                for chunk_id, text in chunks:
                    # Sanitize chunk_id to avoid Milvus apostrophe bug
                    safe_id = chunk_id.replace("'", "’")
                    dense = await embedder.embed(text)
                    if dense is None:
                        continue
                    sparse = bm25.encode(text) if bm25.is_fitted else {}
                    rows.append({
                        "chunk_id": safe_id,
                        "dense": dense,
                        "sparse": sparse,
                        "record_type": record.record_type,
                        "name": record.frontmatter.get("name", rel_path),
                        "chunk_index": len(chunk_ids),
                    })
                    chunk_ids.append(safe_id)
            except EmbeddingBackendUnavailable as e:
                # Abandon the whole tick before the delete-and-record block
                # below. Falling through would read an empty `rows` as "this
                # file has no embeddable content", delete its existing vectors
                # as stale, and write FileState(md5=current, chunk_ids=[]) —
                # after which the md5 matches and _compute_diff never revisits
                # it. That is silent, permanent removal from search.
                #
                # Returning (not continuing) leaves every file this tick has
                # not reached untouched in state, so the next tick redoes the
                # remainder. Files already committed above keep their state:
                # _tick still saves.
                self.log.warning(
                    "surveyor.embedder_unavailable",
                    error=str(e),
                    deferred_from=rel_path,
                    indexed_before_stop=len(state.files),
                )
                return
            # One batched commit per file instead of one per chunk — collapses
            # ~N manifest writes into a single Lance commit, shrinking the
            # interrupted-write corruption window that crash-looped the daemon.
            if rows:
                try:
                    self.store.upsert_many(rows)
                except Exception as e:
                    self.log.warning(
                        "surveyor.upsert_failed",
                        path=rel_path, count=len(rows), error=str(e),
                    )
                    # Leave state.files[rel_path] (and its chunk_ids) untouched
                    # so the old md5 is preserved and the file is retried next
                    # tick.  Deleting the old chunks is deferred until *after*
                    # a successful upsert (below) specifically so this failure
                    # path never leaves state pointing at chunk_ids that have
                    # already been removed from the vector store — recording
                    # md5=current here, or deleting the old chunks up front,
                    # would silently drop the file from search while state
                    # still reports it as indexed.
                    continue
            # (rows empty → file has no embeddable content; fall through and
            # record state so we don't retry an empty file forever.)

            # Only remove chunks from the *previous* version of this file once
            # the new ones are confirmed written (or confirmed unnecessary,
            # for the empty-rows case above) — never before, so a failed
            # upsert can't leave state referencing chunk_ids that no longer
            # exist in the store. chunk_ids are deterministic per index
            # (rel_path::chunk_NN), so any id shared with the just-written
            # `chunk_ids` was already refreshed by upsert_many's merge_insert
            # — only the leftover ids (e.g. the file got shorter) are stale
            # and need an explicit delete.
            if rel_path in diff["changed"]:
                old_fs = state.files.get(rel_path)
                if old_fs:
                    stale_ids = [cid for cid in old_fs.chunk_ids if cid not in chunk_ids]
                    if stale_ids:
                        try:
                            self.store.delete_file(rel_path, stale_ids)
                        except Exception as e:
                            self.log.warning(
                                "surveyor.delete_failed", path=rel_path, error=str(e),
                            )

            now = datetime.now(timezone.utc).isoformat()
            state.files[rel_path] = FileState(
                md5=current[rel_path],
                last_embedded=now,
                chunk_ids=chunk_ids,
                semantic_cluster_id=state.files.get(rel_path, FileState(md5="")).semantic_cluster_id,
            )
            # Update graph edges
            try:
                from alfred.store.graph import GraphStore
                graph = GraphStore(self.cfg.graph_path)
                with graph.transaction():
                    graph.load()
                    graph.add_edges_from_wikilinks(rel_path, record.wikilinks)
                    graph.save()
            except Exception as e:
                # The file stays indexed but its wikilinks never reach the
                # graph, so related-note lookups quietly degrade with nothing
                # anywhere indicating why.
                record_failure("surveyor.graph_update_failed", error=e, path=rel_path)

            self.log.info("surveyor.embedded", path=rel_path, chunks=len(chunk_ids))

            # Bound the vectors-committed-but-state-unsaved window. Ordering
            # matters: this runs AFTER state.files[rel_path] is assigned, so a
            # save here always covers every file whose vectors are already in
            # the store.
            since_save += 1
            if since_save >= STATE_SAVE_EVERY:
                await self.save_state()
                since_save = 0

        if since_save:
            await self.save_state()

        self.emit("files_embedded", paths=diff["new"] + diff["changed"])

    async def _recluster(self) -> None:
        """HDBSCAN clustering over current embeddings.

        The vector-store query_all (~48s for 4k vectors) and HDBSCAN run in a
        thread pool so the event loop stays responsive for other daemons.
        """
        try:
            import numpy as np
            from sklearn.cluster import HDBSCAN
        except ImportError as e:
            self.log.warning("surveyor.cluster_skip", reason=str(e))
            return

        min_cluster_size = self.cfg.hdbscan_min_cluster_size
        min_samples = self.cfg.hdbscan_min_samples
        store = self.store

        def _compute() -> tuple[dict[int, list[str]], list[tuple[str, int]]] | None:
            rows = store.query_all(output_fields=["id", "embedding"])
            if not rows:
                return None

            # One vector per file — its lowest-numbered chunk — in path order,
            # never in the store's row order. HDBSCAN's output depends on input
            # order: the neuro vault's own vectors, shuffled, came back with 13
            # of 30 clusters intact and nearly every key renumbered (measured
            # 2026-09-24). Re-embeds rewrite rows, so the order drifted between
            # passes and the consolidator saw "changed" membership, and paid an
            # LLM call to relabel it, for clusters nobody had touched.
            seen: dict[str, tuple[tuple[int, str], list[float]]] = {}
            for r in rows:
                rel_path = r["id"].rsplit("::", 1)[0]
                rank = _chunk_rank(r["id"])
                if rel_path not in seen or rank < seen[rel_path][0]:
                    seen[rel_path] = (rank, r["embedding"])

            paths = sorted(seen)
            vectors = np.array([seen[p][1] for p in paths], dtype=np.float32)

            if len(paths) < min_cluster_size:
                return None

            labels = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="cosine",
                # Pin current (False) behaviour; the default flips to True in
                # sklearn 1.10 and otherwise emits a FutureWarning every cluster.
                copy=False,
            ).fit_predict(vectors)

            cluster_members: dict[int, list[str]] = {}
            path_labels: list[tuple[str, int]] = []
            for path, cid in zip(paths, labels):
                cid_int = int(cid)
                path_labels.append((path, cid_int))
                if cid_int != -1:
                    cluster_members.setdefault(cid_int, []).append(path)

            return cluster_members, path_labels

        try:
            result = await asyncio.to_thread(_compute)
            if result is None:
                return
            cluster_members, path_labels = result
        except Exception as e:
            self.log.error("surveyor.cluster_failed", error=str(e))
            return

        state = self.state.state
        from alfred.core.models import ClusterState

        for path, cid_int in path_labels:
            if path in state.files:
                state.files[path].semantic_cluster_id = cid_int

        for cid, members in cluster_members.items():
            key = f"semantic_{cid}"
            existing = state.clusters.get(key)
            if existing is None:
                state.clusters[key] = ClusterState(
                    cluster_id=cid,
                    cluster_type="semantic",
                    member_files=members,
                )
            else:
                # In place: the label/synthesis bookkeeping rides along, and a
                # consolidator pass holding this object keeps writing to the
                # one that gets saved.
                existing.cluster_id = cid
                existing.cluster_type = "semantic"
                existing.member_files = members

        try:
            from alfred.store.graph import GraphStore
            graph = GraphStore(self.cfg.graph_path)
            with graph.transaction():
                graph.load()
                cleared = graph.clear_cluster_edges()
                self.log.debug("surveyor.cluster_edges_cleared", count=cleared)
                for members in cluster_members.values():
                    graph.add_cluster_edges(members)
                graph.save()
        except Exception as e:
            self.log.warning("surveyor.graph_update_failed", error=str(e))

        self.emit("clusters_updated", cluster_count=len(cluster_members))
        self.log.info("surveyor.clustered", clusters=len(cluster_members), files=len(path_labels))
