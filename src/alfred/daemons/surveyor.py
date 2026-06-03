"""SurveyorDaemon — watch → embed → cluster → graph → update state."""
from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING

from alfred.core.vault import VaultRecord, chunk_record, parse_file
from alfred.daemons.base import BaseDaemon, DaemonEvent
from alfred.store.milvus import MilvusStore

if TYPE_CHECKING:
    from alfred.config import AlfredConfig
    from alfred.store.state import StateStore

WATCH_INTERVAL = 60.0      # seconds between filesystem polls
CLUSTER_INTERVAL = 1800.0  # re-cluster every 30 minutes


class SurveyorDaemon(BaseDaemon):
    name = "surveyor"

    def __init__(self, cfg, state, events, milvus: MilvusStore) -> None:
        super().__init__(cfg, state, events)
        self.milvus = milvus
        self._embedder = None
        self._bm25 = None
        self._last_cluster = float("-inf")

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
        """One-shot recluster — called by APScheduler on CLUSTER_INTERVAL."""
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

        if time.time() - self._last_cluster > CLUSTER_INTERVAL:
            await self._recluster()
            self._last_cluster = time.time()

    def _compute_diff(self) -> dict[str, list[str]]:
        """Scan vault for new/changed/deleted files."""
        vault_path = self.cfg.vault_path
        ignore = set(self.cfg.ignore_dirs)
        current: dict[str, str] = {}

        for md_file in vault_path.rglob("*.md"):
            rel = md_file.relative_to(vault_path)
            if any(part in ignore for part in rel.parts):
                continue
            rel_str = str(rel).replace("\\", "/")
            try:
                raw = md_file.read_bytes()
                # Skip LLM-generated files — they must not feed back into the index
                if b"generated_by: llm" in raw or b"generated_by: \"llm\"" in raw:
                    continue
                current[rel_str] = hashlib.md5(raw).hexdigest()
            except OSError:
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

        # Delete removed files
        for rel_path in diff["deleted"]:
            fs = state.files.get(rel_path)
            chunk_ids = fs.chunk_ids if fs else None
            try:
                self.milvus.delete_file(rel_path, chunk_ids)
            except Exception as e:
                self.log.warning("surveyor.delete_failed", path=rel_path, error=str(e))
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

            # Delete old chunks before re-embedding
            if rel_path in diff["changed"]:
                old_fs = state.files.get(rel_path)
                if old_fs:
                    self.milvus.delete_file(rel_path, old_fs.chunk_ids)

            chunk_ids: list[str] = []
            rows: list[dict] = []
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
            # One batched commit per file instead of one per chunk — collapses
            # ~N manifest writes into a single Lance commit, shrinking the
            # interrupted-write corruption window that crash-looped the daemon.
            if rows:
                try:
                    self.milvus.upsert_many(rows)
                except Exception as e:
                    self.log.warning(
                        "surveyor.upsert_failed",
                        path=rel_path, count=len(rows), error=str(e),
                    )
                    # Leave state.files[rel_path] untouched so the old md5 is
                    # preserved and the file is retried next tick.  Recording
                    # md5=current here would mark it "done" with no chunks —
                    # and for a changed file its old chunks are already gone
                    # (deleted above), so it would silently drop from search.
                    continue
            # (rows empty → file has no embeddable content; fall through and
            # record state so we don't retry an empty file forever.)

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
                graph.load()
                graph.add_edges_from_wikilinks(rel_path, record.wikilinks)
                graph.save()
            except Exception:
                pass

            self.log.info("surveyor.embedded", path=rel_path, chunks=len(chunk_ids))

        self.emit("files_embedded", paths=diff["new"] + diff["changed"])

    async def _recluster(self) -> None:
        """HDBSCAN clustering over current embeddings.

        The Milvus query_all (~48s for 4k vectors) and HDBSCAN run in a thread
        pool so the event loop stays responsive for other daemons.
        """
        try:
            import numpy as np
            from sklearn.cluster import HDBSCAN
        except ImportError as e:
            self.log.warning("surveyor.cluster_skip", reason=str(e))
            return

        min_cluster_size = self.cfg.hdbscan_min_cluster_size
        min_samples = self.cfg.hdbscan_min_samples
        milvus = self.milvus

        def _compute() -> tuple[dict[int, list[str]], list[tuple[str, int]]] | None:
            rows = milvus.query_all(output_fields=["id", "embedding"])
            if not rows:
                return None

            seen: dict[str, list[float]] = {}
            for r in rows:
                rel_path = r["id"].rsplit("::", 1)[0]
                if rel_path not in seen:
                    seen[rel_path] = r["embedding"]

            paths = list(seen.keys())
            vectors = np.array(list(seen.values()), dtype=np.float32)

            if len(paths) < min_cluster_size:
                return None

            labels = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="cosine",
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
            state.clusters[key] = ClusterState(
                cluster_id=cid,
                cluster_type="semantic",
                label=existing.label if existing else [],
                member_files=members,
                last_labeled=existing.last_labeled if existing else "",
                consolidated_chunk_id=existing.consolidated_chunk_id if existing else "",
            )

        try:
            from alfred.store.graph import GraphStore
            graph = GraphStore(self.cfg.graph_path)
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
