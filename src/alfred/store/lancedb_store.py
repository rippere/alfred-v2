"""LanceDB vector store — drop-in replacement for MilvusStore.

LanceDB is an embedded columnar vector database that uses memory-mapped files
instead of a gRPC subprocess, so multiple readers can open the same database
simultaneously without file-lock contention.  This eliminates the daemon vs.
CLI lock conflict that existed with Milvus Lite.

Interface mirrors MilvusStore exactly so callers don't need to change.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

# Re-export SearchHit from milvus so importers don't have to change.
from alfred.store.milvus import SearchHit  # noqa: F401


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

        if collection in self._db.table_names():
            self._tbl = self._db.open_table(collection)
        else:
            self._tbl = self._db.create_table(
                collection,
                schema=self._schema,
                mode="create",
            )
            log.info("lancedb.table_created", name=collection)

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
        BM25 ranking is handled entirely by BM25Store.
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
        """Dense cosine search.  BM25/sparse component is handled upstream.

        ``sparse_vec`` is accepted for interface parity but ignored here —
        the BM25Store already does sparse scoring and the QueryEngine merges
        results from both paths via RRF-style re-ranking.
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
