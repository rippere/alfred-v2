"""Milvus Lite store — hybrid dense+sparse schema."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

try:
    from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
except ImportError:
    CollectionSchema = DataType = FieldSchema = MilvusClient = None  # type: ignore[assignment,misc]

log = structlog.get_logger()


def _is_channel_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "closed channel" in msg or "invoke rpc" in msg or "channel" in msg and "grpc" in msg


def _auto_reconnect(method):
    """Retry once after reconnecting if the gRPC channel to milvus-lite is dead."""
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as e:
            if _is_channel_error(e):
                log.warning("milvus.channel_dead", error=str(e)[:120])
                self._reconnect()
                return method(self, *args, **kwargs)
            raise
    return wrapper

COLLECTION = "vault_v2"


@dataclass
class SearchHit:
    chunk_id: str
    rel_path: str
    score: float
    record_type: str = ""
    name: str = ""
    rerank_score: float = 0.0


class MilvusStore:
    def __init__(self, uri: str, embed_dims: int = 768, collection: str = COLLECTION) -> None:
        if MilvusClient is None:
            raise RuntimeError("pymilvus not installed — install with: uv pip install 'alfred-v2[embed]'")
        self.uri = uri
        self.embed_dims = embed_dims
        self.collection = collection
        Path(uri).parent.mkdir(parents=True, exist_ok=True)

        # Retry on lock contention from a prior process
        for attempt in range(4):
            try:
                self._client = MilvusClient(uri=uri)
                break
            except Exception as e:
                if attempt < 3:
                    delay = 2.0 * (2 ** attempt)
                    log.warning("milvus.open_retry", attempt=attempt + 1, delay=delay, error=str(e))
                    time.sleep(delay)
                else:
                    raise

        self._ensure_collection()

    def _reconnect(self) -> None:
        """Re-open the MilvusClient after the internal subprocess crashes."""
        try:
            self._client.close()
        except Exception as e:
            # Expected: the client's subprocess already crashed — we're about to
            # reconnect. Record it at debug so it's greppable without log spam.
            log.debug("milvus.close_failed", error=str(e))
        for attempt in range(4):
            try:
                self._client = MilvusClient(uri=self.uri)
                self._ensure_collection()
                log.info("milvus.reconnected")
                return
            except Exception as e:
                if attempt < 3:
                    delay = 2.0 * (2 ** attempt)
                    log.warning("milvus.reconnect_retry", attempt=attempt + 1, delay=delay, error=str(e))
                    time.sleep(delay)
                else:
                    log.error("milvus.reconnect_failed", error=str(e))
                    raise

    def _ensure_collection(self) -> None:
        if self._client.has_collection(self.collection):
            return

        schema = CollectionSchema(
            fields=[
                FieldSchema("id",               DataType.VARCHAR,           is_primary=True, max_length=512),
                FieldSchema("embedding",        DataType.FLOAT_VECTOR,      dim=self.embed_dims),
                FieldSchema("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema("record_type",      DataType.VARCHAR,           max_length=64),
                FieldSchema("name",             DataType.VARCHAR,           max_length=512),
                FieldSchema("chunk_index",      DataType.INT32),
            ],
            description="Alfred vault hybrid embeddings v2",
        )
        self._client.create_collection(collection_name=self.collection, schema=schema)

        index_params = self._client.prepare_index_params()
        index_params.add_index(field_name="embedding",        index_type="FLAT",                  metric_type="COSINE")
        index_params.add_index(field_name="sparse_embedding", index_type="SPARSE_INVERTED_INDEX",  metric_type="IP")
        self._client.create_index(collection_name=self.collection, index_params=index_params)
        log.info("milvus.collection_created", name=self.collection)

    @_auto_reconnect
    def upsert(
        self,
        chunk_id: str,
        dense: list[float],
        sparse: dict[int, float],
        record_type: str,
        name: str,
        chunk_index: int = 0,
    ) -> None:
        self._client.upsert(
            collection_name=self.collection,
            data=[{
                "id":               chunk_id,
                "embedding":        dense,
                "sparse_embedding": sparse,
                "record_type":      record_type,
                "name":             name,
                "chunk_index":      chunk_index,
            }],
        )

    @_auto_reconnect
    def upsert_many(self, rows: list[dict]) -> None:
        """Batch upsert — interface parity with LanceDBStore.upsert_many.

        Each row dict needs: ``chunk_id``, ``dense``, ``sparse``,
        ``record_type``, ``name``, ``chunk_index``.
        """
        if not rows:
            return
        self._client.upsert(
            collection_name=self.collection,
            data=[{
                "id":               r["chunk_id"],
                "embedding":        r["dense"],
                "sparse_embedding": r.get("sparse", {}),
                "record_type":      r.get("record_type", ""),
                "name":             r.get("name", ""),
                "chunk_index":      r.get("chunk_index", 0),
            } for r in rows],
        )

    @_auto_reconnect
    def delete_file(self, rel_path: str, chunk_ids: list[str] | None = None) -> None:
        """Delete all chunks for a file. Uses known chunk_ids when available (fast path)."""
        if chunk_ids:
            self._client.delete(collection_name=self.collection, ids=chunk_ids)
        else:
            # Fallback: range filter on id — slower but safe
            self._client.delete(
                collection_name=self.collection,
                filter=f'id >= "{rel_path}::chunk_" and id < "{rel_path}::chunk_~"',
            )

    @_auto_reconnect
    def search(
        self,
        dense_vec: list[float],
        sparse_vec: dict[int, float],
        top_k: int = 8,
        include_inbox: bool = False,
    ) -> list[SearchHit]:
        from pymilvus import AnnSearchRequest, RRFRanker

        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {}},
            limit=top_k * 3,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_vec],
            anns_field="sparse_embedding",
            param={"metric_type": "IP", "params": {}},
            limit=top_k * 3,
        )

        results = self._client.hybrid_search(
            collection_name=self.collection,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(),
            limit=top_k,
            output_fields=["record_type", "name"],
        )

        hits: list[SearchHit] = []
        for r in results[0]:
            chunk_id = r["id"]
            rel_path = chunk_id.rsplit("::", 1)[0]
            if not include_inbox and rel_path.startswith("inbox/"):
                continue
            hits.append(SearchHit(
                chunk_id=chunk_id,
                rel_path=rel_path,
                score=r["distance"],
                record_type=r["entity"].get("record_type", ""),
                name=r["entity"].get("name", ""),
            ))
        return hits

    @_auto_reconnect
    def query_all(self, output_fields: list[str] | None = None) -> list[dict[str, Any]]:
        """Page through entire collection. Used for migration and cluster building."""
        PAGE = 16_000
        out_fields = output_fields or ["id", "embedding", "record_type", "name"]
        all_rows: list[dict] = []
        offset = 0
        while True:
            page = self._client.query(
                collection_name=self.collection,
                filter="",
                output_fields=out_fields,
                limit=PAGE,
                offset=offset,
            )
            if not page:
                break
            all_rows.extend(page)
            if len(page) < PAGE:
                break
            offset += PAGE
        return all_rows

    @_auto_reconnect
    def count(self) -> int:
        stats = self._client.get_collection_stats(self.collection)
        return int(stats.get("row_count", 0))
