"""QueryEngine — retrieval pipeline (dense vector search + graph/Hopfield/rerank stages).

Retrieval on the live LanceDB backend is dense-only; BM25 is used solely by
the separate ``bm25_only`` offline path (and by the legacy Milvus backend's
hybrid search).
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from alfred.config import AlfredConfig
from alfred.query.context import SourceRef, assemble, chunk_preview
from alfred.query.wiki import WikiFastPath, WikiHit
from alfred.store.types import SearchHit

if TYPE_CHECKING:
    pass

log = structlog.get_logger()


@dataclass
class QueryOptions:
    top_k: int = 8
    use_hopfield: bool = True
    use_graph: bool = True
    use_ebbinghaus: bool = True
    include_synthesis: bool = False
    include_inbox: bool = False


@dataclass
class QueryResult:
    query: str
    wiki_hit: WikiHit | None
    hits: list[SearchHit]
    sources: list[SourceRef]
    context: str
    answer: str = ""
    synthesis_backend: str = ""
    synthesis_model: str = ""
    elapsed: dict[str, float] = field(default_factory=dict)
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class QueryEngine:
    def __init__(self, cfg: AlfredConfig) -> None:
        self.cfg = cfg
        self._store = None  # LanceDBStore or MilvusStore depending on cfg.vector_store
        self._bm25 = None
        self._embedder = None
        self._reranker_loaded = False
        self._hopfield = None
        self._graph = None
        self._wiki = WikiFastPath(cfg.vault_path, cfg.wiki_dir)
        self._state_store = None

    def _get_milvus(self):
        """Return the active vector store (LanceDB or Milvus) — named _get_milvus for back-compat."""
        if self._store is None:
            if self.cfg.vector_store == "lancedb":
                from alfred.store.lancedb_store import LanceDBStore
                self._store = LanceDBStore(
                    uri=self.cfg.lancedb_uri,
                    collection=self.cfg.milvus_collection,
                    dims=self.cfg.embed_dims,
                )
            else:
                from alfred.store.milvus import MilvusStore
                self._store = MilvusStore(
                    uri=self.cfg.milvus_uri,
                    embed_dims=self.cfg.embed_dims,
                    collection=self.cfg.milvus_collection,
                )
        return self._store

    def _get_bm25(self):
        if self._bm25 is None:
            from alfred.store.bm25 import BM25Store
            store = BM25Store(self.cfg.bm25_path)
            if not store.load():
                raise RuntimeError(
                    f"BM25 index not found at {self.cfg.bm25_path}. "
                    "Build it from the vault corpus with BM25Store.fit_and_store() + save() "
                    "(see src/alfred/store/bm25.py), or copy bm25.pkl from an existing "
                    "deployment's data dir."
                )
            self._bm25 = store
        return self._bm25

    def _get_embedder(self):
        if self._embedder is None:
            import httpx
            self._embedder = _SyncEmbedder(
                url=f"{self.cfg.ollama_base_url}/api/embeddings",
                model=self.cfg.ollama_embed_model,
            )
        return self._embedder

    def _get_state(self):
        if self._state_store is None:
            from alfred.store.state import StateStore
            self._state_store = StateStore(self.cfg.state_path)
            self._state_store.load()
        return self._state_store

    def query(self, text: str, opts: QueryOptions | None = None) -> QueryResult:
        if opts is None:
            opts = QueryOptions(top_k=self.cfg.default_top_k)

        if self.cfg.bm25_only:
            log.warning(
                "query.bm25_only_deprecated",
                message=(
                    "bm25_only is a deprecated/unsupported path: the BM25 corpus at "
                    "cfg.bm25_path is static and is NOT updated by the live surveyor "
                    "pipeline. It was last populated by the archived one-off script "
                    "scripts/_archive/phase4_rebuild_milvus.py and will silently drift "
                    "stale as the vault changes. Do not rely on this path for "
                    "production retrieval quality."
                ),
                bm25_path=str(self.cfg.bm25_path),
            )
            return self._query_bm25_only(text, opts)

        t = {}

        # ── Step 0: Wiki fast-path ─────────────────────────────────────────────
        t0 = time.perf_counter()
        wiki_hit = self._wiki.lookup(text)
        t["wiki"] = time.perf_counter() - t0

        # ── Step 1: Embed query ────────────────────────────────────────────────
        t0 = time.perf_counter()
        dense_vec = self._get_embedder().embed(text)
        t["embed"] = time.perf_counter() - t0

        # ── Step 2: BM25 sparse vector (legacy Milvus backend only) ──────────
        # LanceDBStore.search() ignores its sparse argument entirely (retrieval
        # is dense-only; BM25 serves only the separate bm25_only offline path),
        # so encoding the query here would be dead work on every call. Only the
        # legacy Milvus backend's hybrid_search still consumes a sparse vector.
        sparse_vec: dict[int, float] = {}
        if self.cfg.vector_store != "lancedb":
            t0 = time.perf_counter()
            sparse_vec = self._get_bm25().encode(text)
            t["bm25"] = time.perf_counter() - t0

        # ── Step 3: Hopfield refinement (optional) ────────────────────────────
        if opts.use_hopfield:
            t0 = time.perf_counter()
            dense_vec = self._hopfield_refine(dense_vec, opts.top_k)
            t["hopfield"] = time.perf_counter() - t0

        # ── Step 4: Vector search (dense-only on LanceDB; hybrid on legacy Milvus) ─
        t0 = time.perf_counter()
        hits = self._get_milvus().search(
            dense_vec=dense_vec,
            sparse_vec=sparse_vec,
            top_k=opts.top_k,
            include_inbox=opts.include_inbox,
        )
        t["search"] = time.perf_counter() - t0

        # ── Step 5: Graph spreading activation (optional) ─────────────────────
        if opts.use_graph:
            t0 = time.perf_counter()
            hits = self._spread_activate(hits, opts.top_k)
            t["graph"] = time.perf_counter() - t0

        # ── Step 6: FlashRank reranking ───────────────────────────────────────
        t0 = time.perf_counter()
        from alfred.query.context import _chunk_text
        texts = {h.chunk_id: (_chunk_text(self.cfg.vault_path, h.chunk_id) or "") for h in hits}
        from alfred.embed.reranker import rerank
        hits = rerank(text, hits, texts, top_n=opts.top_k)
        t["rerank"] = time.perf_counter() - t0

        # ── Step 7: Ebbinghaus late-stage reranking (optional) ────────────────
        # Applied AFTER FlashRank so recency signal only adjusts final ordering
        # among semantically relevant results, not the retrieval pool itself.
        if opts.use_ebbinghaus:
            t0 = time.perf_counter()
            state_store = self._get_state()
            from alfred.query.memory import adjust_scores
            adjust_scores(hits, state_store.state)
            hits.sort(key=lambda h: h.score, reverse=True)
            t["ebbinghaus"] = time.perf_counter() - t0

        # ── Step 8: Context assembly ──────────────────────────────────────────
        t0 = time.perf_counter()
        # Prepend wiki hit context if present
        context, sources = assemble(hits, self.cfg.vault_path)
        if wiki_hit:
            wiki_block = f"[Wiki: {wiki_hit.rel_path}  score=0.95]\n{wiki_hit.content}"
            context = wiki_block + "\n\n---\n\n" + context if context else wiki_block
        t["assemble"] = time.perf_counter() - t0

        # ── Step 9: Synthesis (optional) ─────────────────────────────────────
        answer, backend, model = "", "", ""
        if opts.include_synthesis and context:
            t0 = time.perf_counter()
            from alfred.query.synth import synthesize
            answer, backend, model = synthesize(
                query=text,
                context=context,
                anthropic_model=self.cfg.anthropic_model,
                openrouter_model=self.cfg.openrouter_model,
                ollama_base_url=self.cfg.ollama_base_url,
                ollama_model=self.cfg.ollama_llm_model,
            )
            t["synth"] = time.perf_counter() - t0

        # Record memory access asynchronously
        if opts.use_ebbinghaus and hits:
            from alfred.query.memory import record_access
            state_store = self._get_state()
            record_access(hits, state_store.state)
            state_store.save()

        result = QueryResult(
            query=text,
            wiki_hit=wiki_hit,
            hits=hits,
            sources=sources,
            context=context,
            answer=answer,
            synthesis_backend=backend,
            synthesis_model=model,
            elapsed=t,
        )
        self._log_query(result)
        return result

    def _log_query(self, result: QueryResult) -> None:
        try:
            log_path = self.cfg.data_dir / "query_log.jsonl"
            entry = {
                "query_id": result.query_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": result.query,
                "top_paths": [h.rel_path for h in result.hits[:5]],
                "synthesis_used": bool(result.answer),
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def _query_bm25_only(self, text: str, opts: QueryOptions) -> QueryResult:
        """Lightweight query path: BM25 corpus search only, no Milvus or Ollama.

        Requires the BM25 index to have been built with BM25Store.fit_and_store()
        so the corpus matrix is stored alongside the vectorizer.
        Returns results ranked by BM25 score then reranked by FlashRank.
        """
        t = {}

        t0 = time.perf_counter()
        wiki_hit = self._wiki.lookup(text)
        t["wiki"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        bm25 = self._get_bm25()
        if not bm25.has_corpus:
            raise RuntimeError(
                "BM25-only mode requires corpus storage. "
                "Rebuild the index at cfg.bm25_path with BM25Store.fit_and_store() — "
                "not fit() — so the corpus matrix is saved (see src/alfred/store/bm25.py)."
            )
        raw_hits = bm25.search(text, top_k=opts.top_k * 3)
        t["bm25"] = time.perf_counter() - t0

        hits: list[SearchHit] = []
        seen_paths: set[str] = set()
        for chunk_id, score in raw_hits:
            rel_path = chunk_id.rsplit("::", 1)[0]
            if not opts.include_inbox and rel_path.startswith("inbox/"):
                continue
            if rel_path in seen_paths:
                continue
            seen_paths.add(rel_path)
            hits.append(SearchHit(
                chunk_id=chunk_id,
                rel_path=rel_path,
                score=score,
                record_type="",
                name=Path(rel_path).stem,
            ))

        # FlashRank reranking
        t0 = time.perf_counter()
        from alfred.query.context import _chunk_text
        from alfred.embed.reranker import rerank
        texts = {h.chunk_id: (_chunk_text(self.cfg.vault_path, h.chunk_id) or "") for h in hits}
        hits = rerank(text, hits, texts, top_n=opts.top_k)
        t["rerank"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        context, sources = assemble(hits, self.cfg.vault_path)
        if wiki_hit:
            wiki_block = f"[Wiki: {wiki_hit.rel_path}  score=0.95]\n{wiki_hit.content}"
            context = wiki_block + "\n\n---\n\n" + context if context else wiki_block
        t["assemble"] = time.perf_counter() - t0

        answer, backend, model = "", "", ""
        if opts.include_synthesis and context:
            t0 = time.perf_counter()
            from alfred.query.synth import synthesize
            answer, backend, model = synthesize(
                query=text,
                context=context,
                anthropic_model=self.cfg.anthropic_model,
                openrouter_model=self.cfg.openrouter_model,
                ollama_base_url=self.cfg.ollama_base_url,
                ollama_model=self.cfg.ollama_llm_model,
            )
            t["synth"] = time.perf_counter() - t0

        return QueryResult(
            query=text,
            wiki_hit=wiki_hit,
            hits=hits,
            sources=sources,
            context=context,
            answer=answer,
            synthesis_backend=backend,
            synthesis_model=model,
            elapsed=t,
        )

    def _hopfield_refine(self, dense_vec: list[float], top_k: int) -> list[float]:
        """First-pass retrieval → Hopfield refinement → return refined vector."""
        try:
            import numpy as np
            from alfred.embed.hopfield import HopfieldRefiner

            if self._hopfield is None:
                self._hopfield = HopfieldRefiner(
                    iterations=self.cfg.hopfield_iterations,
                    beta=self.cfg.hopfield_beta,
                )

            store = self._get_milvus()
            limit = min(100, store.count())
            if limit == 0:
                return dense_vec

            # Retrieve top candidates with their raw embeddings.
            # LanceDBStore exposes this via its search() + query_all() API;
            # MilvusStore exposes it via the internal _client.search() call.
            if self.cfg.vector_store == "lancedb":
                from alfred.store.lancedb_store import LanceDBStore
                assert isinstance(store, LanceDBStore)
                rows = (
                    store._tbl
                    .search(dense_vec, vector_column_name="vector")
                    .metric("cosine")
                    .limit(limit)
                    .select(["vector", "_distance"])
                    .to_list()
                )
                if not rows:
                    return dense_vec
                raw_embs = np.array([r["vector"] for r in rows], dtype=np.float32)
            else:
                candidates = store._client.search(
                    collection_name=self.cfg.milvus_collection,
                    data=[dense_vec],
                    anns_field="embedding",
                    limit=limit,
                    search_params={"metric_type": "COSINE", "params": {}},
                    output_fields=["embedding"],
                )
                if not candidates or not candidates[0]:
                    return dense_vec
                raw_embs = np.array([r["entity"]["embedding"] for r in candidates[0]], dtype=np.float32)

            refined = self._hopfield.refine(np.array(dense_vec, dtype=np.float32), raw_embs)
            return refined.tolist()
        except Exception as e:
            log.warning("engine.hopfield_skip", error=str(e))
            return dense_vec

    def _spread_activate(self, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        """Spread activation through NetworkX graph to surface associated nodes."""
        try:
            if self._graph is None:
                from alfred.store.graph import GraphStore
                self._graph = GraphStore(self.cfg.graph_path)
                self._graph.load()

            if self._graph.is_empty():
                return hits

            seed_paths = [h.rel_path for h in hits]
            activated = self._graph.spreading_activation(
                seed_paths,
                hops=self.cfg.graph_hops,
                decay=self.cfg.graph_decay,
            )

            existing = {h.rel_path for h in hits}
            for rel_path, score in activated.items():
                if rel_path not in existing:
                    hits.append(SearchHit(
                        chunk_id=f"{rel_path}::chunk_00",
                        rel_path=rel_path,
                        score=score,
                        record_type="",
                        name=rel_path,
                    ))
            return hits
        except Exception as e:
            log.warning("engine.graph_skip", error=str(e))
            return hits


class _SyncEmbedder:
    """Synchronous Ollama embed wrapper for use in the query CLI."""
    def __init__(self, url: str, model: str) -> None:
        self.url = url
        self.model = model

    def embed(self, text: str) -> list[float]:
        import httpx
        resp = httpx.post(self.url, json={"model": self.model, "prompt": text}, timeout=30.0)
        resp.raise_for_status()
        return resp.json()["embedding"]
