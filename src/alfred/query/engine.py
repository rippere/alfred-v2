"""QueryEngine — full hybrid retrieval pipeline."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from alfred.config import AlfredConfig
from alfred.query.context import SourceRef, assemble, chunk_preview
from alfred.query.wiki import WikiFastPath, WikiHit
from alfred.store.milvus import MilvusStore, SearchHit

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


class QueryEngine:
    def __init__(self, cfg: AlfredConfig) -> None:
        self.cfg = cfg
        self._milvus: MilvusStore | None = None
        self._bm25 = None
        self._embedder = None
        self._reranker_loaded = False
        self._hopfield = None
        self._graph = None
        self._wiki = WikiFastPath(cfg.vault_path, cfg.wiki_dir)
        self._state_store = None

    def _get_milvus(self) -> MilvusStore:
        if self._milvus is None:
            self._milvus = MilvusStore(
                uri=self.cfg.milvus_uri,
                embed_dims=self.cfg.embed_dims,
                collection=self.cfg.milvus_collection,
            )
        return self._milvus

    def _get_bm25(self):
        if self._bm25 is None:
            from alfred.store.bm25 import BM25Store
            store = BM25Store(self.cfg.bm25_path)
            if not store.load():
                raise RuntimeError(
                    f"BM25 index not found at {self.cfg.bm25_path}. "
                    "Run: uv run python scripts/migrate_milvus.py"
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
        t = {}

        # ── Step 0: Wiki fast-path ─────────────────────────────────────────────
        t0 = time.perf_counter()
        wiki_hit = self._wiki.lookup(text)
        t["wiki"] = time.perf_counter() - t0

        # ── Step 1: Embed query ────────────────────────────────────────────────
        t0 = time.perf_counter()
        dense_vec = self._get_embedder().embed(text)
        t["embed"] = time.perf_counter() - t0

        # ── Step 2: BM25 sparse vector ─────────────────────────────────────────
        t0 = time.perf_counter()
        sparse_vec = self._get_bm25().encode(text)
        t["bm25"] = time.perf_counter() - t0

        # ── Step 3: Hopfield refinement (optional) ────────────────────────────
        if opts.use_hopfield:
            t0 = time.perf_counter()
            dense_vec = self._hopfield_refine(dense_vec, opts.top_k)
            t["hopfield"] = time.perf_counter() - t0

        # ── Step 4: Hybrid search ─────────────────────────────────────────────
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

        # ── Step 6: Ebbinghaus score adjustment (optional) ───────────────────
        if opts.use_ebbinghaus:
            t0 = time.perf_counter()
            state_store = self._get_state()
            from alfred.query.memory import adjust_scores, record_access
            adjust_scores(hits, state_store.state)
            t["ebbinghaus"] = time.perf_counter() - t0

        # ── Step 7: FlashRank reranking ───────────────────────────────────────
        t0 = time.perf_counter()
        from alfred.query.context import _chunk_text
        texts = {h.chunk_id: (_chunk_text(self.cfg.vault_path, h.chunk_id) or "") for h in hits}
        from alfred.embed.reranker import rerank
        hits = rerank(text, hits, texts, top_n=opts.top_k)
        t["rerank"] = time.perf_counter() - t0

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

            # Pass 1: get top-100 candidates with raw embeddings
            candidates = self._get_milvus()._client.search(
                collection_name=self.cfg.milvus_collection,
                data=[dense_vec],
                anns_field="embedding",
                limit=min(100, self._get_milvus().count()),
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
