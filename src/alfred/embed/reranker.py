"""FlashRank CPU cross-encoder reranker."""
from __future__ import annotations

import structlog
from alfred.store.types import SearchHit

log = structlog.get_logger()

_ranker = None
_RERANK_BATCH = 16


def _get_ranker():
    global _ranker
    if _ranker is None:
        import onnxruntime as ort
        from flashrank import Ranker
        # ms-marco-MiniLM-L-12-v2 (~34MB) — meaningfully better than the 4MB default
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        # FlashRank builds its InferenceSession with ORT defaults: a CPU memory
        # arena that grows to fit every new (batch, seq_len) shape and never
        # shrinks. With one MCP server per Claude session, that retained ~1.6 GB
        # per process and drove global OOM / desktop freezes. Arena-off keeps
        # rerank RSS flat (~200 MB) at a small latency cost.
        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        opts.intra_op_num_threads = 4
        _ranker.session = ort.InferenceSession(
            _ranker.session._model_path, opts, providers=["CPUExecutionProvider"]
        )
    return _ranker


def rerank(query: str, hits: list[SearchHit], texts: dict[str, str], top_n: int) -> list[SearchHit]:
    """Rerank hits using FlashRank cross-encoder. Returns top_n hits in new order.

    texts: {chunk_id: text} — must be pre-fetched by the caller.
    Hits with no text entry are passed through with their original score.
    """
    if not hits:
        return hits

    ranker = _get_ranker()
    from flashrank import RerankRequest

    passages = []
    has_text: list[int] = []
    no_text: list[SearchHit] = []

    for i, h in enumerate(hits):
        text = texts.get(h.chunk_id, "")
        if text:
            passages.append({"id": i, "text": text, "meta": {"chunk_id": h.chunk_id}})
            has_text.append(i)
        else:
            no_text.append(h)

    if not passages:
        return hits

    # Score in small batches: graph expansion hands us ~150-200 passages, and a
    # single ONNX batch padded to 512 tokens peaks at 1.5-2+ GB transient. Each
    # (query, passage) score is independent, so batching is result-identical.
    reranked = []
    for start in range(0, len(passages), _RERANK_BATCH):
        batch = passages[start:start + _RERANK_BATCH]
        reranked.extend(ranker.rerank(RerankRequest(query=query, passages=batch)))
    reranked.sort(key=lambda r: r["score"], reverse=True)

    # Build reranked list using returned scores
    idx_to_hit = {i: hits[i] for i in has_text}
    result: list[SearchHit] = []
    for r in reranked:
        original_idx = r["id"]
        hit = idx_to_hit[original_idx]
        hit.rerank_score = float(r["score"])
        result.append(hit)

    # Append no-text hits at the end (lower priority)
    result.extend(no_text)
    return result[:top_n]
