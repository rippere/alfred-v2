"""FlashRank CPU cross-encoder reranker."""
from __future__ import annotations

import structlog
from alfred.store.milvus import SearchHit

log = structlog.get_logger()

_ranker = None


def _get_ranker():
    global _ranker
    if _ranker is None:
        from flashrank import Ranker
        # ms-marco-MiniLM-L-12-v2 (~34MB) — meaningfully better than the 4MB default
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
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

    reranked = ranker.rerank(RerankRequest(query=query, passages=passages))

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
