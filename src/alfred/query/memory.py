"""Ebbinghaus memory scoring: boost/penalize hits by access history."""
from __future__ import annotations

from alfred.core.models import MemoryStrength, PipelineState
from alfred.store.milvus import SearchHit


def adjust_scores(hits: list[SearchHit], state: PipelineState) -> list[SearchHit]:
    """Apply Ebbinghaus modifier to each hit's score. Modifies hits in-place."""
    for hit in hits:
        strength = state.memory.get(hit.rel_path)
        if strength:
            hit.score = hit.score * strength.score_modifier()
    return hits


def record_access(hits: list[SearchHit], state: PipelineState) -> None:
    """Update MemoryStrength for each returned source file."""
    for hit in hits:
        if hit.rel_path not in state.memory:
            state.memory[hit.rel_path] = MemoryStrength(rel_path=hit.rel_path)
        state.memory[hit.rel_path].update()
