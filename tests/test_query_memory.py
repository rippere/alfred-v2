"""Coverage for alfred.query.memory: Ebbinghaus score adjustment and access
tracking. Pure dataclass logic, no LLM/network calls to mock."""
from __future__ import annotations

from alfred.core.models import MemoryStrength, PipelineState
from alfred.query.memory import adjust_scores, record_access
from alfred.store.types import SearchHit


def test_adjust_scores_leaves_hits_without_memory_untouched():
    state = PipelineState()
    hits = [SearchHit(chunk_id="a::chunk_00", rel_path="notes/a.md", score=1.0)]

    result = adjust_scores(hits, state)

    assert result[0].score == 1.0


def test_adjust_scores_applies_modifier_when_strength_present():
    state = PipelineState()
    strength = MemoryStrength(rel_path="notes/a.md")
    state.memory["notes/a.md"] = strength
    hits = [SearchHit(chunk_id="a::chunk_00", rel_path="notes/a.md", score=1.0)]

    result = adjust_scores(hits, state)

    assert result[0].score == strength.score_modifier() * 1.0
    assert result[0].score != 1.0 or strength.score_modifier() == 1.0


def test_adjust_scores_modifies_hits_in_place_and_returns_same_list():
    state = PipelineState()
    hits = [SearchHit(chunk_id="a::chunk_00", rel_path="notes/a.md", score=1.0)]

    result = adjust_scores(hits, state)

    assert result is hits


def test_record_access_creates_new_memory_strength_for_unseen_file():
    state = PipelineState()
    hits = [SearchHit(chunk_id="a::chunk_00", rel_path="notes/a.md", score=0.5)]

    record_access(hits, state)

    assert "notes/a.md" in state.memory
    assert state.memory["notes/a.md"].access_count == 1
    assert state.memory["notes/a.md"].last_accessed != ""


def test_record_access_increments_existing_strength():
    state = PipelineState()
    state.memory["notes/a.md"] = MemoryStrength(rel_path="notes/a.md", access_count=3)
    hits = [SearchHit(chunk_id="a::chunk_00", rel_path="notes/a.md", score=0.5)]

    record_access(hits, state)

    assert state.memory["notes/a.md"].access_count == 4


def test_record_access_handles_multiple_hits_independently():
    state = PipelineState()
    hits = [
        SearchHit(chunk_id="a::chunk_00", rel_path="notes/a.md", score=0.5),
        SearchHit(chunk_id="b::chunk_00", rel_path="notes/b.md", score=0.5),
    ]

    record_access(hits, state)

    assert set(state.memory.keys()) == {"notes/a.md", "notes/b.md"}
    assert state.memory["notes/a.md"].access_count == 1
    assert state.memory["notes/b.md"].access_count == 1


def test_record_access_same_file_hit_twice_in_one_call_increments_twice():
    state = PipelineState()
    hits = [
        SearchHit(chunk_id="a::chunk_00", rel_path="notes/a.md", score=0.5),
        SearchHit(chunk_id="a::chunk_01", rel_path="notes/a.md", score=0.4),
    ]

    record_access(hits, state)

    assert state.memory["notes/a.md"].access_count == 2
