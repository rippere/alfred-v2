"""Deterministic-logic coverage for alfred.query.context: dedup, truncation,
chunk-id parsing, and preview formatting. No LLM/network calls in this module,
so no mocking is required beyond building small real vault fixtures."""
from __future__ import annotations

import alfred.query.context as context
from alfred.query.context import SourceRef, assemble, chunk_preview
from alfred.store.types import SearchHit


def _write_note(vault, rel_path: str, body: str, record_type: str = "note") -> None:
    full = vault / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(f"---\ntype: {record_type}\n---\n{body}\n", encoding="utf-8")


def test_assemble_keeps_highest_scoring_chunk_per_source_file(tmp_path):
    vault = tmp_path / "vault"
    _write_note(vault, "notes/alpha.md", "alpha body text")

    hits = [
        SearchHit(chunk_id="notes/alpha.md::chunk_00", rel_path="notes/alpha.md", score=0.4, name="alpha"),
        SearchHit(chunk_id="notes/alpha.md::chunk_00", rel_path="notes/alpha.md", score=0.9, name="alpha"),
    ]
    text, sources = assemble(hits, vault)

    assert len(sources) == 1, "duplicate rel_path hits must collapse to one source"
    assert sources[0].score == 0.9, "the higher-scoring duplicate must win"
    assert "alpha body text" in text


def test_assemble_orders_sources_by_score_descending(tmp_path):
    vault = tmp_path / "vault"
    _write_note(vault, "notes/alpha.md", "alpha body")
    _write_note(vault, "notes/beta.md", "beta body")

    hits = [
        SearchHit(chunk_id="notes/alpha.md::chunk_00", rel_path="notes/alpha.md", score=0.2, name="alpha"),
        SearchHit(chunk_id="notes/beta.md::chunk_00", rel_path="notes/beta.md", score=0.8, name="beta"),
    ]
    _, sources = assemble(hits, vault)

    assert [s.rel_path for s in sources] == ["notes/beta.md", "notes/alpha.md"]


def test_assemble_skips_hits_with_unresolvable_chunk_text(tmp_path):
    vault = tmp_path / "vault"
    _write_note(vault, "notes/alpha.md", "alpha body")

    hits = [
        SearchHit(chunk_id="notes/missing.md::chunk_00", rel_path="notes/missing.md", score=0.9, name="missing"),
        SearchHit(chunk_id="notes/alpha.md::chunk_00", rel_path="notes/alpha.md", score=0.5, name="alpha"),
    ]
    text, sources = assemble(hits, vault)

    assert len(sources) == 1
    assert sources[0].rel_path == "notes/alpha.md"
    assert "alpha body" in text


def test_assemble_truncates_at_max_context_chars(tmp_path, monkeypatch):
    monkeypatch.setattr(context, "MAX_CONTEXT_CHARS", 300)
    vault = tmp_path / "vault"
    _write_note(vault, "notes/big1.md", "A" * 1000)

    hits = [
        SearchHit(chunk_id="notes/big1.md::chunk_00", rel_path="notes/big1.md", score=0.9, name="big1"),
    ]
    text, sources = assemble(hits, vault)

    assert len(sources) == 1
    assert sources[0].text_len == 300 + len("...[truncated]")
    assert "[truncated]" in text


def test_assemble_stops_when_remaining_budget_too_small(tmp_path, monkeypatch):
    # remaining < 200 after the first block must break rather than emit a
    # near-empty truncated fragment
    monkeypatch.setattr(context, "MAX_CONTEXT_CHARS", 410)
    vault = tmp_path / "vault"
    _write_note(vault, "notes/big1.md", "A" * 400)
    _write_note(vault, "notes/big2.md", "B" * 400)

    hits = [
        SearchHit(chunk_id="notes/big1.md::chunk_00", rel_path="notes/big1.md", score=0.9, name="big1"),
        SearchHit(chunk_id="notes/big2.md::chunk_00", rel_path="notes/big2.md", score=0.8, name="big2"),
    ]
    _, sources = assemble(hits, vault)

    assert len(sources) == 1, "second source must be dropped once remaining budget < 200 chars"


def test_assemble_empty_hits_returns_empty_context():
    text, sources = assemble([], __import__("pathlib").Path("/nonexistent"))
    assert text == ""
    assert sources == []


def test_chunk_preview_truncates_and_flattens_newlines(tmp_path):
    vault = tmp_path / "vault"
    _write_note(vault, "notes/alpha.md", "line one\nline two\n" + ("x" * 300))

    sources = [SourceRef(
        chunk_id="notes/alpha.md::chunk_00",
        rel_path="notes/alpha.md",
        score=0.9,
        record_type="note",
        name="alpha",
    )]
    previews = chunk_preview(sources, vault)

    assert "notes/alpha.md" in previews
    preview = previews["notes/alpha.md"]
    assert "\n" not in preview
    assert len(preview) <= context.CHUNK_PREVIEW_CHARS


def test_chunk_preview_missing_file_yields_empty_string(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    sources = [SourceRef(
        chunk_id="notes/gone.md::chunk_00",
        rel_path="notes/gone.md",
        score=0.9,
        record_type="note",
        name="gone",
    )]
    previews = chunk_preview(sources, vault)
    assert previews["notes/gone.md"] == ""


def test_chunk_text_rejects_ids_without_double_colon(tmp_path):
    vault = tmp_path / "vault"
    _write_note(vault, "notes/alpha.md", "alpha body")
    assert context._chunk_text(vault, "notes/alpha.md_no_separator") is None


def test_chunk_text_rejects_non_integer_chunk_suffix(tmp_path):
    vault = tmp_path / "vault"
    _write_note(vault, "notes/alpha.md", "alpha body")
    assert context._chunk_text(vault, "notes/alpha.md::chunk_notanumber") is None


def test_chunk_text_returns_none_for_out_of_range_index(tmp_path):
    vault = tmp_path / "vault"
    _write_note(vault, "notes/alpha.md", "alpha body")
    # a short note only produces chunk_00 — chunk_05 is out of range
    assert context._chunk_text(vault, "notes/alpha.md::chunk_05") is None


def test_chunk_text_returns_none_for_missing_file(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    assert context._chunk_text(vault, "notes/missing.md::chunk_00") is None


def test_chunk_text_handles_parse_failure_gracefully(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "notes").mkdir()
    # A file that exists but whose frontmatter/parsing blows up should be
    # swallowed and return None, not raise.
    bad = vault / "notes" / "broken.md"
    bad.write_bytes(b"\xff\xfe not valid utf-8 frontmatter")
    assert context._chunk_text(vault, "notes/broken.md::chunk_00") is None
