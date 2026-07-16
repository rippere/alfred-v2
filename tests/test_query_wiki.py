"""Coverage for alfred.query.wiki.WikiFastPath: filesystem lookup + regex
entity extraction + slug conversion. No LLM/network calls in this module."""
from __future__ import annotations

from alfred.query.wiki import WikiFastPath, _safe_name


def _make_wiki(vault, pages: dict[str, str]) -> None:
    wiki_dir = vault / "wiki"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    for slug, content in pages.items():
        (wiki_dir / f"{slug}.md").write_text(content, encoding="utf-8")


def test_lookup_returns_none_when_wiki_dir_absent(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    fp = WikiFastPath(vault)
    assert fp.lookup("Tell me about John Smith") is None


def test_lookup_finds_matching_capitalized_entity(tmp_path):
    vault = tmp_path / "vault"
    _make_wiki(vault, {"john_smith": "# John Smith\n\nFacts."})
    fp = WikiFastPath(vault)

    hit = fp.lookup("What does John Smith think about this?")

    assert hit is not None
    assert hit.entity_name == "John Smith"
    assert hit.rel_path == "wiki/john_smith.md"
    assert "Facts." in hit.content
    assert hit.score == 0.95


def test_lookup_prefers_longest_candidate_match(tmp_path):
    vault = tmp_path / "vault"
    _make_wiki(vault, {"john": "# John\n\nShort page."})
    fp = WikiFastPath(vault)

    # "John Smith" as a full n-gram has no page, but the shorter "John" does —
    # longest-first ordering must still fall through to a shorter candidate
    # that actually has a page.
    hit = fp.lookup("Ask John Smith about John")
    assert hit is not None
    assert hit.entity_name == "John"


def test_lookup_returns_none_when_no_candidate_has_a_page(tmp_path):
    vault = tmp_path / "vault"
    _make_wiki(vault, {"someone_else": "# Someone Else\n"})
    fp = WikiFastPath(vault)

    assert fp.lookup("Ask John Smith about things") is None


def test_lookup_returns_none_for_query_with_no_capitalized_ngrams(tmp_path):
    vault = tmp_path / "vault"
    _make_wiki(vault, {"alpha": "# alpha\n"})
    fp = WikiFastPath(vault)

    assert fp.lookup("what is the lowercase query here") is None


def test_get_page_direct_lookup_hit(tmp_path):
    vault = tmp_path / "vault"
    _make_wiki(vault, {"acme_corp": "# Acme Corp\n\nDetails."})
    fp = WikiFastPath(vault)

    hit = fp.get_page("Acme Corp")
    assert hit is not None
    assert hit.entity_name == "Acme Corp"
    assert hit.rel_path == "wiki/acme_corp.md"


def test_get_page_returns_none_when_missing(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "wiki").mkdir()
    fp = WikiFastPath(vault)

    assert fp.get_page("Nobody Here") is None


def test_safe_name_lowercases_and_underscores_spaces():
    assert _safe_name("John Smith") == "john_smith"


def test_safe_name_strips_punctuation():
    assert _safe_name("Acme, Corp.!") == "acme_corp"


def test_safe_name_collapses_surrounding_whitespace():
    assert _safe_name("  Jane Doe  ") == "jane_doe"
