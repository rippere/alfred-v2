"""Coverage for alfred.wiki.writer.WikiWriter: page creation/update bookkeeping
and LLM-backed enrichment. Mocking strategy mirrors test_query_engine.py —
the vault filesystem ops (vault_create/vault_read/vault_edit) are exercised
for real against a tmp_path vault since they're deterministic; only the
Anthropic client call in enrich_page() is mocked, matching how the existing
suite avoids real model calls without re-testing model output quality."""
from __future__ import annotations

import json

import pytest

from alfred.config import AlfredConfig
from alfred.core.vault_ops import vault_create
from alfred.store.state import StateStore
from alfred.wiki.writer import WikiWriter


@pytest.fixture
def cfg(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return AlfredConfig(vault_path=vault, data_dir=data_dir)


@pytest.fixture
def state_store(tmp_path):
    return StateStore(tmp_path / "data" / "state.json")


@pytest.fixture
def ww(cfg, state_store):
    return WikiWriter(cfg, state_store)


def test_ensure_page_creates_new_wiki_page_and_state_entry(ww, cfg):
    rel_path = ww.ensure_page("Acme Corp", "org", "notes/source.md")

    assert rel_path == "wiki/acme-corp.md"
    assert (cfg.vault_path / rel_path).exists()

    key = "acme corp"
    assert key in ww.state.state.wiki_pages
    page = ww.state.state.wiki_pages[key]
    assert page.entity_name == "Acme Corp"
    assert page.entity_type == "org"
    assert page.sources == ["notes/source.md"]


def test_ensure_page_existing_entity_appends_new_source(ww):
    ww.ensure_page("Acme Corp", "org", "notes/one.md")
    rel_path = ww.ensure_page("Acme Corp", "org", "notes/two.md")

    page = ww.state.state.wiki_pages["acme corp"]
    assert rel_path == page.rel_path
    assert page.sources == ["notes/one.md", "notes/two.md"]


def test_ensure_page_existing_entity_does_not_duplicate_same_source(ww):
    ww.ensure_page("Acme Corp", "org", "notes/one.md")
    ww.ensure_page("Acme Corp", "org", "notes/one.md")

    page = ww.state.state.wiki_pages["acme corp"]
    assert page.sources == ["notes/one.md"]


def test_ensure_page_lookup_is_case_insensitive_on_entity_name(ww):
    ww.ensure_page("Acme Corp", "org", "notes/one.md")
    rel_path = ww.ensure_page("acme corp", "org", "notes/two.md")

    assert rel_path == "wiki/acme-corp.md"
    assert len(ww.state.state.wiki_pages) == 1


def test_ensure_page_survives_vault_create_race(ww, cfg, monkeypatch):
    """If the on-disk file already exists (e.g. another process created it),
    vault_create raises VaultError — ensure_page must swallow it and still
    record state, not blow up the caller."""
    vault_create(cfg.vault_path, "wiki", "acme-corp", body="# Acme Corp\n\n")

    rel_path = ww.ensure_page("Acme Corp", "org", "notes/source.md")

    assert rel_path == "wiki/acme-corp.md"
    assert "acme corp" in ww.state.state.wiki_pages


def test_enrich_page_backend_unavailable_returns_false(ww, cfg, monkeypatch):
    """A paused backend leaves the page untouched and reports no update.

    Replaces the old "no ANTHROPIC_API_KEY" gate. The page must keep its
    existing facts so the next pass can retry, rather than being recorded as
    enriched-with-nothing.
    """
    from alfred.core.local_llm import LocalLLMUnavailable

    ww.ensure_page("Acme Corp", "org", "notes/one.md")
    (cfg.vault_path / "notes").mkdir()
    (cfg.vault_path / "notes" / "one.md").write_text(
        "---\ntype: note\n---\nAcme Corp signed a big deal.\n", encoding="utf-8"
    )

    def _down(*a, **kw):
        raise LocalLLMUnavailable("connection refused")

    import alfred.core.local_llm as local_llm_mod
    monkeypatch.setattr(local_llm_mod, "complete", _down)

    assert ww.enrich_page("Acme Corp", ["notes/one.md"]) is False
    assert ww.state.state.wiki_pages["acme corp"].known_facts == []


def test_enrich_page_unknown_entity_returns_false(ww):
    assert ww.enrich_page("Nonexistent Entity", ["notes/one.md"]) is False


class _FakeContentBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeClient:
    """Stands in for the local backend's raw completion text.

    enrich_page imports complete_json lazily from alfred.core.local_llm, and
    complete_json calls complete() within that same module — so patching
    `alfred.core.local_llm.complete` intercepts the whole path while still
    exercising the real JSON parsing and fence-stripping.
    """

    def __init__(self, response_text: str) -> None:
        self.text = response_text
        self.calls: list[dict] = []

    def as_complete(self):
        def _complete(*args, **kwargs):
            self.calls.append(kwargs)
            return self.text
        return _complete


def test_enrich_page_extracts_new_facts_and_updates_vault(ww, cfg, monkeypatch):
    ww.ensure_page("Acme Corp", "org", "notes/one.md")
    (cfg.vault_path / "notes").mkdir()
    (cfg.vault_path / "notes" / "one.md").write_text(
        "---\ntype: note\n---\nAcme Corp signed a big deal.\n", encoding="utf-8"
    )

    fake_client = _FakeClient(json.dumps({
        "new_facts": ["Signed a big deal"],
        "related": ["Beta LLC"],
    }))
    # writer.enrich_page imports get_client lazily inside the method body, so
    # patch the source module rather than a (nonexistent) module-level name
    # on alfred.wiki.writer.
    import alfred.core.local_llm as local_llm_mod
    monkeypatch.setattr(local_llm_mod, "complete", fake_client.as_complete())

    updated = ww.enrich_page("Acme Corp", ["notes/one.md"])

    assert updated is True
    page = ww.state.state.wiki_pages["acme corp"]
    assert page.known_facts == ["Signed a big deal"]
    assert page.related == ["Beta LLC"]

    body = (cfg.vault_path / page.rel_path).read_text(encoding="utf-8")
    assert "Signed a big deal" in body
    assert "[[Beta LLC]]" in body


def test_enrich_page_dedupes_against_existing_facts_and_related(ww, cfg, monkeypatch):
    ww.ensure_page("Acme Corp", "org", "notes/one.md")
    page = ww.state.state.wiki_pages["acme corp"]
    page.known_facts.append("Already known fact")
    page.related.append("Beta LLC")

    (cfg.vault_path / "notes").mkdir()
    (cfg.vault_path / "notes" / "one.md").write_text(
        "---\ntype: note\n---\nsome content\n", encoding="utf-8"
    )

    fake_client = _FakeClient(json.dumps({
        "new_facts": ["Already known fact", "Brand new fact"],
        "related": ["Beta LLC", "Gamma Inc"],
    }))
    import alfred.core.local_llm as local_llm_mod
    monkeypatch.setattr(local_llm_mod, "complete", fake_client.as_complete())

    ww.enrich_page("Acme Corp", ["notes/one.md"])

    assert page.known_facts == ["Already known fact", "Brand new fact"]
    assert page.related == ["Beta LLC", "Gamma Inc"]


def test_enrich_page_no_new_facts_or_related_returns_false(ww, cfg, monkeypatch):
    ww.ensure_page("Acme Corp", "org", "notes/one.md")
    page = ww.state.state.wiki_pages["acme corp"]
    page.known_facts.append("Already known fact")

    (cfg.vault_path / "notes").mkdir()
    (cfg.vault_path / "notes" / "one.md").write_text(
        "---\ntype: note\n---\nsome content\n", encoding="utf-8"
    )

    fake_client = _FakeClient(json.dumps({
        "new_facts": ["Already known fact"],
        "related": [],
    }))
    import alfred.core.local_llm as local_llm_mod
    monkeypatch.setattr(local_llm_mod, "complete", fake_client.as_complete())

    assert ww.enrich_page("Acme Corp", ["notes/one.md"]) is False


def test_enrich_page_malformed_json_response_logged_and_returns_false(ww, cfg, monkeypatch):
    ww.ensure_page("Acme Corp", "org", "notes/one.md")

    (cfg.vault_path / "notes").mkdir()
    (cfg.vault_path / "notes" / "one.md").write_text(
        "---\ntype: note\n---\nsome content\n", encoding="utf-8"
    )

    fake_client = _FakeClient("not valid json at all")
    import alfred.core.local_llm as local_llm_mod
    monkeypatch.setattr(local_llm_mod, "complete", fake_client.as_complete())

    assert ww.enrich_page("Acme Corp", ["notes/one.md"]) is False


def test_enrich_page_strips_markdown_code_fences_from_response(ww, cfg, monkeypatch):
    ww.ensure_page("Acme Corp", "org", "notes/one.md")

    (cfg.vault_path / "notes").mkdir()
    (cfg.vault_path / "notes" / "one.md").write_text(
        "---\ntype: note\n---\nsome content\n", encoding="utf-8"
    )

    fenced = "```json\n" + json.dumps({"new_facts": ["Fenced fact"], "related": []}) + "\n```"
    fake_client = _FakeClient(fenced)
    import alfred.core.local_llm as local_llm_mod
    monkeypatch.setattr(local_llm_mod, "complete", fake_client.as_complete())

    updated = ww.enrich_page("Acme Corp", ["notes/one.md"])

    assert updated is True
    assert "Fenced fact" in ww.state.state.wiki_pages["acme corp"].known_facts


def test_enrich_page_no_readable_sources_returns_false(ww, cfg, monkeypatch):
    ww.ensure_page("Acme Corp", "org", "notes/one.md")
    # source_rel_path points at a file that doesn't exist -> vault_read raises,
    # gets caught, source_texts stays empty -> should bail before calling the LLM
    called = []
    import alfred.core.local_llm as local_llm_mod
    monkeypatch.setattr(
        local_llm_mod, "complete",
        lambda *a, **kw: called.append(True) or "{}",
    )

    result = ww.enrich_page("Acme Corp", ["notes/does-not-exist.md"])

    assert result is False
    assert called == [], "must not call the LLM when there are no readable source texts"
