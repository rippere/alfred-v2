"""Integration-style test for alfred.bridge.enrich_crm — the full dry-run
flow end to end, with every external dependency mocked: CRM GET calls
(httpx.get/httpx.post monkeypatched module-wide), vault entity resolution,
and brief/LLM synthesis. No real network call is made anywhere in this file.

The core assertion this file exists to make: a full pass over several deals
(mixed closed/active, matched/unmatched, fresh/duplicate) produces correct
summary counts AND issues zero note-creation POST calls — dry_run is a hard
guarantee, not a best-effort one.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from alfred.bridge import enrich_crm
from alfred.bridge.brief import Brief
from alfred.bridge.resolve import EntityMatch

ALL_ENV_KEYS = (
    "LEDGER_SUPABASE_URL",
    "LEDGER_SUPABASE_ANON_KEY",
    "LEDGER_API_URL",
    "LEDGER_WORKSPACE_ID",
    "LEDGER_EMAIL",
    "LEDGER_PASSWORD",
)


@pytest.fixture(autouse=True)
def isolate_env(tmp_path, monkeypatch):
    """Point the shared ledger env file at a scratch dir and clear process
    env vars, so tests never read (or are affected by) a real
    ~/.config/alfred-ledger/env on the host machine."""
    from alfred.ledger import config as ledger_config

    cfg_dir = tmp_path / "alfred-ledger-cfg"
    monkeypatch.setattr(ledger_config, "ENV_DIR", cfg_dir)
    monkeypatch.setattr(ledger_config, "ENV_FILE", cfg_dir / "env")
    monkeypatch.setattr(ledger_config, "ENV_EXAMPLE", cfg_dir / "env.example")
    for key in ALL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _set_all_creds(monkeypatch):
    monkeypatch.setenv("LEDGER_SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("LEDGER_SUPABASE_ANON_KEY", "anon-key")
    monkeypatch.setenv("LEDGER_API_URL", "https://api.example.com")
    monkeypatch.setenv("LEDGER_WORKSPACE_ID", "ws-123")
    monkeypatch.setenv("LEDGER_EMAIL", "bot@example.com")
    monkeypatch.setenv("LEDGER_PASSWORD", "hunter2")


class _FakeResponse:
    def __init__(self, *, json_body=None, status_code=200, text=""):
        self._json_body = json_body
        self.status_code = status_code
        self.text = text
        self.is_success = 200 <= status_code < 300

    def json(self):
        return self._json_body

    def raise_for_status(self):
        if not self.is_success:
            raise RuntimeError(f"HTTP {self.status_code}")


# 5 deals: one closed (excluded client-side), one with no contact (skipped),
# and three with contacts covering "matched + fresh", "no vault match", and
# "matched + already-posted duplicate".
DEALS = [
    {"id": "deal-1", "contact_id": "contact-1", "stage": "discovery"},
    {"id": "deal-2", "contact_id": "contact-2", "stage": "proposal"},
    {"id": "deal-3", "contact_id": "contact-3", "stage": "closed_won"},
    {"id": "deal-4", "contact_id": None, "stage": "lead"},
    {"id": "deal-5", "contact_id": "contact-5", "stage": "negotiation"},
]

CONTACTS = {
    "contact-1": {"id": "contact-1", "email": "alice@example.com", "name": "Alice"},
    "contact-2": {"id": "contact-2", "email": "bob@example.com", "name": "Bob"},
    "contact-5": {"id": "contact-5", "email": "eve@example.com", "name": "Eve"},
}


def _existing_notes():
    return {
        "deal-1": [],
        "deal-5": [{"body": "prior brief\n\n<!-- alfred:brief hash=hashEEE555555 -->"}],
    }


class _CrmFake:
    """Routes httpx.get/httpx.post calls by URL for the full enrich flow."""

    def __init__(self, existing_notes=None):
        self.get_calls: list[str] = []
        self.post_calls: list[str] = []
        self.existing_notes = existing_notes if existing_notes is not None else _existing_notes()

    def fake_get(self, url, *, headers=None, timeout=None, params=None):
        self.get_calls.append(url)
        if url.endswith("/deals"):
            return _FakeResponse(json_body=DEALS)
        if "/contacts/" in url:
            contact_id = url.rstrip("/").rsplit("/", 1)[-1]
            if contact_id in CONTACTS:
                return _FakeResponse(json_body=CONTACTS[contact_id])
            return _FakeResponse(json_body=None, status_code=404)
        if "/deals/" in url and url.endswith("/notes"):
            deal_id = url.split("/deals/")[1].split("/")[0]
            return _FakeResponse(json_body=self.existing_notes.get(deal_id, []))
        raise AssertionError(f"unexpected GET {url}")

    def fake_post(self, url, *, headers=None, json=None, timeout=None):
        self.post_calls.append(url)
        if "auth/v1/token" in url:
            return _FakeResponse(json_body={"access_token": "fake-token"})
        # Anything else is a note-creation write.
        return _FakeResponse(json_body={"id": "note-x", "body": (json or {}).get("body", "")}, status_code=201)

    @property
    def note_write_calls(self):
        return [u for u in self.post_calls if "auth/v1/token" not in u]


def _fake_resolve(vault_path, *, email, name=None, ignore_dirs=None):
    if email == "alice@example.com":
        return EntityMatch(rel_path="person/Alice.md", entity_type="person", name="Alice")
    if email == "eve@example.com":
        return EntityMatch(rel_path="person/Eve.md", entity_type="person", name="Eve")
    return None  # bob: no vault match


def _fake_synthesize(engine, entity, *, top_k=6, **kwargs):
    if entity.name == "Alice":
        return Brief(
            entity_rel_path=entity.rel_path, entity_name="Alice",
            text="Alice brief text.", note_hash="hashAAA111111", source_paths=[],
        )
    if entity.name == "Eve":
        return Brief(
            entity_rel_path=entity.rel_path, entity_name="Eve",
            text="Eve brief text.", note_hash="hashEEE555555", source_paths=[],
        )
    raise AssertionError(f"unexpected entity passed to synthesize: {entity.name}")


def _patch_flow(monkeypatch, crm):
    monkeypatch.setattr("httpx.get", crm.fake_get)
    monkeypatch.setattr("httpx.post", crm.fake_post)
    monkeypatch.setattr(enrich_crm, "resolve_contact_to_entity", _fake_resolve)
    monkeypatch.setattr(enrich_crm, "synthesize_entity_brief", _fake_synthesize)


def test_run_enrich_full_dry_run_flow(monkeypatch):
    _set_all_creds(monkeypatch)
    crm = _CrmFake()
    _patch_flow(monkeypatch, crm)

    cfg = SimpleNamespace(vault_path=Path("/does/not/matter"))
    summary = enrich_crm.run_enrich(cfg, object(), dry_run=True)

    assert summary.deals_seen == 4          # 5 total minus 1 closed_won
    assert summary.contacts_seen == 3        # deal-4 has no contact_id at all
    assert summary.matched == 2              # alice + eve; bob has no vault match
    assert summary.skipped_no_match == 1
    assert summary.briefs_synthesized == 2
    assert summary.skipped_no_brief == 0
    assert summary.would_post == 1           # alice: no existing notes on deal-1
    assert summary.skipped_duplicate == 1    # eve: hash already present on deal-5
    assert summary.posted == 0
    assert summary.errors == 0

    # The hard guarantee: zero writes, no matter what dry_run's internals did.
    assert crm.note_write_calls == []

    # Per-contact decisions are recorded for CLI/dry-run inspection.
    outcomes = {d["deal_id"]: d.get("outcome") for d in summary.decisions}
    assert outcomes["deal-1"] == "would_post"
    assert outcomes["deal-2"] == "no_match"
    assert outcomes["deal-5"] == "skipped_duplicate"
    assert "deal-4" not in outcomes  # skipped before any decision was recorded
    assert "deal-3" not in outcomes  # filtered out as closed_won, never fetched


def test_run_enrich_missing_creds_makes_zero_http_calls(monkeypatch):
    crm = _CrmFake()
    _patch_flow(monkeypatch, crm)

    cfg = SimpleNamespace(vault_path=Path("/does/not/matter"))
    summary = enrich_crm.run_enrich(cfg, object(), dry_run=True)

    assert summary.as_dict() == {
        "deals_seen": 0, "contacts_seen": 0, "matched": 0, "skipped_no_match": 0,
        "briefs_synthesized": 0, "skipped_no_brief": 0, "would_post": 0,
        "posted": 0, "skipped_duplicate": 0, "errors": 0,
    }
    assert summary.decisions == []
    assert crm.get_calls == []
    assert crm.post_calls == []


def test_run_enrich_dry_run_true_never_posts_even_when_push_flag_semantics_checked(monkeypatch):
    """Explicit call-count proof that dry_run=True issues no note-creation
    POST, mirroring the same style of assertion used in test_bridge_notes.py."""
    _set_all_creds(monkeypatch)
    crm = _CrmFake()
    _patch_flow(monkeypatch, crm)

    cfg = SimpleNamespace(vault_path=Path("/does/not/matter"))
    enrich_crm.run_enrich(cfg, object(), dry_run=True)

    # Every POST that happened was a Supabase auth grant, never a CRM write.
    assert crm.post_calls, "expected at least the auth POST(s) to have happened"
    assert all("auth/v1/token" in u for u in crm.post_calls)


def test_run_enrich_push_true_still_skips_all_duplicates(monkeypatch):
    """Even with dry_run=False, a run where every match is already a
    duplicate must perform zero note-creation POSTs — proves the dedup gate
    (not dry_run alone) is what prevents reposting."""
    _set_all_creds(monkeypatch)
    notes = _existing_notes()
    notes["deal-1"] = [{"body": "x\n\n<!-- alfred:brief hash=hashAAA111111 -->"}]
    crm = _CrmFake(existing_notes=notes)
    _patch_flow(monkeypatch, crm)

    cfg = SimpleNamespace(vault_path=Path("/does/not/matter"))
    summary = enrich_crm.run_enrich(cfg, object(), dry_run=False)

    assert summary.would_post == 0
    assert summary.skipped_duplicate == 2  # alice + eve both duplicates now
    assert summary.posted == 0
    assert crm.note_write_calls == []


def test_run_enrich_auth_failure_reports_error_and_makes_no_deal_calls(monkeypatch):
    _set_all_creds(monkeypatch)

    def failing_post(url, *, headers=None, json=None, timeout=None):
        if "auth/v1/token" in url:
            return _FakeResponse(status_code=401, text="invalid credentials")
        raise AssertionError("should never reach a CRM write when auth fails")

    def unexpected_get(url, *, headers=None, timeout=None, params=None):
        raise AssertionError(f"should never GET {url} when auth fails")

    monkeypatch.setattr("httpx.get", unexpected_get)
    monkeypatch.setattr("httpx.post", failing_post)
    monkeypatch.setattr(enrich_crm, "resolve_contact_to_entity", _fake_resolve)
    monkeypatch.setattr(enrich_crm, "synthesize_entity_brief", _fake_synthesize)

    cfg = SimpleNamespace(vault_path=Path("/does/not/matter"))
    summary = enrich_crm.run_enrich(cfg, object(), dry_run=True)

    assert summary.errors == 1
    assert summary.deals_seen == 0
    assert summary.decisions == []
