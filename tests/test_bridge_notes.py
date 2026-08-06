"""Unit tests for alfred.bridge.notes — note posting, dedup, dry-run split.

All HTTP is mocked (httpx.get / httpx.post monkeypatched module-wide); no
test in this file makes a real network call. The env file is isolated per
test via the `isolate_env` autouse fixture so these tests never read (or are
affected by) a real ``~/.config/alfred-ledger/env`` on the host machine.
"""
from __future__ import annotations

import pytest

from alfred.bridge import config as C
from alfred.bridge import notes


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
    env vars, so tests never read or depend on the real ~/.config file."""
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


class _HttpRecorder:
    """Records every httpx.get/httpx.post call and routes canned responses
    by URL, so tests can assert exactly which endpoints were (or weren't)
    hit — in particular, that the note-creation POST never fires under
    dry_run / duplicate-hash scenarios."""

    def __init__(self, *, existing_notes=None, create_status=201):
        self.get_calls: list[dict] = []
        self.post_calls: list[dict] = []
        self.existing_notes = existing_notes if existing_notes is not None else []
        self.create_status = create_status

    def fake_get(self, url, *, headers=None, timeout=None):
        self.get_calls.append({"url": url, "headers": headers, "timeout": timeout})
        return _FakeResponse(json_body=self.existing_notes)

    def fake_post(self, url, *, headers=None, json=None, timeout=None):
        self.post_calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        if "auth/v1/token" in url:
            return _FakeResponse(json_body={"access_token": "fake-token"})
        # A note-creation POST.
        return _FakeResponse(
            json_body={"id": "note-1", "body": (json or {}).get("body", "")},
            status_code=self.create_status,
        )

    @property
    def note_post_calls(self):
        return [c for c in self.post_calls if "auth/v1/token" not in c["url"]]


def test_build_note_body_embeds_marker():
    body = notes.build_note_body("Some brief prose.", "abc123def456")
    assert body.startswith("Some brief prose.")
    assert f"{C.NOTE_MARKER_PREFIX}abc123def456{C.NOTE_MARKER_SUFFIX}" in body


def test_build_note_payload_shape():
    payload = notes.build_note_payload("body text", author="alfred")
    assert payload == {"body": "body text", "author": "alfred"}


def test_has_matching_hash_true_and_false():
    marker = f"{C.NOTE_MARKER_PREFIX}deadbeef1234{C.NOTE_MARKER_SUFFIX}"
    existing = [{"body": f"prose\n\n{marker}"}]
    assert notes.has_matching_hash(existing, "deadbeef1234") is True
    assert notes.has_matching_hash(existing, "other000000") is False
    assert notes.has_matching_hash([], "deadbeef1234") is False


def test_post_note_missing_creds_makes_zero_http_calls(monkeypatch):
    rec = _HttpRecorder()
    monkeypatch.setattr("httpx.get", rec.fake_get)
    monkeypatch.setattr("httpx.post", rec.fake_post)

    status, detail = notes.post_note("deals", "deal-1", "brief text", "hash1234abcd", dry_run=True)

    assert status == "dry"
    assert "missing" in detail.lower()
    assert rec.get_calls == []
    assert rec.post_calls == []


def test_post_note_dry_run_checks_dedup_but_never_posts(monkeypatch):
    _set_all_creds(monkeypatch)
    rec = _HttpRecorder(existing_notes=[])  # no existing notes -> not a dup
    monkeypatch.setattr("httpx.get", rec.fake_get)
    monkeypatch.setattr("httpx.post", rec.fake_post)

    status, detail = notes.post_note(
        "deals", "deal-1", "brief text", "hash1234abcd", dry_run=True,
    )

    assert status == "dry"
    assert "deals/deal-1" in detail
    # Auth happened and existing notes were listed (the dedup check runs)...
    assert len(rec.get_calls) == 1
    assert "deals/deal-1/notes" in rec.get_calls[0]["url"]
    # ...but the actual note-creation POST was never issued.
    assert rec.note_post_calls == []


def test_post_note_dry_run_true_makes_zero_write_calls_regardless_of_dedup_state(monkeypatch):
    """Explicit assertion (via mock call count) that dry_run=True never
    performs a real write, whether or not a duplicate exists."""
    _set_all_creds(monkeypatch)
    rec = _HttpRecorder(existing_notes=[])
    post_spy = []
    orig_fake_post = rec.fake_post

    def counting_post(*args, **kwargs):
        post_spy.append((args, kwargs))
        return orig_fake_post(*args, **kwargs)

    monkeypatch.setattr("httpx.get", rec.fake_get)
    monkeypatch.setattr("httpx.post", counting_post)

    notes.post_note("contacts", "c-1", "brief text", "hashabc123456", dry_run=True)

    # Exactly one POST (the Supabase auth grant) — zero writes to the CRM.
    assert len(post_spy) == 1
    assert "auth/v1/token" in post_spy[0][0][0]


def test_post_note_skips_when_hash_already_present(monkeypatch):
    _set_all_creds(monkeypatch)
    marker = f"{C.NOTE_MARKER_PREFIX}hash1234abcd{C.NOTE_MARKER_SUFFIX}"
    rec = _HttpRecorder(existing_notes=[{"body": f"prior brief\n\n{marker}"}])
    monkeypatch.setattr("httpx.get", rec.fake_get)
    monkeypatch.setattr("httpx.post", rec.fake_post)

    # dry_run=False here specifically to prove the SKIP is due to dedup, not
    # the dry_run guard — a duplicate must never repost even when pushing.
    status, detail = notes.post_note(
        "deals", "deal-1", "brief text", "hash1234abcd", dry_run=False,
    )

    assert status == "skip"
    assert "hash1234abcd" in detail
    assert rec.note_post_calls == []


def test_post_note_push_posts_when_not_duplicate(monkeypatch):
    _set_all_creds(monkeypatch)
    rec = _HttpRecorder(existing_notes=[], create_status=201)
    monkeypatch.setattr("httpx.get", rec.fake_get)
    monkeypatch.setattr("httpx.post", rec.fake_post)

    status, detail = notes.post_note(
        "contacts", "c-42", "brief text", "hashabc123456", dry_run=False, author="alfred",
    )

    assert status == "ok"
    assert len(rec.note_post_calls) == 1
    call = rec.note_post_calls[0]
    assert "contacts/c-42/notes" in call["url"]
    assert call["json"]["author"] == "alfred"
    assert notes.C.NOTE_MARKER_PREFIX.strip("<!- ") in call["json"]["body"] or "hashabc123456" in call["json"]["body"]


def test_post_note_push_reports_error_on_http_failure(monkeypatch):
    _set_all_creds(monkeypatch)
    rec = _HttpRecorder(existing_notes=[], create_status=500)
    monkeypatch.setattr("httpx.get", rec.fake_get)
    monkeypatch.setattr("httpx.post", rec.fake_post)

    status, detail = notes.post_note(
        "deals", "deal-1", "brief text", "hashabc123456", dry_run=False,
    )

    assert status == "error"
    assert "500" in detail
