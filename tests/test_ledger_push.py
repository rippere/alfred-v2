"""push.py credential-presence branching.

Missing NovaCRM/Supabase credentials must degrade `push_snapshot` to a dry run
("dry" status, summary-only detail) without ever attempting a network call —
no Supabase auth POST, no KPI PUT.
"""
from __future__ import annotations

import pytest

from alfred.ledger import config as ledger_config
from alfred.ledger import push as ledger_push

_ROWS = [
    {"domain": "engineering", "metric": "git_commits", "value": 3, "meta": None},
]


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Point config at a scratch dir with no env file, and scrub any real
    LEDGER_* creds that might be sitting in the actual process environment."""
    monkeypatch.setattr(ledger_config, "ENV_DIR", tmp_path / "alfred-ledger")
    monkeypatch.setattr(ledger_config, "ENV_FILE", tmp_path / "alfred-ledger" / "env")
    monkeypatch.setattr(ledger_config, "ENV_EXAMPLE", tmp_path / "alfred-ledger" / "env.example")
    for key in ledger_config.PUSH_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _boom_network(*_a, **_kw):
    raise AssertionError("no network call should be attempted in dry-run mode")


def test_push_snapshot_is_dry_when_no_credentials(monkeypatch):
    monkeypatch.setattr(ledger_push.httpx, "post", _boom_network)
    monkeypatch.setattr(ledger_push.httpx, "put", _boom_network)

    status, detail = ledger_push.push_snapshot("2026-01-01", _ROWS)

    assert status == "dry"
    assert "DRY RUN" in detail
    assert "2026-01-01" in detail


def test_push_snapshot_dry_detail_lists_missing_keys(monkeypatch):
    monkeypatch.setattr(ledger_push.httpx, "post", _boom_network)
    monkeypatch.setattr(ledger_push.httpx, "put", _boom_network)

    status, detail = ledger_push.push_snapshot("2026-01-01", _ROWS)

    assert status == "dry"
    for key in ledger_push._REQUIRED:
        assert key in detail


def test_push_snapshot_dry_with_partial_credentials(monkeypatch):
    """Even one missing required key must still short-circuit to dry-run."""
    monkeypatch.setattr(ledger_push.httpx, "post", _boom_network)
    monkeypatch.setattr(ledger_push.httpx, "put", _boom_network)
    monkeypatch.setenv("LEDGER_SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("LEDGER_SUPABASE_ANON_KEY", "anon-key")
    monkeypatch.setenv("LEDGER_API_URL", "https://api.example.com")
    monkeypatch.setenv("LEDGER_WORKSPACE_ID", "ws-1")
    monkeypatch.setenv("LEDGER_EMAIL", "you@example.com")
    # LEDGER_PASSWORD deliberately left unset.

    status, detail = ledger_push.push_snapshot("2026-01-01", _ROWS)

    assert status == "dry"
    assert "LEDGER_PASSWORD" in detail


def test_push_snapshot_ensures_env_scaffold_even_when_dry(monkeypatch):
    monkeypatch.setattr(ledger_push.httpx, "post", _boom_network)
    monkeypatch.setattr(ledger_push.httpx, "put", _boom_network)

    ledger_push.push_snapshot("2026-01-01", _ROWS)

    assert ledger_config.ENV_EXAMPLE.exists()


def test_push_snapshot_attempts_network_when_credentials_present(monkeypatch):
    """Sanity check for the flip side: full credentials DO trigger a real call,
    confirming the dry-run branch above is actually gating the network path
    (rather than e.g. httpx being mocked out entirely upstream)."""
    called = {"post": False}

    class _FakeAuthResp:
        is_success = True

        def json(self):
            return {"access_token": "tok-123"}

    class _FakePutResp:
        is_success = True
        status_code = 200
        text = "ok"

    def _fake_post(*a, **kw):
        called["post"] = True
        return _FakeAuthResp()

    def _fake_put(*a, **kw):
        return _FakePutResp()

    monkeypatch.setattr(ledger_push.httpx, "post", _fake_post)
    monkeypatch.setattr(ledger_push.httpx, "put", _fake_put)
    monkeypatch.setenv("LEDGER_SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("LEDGER_SUPABASE_ANON_KEY", "anon-key")
    monkeypatch.setenv("LEDGER_API_URL", "https://api.example.com")
    monkeypatch.setenv("LEDGER_WORKSPACE_ID", "ws-1")
    monkeypatch.setenv("LEDGER_EMAIL", "you@example.com")
    monkeypatch.setenv("LEDGER_PASSWORD", "secret")

    status, detail = ledger_push.push_snapshot("2026-01-01", _ROWS)

    assert called["post"] is True
    assert status == "ok"
