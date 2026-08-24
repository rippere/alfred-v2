"""Exit-code behavior of `alfred ledger collect`/`backfill` around --push status.

ledger-collect.service relies on a non-zero process exit to trigger its
OnFailure= systemd alert. A real push failure ("error") must exit non-zero;
a "dry" run (no creds configured — intentional) and "ok" must not.
"""
from __future__ import annotations

from typing import Optional

import pytest
from typer.testing import CliRunner

from alfred.ledger import cli as ledger_cli
from alfred.ledger import collect as ledger_collect
from alfred.ledger import config as ledger_config
from alfred.ledger import push as ledger_push

runner = CliRunner()

_ROWS = [
    {"domain": "engineering", "metric": "git_commits", "value": 3, "meta": None},
]


class _StubComputer:
    """Stand-in for LedgerComputer that never touches the real filesystem."""

    def __init__(self) -> None:
        pass

    def snapshot(self, date: str) -> list[dict]:
        return list(_ROWS)


@pytest.fixture(autouse=True)
def _stub_heavy_deps(tmp_path, monkeypatch):
    """Redirect the DB to a tmp file and stub out the filesystem-scanning computer."""
    monkeypatch.setattr(ledger_config, "LEDGER_DB", tmp_path / "ledger.db")
    monkeypatch.setattr(ledger_collect, "LedgerComputer", _StubComputer)


def _stub_push(status: str, detail: str = "stubbed"):
    def _fake_push_snapshot(date: str, rows: list[dict], *, timeout: float = 30.0):
        return status, detail

    return _fake_push_snapshot


@pytest.mark.parametrize(
    "status,expect_nonzero",
    [
        ("error", True),
        ("dry", False),
        ("ok", False),
    ],
)
def test_collect_push_exit_code(monkeypatch, status, expect_nonzero):
    monkeypatch.setattr(ledger_push, "push_snapshot", _stub_push(status))
    result = runner.invoke(ledger_cli.ledger_app, ["collect", "--date", "2026-01-01", "--push"])
    if expect_nonzero:
        assert result.exit_code != 0
    else:
        assert result.exit_code == 0


def test_collect_no_push_always_exits_zero(monkeypatch):
    # Push status must not even be consulted when --push wasn't requested.
    def _boom(*a, **kw):
        raise AssertionError("push_snapshot should not be called without --push")

    monkeypatch.setattr(ledger_push, "push_snapshot", _boom)
    result = runner.invoke(ledger_cli.ledger_app, ["collect", "--date", "2026-01-01"])
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "status,expect_nonzero",
    [
        ("error", True),
        ("dry", False),
        ("ok", False),
    ],
)
def test_backfill_push_exit_code(monkeypatch, status, expect_nonzero):
    monkeypatch.setattr(ledger_push, "push_snapshot", _stub_push(status))
    result = runner.invoke(
        ledger_cli.ledger_app,
        ["backfill", "--since", "2026-01-01", "--until", "2026-01-02", "--push"],
    )
    if expect_nonzero:
        assert result.exit_code != 0
    else:
        assert result.exit_code == 0


def test_backfill_reports_and_continues_across_range(monkeypatch):
    """One bad day's push shouldn't stop the rest of the range from running."""
    calls: list[str] = []

    def _flaky_push(date: str, rows: list[dict], *, timeout: float = 30.0):
        calls.append(date)
        # Fail only the first day; the rest should still be attempted.
        return ("error" if date == "2026-01-01" else "ok"), "stubbed"

    monkeypatch.setattr(ledger_push, "push_snapshot", _flaky_push)
    result = runner.invoke(
        ledger_cli.ledger_app,
        ["backfill", "--since", "2026-01-01", "--until", "2026-01-03", "--push"],
    )
    assert calls == ["2026-01-01", "2026-01-02", "2026-01-03"]
    assert result.exit_code != 0
