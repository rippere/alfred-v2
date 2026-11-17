"""db.py upsert-by-(date,metric) behavior.

`upsert_snapshot` must be idempotent: re-running collect/backfill for the same
date should replace existing rows in place, never duplicate them, and must
overwrite stale values/meta/domain rather than leaving the old row behind.
"""
from __future__ import annotations

import json

import pytest

from alfred.ledger import db as L_db


@pytest.fixture
def conn(tmp_path):
    c = L_db.connect(tmp_path / "ledger.db")
    yield c
    c.close()


def _row_count(conn) -> int:
    cur = conn.execute("SELECT COUNT(*) FROM kpi_daily")
    return cur.fetchone()[0]


def test_upsert_same_date_metric_twice_does_not_duplicate(conn):
    rows = [{"domain": "engineering", "metric": "git_commits", "value": 3, "meta": None}]

    n1 = L_db.upsert_snapshot(conn, "2026-01-01", rows)
    n2 = L_db.upsert_snapshot(conn, "2026-01-01", rows)

    assert n1 == 1
    assert n2 == 1
    assert _row_count(conn) == 1

    stored = L_db.fetch_snapshot(conn, "2026-01-01")
    assert len(stored) == 1
    assert stored[0]["metric"] == "git_commits"
    assert stored[0]["value"] == 3


def test_upsert_overwrites_value_domain_and_meta(conn):
    L_db.upsert_snapshot(
        conn,
        "2026-01-01",
        [{"domain": "engineering", "metric": "git_commits", "value": 3, "meta": {"a": 1}}],
    )
    L_db.upsert_snapshot(
        conn,
        "2026-01-01",
        [{"domain": "knowledge", "metric": "git_commits", "value": 9, "meta": {"b": 2}}],
    )

    assert _row_count(conn) == 1
    row = L_db.fetch_snapshot(conn, "2026-01-01")[0]
    assert row["domain"] == "knowledge"
    assert row["value"] == 9
    assert row["meta"] == {"b": 2}


def test_upsert_bumps_computed_at_on_reupsert(conn):
    L_db.upsert_snapshot(
        conn, "2026-01-01", [{"domain": "engineering", "metric": "m", "value": 1, "meta": None}]
    )
    first = L_db.fetch_snapshot(conn, "2026-01-01")[0]["computed_at"]

    L_db.upsert_snapshot(
        conn, "2026-01-01", [{"domain": "engineering", "metric": "m", "value": 2, "meta": None}]
    )
    second = L_db.fetch_snapshot(conn, "2026-01-01")[0]["computed_at"]

    assert second >= first


def test_upsert_distinct_metrics_same_date_do_not_collide(conn):
    rows = [
        {"domain": "engineering", "metric": "git_commits", "value": 3, "meta": None},
        {"domain": "knowledge", "metric": "sessions", "value": 5, "meta": None},
    ]
    L_db.upsert_snapshot(conn, "2026-01-01", rows)
    L_db.upsert_snapshot(conn, "2026-01-01", rows)

    assert _row_count(conn) == 2
    metrics = {r["metric"] for r in L_db.fetch_snapshot(conn, "2026-01-01")}
    assert metrics == {"git_commits", "sessions"}


def test_upsert_same_metric_different_dates_creates_separate_rows(conn):
    row = [{"domain": "engineering", "metric": "git_commits", "value": 1, "meta": None}]
    L_db.upsert_snapshot(conn, "2026-01-01", row)
    L_db.upsert_snapshot(conn, "2026-01-02", row)

    assert _row_count(conn) == 2
    assert L_db.distinct_dates(conn) == ["2026-01-01", "2026-01-02"]


def test_upsert_persists_across_reconnect(tmp_path):
    db_path = tmp_path / "ledger.db"
    c1 = L_db.connect(db_path)
    L_db.upsert_snapshot(
        c1, "2026-01-01", [{"domain": "engineering", "metric": "git_commits", "value": 3, "meta": None}]
    )
    c1.close()

    c2 = L_db.connect(db_path)
    try:
        L_db.upsert_snapshot(
            c2, "2026-01-01", [{"domain": "engineering", "metric": "git_commits", "value": 3, "meta": None}]
        )
        assert _row_count(c2) == 1
    finally:
        c2.close()


def test_upsert_meta_json_round_trips(conn):
    meta = {"branch": "main", "n": 2}
    L_db.upsert_snapshot(
        conn, "2026-01-01", [{"domain": "engineering", "metric": "git_commits", "value": 1, "meta": meta}]
    )
    row = L_db.fetch_snapshot(conn, "2026-01-01")[0]
    assert row["meta"] == meta

    # Verify it's genuinely stored as a JSON string under the hood, not the dict itself.
    cur = conn.execute("SELECT meta FROM kpi_daily WHERE date=? AND metric=?", ("2026-01-01", "git_commits"))
    raw = cur.fetchone()[0]
    assert json.loads(raw) == meta
