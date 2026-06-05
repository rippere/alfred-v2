"""SQLite time-series store for the life ledger.

Schema:
    kpi_daily(date TEXT, domain TEXT, metric TEXT, value REAL,
              meta TEXT, computed_at TEXT, PRIMARY KEY(date, metric))
    push_log(date TEXT, pushed_at TEXT, status TEXT, detail TEXT)

`meta` is JSON-encoded (or NULL). Upserting a snapshot replaces same-day rows
for the same metric so re-running collect/backfill is idempotent.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from alfred.ledger import config as C

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kpi_daily (
    date        TEXT NOT NULL,
    domain      TEXT NOT NULL,
    metric      TEXT NOT NULL,
    value       REAL,
    meta        TEXT,
    computed_at TEXT NOT NULL,
    PRIMARY KEY (date, metric)
);
CREATE INDEX IF NOT EXISTS idx_kpi_daily_date ON kpi_daily(date);
CREATE INDEX IF NOT EXISTS idx_kpi_daily_domain ON kpi_daily(domain);

CREATE TABLE IF NOT EXISTS push_log (
    date      TEXT NOT NULL,
    pushed_at TEXT NOT NULL,
    status    TEXT NOT NULL,
    detail    TEXT
);
"""


def connect(path: Optional[Path] = None) -> sqlite3.Connection:
    """Open (creating if needed) the ledger DB and ensure the schema exists."""
    db_path = Path(path) if path else C.LEDGER_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_snapshot(
    conn: sqlite3.Connection,
    date: str,
    rows: Iterable[dict],
) -> int:
    """Insert/replace all metric rows for `date`. Returns the row count written."""
    computed_at = _now()
    payload = [
        (
            date,
            r["domain"],
            r["metric"],
            r["value"],
            json.dumps(r["meta"]) if r.get("meta") is not None else None,
            computed_at,
        )
        for r in rows
    ]
    conn.executemany(
        "INSERT INTO kpi_daily (date, domain, metric, value, meta, computed_at) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(date, metric) DO UPDATE SET "
        "  domain=excluded.domain, value=excluded.value, "
        "  meta=excluded.meta, computed_at=excluded.computed_at",
        payload,
    )
    conn.commit()
    return len(payload)


def fetch_snapshot(conn: sqlite3.Connection, date: str) -> list[dict]:
    """All rows for a single date, ordered by domain then metric."""
    cur = conn.execute(
        "SELECT date, domain, metric, value, meta, computed_at "
        "FROM kpi_daily WHERE date = ? ORDER BY domain, metric",
        (date,),
    )
    return [_row_to_dict(r) for r in cur.fetchall()]


def fetch_range(conn: sqlite3.Connection, since: str, until: str) -> list[dict]:
    """All rows with `since <= date <= until`, ordered by date/domain/metric."""
    cur = conn.execute(
        "SELECT date, domain, metric, value, meta, computed_at "
        "FROM kpi_daily WHERE date >= ? AND date <= ? "
        "ORDER BY date, domain, metric",
        (since, until),
    )
    return [_row_to_dict(r) for r in cur.fetchall()]


def distinct_dates(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute("SELECT DISTINCT date FROM kpi_daily ORDER BY date")
    return [r[0] for r in cur.fetchall()]


def domain_row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """{domain: total rows} across the whole DB (for backfill reporting)."""
    cur = conn.execute(
        "SELECT domain, COUNT(*) FROM kpi_daily GROUP BY domain ORDER BY domain"
    )
    return {r[0]: r[1] for r in cur.fetchall()}


def record_push(
    conn: sqlite3.Connection,
    date: str,
    status: str,
    detail: str = "",
) -> None:
    """Append a row to push_log."""
    conn.execute(
        "INSERT INTO push_log (date, pushed_at, status, detail) VALUES (?, ?, ?, ?)",
        (date, _now(), status, detail),
    )
    conn.commit()


def _row_to_dict(r: sqlite3.Row) -> dict:
    return {
        "date": r["date"],
        "domain": r["domain"],
        "metric": r["metric"],
        "value": r["value"],
        "meta": json.loads(r["meta"]) if r["meta"] else None,
        "computed_at": r["computed_at"],
    }
