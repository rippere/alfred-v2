"""Alfred "life ledger" collector.

A daily snapshot of personal KPIs computed from Alfred's vault data, local git
repositories, and PM-agent briefs. Snapshots are stored in a local SQLite
time-series (``data/ledger.db``) and can optionally be pushed to a NovaCRM
workspace KPI endpoint.

Public surface:
    - ``alfred ledger collect [--date YYYY-MM-DD] [--push]``
    - ``alfred ledger backfill --since YYYY-MM-DD [--until YYYY-MM-DD] [--push]``
    - ``alfred ledger show [--days N]``

Everything is read-only against the source systems; the only writes are to the
ledger SQLite DB and (optionally) the remote CRM API.
"""

from __future__ import annotations

__all__ = ["config", "sources", "collect", "db", "push"]
