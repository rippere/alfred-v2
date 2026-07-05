"""Meetily integration config.

Resolution order for the Meetily SQLite path:
  1. explicit --db CLI flag
  2. $MEETILY_DB environment variable
  3. `meetily.db_path` in the Alfred config.yaml
  4. the platform default guesses below

Meetily stores its DB under the app data dir. Exact location depends on the
build (legacy FastAPI vs. current Tauri) and OS, so we probe a few candidates
rather than hard-coding one. Override with $MEETILY_DB if none match.
"""
from __future__ import annotations

import os
from pathlib import Path

# Sidecar recording which Meetily meeting ids Alfred has already ingested,
# so re-running `sync` is idempotent. Lives in Alfred's data_dir.
SYNCED_STATE_FILENAME = "meetily_synced.json"

# Best-effort default locations for Meetily's SQLite DB, newest-build first.
# TODO(ben): confirm the real path on your machine and pin it in config.yaml.
_DEFAULT_DB_CANDIDATES = [
    "~/.local/share/meetily/meeting_minutes.db",          # Linux (Tauri appdata)
    "~/.local/share/com.meetily.app/meeting_minutes.db",
    "~/Library/Application Support/meetily/meeting_minutes.db",  # macOS
    "~/Library/Application Support/com.meetily.app/meeting_minutes.db",
    "~/meeting-minutes/backend/app/meeting_minutes.db",   # legacy FastAPI checkout
]


def resolve_db_path(explicit: str | None = None, cfg_path: str | None = None) -> Path | None:
    """Return the first Meetily DB path that exists, or None if none found."""
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    if env := os.environ.get("MEETILY_DB"):
        candidates.append(env)
    if cfg_path:
        candidates.append(cfg_path)
    candidates.extend(_DEFAULT_DB_CANDIDATES)

    for c in candidates:
        p = Path(c).expanduser()
        if p.exists():
            return p
    return None
