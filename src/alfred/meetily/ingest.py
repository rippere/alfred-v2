"""Ingest Meetily meetings into Alfred's vault inbox.

For each Meetily meeting not previously synced, render a `type: meeting`
markdown note into `<vault>/inbox/`. A JSON sidecar in Alfred's data_dir tracks
which `meetily_id`s have been ingested so re-running is idempotent.

We deliberately write to `inbox/` rather than straight to `meeting/`, so the
existing Curator daemon owns filing, slug-collision handling, and project
auto-linking — this module stays a pure producer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from alfred.meetily import record as R
from alfred.meetily.config import SYNCED_STATE_FILENAME
from alfred.meetily.reader import Meeting, read_meetings


@dataclass
class SyncResult:
    scanned: int = 0
    ingested: int = 0
    skipped: int = 0
    ingested_titles: list[str] | None = None


def _synced_path(data_dir: Path) -> Path:
    return Path(data_dir) / SYNCED_STATE_FILENAME


def load_synced(data_dir: Path) -> set[str]:
    p = _synced_path(data_dir)
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text()))
    except (json.JSONDecodeError, OSError):
        return set()


def save_synced(data_dir: Path, ids: set[str]) -> None:
    p = _synced_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sorted(ids), indent=0))


def sync(
    db_path: Path,
    vault_path: Path,
    data_dir: Path,
    *,
    since: str | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> SyncResult:
    """Ingest new Meetily meetings into the vault inbox. Idempotent."""
    meetings: list[Meeting] = read_meetings(db_path, since=since)
    synced = set() if force else load_synced(data_dir)

    inbox = Path(vault_path) / "inbox"
    result = SyncResult(scanned=len(meetings), ingested_titles=[])

    for m in meetings:
        if m.id in synced:
            result.skipped += 1
            continue
        note = R.to_markdown(m)
        dest = inbox / R.inbox_filename(m)
        if not dry_run:
            inbox.mkdir(parents=True, exist_ok=True)
            dest.write_text(note, encoding="utf-8")
            synced.add(m.id)
        result.ingested += 1
        result.ingested_titles.append(m.title)

    if not dry_run and result.ingested:
        save_synced(data_dir, synced)
    return result
