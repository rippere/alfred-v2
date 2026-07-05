"""Read Meetily's local SQLite store.

Schema (as of Meetily main; both the legacy FastAPI backend and the current
Tauri build use the same shape):

    meetings(
        id TEXT PRIMARY KEY, title TEXT, created_at TEXT,
        updated_at TEXT, folder_path TEXT)

    transcripts(
        id TEXT PRIMARY KEY, meeting_id TEXT, transcript TEXT, timestamp TEXT,
        summary TEXT, action_items TEXT, key_points TEXT,
        audio_start_time REAL, audio_end_time REAL, duration REAL)

    summary_processes(
        meeting_id TEXT PRIMARY KEY, status TEXT, created_at TEXT,
        updated_at TEXT, error TEXT, result TEXT,  -- result = json.dumps(summary)
        start_time TEXT, end_time TEXT, chunk_count INTEGER,
        processing_time REAL, metadata TEXT)

We open the DB read-only (`mode=ro`) — Alfred must never mutate Meetily's data.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Meeting:
    """A denormalised Meetily meeting: metadata + stitched transcript + summary."""
    id: str
    title: str
    created_at: str
    updated_at: str
    # Concatenated transcript text across all transcript rows for this meeting.
    transcript: str = ""
    # Per-row summary/action/key-point text (Meetily writes these onto transcripts).
    summary: str = ""
    action_items: str = ""
    key_points: str = ""
    duration_s: float = 0.0
    # Structured summary from summary_processes.result (json), if present.
    structured_summary: dict | None = None
    summary_status: str = ""


def _open_ro(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        return {r[1] for r in cur.fetchall()}
    except sqlite3.Error:
        return set()


def read_meetings(db_path: Path, since: str | None = None) -> list[Meeting]:
    """Return all meetings (optionally created on/after ISO `since`), newest last.

    Tolerant of minor schema drift: only selects columns that actually exist.
    """
    conn = _open_ro(db_path)
    try:
        m_cols = _table_columns(conn, "meetings")
        if not m_cols:
            return []

        where = "WHERE created_at >= ?" if since else ""
        params = (since,) if since else ()
        rows = conn.execute(
            f"SELECT * FROM meetings {where} ORDER BY created_at ASC", params
        ).fetchall()

        t_cols = _table_columns(conn, "transcripts")
        s_cols = _table_columns(conn, "summary_processes")

        meetings: list[Meeting] = []
        for r in rows:
            m = Meeting(
                id=str(r["id"]),
                title=(r["title"] if "title" in r.keys() else "") or "Untitled meeting",
                created_at=r["created_at"] if "created_at" in r.keys() else "",
                updated_at=r["updated_at"] if "updated_at" in r.keys() else "",
            )
            if t_cols:
                _attach_transcripts(conn, m, t_cols)
            if s_cols:
                _attach_structured_summary(conn, m, s_cols)
            meetings.append(m)
        return meetings
    finally:
        conn.close()


def _attach_transcripts(conn: sqlite3.Connection, m: Meeting, cols: set[str]) -> None:
    trows = conn.execute(
        "SELECT * FROM transcripts WHERE meeting_id = ? ORDER BY timestamp ASC",
        (m.id,),
    ).fetchall()
    parts, summaries, actions, keypoints = [], [], [], []
    for t in trows:
        keys = t.keys()
        if "transcript" in keys and t["transcript"]:
            parts.append(str(t["transcript"]).strip())
        if "summary" in keys and t["summary"]:
            summaries.append(str(t["summary"]).strip())
        if "action_items" in keys and t["action_items"]:
            actions.append(str(t["action_items"]).strip())
        if "key_points" in keys and t["key_points"]:
            keypoints.append(str(t["key_points"]).strip())
        if "duration" in keys and t["duration"]:
            try:
                m.duration_s += float(t["duration"])
            except (TypeError, ValueError):
                pass
    m.transcript = "\n\n".join(p for p in parts if p)
    m.summary = "\n\n".join(dict.fromkeys(summaries))       # de-dup, keep order
    m.action_items = "\n".join(dict.fromkeys(actions))
    m.key_points = "\n".join(dict.fromkeys(keypoints))


def _attach_structured_summary(conn: sqlite3.Connection, m: Meeting, cols: set[str]) -> None:
    row = conn.execute(
        "SELECT * FROM summary_processes WHERE meeting_id = ?", (m.id,)
    ).fetchone()
    if not row:
        return
    keys = row.keys()
    if "status" in keys:
        m.summary_status = str(row["status"] or "")
    if "result" in keys and row["result"]:
        try:
            parsed = json.loads(row["result"])
            if isinstance(parsed, dict):
                m.structured_summary = parsed
        except (json.JSONDecodeError, TypeError):
            pass
