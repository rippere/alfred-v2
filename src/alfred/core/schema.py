"""Vault schema constants — ported from personal-alfred vault/schema.py."""
from __future__ import annotations

KNOWN_TYPES: set[str] = {
    "project", "task", "session", "input", "person", "org",
    "location", "note", "decision", "process", "run", "event",
    "account", "asset", "conversation", "assumption", "constraint",
    "contradiction", "synthesis",
}

LEARN_TYPES: set[str] = {
    "assumption", "decision", "constraint", "contradiction", "synthesis",
}

STATUS_BY_TYPE: dict[str, set[str]] = {
    "project": {"active", "paused", "completed", "abandoned", "proposed"},
    "task": {"todo", "active", "blocked", "done", "cancelled"},
    "session": {"active", "completed"},
    "input": {"unprocessed", "processed", "deferred"},
    "person": {"active", "inactive"},
    "org": {"active", "inactive"},
    "location": {"active", "inactive"},
    "note": {"draft", "active", "review", "final"},
    "decision": {"draft", "final", "superseded", "reversed"},
    "process": {"active", "proposed", "design", "deprecated"},
    "run": {"active", "completed", "blocked", "cancelled"},
    "event": set(),
    "account": {"active", "suspended", "closed", "pending"},
    "asset": {"active", "retired", "maintenance", "disposed"},
    "conversation": {"active", "waiting", "resolved", "closed", "archived"},
    "assumption": {"active", "challenged", "invalidated", "confirmed"},
    "constraint": {"active", "expired", "waived", "superseded"},
    "contradiction": {"unresolved", "resolved", "accepted"},
    "synthesis": {"draft", "active", "superseded"},
}

TYPE_DIRECTORY: dict[str, str] = {
    "project": "project", "task": "task", "person": "person", "org": "org",
    "location": "location", "note": "note", "decision": "decision",
    "process": "process", "run": "run", "event": "event", "account": "account",
    "asset": "asset", "conversation": "conversation", "assumption": "assumption",
    "constraint": "constraint", "contradiction": "contradiction", "synthesis": "synthesis",
}

LIST_FIELDS: set[str] = {
    "tags", "aliases", "related", "relationships", "participants",
    "outputs", "depends_on", "blocked_by", "based_on", "supports",
    "challenged_by", "approved_by", "confirmed_by", "invalidated_by",
    "cluster_sources", "governed_by", "references", "project",
}

REQUIRED_FIELDS: list[str] = ["type", "created"]

NAME_FIELD_BY_TYPE: dict[str, str] = {
    "conversation": "subject",
    "input": "subject",
}

_TYPE_CORRECTIONS: dict[str, str] = {
    "persons": "person", "people": "person", "organisation": "org",
    "organization": "org", "company": "org", "projects": "project",
    "tasks": "task", "todo": "task", "notes": "note", "decisions": "decision",
    "processes": "process", "workflow": "process", "events": "event",
    "accounts": "account", "assets": "asset", "conversations": "conversation",
    "chat": "conversation", "thread": "conversation", "runs": "run",
    "sessions": "session", "inputs": "input", "assumptions": "assumption",
    "constraints": "constraint", "contradictions": "contradiction",
    "syntheses": "synthesis",
}

_STATUS_CORRECTIONS: dict[str, str] = {
    "open": "active", "opened": "active", "closed": "completed",
    "complete": "completed", "finished": "completed", "done": "done",
    "pending": "todo", "in-progress": "active", "in_progress": "active",
    "wip": "active", "on-hold": "paused", "on_hold": "paused",
    "hold": "paused", "frozen": "paused", "canceled": "cancelled",
    "archived": "inactive", "archive": "inactive", "retired": "inactive",
    "new": "active", "started": "active", "draft": "draft",
    "final": "final", "confirmed": "confirmed", "challenged": "challenged",
    "superseded": "superseded", "expired": "expired",
}


def correct_type(t: str) -> str | None:
    n = t.lower().strip()
    if n in KNOWN_TYPES:
        return n
    return _TYPE_CORRECTIONS.get(n)


def correct_status(status: str, record_type: str) -> str | None:
    valid = STATUS_BY_TYPE.get(record_type, set())
    if not valid:
        return None
    n = status.lower().strip()
    if n in valid:
        return n
    candidate = _STATUS_CORRECTIONS.get(n, "")
    if candidate in valid:
        return candidate
    return None
