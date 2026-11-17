"""Vault schema constants — ported from personal-alfred vault/schema.py."""
from __future__ import annotations

KNOWN_TYPES: set[str] = {
    "project", "task", "session", "input", "person", "org",
    "location", "note", "decision", "process", "run", "event",
    "account", "asset", "conversation", "assumption", "constraint",
    "contradiction", "synthesis", "wiki", "topic",
    # Content creation types
    "idea", "script", "hook",
    # Legacy — recognised for read-path compat but no longer written as directories
    "learn", "ai-dialogue",
}

# Epistemic types that consolidate into topic/ directory (tracked via tags)
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
    "learn": {"active", "absorbed", "superseded"},
    "topic": {"active", "archived"},
    "idea": {"raw", "researched", "scripted", "published", "shelved"},
    "script": {"draft", "review", "final", "published"},
    "hook": {"active", "retired"},
}

TYPE_DIRECTORY: dict[str, str] = {
    "project": "project", "task": "task", "person": "person", "org": "org",
    "location": "location", "note": "note",
    "process": "process", "run": "run", "event": "event", "account": "account",
    "asset": "asset", "synthesis": "synthesis",
    "wiki": "wiki", "topic": "topic",
    # Content creation types
    "idea": "ideas", "script": "drafts", "hook": "hooks",
    # Session types — all go to session/
    "session": "session", "conversation": "session", "ai-dialogue": "session",
    # Epistemic types — consolidated into topic/ (subtype tracked via tags)
    "decision": "topic", "assumption": "topic",
    "constraint": "topic", "contradiction": "topic",
    # Legacy type — no longer written, but kept for read-path compatibility
    "learn": "topic",
}


def _build_directory_to_type() -> dict[str, str]:
    """Directory name -> canonical record type.

    TYPE_DIRECTORY is many-to-one (session/ <- session, conversation,
    ai-dialogue; topic/ <- topic, decision, assumption, constraint,
    contradiction, learn), so a plain `{v: k for k, v in ...}` inversion is
    lossy: it keeps whichever type happened to be *declared last* and silently
    resolves session/ to "ai-dialogue" and topic/ to "learn". Both are legacy
    read-path-only types that are never written, so autofix was stamping the
    two largest content types in the vault with the wrong value.

    Resolve collisions explicitly: a directory whose name is itself a record
    type maps to that type. Directories with a single contributing type
    (ideas/ <- idea, drafts/ <- script, hooks/ <- hook) map to it unambiguously.
    """
    out: dict[str, str] = {}
    for record_type, directory in TYPE_DIRECTORY.items():
        if out.get(directory) == directory:
            continue          # already resolved to the self-named canonical type
        if directory not in out or record_type == directory:
            out[directory] = record_type
    return out


# Directory name -> canonical record type. Use this instead of inverting
# TYPE_DIRECTORY at the call site.
DIRECTORY_TO_TYPE: dict[str, str] = _build_directory_to_type()

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
    "accounts": "account", "assets": "asset",
    # conversation aliases → session
    "conversations": "session", "chat": "session", "thread": "session",
    "ai-dialogue": "session", "ai_dialogue": "session", "dialogue": "session",
    "runs": "run",
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
