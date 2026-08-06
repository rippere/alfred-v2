"""Process-wide failure counters, so a swallowed exception leaves a trace.

The codebase has ~23 handlers whose entire body is `pass`/`continue` — a
file that failed to parse, a vector-store table that wouldn't open, an
autofix that raised. Each one is individually defensible (the caller really
does want to keep going), but collectively they mean "zero errors" is not a
claim anyone can check: the failure happened, nothing recorded it, and the
next status read looks clean.

This module is the cheapest possible floor under that. A handler calls
``record_failure("janitor.autofix_failed", error=e, path=p)``, which:

  * logs it at WARNING with structured fields (so it shows up in the journal
    at the moment it happens), and
  * bumps an in-process counter.

StateStore.save() drains those counters into ``state.error_counts``, so the
count survives the process and ``alfred status`` can show a non-zero number
instead of silence. Counters are drained (read-and-zero) rather than read,
so each count is folded into persistent state exactly once no matter how
many StateStore instances a process holds.

Deliberately NOT a replacement for handling the error — it is the floor, not
the ceiling. A handler that can do something better should.
"""
from __future__ import annotations

import threading
from collections import Counter

import structlog

_log = structlog.get_logger(__name__)

# Guards _counts against concurrent mutation. The daemons are asyncio, but
# StateStore.save() is called from threads in the concurrency tests and from
# separate MCP/CLI processes, and Counter mutation is not atomic under
# arbitrary interleaving.
_lock = threading.Lock()
_counts: Counter[str] = Counter()


def record_failure(key: str, error: BaseException | str | None = None, **fields) -> None:
    """Record one swallowed failure under a stable dotted ``key``.

    ``key`` should be ``<component>.<what_failed>`` (e.g.
    ``"surveyor.stat_failed"``) and must come from a bounded, hand-written
    set — never interpolate a path or an error message into it, or
    state.json's error_counts grows without limit.
    """
    with _lock:
        _counts[key] += 1
    if error is not None:
        fields["error"] = str(error)
        fields["error_type"] = type(error).__name__ if isinstance(error, BaseException) else "str"
    _log.warning(key, **fields)


def drain_failures() -> dict[str, int]:
    """Return the counts accumulated since the last drain, and zero them.

    Read-and-zero so StateStore.save() can add the delta to the persisted
    total without double counting across repeated saves.
    """
    with _lock:
        drained = dict(_counts)
        _counts.clear()
    return drained


def restore_failures(counts: dict[str, int]) -> None:
    """Put drained counts back — used when the save that drained them failed,
    so a write error doesn't also destroy the record of earlier errors."""
    if not counts:
        return
    with _lock:
        _counts.update(counts)


def peek_failures() -> dict[str, int]:
    """Current in-process counts without draining (for status/debugging)."""
    with _lock:
        return dict(_counts)


def reset_failures() -> None:
    """Drop all in-process counts. For test isolation only."""
    with _lock:
        _counts.clear()
