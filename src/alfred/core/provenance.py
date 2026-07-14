"""Provenance predicate — the "never re-ingest daemon output" invariant.

Daemon-generated content (distiller learn entries, consolidator syntheses,
topic pages) must never feed back into ingestion, distillation, or synthesis:
re-processing LLM output produces compounding noise, not knowledge.

This module is the single source of truth for that invariant.  It replaces
four independent implementations that previously lived in curator/distiller
(record-type check + path-prefix/`generated_by`-field checks), consolidator
(a local `daemon_prefixes` tuple), and surveyor (a raw byte-grep on the diff
scan's hot path).  A schema change (new output directory, new record type,
new marker) now needs exactly one edit here.
"""
from __future__ import annotations

# Vault directories whose contents are produced by daemons.
DAEMON_OUTPUT_PREFIXES: tuple[str, ...] = ("learn/", "topic/", "synthesis/")

# Record types produced by daemons.
DAEMON_OUTPUT_TYPES: frozenset[str] = frozenset({"learn", "topic", "synthesis"})

# `generated_by` frontmatter value stamped on LLM output by vault_ops.
LLM_MARKER = "llm"

# Raw byte signatures of the LLM marker, for the pre-parse fast check used on
# the surveyor's filesystem diff scan (avoids frontmatter parsing per file).
_RAW_MARKERS: tuple[bytes, ...] = (
    b"generated_by: llm",
    b'generated_by: "llm"',
)


def is_daemon_generated(
    rel_path: str | None = None,
    *,
    record_type: str | None = None,
    generated_by: str | None = None,
) -> bool:
    """Return True if any provided signal marks the record as daemon output.

    Callers pass whichever signals they have; unsupplied signals are skipped:

    - ``rel_path``: vault-relative path — matches ``DAEMON_OUTPUT_PREFIXES``.
    - ``record_type``: frontmatter ``type`` — matches ``DAEMON_OUTPUT_TYPES``.
    - ``generated_by``: frontmatter ``generated_by`` — matches ``LLM_MARKER``.
    """
    if rel_path is not None and rel_path.startswith(DAEMON_OUTPUT_PREFIXES):
        return True
    if record_type is not None and record_type in DAEMON_OUTPUT_TYPES:
        return True
    if generated_by is not None and generated_by == LLM_MARKER:
        return True
    return False


def is_daemon_generated_raw(raw: bytes) -> bool:
    """Fast pre-parse check on raw file bytes (surveyor's diff-scan hot path).

    Greps for the ``generated_by: llm`` marker without parsing frontmatter —
    the byte-level equivalent of ``is_daemon_generated(generated_by="llm")``.
    """
    return any(marker in raw for marker in _RAW_MARKERS)
