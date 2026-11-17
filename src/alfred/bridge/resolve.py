"""Contact -> vault-entity resolution for the CRM bridge.

Matches a CRM contact's email against the ``email`` frontmatter field of
``person/*.md`` vault records. This is the ONLY trusted signal: email
equality is exact (case-insensitive) or there is no match. Name-only
matching is deliberately NOT implemented — a shared first name ("John") is a
classic false-positive trap, and Pattern A's spec is explicit that returning
no match beats guessing wrong. `name` is accepted for logging/future use but
never used to establish or disambiguate a match.

Two-step "grep then confirm" resolution, using only existing vault_ops
primitives (no embeddings, no new indexing infra — this is exact-match
lookup, not semantic search):

  1. `vault_search(..., glob_pattern="person/*.md", grep_pattern=email)` —
     a cheap case-insensitive substring prefilter over each candidate
     file's raw text, narrowing the scan to files that mention the email
     string at all.
  2. `vault_read()` each candidate and confirm its frontmatter `email` key
     equals the target exactly. Step 1 alone is not sufficient: the email
     could appear in the file's body/description prose rather than the
     frontmatter `email:` key, so the frontmatter-field confirmation is the
     actual "drop low-confidence matches" gate.

Ambiguity handling: if more than one person record's frontmatter email
matches, that is a vault data-integrity problem (duplicate/conflicting
records), not something this function arbitrates — it returns None rather
than guessing between candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from alfred.core.failures import record_failure
from alfred.core.vault_ops import VaultError, vault_read, vault_search

PERSON_GLOB = "person/*.md"


@dataclass(frozen=True)
class EntityMatch:
    """A vault entity resolved from a CRM contact, with a confidence signal."""

    rel_path: str
    entity_type: str
    name: str
    frontmatter: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    reason: str = "email_exact"


def _normalize_email(email: Optional[str]) -> str:
    return (email or "").strip().lower()


def resolve_contact_to_entity(
    vault_path: Path,
    *,
    email: Optional[str],
    name: Optional[str] = None,
    ignore_dirs: Optional[list[str]] = None,
) -> Optional[EntityMatch]:
    """Resolve a CRM contact (email, name) to a vault ``person/*.md`` entity.

    Returns ``None`` unless there is exactly one person record whose
    frontmatter ``email`` field case-insensitively equals ``email``. A
    contact with no email, no matching record, or more than one matching
    record all resolve to ``None`` — bias toward "no match" over a guess.

    ``name`` is currently informational only (useful to callers for
    logging/dry-run display); it does not participate in matching.
    """
    target = _normalize_email(email)
    if not target:
        return None  # nothing safe to match on

    candidates = vault_search(
        vault_path,
        glob_pattern=PERSON_GLOB,
        grep_pattern=target,
        ignore_dirs=ignore_dirs,
    )

    matches: list[tuple[dict, dict]] = []
    for candidate in candidates:
        try:
            record = vault_read(vault_path, candidate["path"])
        except VaultError as e:
            # An unreadable candidate silently drops out of the match set, so
            # a contact can fail to resolve and the caller sees "no match"
            # rather than "one candidate couldn't be read".
            record_failure("bridge.candidate_read_failed", error=e, path=candidate["path"])
            continue
        fm_email = _normalize_email(record["frontmatter"].get("email"))
        if fm_email and fm_email == target:
            matches.append((candidate, record))

    if len(matches) != 1:
        # 0 matches -> no match. >1 matches -> ambiguous; don't arbitrate.
        return None

    candidate, record = matches[0]
    fm = record["frontmatter"]
    resolved_name = fm.get("name") or candidate.get("name") or Path(candidate["path"]).stem
    return EntityMatch(
        rel_path=candidate["path"],
        entity_type=fm.get("type", "person"),
        name=resolved_name,
        frontmatter=fm,
        confidence=1.0,
        reason="email_exact",
    )
