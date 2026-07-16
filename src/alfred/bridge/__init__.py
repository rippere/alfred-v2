"""Alfred -> NovaCRM deal/contact brief enrichment ("Pattern A").

For each contact on an active NovaCRM deal: resolve the contact to a vault
entity (email match only — see `alfred.bridge.resolve`), and if relevant
vault knowledge exists about that entity, synthesize a short brief and post
it as a note on the CRM deal/contact via the existing REST endpoints.

Non-destructive and append-only: notes are created, never edited or
deleted. Idempotent via an embedded `<!-- alfred:brief hash=... -->` marker
checked against existing notes on the target (read via the list endpoints)
before posting — the CRM itself is the source of truth for what has already
been posted, not a local state file, so a restart/reinstall of Alfred can't
cause duplicate notes.

Public surface (once cli.py lands):
    - `alfred bridge enrich [--push]` (dry-run by default; --push performs
      the actual write and requires credentials in the shared ledger env
      file).

Reuses the ledger module's Supabase auth pattern and credentials verbatim —
same bot user, same workspace, same API. See `alfred.bridge.config` and
`alfred.ledger.push` for the pattern this mirrors.
"""
from __future__ import annotations

__all__ = ["config", "resolve"]
