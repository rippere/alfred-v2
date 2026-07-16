"""Alfred -> NovaCRM deal/contact brief enrichment ("Pattern A").

For each contact on an active NovaCRM deal: resolve the contact to a vault
entity (email match only — see `alfred.bridge.resolve`), and if relevant
vault knowledge exists about that entity, synthesize a short brief and post
it as a note on the CRM deal via the existing REST endpoints.

Non-destructive and append-only: notes are created, never edited or
deleted. Idempotent via an embedded `<!-- alfred:brief hash=... -->` marker
checked against existing notes on the target (read via the list endpoints)
before posting — the CRM itself is the source of truth for what has already
been posted, not a local state file, so a restart/reinstall of Alfred can't
cause duplicate notes.

Module map (mirrors `alfred.ledger`'s shape):
    config.py      — paths, env keys (reuses the ledger's env file/creds
                      verbatim — same bot user, same workspace, same API),
                      match-confidence + note-marker + closed-stage constants.
    resolve.py      — CRM contact -> vault `person/*.md` entity, email-exact
                       match only; ambiguous or absent -> None, never a guess.
    brief.py        — vault-entity -> synthesized brief (QueryEngine +
                       `anthropic_client.get_client()`); returns None when
                       there's no relevant vault knowledge to write about.
    notes.py        — GET/POST NovaCRM deal & contact notes, the idempotency
                       hash-marker check, and the dry-run/--push split.
    enrich_crm.py    — orchestrates the above end to end: list active deals ->
                       resolve each deal's contact -> synthesize -> post
                       (or simulate posting), with structlog logging of every
                       decision and a summary of counts.
    cli.py           — `alfred bridge enrich [--push]` Typer sub-app, wired
                       into the top-level app in `alfred.cli`.

Public surface:
    `alfred bridge enrich`          — dry run (default): resolves contacts,
                                       synthesizes briefs, checks for
                                       duplicates, PRINTS what would be
                                       posted. Makes zero write calls.
    `alfred bridge enrich --push`   — performs the actual POST. Requires
                                       credentials in the shared ledger env
                                       file (`~/.config/alfred-ledger/env`,
                                       same `LEDGER_*` keys as `alfred ledger
                                       collect --push`); missing credentials
                                       degrade back to a dry run.

Going live: a systemd timer (analogous to `deploy/systemd/ledger-collect.
{service,timer}`) is deliberately NOT installed yet. `--push` exists in the
CLI and is exercised by nothing in this repo's test suite or automation —
it's there for a human to run by hand and review the dry-run output first.
Once dry-run output has been eyeballed against real deals/contacts, wiring a
`bridge-enrich.service`/`.timer` pair that runs `alfred bridge enrich --push`
on a cadence is the natural next step, following the ledger timer's template
(`Type=oneshot`, `OnFailure=alfred-alert@%n.service`, `WantedBy=default.
target`) — but that installation is out of scope here and requires explicit
human review first.

Reuses the ledger module's Supabase auth pattern and credentials verbatim —
same bot user, same workspace, same API. See `alfred.bridge.config` and
`alfred.ledger.push` for the pattern this mirrors.
"""
from __future__ import annotations

__all__ = ["config", "resolve", "brief", "notes", "enrich_crm"]
