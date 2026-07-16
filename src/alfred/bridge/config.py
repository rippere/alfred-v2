"""Single source of truth for every path / setting the CRM bridge touches.

Mirrors `alfred.ledger.config`'s conventions exactly, with one deliberate
difference: this module does NOT define a separate credential set. The
bridge writes to the same NovaCRM workspace, as the same bot user, as the
existing life-ledger KPI push — so it reuses that module's env file
(``~/.config/alfred-ledger/env``) and its ``LEDGER_*`` keys verbatim rather
than inventing a parallel ``BRIDGE_*`` credential set for what is, from
Supabase's point of view, the exact same authenticated identity.

If credentials are absent, the push step degrades to a dry run (same
degrade-to-dry-run behavior as ledger.push) — see `alfred.bridge.enrich_crm`
once that module lands.
"""

from __future__ import annotations

from pathlib import Path

from alfred.ledger import config as LEDGER_C

# This file is src/alfred/bridge/config.py -> repo root is three parents up.
REPO_ROOT: Path = Path(__file__).resolve().parents[3]

# ── Remote credentials — SAME file, SAME keys as the ledger push path ───────
# Deliberately not renamed/duplicated: same Supabase project, same bot user,
# same workspace. See module docstring.
ENV_DIR: Path = LEDGER_C.ENV_DIR
ENV_FILE: Path = LEDGER_C.ENV_FILE
ENV_EXAMPLE: Path = LEDGER_C.ENV_EXAMPLE
PUSH_ENV_KEYS: tuple[str, ...] = LEDGER_C.PUSH_ENV_KEYS
load_push_env = LEDGER_C.load_push_env

# ── Vault ────────────────────────────────────────────────────────────────
# v1 targets the main knowledge vault only (person/org records live here).
VAULT_PATH: Path = LEDGER_C.VAULTS["main"]["root"]
VAULT_IGNORE_DIRS: list[str] = [
    "inbox", "wiki", "_archived", "_templates", "_bases", ".obsidian",
]

# ── Entity resolution ────────────────────────────────────────────────────
# Only an exact (case-insensitive) match on a person/*.md record's
# frontmatter `email` field is trusted. There is no partial-credit score
# below this — resolve.py either returns a match at this confidence or
# returns None. Kept as a named constant (rather than a bare 1.0 literal at
# call sites) so a future looser signal has an explicit threshold to compare
# against instead of a magic number.
MIN_MATCH_CONFIDENCE: float = 1.0

# ── Note idempotency ─────────────────────────────────────────────────────
# Every posted note body embeds this marker with a content hash appended
# before the suffix, e.g. "<!-- alfred:brief hash=abcdef123456 -->". Before
# posting, existing notes (GET .../notes) are substring-checked for a
# matching marker; a match means "already posted, skip" — no local state
# file is consulted, the CRM itself is the source of truth.
NOTE_MARKER_PREFIX: str = "<!-- alfred:brief hash="
NOTE_MARKER_SUFFIX: str = " -->"

# ── Deal stage filter ────────────────────────────────────────────────────
# `GET .../deals` has no server-side "active" filter (stage is a free
# string column) — these are excluded client-side to approximate "active".
CLOSED_DEAL_STAGES: set[str] = {"closed_won", "closed_lost"}

# ── Brief synthesis ──────────────────────────────────────────────────────
# No shared LLM-call helper exists in alfred.core.anthropic_client beyond
# get_client() itself (see query/synth.py for the ad hoc call-shape every
# caller repeats) — the bridge defines its own default model here so it
# isn't hardcoded inside enrich_crm.py's call site.
BRIEF_SYNTHESIS_MODEL: str = "claude-sonnet-4-6"
BRIEF_MAX_TOKENS: int = 512
