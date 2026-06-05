"""Single source of truth for every path / endpoint the ledger touches.

All filesystem locations, vault roots, git repos, and remote-API settings live
here so there is exactly one place to edit when something moves.

Remote credentials are NOT stored here. They are read at runtime from
``~/.config/alfred-ledger/env`` (KEY=VALUE lines). A template lives beside it as
``env.example``. If credentials are absent, the push step degrades to a dry run.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Repo / data locations ────────────────────────────────────────────────────
# This file is src/alfred/ledger/config.py → repo root is three parents up.
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
DATA_DIR: Path = REPO_ROOT / "data"
LEDGER_DB: Path = DATA_DIR / "ledger.db"

# Main vault state (curator_processed + api cost live here). Read-only.
MAIN_STATE_PATH: Path = DATA_DIR / "state.json"

# ── Vaults ───────────────────────────────────────────────────────────────────
# label -> {root: vault filesystem root, state: that vault's state.json}.
# Roots come from the per-vault config yamls (verified live, not assumed).
# `domain` maps each vault's record count into a ledger domain.
VAULTS: dict[str, dict] = {
    "main": {
        "root": Path("/mnt/external/obsidian-vault"),
        "state": DATA_DIR / "state.json",
        "domain": "knowledge",
    },
    "neuroscience": {
        "root": Path("/mnt/external/vault-neuroscience"),
        "state": REPO_ROOT / "data-neuroscience" / "state.json",
        "domain": "knowledge",
    },
    "content": {
        "root": Path("/mnt/external/vault-content"),
        "state": REPO_ROOT / "data-content" / "state.json",
        "domain": "knowledge",
    },
    "finance": {
        "root": Path("/mnt/external/vault-finance"),
        "state": REPO_ROOT / "data-finance" / "state.json",
        "domain": "life",
    },
    "personal": {
        "root": Path("/mnt/external/vault-personal"),
        "state": REPO_ROOT / "data-personal" / "state.json",
        "domain": "life",
    },
}

# Vault sub-directories that are NOT knowledge records (don't count them).
VAULT_EXCLUDE_DIRS: set[str] = {
    "inbox", "wiki", "_archived", "_templates", "_bases", ".obsidian",
    "view", "Excalidraw", "published", "_published",
}

# Sessions live in the main vault's session/ directory; daily session count is
# derived from frontmatter `created` here (see sources.py for rationale).
SESSION_DIR: Path = VAULTS["main"]["root"] / "session"

# topic/ records (decisions/assumptions/constraints/contradictions consolidate
# here) — topics_distilled counts records created here per day.
TOPIC_DIR: Path = VAULTS["main"]["root"] / "topic"

# ── PM briefs ────────────────────────────────────────────────────────────────
INBOX_DIR: Path = VAULTS["main"]["root"] / "inbox"
INBOX_PROCESSED_DIR: Path = INBOX_DIR / "processed"
# Curator moves briefs into processed/ — that subdir is the backfill history.
BRIEF_DIRS: list[Path] = [INBOX_DIR, INBOX_PROCESSED_DIR]

# ── Git repos (commits/day, local dates, all branches deduped) ───────────────
GIT_REPOS: dict[str, Path] = {
    "alfred-v2": Path("/home/rippere/alfred-v2"),
    "tribe-social": Path("/mnt/external/Projects/tribe-social"),
    "tribe-social-lab": Path("/mnt/external/Projects/tribe-social-lab"),
    "crm-agentic": Path("/mnt/external/Projects/crm-agentic"),
    "sector-flow-analyzer": Path("/mnt/external/Projects/sector-flow-analyzer"),
    "resume-pipeline": Path("/mnt/external/Projects/resume-pipeline"),
    "digital-twin": Path("/mnt/external/digital-twin"),
    "canvas-autopilot": Path("/mnt/external/Projects/canvas-autopilot"),
    "portfolio-narrative": Path("/mnt/external/Projects/portfolio-narrative"),
    "executive-mind-matrix": Path("/mnt/external/executive-mind-matrix"),
}

# ── Remote push (NovaCRM / Supabase) ─────────────────────────────────────────
ENV_DIR: Path = Path.home() / ".config" / "alfred-ledger"
ENV_FILE: Path = ENV_DIR / "env"
ENV_EXAMPLE: Path = ENV_DIR / "env.example"

# Keys read from ENV_FILE (or the real process environment as a fallback).
PUSH_ENV_KEYS: tuple[str, ...] = (
    "LEDGER_SUPABASE_URL",
    "LEDGER_SUPABASE_ANON_KEY",
    "LEDGER_API_URL",
    "LEDGER_WORKSPACE_ID",
    "LEDGER_EMAIL",
    "LEDGER_PASSWORD",
)


def load_push_env() -> dict[str, str]:
    """Return push credentials from ENV_FILE, falling back to os.environ.

    Missing keys simply aren't present in the returned dict; callers decide
    whether enough is available to perform a real push.
    """
    env: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            env[key.strip()] = val.strip().strip('"').strip("'")
    # Process environment overrides / supplements the file.
    for key in PUSH_ENV_KEYS:
        if os.environ.get(key):
            env[key] = os.environ[key]
    return {k: v for k, v in env.items() if v}
