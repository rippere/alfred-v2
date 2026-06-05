"""Readers that turn raw data sources into per-day numbers.

Every function here is read-only and defensive: a missing repo, vault, or
malformed file yields a skipped/None value rather than an exception, and values
are NEVER fabricated.

Source-semantics notes (verified against live data 2026-06-05):

* ``curator_processed`` (in each vault's state.json) maps ``inbox/<file>.md`` →
  the **UTC timestamp curator processed it**, i.e. batch-ingestion time, not the
  note's authored date. It frequently lags creation by days/weeks (e.g. a
  session named ``...2026-04-14...`` was processed 2026-05-13), and only the
  *main* vault has it meaningfully populated (satellites had 4/4/3/0 entries).
  It is therefore unsuitable for a reproducible daily/backfilled "records
  created" series.

* Authoritative, reproducible day attribution comes from each record's
  frontmatter ``created: YYYY-MM-DD`` field. 100% of session records carry it.
  So both ``sessions`` and ``records.<vault>`` are computed by scanning record
  frontmatter ``created`` dates — identical logic for live and historical days.

* Git: commits are counted across ALL local branches in the repo's LOCAL
  timezone, deduped by sha (``git log --all`` already dedupes by sha within one
  invocation). Missing / non-git dirs are skipped.

* PM briefs: tolerant regexes over the markdown body; a value that isn't found
  is omitted (not zero-filled) so it can be skipped downstream.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import frontmatter

from alfred.ledger import config as C


# ── frontmatter created-date scanning ────────────────────────────────────────

def _created_date(md_path: Path) -> Optional[str]:
    """Return the YYYY-MM-DD `created` date from a record's frontmatter, or None."""
    try:
        post = frontmatter.load(str(md_path))
    except Exception:
        return None
    created = post.metadata.get("created")
    if created is None:
        return None
    # `created` may be a date, datetime, or string — normalise to YYYY-MM-DD.
    text = str(created).strip().strip("'\"")
    return text[:10] if len(text) >= 10 else None


def _iter_record_files(root: Path, only_dir: Optional[str] = None):
    """Yield record .md files under a vault root, skipping excluded directories."""
    if not root.exists():
        return
    if only_dir is not None:
        search_dirs = [root / only_dir]
    else:
        search_dirs = [
            d for d in root.iterdir()
            if d.is_dir() and d.name not in C.VAULT_EXCLUDE_DIRS
        ]
    for d in search_dirs:
        if not d.exists():
            continue
        for md in d.rglob("*.md"):
            rel_parts = md.relative_to(root).parts[:-1]
            if any(part in C.VAULT_EXCLUDE_DIRS for part in rel_parts):
                continue
            yield md


def created_counts_by_day(root: Path, only_dir: Optional[str] = None) -> Counter:
    """Count records per `created` day under a vault root (optionally one dir).

    Returns a Counter {YYYY-MM-DD: count}. Scanning the whole vault once and
    bucketing by day is far cheaper than re-scanning per date during backfill.
    """
    counts: Counter = Counter()
    for md in _iter_record_files(root, only_dir=only_dir):
        day = _created_date(md)
        if day:
            counts[day] += 1
    return counts


# ── sessions ─────────────────────────────────────────────────────────────────

# Cron-driven agent sessions (crm-self-healer every 15 min, daily PM agents,
# weekly retro) all start from a systemd prompt of the form "Run the X agent…",
# which slugifies to a "run-the-…" record name. Counting them alongside human
# sessions inflated the metric ~3x, so they get their own series.
AGENT_SESSION_PREFIXES = ("run-the-",)


def session_counts_by_day() -> tuple[Counter, Counter]:
    """(human, agent) sessions per day, by frontmatter `created` in the main
    vault session/ dir.

    Chosen over curator_processed because `created` is the authored date, is
    present on 100% of session records, and is identical for live and backfill
    runs. curator_processed smears across processing batches (e.g. 97 sessions
    'processed' on 2026-06-04 vs 86 actually 'created' that day).
    """
    human: Counter = Counter()
    agent: Counter = Counter()
    if not C.SESSION_DIR.exists():
        return human, agent
    for md in C.SESSION_DIR.glob("*.md"):
        day = _created_date(md)
        if not day:
            continue
        if md.name.startswith(AGENT_SESSION_PREFIXES):
            agent[day] += 1
        else:
            human[day] += 1
    return human, agent


# ── git commits ──────────────────────────────────────────────────────────────

def git_commit_counts(repo: Path) -> Counter:
    """Commits per LOCAL day across all branches for one repo (deduped by sha).

    `git log --all` walks every ref once and emits each commit a single time, so
    shas are already deduped. Returns {YYYY-MM-DD: count}; empty Counter for a
    missing or non-git directory.
    """
    counts: Counter = Counter()
    if not (repo / ".git").exists():
        return counts
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "log", "--all",
             "--pretty=format:%ad", "--date=format-local:%Y-%m-%d"],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.SubprocessError, OSError):
        return counts
    if out.returncode != 0:
        return counts
    for line in out.stdout.splitlines():
        line = line.strip()
        if line:
            counts[line] += 1
    return counts


def all_git_counts() -> dict[str, Counter]:
    """{repo_name: Counter{day: commits}} for every configured repo that exists."""
    result: dict[str, Counter] = {}
    for name, path in C.GIT_REPOS.items():
        c = git_commit_counts(path)
        if c:  # only include repos that produced commits (skips missing/non-git)
            result[name] = c
    return result


# ── PM briefs ────────────────────────────────────────────────────────────────

# Brief filename date forms:
#   crm-pm-20260605-0848.md   -> 20260605  (compact)
#   tribe-pm-2026-06-05.md     -> 2026-06-05 (dashed)
_CRM_DATE_RE = re.compile(r"crm-pm-(\d{8})", re.IGNORECASE)
_TRIBE_DATE_RE = re.compile(r"tribe-pm-(\d{4}-\d{2}-\d{2})", re.IGNORECASE)

# Status tokens → numeric. GREEN=1, YELLOW/DEGRADED=0.5, RED=0.
_STATUS_MAP = {"GREEN": 1.0, "YELLOW": 0.5, "DEGRADED": 0.5, "RED": 0.0}
_STATUS_RE = re.compile(r"\*\*Status:\*\*\s*([A-Za-z]+)")

# CRM users: e.g. "users: 3 real" / "2 real (excluding ...)".
_CRM_USERS_RE = re.compile(r"users?:\s*(\d+)\s*real", re.IGNORECASE)

# Tribe corpus line: "Corpus: 24 videos, avg score 63.8".
_TRIBE_CORPUS_RE = re.compile(
    r"Corpus:\s*(\d+)\s*videos.*?avg\s*score\s*([\d.]+)", re.IGNORECASE | re.DOTALL
)


def _brief_date_from_name(name: str) -> Optional[str]:
    """Map a brief filename to its YYYY-MM-DD day, or None."""
    m = _CRM_DATE_RE.search(name)
    if m:
        raw = m.group(1)
        return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"
    m = _TRIBE_DATE_RE.search(name)
    if m:
        return m.group(1)
    return None


def _status_token(text: str) -> Optional[float]:
    """Extract an explicit GREEN/YELLOW/DEGRADED/RED status token → numeric."""
    m = _STATUS_RE.search(text)
    if not m:
        return None
    return _STATUS_MAP.get(m.group(1).upper())


def _crm_status_heuristic(text: str) -> Optional[float]:
    """Derive CRM health when no explicit status token is present.

    CRM briefs convey health in prose, not a GREEN/RED label. Conservative
    heuristic: an open CRITICAL or a FAILED deploy this run → DEGRADED (0.5);
    otherwise, if the brief clearly reports healthy endpoints → GREEN (1.0).
    Returns None if neither signal is clear (value then skipped, never faked).
    """
    lower = text.lower()
    degraded = ("critical" in lower and "open" in lower) or "failed" in lower
    healthy = "health: " in lower or '"status":"ok"' in lower or "api health" in lower
    if degraded:
        return 0.5
    if healthy:
        return 1.0
    return None


def _iter_briefs(prefix: str):
    """Yield (day, text) for every brief with the given filename prefix.

    Scans both inbox/ and inbox/processed/. When multiple briefs exist for one
    day, the caller's dict assignment keeps the LAST one encountered; we sort by
    filename so the lexically-latest (newest timestamp) brief wins per day.
    """
    seen: dict[str, Path] = {}
    for d in C.BRIEF_DIRS:
        if not d.exists():
            continue
        for md in d.glob(f"{prefix}*.md"):
            day = _brief_date_from_name(md.name)
            if not day:
                continue
            # Latest filename for the day wins (timestamps sort lexically).
            if day not in seen or md.name > seen[day].name:
                seen[day] = md
    for day in sorted(seen):
        try:
            text = seen[day].read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        yield day, text


def crm_briefs_by_day() -> dict[str, dict]:
    """{day: {crm_users, crm_status}} parsed from crm-pm-*.md (latest per day)."""
    result: dict[str, dict] = {}
    for day, text in _iter_briefs("crm-pm-"):
        rec: dict = {}
        m = _CRM_USERS_RE.search(text)
        if m:
            rec["crm_users"] = float(m.group(1))
        status = _status_token(text)
        if status is None:
            status = _crm_status_heuristic(text)
        if status is not None:
            rec["crm_status"] = status
        if rec:
            result[day] = rec
    return result


def tribe_briefs_by_day() -> dict[str, dict]:
    """{day: {tribe_corpus_videos, tribe_avg_score, ...}} from tribe-pm-*.md."""
    result: dict[str, dict] = {}
    for day, text in _iter_briefs("tribe-pm-"):
        rec: dict = {}
        m = _TRIBE_CORPUS_RE.search(text)
        if m:
            rec["tribe_corpus_videos"] = float(m.group(1))
            rec["tribe_avg_score"] = float(m.group(2))
        status = _status_token(text)
        if status is not None:
            rec["tribe_status"] = status
        if rec:
            result[day] = rec
    return result


# ── main-state extras (api cost — today only) ────────────────────────────────

def main_state() -> dict:
    """Load the main vault state.json, or {} if unreadable."""
    try:
        return json.loads(C.MAIN_STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def api_cost_for_today(today: str) -> Optional[float]:
    """Return api_cost_usd_today from main state, but only if it is for `today`.

    The state tracks a single rolling 'today' cost that resets on date rollover,
    so it's only valid for the current date — historical backfill cannot recover
    per-day API cost, and this returns None for any non-current date.
    """
    state = main_state()
    if state.get("api_calls_date") != today:
        return None
    cost = state.get("api_cost_usd_today")
    return float(cost) if cost is not None else None


def today_local() -> str:
    """Today's date in the LOCAL timezone as YYYY-MM-DD (matches git local dates)."""
    return datetime.now().astimezone().strftime("%Y-%m-%d")
