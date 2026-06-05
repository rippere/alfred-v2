"""Compute a KPI snapshot for a given date from the readers in sources.py.

A snapshot is a list of metric rows:
    {"domain": str, "metric": str, "value": float, "meta": dict | None}

Metric naming (domain in parens) — v1:
    engineering: git_commits (meta = per-project counts), sessions
    knowledge:   records.main, records.neuroscience, records.content,
                 topics_distilled, api_cost_usd (today-only)
    product:     crm_users, crm_status, tribe_corpus_videos, tribe_avg_score
    life:        records.personal, records.finance

A value is only emitted when its source actually has data for that day; missing
values are omitted (never zero-filled) so downstream truly knows "no data".
``git_commits`` and ``sessions`` are the exception — they legitimately mean 0 on
a day with no activity, so they are always emitted.

To make backfill over a wide range cheap, :class:`LedgerComputer` scans every
source exactly once at construction and buckets results by day; per-date
snapshots are then dict lookups.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from alfred.ledger import config as C
from alfred.ledger import sources as S


class LedgerComputer:
    """Pre-scans all sources once; serves O(1) per-day snapshots."""

    def __init__(self) -> None:
        # Vault record counts per day, per vault label.
        self._vault_counts: dict[str, Counter] = {}
        for label, spec in C.VAULTS.items():
            self._vault_counts[label] = S.created_counts_by_day(spec["root"])

        # Sessions per day (main vault session/ frontmatter `created`), split
        # human vs cron-agent so the healer's 15-min loop can't inflate the KPI.
        self._sessions, self._sessions_agent = S.session_counts_by_day()

        # topic/ records created per day (main vault).
        self._topics: Counter = S.created_counts_by_day(
            C.VAULTS["main"]["root"], only_dir="topic"
        )

        # Git commits per repo per day.
        self._git: dict[str, Counter] = S.all_git_counts()

        # PM briefs per day.
        self._crm: dict[str, dict] = S.crm_briefs_by_day()
        self._tribe: dict[str, dict] = S.tribe_briefs_by_day()

        self._today: str = S.today_local()

    # ── helpers ──────────────────────────────────────────────────────────────

    def observed_days(self) -> set[str]:
        """Union of every day that any source has data for (for reporting)."""
        days: set[str] = set()
        for c in self._vault_counts.values():
            days |= set(c)
        days |= set(self._sessions) | set(self._topics)
        for c in self._git.values():
            days |= set(c)
        days |= set(self._crm) | set(self._tribe)
        return days

    # ── snapshot ─────────────────────────────────────────────────────────────

    def snapshot(self, date: str) -> list[dict]:
        """Return the list of metric rows for `date` (YYYY-MM-DD)."""
        rows: list[dict] = []

        def add(domain: str, metric: str, value: float, meta: Optional[dict] = None):
            rows.append({
                "domain": domain,
                "metric": metric,
                "value": float(value),
                "meta": meta,
            })

        # ── engineering ──────────────────────────────────────────────────────
        per_project = {
            name: counts.get(date, 0)
            for name, counts in self._git.items()
            if counts.get(date, 0) > 0
        }
        total_commits = sum(per_project.values())
        # Always emit commits + sessions: 0 is a meaningful "no activity" reading.
        add("engineering", "git_commits", total_commits,
            meta=per_project or None)
        add("engineering", "sessions", self._sessions.get(date, 0))
        # Cron-agent sessions (healer/PM/retro loops) — own series, only on
        # active days; zero would just restate "the cron ran no sessions".
        if self._sessions_agent.get(date, 0):
            add("engineering", "sessions_agent", self._sessions_agent[date])

        # ── knowledge ────────────────────────────────────────────────────────
        for label in ("main", "neuroscience", "content"):
            cnt = self._vault_counts.get(label, Counter()).get(date)
            if cnt is not None:
                add("knowledge", f"records.{label}", cnt)

        topics = self._topics.get(date)
        if topics is not None:
            add("knowledge", "topics_distilled", topics)

        # api cost: only meaningful for the current local date (state is rolling).
        if date == self._today:
            cost = S.api_cost_for_today(date)
            if cost is not None:
                add("knowledge", "api_cost_usd", cost)

        # ── product (PM briefs; latest brief per day already chosen upstream) ─
        crm = self._crm.get(date, {})
        if "crm_users" in crm:
            add("product", "crm_users", crm["crm_users"])
        if "crm_status" in crm:
            add("product", "crm_status", crm["crm_status"])

        tribe = self._tribe.get(date, {})
        if "tribe_corpus_videos" in tribe:
            add("product", "tribe_corpus_videos", tribe["tribe_corpus_videos"])
        if "tribe_avg_score" in tribe:
            add("product", "tribe_avg_score", tribe["tribe_avg_score"])

        # ── life ─────────────────────────────────────────────────────────────
        for label in ("personal", "finance"):
            cnt = self._vault_counts.get(label, Counter()).get(date)
            if cnt is not None:
                add("life", f"records.{label}", cnt)

        return rows
