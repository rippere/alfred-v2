from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# Days of stability=1.0 memory that decay to R≈0.37 (1/e). Scales the whole
# retention curve: at the default, a file accessed once and untouched for a
# year sits near R≈0.02, while a file accessed ~50 times (stability 5.0)
# takes ~5x longer to reach the same point. Tuned so "forgotten" means
# genuinely cold, not merely quiet for a month.
RETENTION_SCALE_DAYS = 90.0


@dataclass
class MemoryStrength:
    rel_path: str
    access_count: int = 0
    last_accessed: str = ""
    stability: float = 1.0

    def update(self) -> None:
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc).isoformat()
        self.stability = min(5.0, 1.0 + math.log1p(self.access_count))

    def score_modifier(self) -> float:
        if not self.last_accessed:
            return 1.0
        last = datetime.fromisoformat(self.last_accessed)
        days_ago = (datetime.now(timezone.utc) - last).total_seconds() / 86400
        raw = self.stability * math.exp(-0.1 * days_ago / max(self.stability, 1.0))
        return max(0.5, min(1.5, raw))

    def retrievability(self, now: datetime | None = None) -> float:
        """Ebbinghaus retrievability R = exp(-t / (S * RETENTION_SCALE_DAYS)),
        in [0, 1] — the probability this file is still "remembered".

        Distinct from score_modifier(), which is a bounded *ranking* nudge in
        [0.5, 1.5] and deliberately never reaches zero: a stale file should
        rank lower, not vanish from results. retrievability() is the honest
        decay curve, unclamped at the bottom, and it is what the retention
        (forgetting) decision reads. Keeping them separate means tuning the
        forget threshold can't quietly distort search ranking.

        A file that has never been accessed returns 0.0 — no retrieval
        evidence at all is the weakest possible memory, not a neutral one.
        Callers that care about age-since-embed rather than age-since-access
        must supply that themselves; this type only knows about access.
        """
        if not self.last_accessed:
            return 0.0
        now = now or datetime.now(timezone.utc)
        last = datetime.fromisoformat(self.last_accessed)
        days_ago = max(0.0, (now - last).total_seconds() / 86400)
        stability_days = max(self.stability, 1.0) * RETENTION_SCALE_DAYS
        return math.exp(-days_ago / stability_days)


@dataclass
class FileState:
    md5: str
    last_embedded: str = ""
    chunk_ids: list[str] = field(default_factory=list)
    semantic_cluster_id: int = -1
    structural_community_id: int = -1
    last_scanned: str = ""
    open_issues: list[str] = field(default_factory=list)
    learn_records_created: list[str] = field(default_factory=list)
    last_distilled: str = ""
    # ISO-8601 UTC timestamp at which this file's vectors were evicted by the
    # janitor's forget sweep. The FileState entry itself (crucially its md5)
    # is KEPT: the surveyor decides what to re-embed by diffing md5, so an
    # entry that is popped instead of flagged comes back as "new" on the very
    # next tick and re-embeds — which is exactly how the vector store grew
    # back before. Empty string means "not forgotten". Editing the file
    # changes its md5 and it re-embeds normally, which is the intended way
    # back in.
    forgotten: str = ""


@dataclass
class ClusterState:
    cluster_id: int
    cluster_type: str = "semantic"
    label: list[str] = field(default_factory=list)
    member_files: list[str] = field(default_factory=list)
    last_labeled: str = ""
    consolidated_chunk_id: str = ""


@dataclass
class WikiPage:
    entity_name: str
    entity_type: str
    rel_path: str
    created: str
    updated: str
    sources: list[str] = field(default_factory=list)
    known_facts: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)


@dataclass
class GraphEdge:
    source: str
    target: str
    weight: float = 1.0
    edge_type: str = "wikilink"


@dataclass
class PipelineState:
    version: int = 1
    last_run: str = ""
    files: dict[str, FileState] = field(default_factory=dict)
    clusters: dict[str, ClusterState] = field(default_factory=dict)
    memory: dict[str, MemoryStrength] = field(default_factory=dict)
    wiki_pages: dict[str, WikiPage] = field(default_factory=dict)
    curator_processed: dict[str, str] = field(default_factory=dict)
    distiller_runs: list[dict[str, Any]] = field(default_factory=list)
    janitor_sweeps: list[dict[str, Any]] = field(default_factory=list)
    last_dedup: str = ""
    api_calls_today: int = 0
    api_calls_date: str = ""
    api_cost_usd_today: float = 0.0
    # ISO-8601 UTC timestamp until which all API calls are paused (set when a
    # recognized failure signature, e.g. credit exhaustion, is recorded).
    api_paused_until: str = ""
    # Cumulative count of swallowed failures, keyed by the stable dotted key
    # passed to alfred.core.failures.record_failure (e.g.
    # {"janitor.autofix_failed": 3}). Drained into here by StateStore.save()
    # so a handler whose body is `pass` still leaves a countable trace, and
    # "zero errors" becomes a checkable claim rather than an absence of
    # evidence. Monotonic — never reset by normal operation.
    error_counts: dict[str, int] = field(default_factory=dict)
