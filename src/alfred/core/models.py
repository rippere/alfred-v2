from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


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
