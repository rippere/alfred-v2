"""Backend-neutral store types shared across stores, query, and daemons."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchHit:
    chunk_id: str
    rel_path: str
    score: float
    record_type: str = ""
    name: str = ""
    rerank_score: float = 0.0
