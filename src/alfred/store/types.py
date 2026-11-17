"""Backend-neutral store types shared across stores, query, and daemons."""
from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class SearchHit:
    chunk_id: str
    rel_path: str
    score: float
    record_type: str = ""
    name: str = ""
    rerank_score: float = 0.0


def representative_chunks(
    rows: Iterable[tuple[str, int | None]],
) -> dict[str, tuple[int, str]]:
    """Pick one chunk per file from ``(chunk_id, chunk_index)`` pairs.

    Chunk ids are ``<rel_path>::<suffix>``.  The representative is the row with
    the lowest ``chunk_index`` (normally chunk 0, the file's opening — title,
    frontmatter summary, first section); a null index sorts last, and ties keep
    the first row seen.  Returns ``{rel_path: (chunk_index, chunk_id)}``.
    """
    best: dict[str, tuple[int, str]] = {}
    for chunk_id, idx in rows:
        rel_path = chunk_id.rsplit("::", 1)[0]
        key = idx if idx is not None else sys.maxsize
        cur = best.get(rel_path)
        if cur is None or key < cur[0]:
            best[rel_path] = (key, chunk_id)
    return best
