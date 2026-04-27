"""Context assembly: deduplicate hits by source file, build context window."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alfred.core.vault import chunk_record, parse_file
from alfred.store.milvus import SearchHit

MAX_CONTEXT_CHARS = 60_000
CHUNK_PREVIEW_CHARS = 200


@dataclass
class SourceRef:
    chunk_id: str
    rel_path: str
    score: float
    record_type: str
    name: str
    text_len: int = 0


def assemble(
    hits: list[SearchHit],
    vault_path: Path,
) -> tuple[str, list[SourceRef]]:
    """Keep best chunk per source file, assemble up to MAX_CONTEXT_CHARS."""
    # Deduplicate: keep highest-scoring chunk per rel_path
    by_file: dict[str, SearchHit] = {}
    for h in hits:
        if h.rel_path not in by_file or h.score > by_file[h.rel_path].score:
            by_file[h.rel_path] = h

    ranked = sorted(by_file.values(), key=lambda h: h.score, reverse=True)

    parts: list[str] = []
    sources: list[SourceRef] = []
    total = 0

    for hit in ranked:
        text = _chunk_text(vault_path, hit.chunk_id)
        if not text:
            continue
        if total + len(text) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total
            if remaining < 200:
                break
            text = text[:remaining] + "...[truncated]"
        block = f"[Source: {hit.rel_path}  score={hit.score:.3f}]\n{text}"
        parts.append(block)
        sources.append(SourceRef(
            chunk_id=hit.chunk_id,
            rel_path=hit.rel_path,
            score=hit.score,
            record_type=hit.record_type,
            name=hit.name,
            text_len=len(text),
        ))
        total += len(text)

    return "\n\n---\n\n".join(parts), sources


def chunk_preview(sources: list[SourceRef], vault_path: Path) -> dict[str, str]:
    """Return {rel_path: preview_text} for display."""
    previews: dict[str, str] = {}
    for src in sources:
        text = _chunk_text(vault_path, src.chunk_id) or ""
        previews[src.rel_path] = text[:CHUNK_PREVIEW_CHARS].replace("\n", " ")
    return previews


def _chunk_text(vault_path: Path, chunk_id: str) -> str | None:
    if "::" not in chunk_id:
        return None
    rel_path, chunk_part = chunk_id.rsplit("::", 1)
    try:
        idx = int(chunk_part.replace("chunk_", ""))
    except ValueError:
        return None
    full = vault_path / rel_path
    if not full.exists():
        return None
    try:
        record = parse_file(vault_path, rel_path)
        chunks = chunk_record(record)
        return chunks[idx][1] if idx < len(chunks) else None
    except Exception:
        return None
